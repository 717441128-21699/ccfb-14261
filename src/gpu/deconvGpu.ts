/**
 * GPU Richardson-Lucy 反卷积(RL-TV 耦合 + Biggs-Andrews 加速)。
 * 与 CPU 参考实现(src/core/deconv.ts)算法一致,卷积经 FFT 完成。
 */
import { GL, Tex, FRAG_HEADER } from './gl';
import { GpuFFT } from './fftGpu';

const COPY_FRAG = FRAG_HEADER + `
uniform sampler2D uIn;
out vec4 frag;
void main(){ frag = texelFetch(uIn, ivec2(gl_FragCoord.xy), 0); }
`;

const CMUL_FRAG = FRAG_HEADER + `
uniform sampler2D uA;
uniform sampler2D uB;
uniform int uConjB;
out vec4 frag;
void main(){
  vec2 a = texelFetch(uA, ivec2(gl_FragCoord.xy), 0).rg;
  vec2 b = texelFetch(uB, ivec2(gl_FragCoord.xy), 0).rg;
  if (uConjB == 1) b.y = -b.y;
  frag = vec4(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x, 0.0, 1.0);
}
`;

const RATIO_FRAG = FRAG_HEADER + `
uniform sampler2D uY;
uniform sampler2D uC;
uniform float uEps;
out vec4 frag;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  float c = max(texelFetch(uC, p, 0).r, uEps);
  frag = vec4(texelFetch(uY, p, 0).r / c, 0.0, 0.0, 1.0);
}
`;

const TVDIV_FRAG = FRAG_HEADER + `
uniform sampler2D uX;
uniform int uN;
out vec4 frag;
float at(ivec2 p){
  p = clamp(p, ivec2(0), ivec2(uN - 1));
  return texelFetch(uX, p, 0).r;
}
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  float c = at(p);
  float eps = 1e-4;
  float d1x = at(p + ivec2(1, 0)) - c, d1y = at(p + ivec2(0, 1)) - c;
  float d2x = c - at(p - ivec2(1, 0)), d2y = at(p + ivec2(-1, 1)) - at(p - ivec2(1, 0));
  float d3x = at(p + ivec2(1, -1)) - at(p - ivec2(0, 1)), d3y = c - at(p - ivec2(0, 1));
  float gx  = d1x / sqrt(d1x * d1x + d1y * d1y + eps * eps);
  float gy  = d1y / sqrt(d1x * d1x + d1y * d1y + eps * eps);
  float gxp = d2x / sqrt(d2x * d2x + d2y * d2y + eps * eps);
  float gyp = d3y / sqrt(d3x * d3x + d3y * d3y + eps * eps);
  frag = vec4((gx - gxp) + (gy - gyp), 0.0, 0.0, 1.0);
}
`;

// RL-TV 更新(MRT):out0 = yk,out1 = gk = yk - x
const UPDATE_FRAG = FRAG_HEADER + `
uniform sampler2D uX;
uniform sampler2D uG;
uniform sampler2D uDiv;
uniform float uTV;
layout(location=0) out vec4 outY;
layout(location=1) out vec4 outG;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  float x = texelFetch(uX, p, 0).r;
  float g = max(texelFetch(uG, p, 0).r, 0.0);
  float denom = max(1.0 - uTV * texelFetch(uDiv, p, 0).r, 0.1);
  float yk = max(x * g / denom, 0.0);
  outY = vec4(yk, 0.0, 0.0, 1.0);
  outG = vec4(yk - x, 0.0, 0.0, 1.0);
}
`;

// 归约第一级:out = (Σ gk·gPrev, Σ gPrev²)(2×2 块求和,尺寸减半)
const REDUCE_FIRST_FRAG = FRAG_HEADER + `
uniform sampler2D uA; // gk
uniform sampler2D uB; // gPrev
out vec4 frag;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy) * 2;
  vec2 s = vec2(0.0);
  for (int dy = 0; dy < 2; dy++)
    for (int dx = 0; dx < 2; dx++){
      float a = texelFetch(uA, p + ivec2(dx, dy), 0).r;
      float b = texelFetch(uB, p + ivec2(dx, dy), 0).r;
      s += vec2(a * b, b * b);
    }
  frag = vec4(s, 0.0, 1.0);
}
`;

const REDUCE_NEXT_FRAG = FRAG_HEADER + `
uniform sampler2D uIn;
out vec4 frag;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy) * 2;
  vec2 s = vec2(0.0);
  for (int dy = 0; dy < 2; dy++)
    for (int dx = 0; dx < 2; dx++)
      s += texelFetch(uIn, p + ivec2(dx, dy), 0).rg;
  frag = vec4(s, 0.0, 1.0);
}
`;

// Biggs-Andrews 外推:x' = max(yk + α·gk, 0)
const BA_APPLY_FRAG = FRAG_HEADER + `
uniform sampler2D uY;  // yk
uniform sampler2D uGk;
uniform float uAlpha;
out vec4 frag;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  float yk = texelFetch(uY, p, 0).r;
  float gk = texelFetch(uGk, p, 0).r;
  frag = vec4(max(yk + uAlpha * gk, 0.0), 0.0, 0.0, 1.0);
}
`;

// 打包为 RGBA8(历史快照/显示用)
const PACK8_FRAG = FRAG_HEADER + `
uniform sampler2D uIn;
uniform float uGain;
out vec4 frag;
void main(){
  float v = texelFetch(uIn, ivec2(gl_FragCoord.xy), 0).r * uGain;
  frag = vec4(vec3(clamp(v, 0.0, 1.0)), 1.0);
}
`;

export interface GpuDeconvOptions {
  iterations: number;
  tv: number;
  accel: number; // 0 关闭 BA 加速,否则为 α 上限(建议 0.9)
  /** 每次迭代后回调(可异步),iter 从 1 开始 */
  onIteration?: (iter: number) => void | Promise<void>;
}

export class GpuDeconvolver {
  readonly n: number;
  private g: GL;
  private fft: GpuFFT;
  private texOTF: Tex;
  private texY: Tex;
  private texX: Tex;
  private texC: Tex;
  private texR: Tex;
  private texDiv: Tex;
  private texYk: Tex;
  private texGk: Tex;
  private texGPrev: Tex;
  private reduceChain: Tex[] = [];
  private reduceFirst: WebGLProgram;
  private reduceNext: WebGLProgram;
  private copyProg: WebGLProgram;
  private cmulProg: WebGLProgram;
  private ratioProg: WebGLProgram;
  private tvdivProg: WebGLProgram;
  private updateProg: WebGLProgram;
  private baProg: WebGLProgram;
  private pack8Prog: WebGLProgram;

  constructor(g: GL, n: number) {
    this.g = g;
    this.n = n;
    this.fft = new GpuFFT(g, n);
    this.texOTF = g.tex(n, n);
    this.texY = g.tex(n, n);
    this.texX = g.tex(n, n);
    this.texC = g.tex(n, n);
    this.texR = g.tex(n, n);
    this.texDiv = g.tex(n, n);
    this.texYk = g.tex(n, n);
    this.texGk = g.tex(n, n);
    this.texGPrev = g.tex(n, n);
    this.copyProg = g.program(COPY_FRAG);
    this.cmulProg = g.program(CMUL_FRAG);
    this.ratioProg = g.program(RATIO_FRAG);
    this.tvdivProg = g.program(TVDIV_FRAG);
    this.updateProg = g.program(UPDATE_FRAG);
    this.baProg = g.program(BA_APPLY_FRAG);
    this.pack8Prog = g.program(PACK8_FRAG);
    this.reduceFirst = g.program(REDUCE_FIRST_FRAG);
    this.reduceNext = g.program(REDUCE_NEXT_FRAG);
  }

  /** 上传 PSF(和为 1,ps×ps)并预计算 OTF */
  setPSF(psf: Float32Array, ps: number): void {
    const n = this.n;
    const data = new Float32Array(n * n * 2);
    const c = Math.floor(ps / 2);
    for (let y = 0; y < ps; y++) {
      for (let x = 0; x < ps; x++) {
        const dx = (x - c + n) % n;
        const dy = (y - c + n) % n;
        data[(dy * n + dx) * 2] = psf[y * ps + x];
      }
    }
    const gl = this.g.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.fft.texA.tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, n, n, gl.RG, gl.FLOAT, data);
    const result = this.fft.fft2(this.fft.texA, false);
    this.copy(result, this.texOTF);
  }

  /** 上传观测图(已填充到 n×n 画布的实数数据) */
  setImage(canvasData: Float32Array): void {
    const n = this.n;
    const gl = this.g.gl;
    const rg = new Float32Array(n * n * 2);
    for (let i = 0; i < n * n; i++) rg[i * 2] = canvasData[i];
    gl.bindTexture(gl.TEXTURE_2D, this.texY.tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, n, n, gl.RG, gl.FLOAT, rg);
    // 初始估计 = 观测图
    this.copy(this.texY, this.texX);
    // gPrev 清零
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.texGPrev.fbo);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  private copy(src: Tex, dst: Tex): void {
    this.g.run(this.copyProg, dst, [src], (gl, p) => {
      gl.uniform1i(gl.getUniformLocation(p, 'uIn'), 0);
    });
  }

  private cmul(a: Tex, b: Tex, conjB: boolean, dst: Tex): void {
    this.g.run(this.cmulProg, dst, [a, b], (gl, p) => {
      gl.uniform1i(gl.getUniformLocation(p, 'uA'), 0);
      gl.uniform1i(gl.getUniformLocation(p, 'uB'), 1);
      gl.uniform1i(gl.getUniformLocation(p, 'uConjB'), conjB ? 1 : 0);
    });
  }

  /** 全局归约求 <gk, gPrev> 与 <gPrev, gPrev> */
  private reduceAlpha(accelMax: number): number {
    const g = this.g;
    const gl = g.gl;
    // 惰性建立归约链
    if (this.reduceChain.length === 0) {
      let size = this.n;
      while (size > 1) {
        size = Math.max(1, size >> 1);
        this.reduceChain.push(g.tex(size, size));
      }
    }
    g.run(this.reduceFirst, this.reduceChain[0], [this.texGk, this.texGPrev], (gl2, p) => {
      gl2.uniform1i(gl2.getUniformLocation(p, 'uA'), 0);
      gl2.uniform1i(gl2.getUniformLocation(p, 'uB'), 1);
    });
    for (let i = 1; i < this.reduceChain.length; i++) {
      g.run(this.reduceNext, this.reduceChain[i], [this.reduceChain[i - 1]], (gl2, p) => {
        gl2.uniform1i(gl2.getUniformLocation(p, 'uIn'), 0);
      });
    }
    const last = this.reduceChain[this.reduceChain.length - 1];
    const px = new Float32Array(4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, last.fbo);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const den = px[1];
    if (!(den > 0)) return 0;
    return Math.min(Math.max(px[0] / den, 0), accelMax);
  }

  /** 执行一次 RL-TV 迭代(含 BA 加速) */
  private iterate(tv: number, accel: number, first: boolean): void {
    const g = this.g;
    // c = H*x:FFT(x) → ×OTF → IFFT
    let t = this.fft.fft2(this.texX, false);
    this.cmul(t, this.texOTF, false, t === this.fft.texA ? this.fft.texB : this.fft.texA);
    t = this.fft.fft2(t === this.fft.texA ? this.fft.texB : this.fft.texA, true);
    this.copy(t, this.texC);
    // r = y / max(c, eps)
    g.run(this.ratioProg, this.texR, [this.texY, this.texC], (gl, p) => {
      gl.uniform1i(gl.getUniformLocation(p, 'uY'), 0);
      gl.uniform1i(gl.getUniformLocation(p, 'uC'), 1);
      gl.uniform1f(gl.getUniformLocation(p, 'uEps'), 1e-6);
    });
    // g = H^T r
    t = this.fft.fft2(this.texR, false);
    this.cmul(t, this.texOTF, true, t === this.fft.texA ? this.fft.texB : this.fft.texA);
    t = this.fft.fft2(t === this.fft.texA ? this.fft.texB : this.fft.texA, true);
    this.copy(t, this.texC); // 复用 texC 存放相关结果
    // TV 散度
    if (tv > 0) {
      g.run(this.tvdivProg, this.texDiv, [this.texX], (gl, p) => {
        gl.uniform1i(gl.getUniformLocation(p, 'uX'), 0);
        gl.uniform1i(gl.getUniformLocation(p, 'uN'), this.n);
      });
    }
    // RL-TV 更新 → yk, gk
    g.runMRT(this.updateProg, [this.texYk, this.texGk], [this.texX, this.texC, tv > 0 ? this.texDiv : this.texX], (gl, p) => {
      gl.uniform1i(gl.getUniformLocation(p, 'uX'), 0);
      gl.uniform1i(gl.getUniformLocation(p, 'uG'), 1);
      gl.uniform1i(gl.getUniformLocation(p, 'uDiv'), 2);
      gl.uniform1f(gl.getUniformLocation(p, 'uTV'), tv);
    });
    if (accel > 0 && !first) {
      const alpha = this.reduceAlpha(accel);
      g.run(this.baProg, this.texX, [this.texYk, this.texGk], (gl, p) => {
        gl.uniform1i(gl.getUniformLocation(p, 'uY'), 0);
        gl.uniform1i(gl.getUniformLocation(p, 'uGk'), 1);
        gl.uniform1f(gl.getUniformLocation(p, 'uAlpha'), alpha);
      });
    } else {
      this.copy(this.texYk, this.texX);
    }
    // gk → gPrev
    this.copy(this.texGk, this.texGPrev);
  }

  async run(opts: GpuDeconvOptions): Promise<void> {
    for (let it = 1; it <= opts.iterations; it++) {
      this.iterate(opts.tv, opts.accel, it === 1);
      if (opts.onIteration) await opts.onIteration(it);
    }
  }

  /** 把当前估计打包到 RGBA8 纹理(历史快照) */
  snapshotTo(dst: Tex, gain = 1): void {
    this.g.run(this.pack8Prog, dst, [this.texX], (gl, p) => {
      gl.uniform1i(gl.getUniformLocation(p, 'uIn'), 0);
      gl.uniform1f(gl.getUniformLocation(p, 'uGain'), gain);
    });
  }

  /** 把任意实数纹理(R 通道)显示到指定 2D 画布 */
  display(src: Tex, canvas: HTMLCanvasElement, gain = 1): void {
    canvas.width = src.w;
    canvas.height = src.h;
    const tmp = this.g.tex(src.w, src.h, this.g.gl.RGBA8);
    this.g.run(this.pack8Prog, tmp, [src], (gl, p) => {
      gl.uniform1i(gl.getUniformLocation(p, 'uIn'), 0);
      gl.uniform1f(gl.getUniformLocation(p, 'uGain'), gain);
    });
    const px = this.g.readBytes(tmp);
    this.g.deleteTex(tmp);
    const ctx = canvas.getContext('2d')!;
    // 纹理行序与图像行序一致,无需翻转
    const img = new ImageData(new Uint8ClampedArray(px.buffer.slice(0)), src.w, src.h);
    ctx.putImageData(img, 0, 0);
  }

  /** 读出当前估计(实部,n×n) */
  readResult(): Float32Array {
    // 打包到 RGBA32F 再读,保证 readPixels 格式组合合法
    const pack = this.g.tex(this.n, this.n, this.g.gl.RGBA32F);
    this.g.run(this.copyProg, pack, [this.texX], (gl, p) => {
      gl.uniform1i(gl.getUniformLocation(p, 'uIn'), 0);
    });
    const rgba = this.g.readFloats(pack);
    this.g.deleteTex(pack);
    const out = new Float32Array(this.n * this.n);
    for (let i = 0; i < out.length; i++) out[i] = rgba[i * 4];
    return out;
  }

  get estimateTex(): Tex {
    return this.texX;
  }

  dispose(): void {
    this.fft.dispose();
    for (const t of [this.texOTF, this.texY, this.texX, this.texC, this.texR, this.texDiv, this.texYk, this.texGk, this.texGPrev, ...this.reduceChain]) {
      this.g.deleteTex(t);
    }
  }
}
