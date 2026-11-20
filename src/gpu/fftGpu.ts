/** GPU 二维 FFT:位反转置换 + Cooley-Tukey 蝶形,沿 x/y 两轴各做一遍 */
import { GL, Tex, FRAG_HEADER } from './gl';

const BITREV_FRAG = FRAG_HEADER + `
uniform sampler2D uIn;
uniform int uAxis;   // 0: 沿 x 反转, 1: 沿 y
uniform int uLogN;
out vec4 frag;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  int i = uAxis == 0 ? p.x : p.y;
  int r = int(bitfieldReverse(uint(i)) >> (32 - uLogN));
  ivec2 q = uAxis == 0 ? ivec2(r, p.y) : ivec2(p.x, r);
  frag = texelFetch(uIn, q, 0);
}
`;

const BUTTERFLY_FRAG = FRAG_HEADER + `
uniform sampler2D uIn;
uniform int uStage;  // 1..logN
uniform int uAxis;
uniform float uDir;  // -1 正变换, +1 逆变换
out vec4 frag;
const float PI = 3.141592653589793;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  int i = uAxis == 0 ? p.x : p.y;
  int m = 1 << uStage;
  int half_ = m >> 1;
  int pos = i & (m - 1);
  int base = i - pos;
  bool upper = pos >= half_;
  int k = upper ? pos - half_ : pos;
  int ia = base + k;
  int ib = ia + half_;
  ivec2 qa = uAxis == 0 ? ivec2(ia, p.y) : ivec2(p.x, ia);
  ivec2 qb = uAxis == 0 ? ivec2(ib, p.y) : ivec2(p.x, ib);
  vec2 A = texelFetch(uIn, qa, 0).rg;
  vec2 B = texelFetch(uIn, qb, 0).rg;
  float ang = uDir * 2.0 * PI * float(k) / float(m);
  vec2 w = vec2(cos(ang), sin(ang));
  vec2 t = vec2(w.x * B.x - w.y * B.y, w.x * B.y + w.y * B.x);
  vec2 o = upper ? A - t : A + t;
  frag = vec4(o, 0.0, 1.0);
}
`;

const SCALE_FRAG = FRAG_HEADER + `
uniform sampler2D uIn;
uniform float uScale;
out vec4 frag;
void main(){
  frag = texelFetch(uIn, ivec2(gl_FragCoord.xy), 0) * uScale;
}
`;

export class GpuFFT {
  readonly texA: Tex;
  readonly texB: Tex;
  private bitrevProg: WebGLProgram;
  private butterflyProg: WebGLProgram;
  private scaleProg: WebGLProgram;
  private logN: number;

  constructor(private g: GL, readonly n: number) {
    if ((n & (n - 1)) !== 0) throw new Error('FFT 尺寸必须是 2 的幂');
    this.logN = Math.log2(n);
    this.texA = g.tex(n, n);
    this.texB = g.tex(n, n);
    this.bitrevProg = g.program(BITREV_FRAG);
    this.butterflyProg = g.program(BUTTERFLY_FRAG);
    this.scaleProg = g.program(SCALE_FRAG);
  }

  private pass(prog: WebGLProgram, src: Tex, dst: Tex, set: (gl: WebGL2RenderingContext, p: WebGLProgram) => void): Tex {
    this.g.run(prog, dst, [src], set);
    return dst;
  }

  /** 一轴:位反转 + logN 级蝶形(+逆变换缩放) */
  private axis(cur: Tex, axis: 0 | 1, inverse: boolean): Tex {
    let src = cur;
    // 位反转
    const dst = src === this.texA ? this.texB : this.texA;
    this.pass(this.bitrevProg, src, dst, (gl, p) => {
      gl.uniform1i(gl.getUniformLocation(p, 'uIn'), 0);
      gl.uniform1i(gl.getUniformLocation(p, 'uAxis'), axis);
      gl.uniform1i(gl.getUniformLocation(p, 'uLogN'), this.logN);
    });
    src = dst;
    for (let s = 1; s <= this.logN; s++) {
      const d = src === this.texA ? this.texB : this.texA;
      this.pass(this.butterflyProg, src, d, (gl, p) => {
        gl.uniform1i(gl.getUniformLocation(p, 'uIn'), 0);
        gl.uniform1i(gl.getUniformLocation(p, 'uStage'), s);
        gl.uniform1i(gl.getUniformLocation(p, 'uAxis'), axis);
        gl.uniform1f(gl.getUniformLocation(p, 'uDir'), inverse ? 1 : -1);
      });
      src = d;
    }
    if (inverse) {
      const d = src === this.texA ? this.texB : this.texA;
      this.pass(this.scaleProg, src, d, (gl, p) => {
        gl.uniform1i(gl.getUniformLocation(p, 'uIn'), 0);
        gl.uniform1f(gl.getUniformLocation(p, 'uScale'), 1 / this.n);
      });
      src = d;
    }
    return src;
  }

  /**
   * 二维 FFT。src 若是 texA/texB 之一会被覆盖;否则先拷入。
   * 返回存放结果的纹理(texA 或 texB)。
   */
  fft2(src: Tex, inverse: boolean): Tex {
    let cur = src;
    if (src !== this.texA && src !== this.texB) {
      // 直接以 src 为输入做第一个位反转 pass,省一次拷贝
      const dst = this.texA;
      this.pass(this.bitrevProg, src, dst, (gl, p) => {
        gl.uniform1i(gl.getUniformLocation(p, 'uIn'), 0);
        gl.uniform1i(gl.getUniformLocation(p, 'uAxis'), 0);
        gl.uniform1i(gl.getUniformLocation(p, 'uLogN'), this.logN);
      });
      // 剩余蝶形
      let curT = dst;
      for (let s = 1; s <= this.logN; s++) {
        const d = curT === this.texA ? this.texB : this.texA;
        this.pass(this.butterflyProg, curT, d, (gl, p) => {
          gl.uniform1i(gl.getUniformLocation(p, 'uIn'), 0);
          gl.uniform1i(gl.getUniformLocation(p, 'uStage'), s);
          gl.uniform1i(gl.getUniformLocation(p, 'uAxis'), 0);
          gl.uniform1f(gl.getUniformLocation(p, 'uDir'), inverse ? 1 : -1);
        });
        curT = d;
      }
      if (inverse) {
        const d = curT === this.texA ? this.texB : this.texA;
        this.pass(this.scaleProg, curT, d, (gl, p) => {
          gl.uniform1i(gl.getUniformLocation(p, 'uIn'), 0);
          gl.uniform1f(gl.getUniformLocation(p, 'uScale'), 1 / this.n);
        });
        curT = d;
      }
      cur = curT;
    } else {
      cur = this.axis(src, 0, inverse);
    }
    cur = this.axis(cur, 1, inverse);
    return cur;
  }

  dispose(): void {
    this.g.deleteTex(this.texA);
    this.g.deleteTex(this.texB);
  }
}
