/** WebGL2 基础设施:上下文、程序编译、浮点纹理、全屏三角形绘制 */

export interface Tex {
  tex: WebGLTexture;
  fbo: WebGLFramebuffer;
  w: number;
  h: number;
}

const VERT_SRC = `#version 300 es
layout(location=0) in vec2 aPos;
void main(){ gl_Position = vec4(aPos, 0.0, 1.0); }
`;

export class GL {
  readonly gl: WebGL2RenderingContext;
  private vertShader: WebGLShader;
  private progCache = new Map<string, WebGLProgram>();

  constructor(canvas?: HTMLCanvasElement) {
    const c = canvas ?? document.createElement('canvas');
    const gl = c.getContext('webgl2', {
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('当前浏览器不支持 WebGL2');
    if (!gl.getExtension('EXT_color_buffer_float')) {
      throw new Error('当前环境不支持浮点渲染目标(EXT_color_buffer_float)');
    }
    this.gl = gl;
    this.vertShader = this.compile(gl.VERTEX_SHADER, VERT_SRC);
    // 全屏三角形
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
  }

  private compile(type: number, src: string): WebGLShader {
    const gl = this.gl;
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error('着色器编译失败: ' + gl.getShaderInfoLog(sh) + '\n' + src);
    }
    return sh;
  }

  /** 按源码缓存编译片元程序 */
  program(fragSrc: string): WebGLProgram {
    let p = this.progCache.get(fragSrc);
    if (p) return p;
    const gl = this.gl;
    p = gl.createProgram()!;
    gl.attachShader(p, this.vertShader);
    gl.attachShader(p, this.compile(gl.FRAGMENT_SHADER, fragSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error('程序链接失败: ' + gl.getProgramInfoLog(p));
    }
    this.progCache.set(fragSrc, p);
    return p;
  }

  /** 创建纹理+帧缓冲。internalFormat 默认 RG32F(复数/实数计算用) */
  tex(w: number, h: number, internalFormat?: number, data?: ArrayBufferView | null): Tex {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const ifmt = internalFormat ?? gl.RG32F;
    const { format, type } = ifmt === gl.RGBA8
      ? { format: gl.RGBA, type: gl.UNSIGNED_BYTE }
      : ifmt === gl.RGBA32F
        ? { format: gl.RGBA, type: gl.FLOAT }
        : { format: gl.RG, type: gl.FLOAT };
    gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, format, type, data ?? null);
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`帧缓冲不完整(尺寸 ${w}×${h})`);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { tex, fbo, w, h };
  }

  deleteTex(t: Tex): void {
    this.gl.deleteTexture(t.tex);
    this.gl.deleteFramebuffer(t.fbo);
  }

  /** 把程序渲染到目标纹理(target 为 null 时渲染到画布) */
  run(prog: WebGLProgram, target: Tex | null, inputs: Tex[], uniforms?: (gl: WebGL2RenderingContext, p: WebGLProgram) => void): void {
    const gl = this.gl;
    gl.useProgram(prog);
    for (let i = 0; i < inputs.length; i++) {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, inputs[i].tex);
    }
    if (uniforms) uniforms(gl, prog);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fbo : null);
    gl.viewport(0, 0, target ? target.w : gl.drawingBufferWidth, target ? target.h : gl.drawingBufferHeight);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** MRT 版本:渲染到两个目标纹理 */
  runMRT(prog: WebGLProgram, targets: [Tex, Tex], inputs: Tex[], uniforms?: (gl: WebGL2RenderingContext, p: WebGLProgram) => void): void {
    const gl = this.gl;
    gl.useProgram(prog);
    for (let i = 0; i < inputs.length; i++) {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, inputs[i].tex);
    }
    if (uniforms) uniforms(gl, prog);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, targets[0].tex, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, targets[1].tex, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, targets[0].w, targets[0].h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.deleteFramebuffer(fbo);
  }

  readFloats(t: Tex): Float32Array {
    const gl = this.gl;
    const out = new Float32Array(t.w * t.h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
    gl.readPixels(0, 0, t.w, t.h, gl.RGBA, gl.FLOAT, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return out;
  }

  readBytes(t: Tex): Uint8Array {
    const gl = this.gl;
    const out = new Uint8Array(t.w * t.h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
    gl.readPixels(0, 0, t.w, t.h, gl.RGBA, gl.UNSIGNED_BYTE, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return out;
  }
}

export const FRAG_HEADER = `#version 300 es
precision highp float;
precision highp int;
`;
