/**
 * 用纯 TS 模拟 GPU FFT 着色器的 pass 调度(位反转 + 蝶形,与
 * src/gpu/fftGpu.ts 中 BITREV_FRAG / BUTTERFLY_FRAG 的索引运算逐一对应),
 * 与 CPU 参考 FFT 对拍,验证 GPU 端索引数学的正确性。
 */
import { describe, it, expect } from 'vitest';
import { fft2 } from '../src/core/fft';

function bitReverse(i: number, logN: number): number {
  let r = 0;
  for (let b = 0; b < logN; b++) {
    r = (r << 1) | ((i >> b) & 1);
  }
  return r;
}

/** 模拟 BUTTERFLY_FRAG 的一级 */
function butterflyPass(
  re: Float64Array, im: Float64Array,
  outRe: Float64Array, outIm: Float64Array,
  n: number, stage: number, axis: 0 | 1, dir: number,
): void {
  const m = 1 << stage;
  const half = m >> 1;
  for (let py = 0; py < n; py++) {
    for (let px = 0; px < n; px++) {
      const i = axis === 0 ? px : py;
      const pos = i & (m - 1);
      const base = i - pos;
      const upper = pos >= half;
      const k = upper ? pos - half : pos;
      const ia = base + k;
      const ib = ia + half;
      const ai = axis === 0 ? py * n + ia : ia * n + px;
      const bi = axis === 0 ? py * n + ib : ib * n + px;
      const ar = re[ai], ai_ = im[ai];
      const br = re[bi], bi_ = im[bi];
      const ang = (dir * 2 * Math.PI * k) / m;
      const wr = Math.cos(ang), wi = Math.sin(ang);
      const tr = wr * br - wi * bi_;
      const ti = wr * bi_ + wi * br;
      const oi = py * n + px;
      if (upper) {
        outRe[oi] = ar - tr;
        outIm[oi] = ai_ - ti;
      } else {
        outRe[oi] = ar + tr;
        outIm[oi] = ai_ + ti;
      }
    }
  }
}

/** 模拟 GpuFFT.fft2 的完整 pass 序列 */
function gpuFftSim(re: Float64Array, im: Float64Array, n: number, inverse: boolean): void {
  const logN = Math.log2(n);
  const dir = inverse ? 1 : -1;
  let curRe = re, curIm = im;
  let bufRe = new Float64Array(n * n), bufIm = new Float64Array(n * n);
  for (const axis of [0, 1] as const) {
    // 位反转 pass
    for (let py = 0; py < n; py++) {
      for (let px = 0; px < n; px++) {
        const i = axis === 0 ? px : py;
        const r = bitReverse(i, logN);
        const src = axis === 0 ? py * n + r : r * n + px;
        const dst = py * n + px;
        bufRe[dst] = curRe[src];
        bufIm[dst] = curIm[src];
      }
    }
    [curRe, bufRe] = [bufRe, curRe];
    [curIm, bufIm] = [bufIm, curIm];
    for (let s = 1; s <= logN; s++) {
      butterflyPass(curRe, curIm, bufRe, bufIm, n, s, axis, dir);
      [curRe, bufRe] = [bufRe, curRe];
      [curIm, bufIm] = [bufIm, curIm];
    }
    if (inverse) {
      for (let i = 0; i < n * n; i++) { curRe[i] /= n; curIm[i] /= n; }
    }
  }
  re.set(curRe);
  im.set(curIm);
}

describe('GPU FFT 索引数学(着色器模拟)', () => {
  it('正变换与 CPU 参考一致', () => {
    const n = 64;
    const re = new Float64Array(n * n);
    const im = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) {
      re[i] = Math.sin(i * 1.7) * Math.cos(i * 0.3);
      im[i] = Math.sin(i * 0.9);
    }
    const refRe = Float64Array.from(re);
    const refIm = Float64Array.from(im);
    fft2(refRe, refIm, n, n, false);
    gpuFftSim(re, im, n, false);
    let maxErr = 0;
    for (let i = 0; i < n * n; i++) {
      maxErr = Math.max(maxErr, Math.abs(re[i] - refRe[i]), Math.abs(im[i] - refIm[i]));
    }
    expect(maxErr).toBeLessThan(1e-8);
  });

  it('逆变换往返还原', () => {
    const n = 32;
    const re = new Float64Array(n * n);
    const im = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) re[i] = Math.cos(i * 0.77);
    const orig = Float64Array.from(re);
    gpuFftSim(re, im, n, false);
    gpuFftSim(re, im, n, true);
    let maxErr = 0;
    for (let i = 0; i < n * n; i++) maxErr = Math.max(maxErr, Math.abs(re[i] - orig[i]));
    expect(maxErr).toBeLessThan(1e-10);
  });
});
