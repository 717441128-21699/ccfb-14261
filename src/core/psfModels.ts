/** 理论 PSF 模型,用于生成测试数据与作为实测 PSF 的初值。输出和为 1。 */

function normalize(psf: Float64Array): Float64Array {
  let s = 0;
  for (const v of psf) s += v;
  if (s > 0) for (let i = 0; i < psf.length; i++) psf[i] /= s;
  return psf;
}

/** 高斯 PSF(离焦/衍射的常用近似) */
export function gaussianPSF(sigma: number, size?: number): { psf: Float64Array; size: number } {
  const n = size ?? (Math.ceil(sigma * 6) | 1);
  const p = new Float64Array(n * n);
  const c = (n - 1) / 2;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const r2 = (x - c) ** 2 + (y - c) ** 2;
      p[y * n + x] = Math.exp(-r2 / (2 * sigma * sigma));
    }
  }
  return { psf: normalize(p), size: n };
}

/** 第一类一阶贝塞尔函数(级数) */
function besselJ1(x: number): number {
  let sum = 0;
  let term = x / 2;
  const xx = (x * x) / 4;
  for (let k = 0; k < 30; k++) {
    if (k > 0) term *= -xx / (k * (k + 1));
    sum += term;
    if (Math.abs(term) < 1e-16) break;
  }
  return sum;
}

/** 艾里斑 PSF(衍射极限),firstZero 为第一暗环半径(像素) */
export function airyPSF(firstZero: number, size?: number): { psf: Float64Array; size: number } {
  const n = size ?? (Math.ceil(firstZero * 4) | 1);
  const p = new Float64Array(n * n);
  const c = (n - 1) / 2;
  // 第一零点在 x = 3.8317
  const scale = 3.8317 / firstZero;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const r = Math.hypot(x - c, y - c) * scale;
      const v = r < 1e-9 ? 1 : (2 * besselJ1(r)) / r;
      p[y * n + x] = v * v;
    }
  }
  return { psf: normalize(p), size: n };
}

/** 匀速直线运动模糊 PSF */
export function motionPSF(len: number, angleDeg: number): { psf: Float64Array; size: number } {
  const n = Math.ceil(len) + 4;
  const p = new Float64Array(n * n);
  const c = (n - 1) / 2;
  const a = (angleDeg * Math.PI) / 180;
  const dx = Math.cos(a), dy = Math.sin(a);
  const half = (len - 1) / 2;
  for (let t = -half; t <= half; t += 0.25) {
    const x = Math.round(c + dx * t);
    const y = Math.round(c + dy * t);
    p[y * n + x] += 1;
  }
  return { psf: normalize(p), size: n };
}

/** 圆盘 PSF(纯离焦) */
export function diskPSF(radius: number): { psf: Float64Array; size: number } {
  const n = Math.ceil(radius * 2) + 3;
  const p = new Float64Array(n * n);
  const c = (n - 1) / 2;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      if (Math.hypot(x - c, y - c) <= radius) p[y * n + x] = 1;
    }
  }
  return { psf: normalize(p), size: n };
}
