/** 确定性随机数与噪声模型(高斯读出噪声 + 泊松光子噪声) */

/** mulberry32 确定性 PRNG,便于测试复现 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 标准正态(Box-Muller) */
export function gaussianSample(rand: () => number): number {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

/** 泊松采样(λ<30 用 Knuth,否则正态近似) */
export function poissonSample(lambda: number, rand: () => number): number {
  if (lambda <= 0) return 0;
  if (lambda < 30) {
    const L = Math.exp(-lambda);
    let k = 0, p = 1;
    do { k++; p *= rand(); } while (p > L);
    return k - 1;
  }
  return Math.max(0, Math.round(lambda + Math.sqrt(lambda) * gaussianSample(rand)));
}

export interface NoiseOptions {
  /** 高斯读出噪声 σ(0..1 灰度单位) */
  gaussianSigma?: number;
  /** 光子数峰值(满阱),>0 时叠加泊松噪声 */
  photonPeak?: number;
  seed?: number;
}

export function addNoise(img: Float64Array, opts: NoiseOptions): Float64Array {
  const rand = mulberry32(opts.seed ?? 42);
  const out = new Float64Array(img.length);
  const sigma = opts.gaussianSigma ?? 0;
  const peak = opts.photonPeak ?? 0;
  for (let i = 0; i < img.length; i++) {
    let v = img[i];
    if (peak > 0) v = poissonSample(Math.max(v, 0) * peak, rand) / peak;
    if (sigma > 0) v += sigma * gaussianSample(rand);
    out[i] = v;
  }
  return out;
}
