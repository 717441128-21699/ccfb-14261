/** 图像质量指标 */

/** 峰值信噪比,灰度范围 0..1 */
export function psnr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let mse = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    mse += d * d;
  }
  mse /= a.length;
  if (mse === 0) return Infinity;
  return 10 * Math.log10(1 / mse);
}

/** 归一化互相关系数(用于比较两个 PSF 形状) */
export function ncc(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let ma = 0, mb = 0;
  for (let i = 0; i < a.length; i++) { ma += a[i]; mb += b[i]; }
  ma /= a.length; mb /= a.length;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < a.length; i++) {
    const xa = a[i] - ma, xb = b[i] - mb;
    num += xa * xb; da += xa * xa; db += xb * xb;
  }
  return num / Math.sqrt(da * db);
}
