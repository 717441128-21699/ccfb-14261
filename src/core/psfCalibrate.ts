/**
 * 从微孔/针孔光源照片提取 PSF:
 * 去背景 → 阈值分割 → 质心定位 → 亚像素对中裁窗 → 截断负值 →
 * 180° 旋转对称化(降噪声) → 归一化(和为 1)。
 */

export interface CalibrateResult {
  psf: Float32Array;
  size: number;
  /** 光源质心在原图中的位置 */
  cx: number;
  cy: number;
  peak: number;
  background: number;
}

/** 估计背景:网格采样的中位数 */
function estimateBackground(img: ArrayLike<number>, w: number, h: number): number {
  const samples: number[] = [];
  const step = Math.max(1, Math.floor(Math.min(w, h) / 32));
  for (let y = step; y < h; y += step * 2) {
    for (let x = step; x < w; x += step * 2) {
      samples.push(img[y * w + x]);
    }
  }
  samples.sort((a, b) => a - b);
  return samples[Math.floor(samples.length / 2)] ?? 0;
}

/**
 * @param img 灰度图(任意非负刻度),w×h 行优先
 * @param size 输出 PSF 边长(奇数),应能容纳 PSF 全部显著能量
 */
export function calibratePSF(
  img: Float32Array | Float64Array, w: number, h: number, size = 65,
): CalibrateResult {
  const bg = estimateBackground(img, w, h);

  // 找峰值与阈值
  let peak = 0;
  for (let i = 0; i < w * h; i++) peak = Math.max(peak, img[i] - bg);
  if (peak <= 0) throw new Error('图像中没有亮于背景的光源信号');
  const thr = bg + peak * 0.3;

  // 阈值以上的加权质心
  let sx = 0, sy = 0, sw = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = img[y * w + x] - thr;
      if (v > 0) { sx += x * v; sy += y * v; sw += v; }
    }
  }
  if (sw === 0) throw new Error('阈值分割失败,请检查光源图像');
  const cx = sx / sw, cy = sy / sw;

  // 以质心为中心、亚像素双线性重采样裁出 size×size 窗口
  const half = (size - 1) / 2;
  const psf = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const gx = cx + x - half, gy = cy + y - half;
      const x0 = Math.floor(gx), y0 = Math.floor(gy);
      const fx = gx - x0, fy = gy - y0;
      const at = (xx: number, yy: number) =>
        xx < 0 || yy < 0 || xx >= w || yy >= h ? bg : img[yy * w + xx];
      const v =
        at(x0, y0) * (1 - fx) * (1 - fy) +
        at(x0 + 1, y0) * fx * (1 - fy) +
        at(x0, y0 + 1) * (1 - fx) * fy +
        at(x0 + 1, y0 + 1) * fx * fy;
      psf[y * size + x] = Math.max(0, v - bg);
    }
  }

  // 窗外残余背景:用窗边缘均值再减一次
  let edge = 0, ne = 0;
  for (let i = 0; i < size; i++) {
    edge += psf[i] + psf[(size - 1) * size + i] + psf[i * size] + psf[i * size + size - 1];
    ne += 4;
  }
  edge /= ne;
  for (let i = 0; i < psf.length; i++) psf[i] = Math.max(0, psf[i] - edge);

  // 180° 旋转对称化,抑制方向性噪声
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      const j = (size - 1 - y) * size + (size - 1 - x);
      const m = (psf[i] + psf[j]) / 2;
      psf[i] = m; psf[j] = m;
    }
  }

  // 归一化
  let sum = 0;
  for (const v of psf) sum += v;
  if (sum <= 0) throw new Error('提取的 PSF 能量为零');
  for (let i = 0; i < psf.length; i++) psf[i] /= sum;

  return { psf, size, cx, cy, peak, background: bg };
}
