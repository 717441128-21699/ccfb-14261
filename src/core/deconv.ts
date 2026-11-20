/**
 * Richardson-Lucy 反卷积(CPU 参考实现),带全变分(TV)正则抑制噪声与振铃。
 * 与 GPU 实现(src/gpu/deconvGpu.ts)算法一致,用于离线验证与小图处理。
 */
import { nextPow2, makeOTF, applyOTF } from './fft';

export interface DeconvOptions {
  iterations: number;
  /** TV 正则强度(RL-TV 耦合),0 关闭 */
  tv?: number;
  /** Biggs-Andrews 加速上限(0 关闭,建议 0.9) */
  accel?: number;
  /** 每次迭代后的回调,x 为裁剪回原尺寸的当前估计 */
  onIteration?: (x: Float64Array, iter: number) => void;
}

/**
 * 边缘渐隐:把距边界 t 像素内的区域替换为「用同一 PSF 模糊后的图像」,
 * 消除边界截断造成的振铃(MATLAB edgetaper 思路)。
 */
export function edgetaper(
  img: Float64Array, w: number, h: number,
  psf: Float64Array | Float32Array, pw: number, ph: number,
): Float64Array {
  const t = Math.max(Math.floor(pw / 2), Math.floor(ph / 2));
  const W = nextPow2(w), H = nextPow2(h);
  const otf = makeOTF(psf, pw, ph, W, H);
  const padded = new Float64Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      padded[y * W + x] = img[Math.min(y, h - 1) * w + Math.min(x, w - 1)];
    }
  }
  const blurred = applyOTF(padded, W, H, otf.re, otf.im, W, H, false);
  const out = new Float64Array(img);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const d = Math.min(x, y, w - 1 - x, h - 1 - y);
      if (d >= t) continue;
      // 余弦渐变:边界处 a=1(完全用模糊图),t 处 a=0
      const a = 0.5 + 0.5 * Math.cos((Math.PI * d) / t);
      const i = y * w + x;
      out[i] = img[i] * (1 - a) + blurred[y * W + x] * a;
    }
  }
  return out;
}

/** 各向同性 TV 的散度项 div(∇x/|∇x|),用于 RL-TV 耦合更新 */
function tvDivergence(x: Float64Array, w: number, h: number, eps = 1e-4): Float64Array {
  const at = (i: number, j: number) => x[Math.min(Math.max(i, 0), h - 1) * w + Math.min(Math.max(j, 0), w - 1)];
  const div = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let i = 0; i < w; i++) {
      const c = at(y, i);
      const gx = (at(y, i + 1) - c) / Math.hypot(at(y, i + 1) - c, at(y + 1, i) - c, eps);
      const gy = (at(y + 1, i) - c) / Math.hypot(at(y, i + 1) - c, at(y + 1, i) - c, eps);
      const gxPrev = (c - at(y, i - 1)) / Math.hypot(c - at(y, i - 1), at(y + 1, i - 1) - at(y, i - 1), eps);
      const gyPrev = (c - at(y - 1, i)) / Math.hypot(at(y - 1, i + 1) - at(y - 1, i), c - at(y - 1, i), eps);
      div[y * w + i] = (gx - gxPrev) + (gy - gyPrev);
    }
  }
  return div;
}

/**
 * Richardson-Lucy 反卷积。
 * @param obs 观测图像(0..1),w×h 行优先
 * @param psf 点扩散函数(和为 1),pw×ph
 */
export function richardsonLucy(
  obs: Float64Array, w: number, h: number,
  psf: Float64Array | Float32Array, pw: number, ph: number,
  opts: DeconvOptions,
): Float64Array {
  const tv = opts.tv ?? 0.002;
  const W = nextPow2(w), H = nextPow2(h);
  const otf = makeOTF(psf, pw, ph, W, H);

  // 边缘渐隐 + 复制边缘填充到 2 的幂画布
  const tapered = edgetaper(obs, w, h, psf, pw, ph);
  const y = new Float64Array(W * H);
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      y[j * W + i] = tapered[Math.min(j, h - 1) * w + Math.min(i, w - 1)];
    }
  }

  let x = Float64Array.from(y);
  const eps = 1e-6;
  const accel = opts.accel ?? 0.9;
  let gPrev: Float64Array | null = null; // 上一轮 RL 修正量
  const rlStep = (src: Float64Array): Float64Array => {
    // c = H*src
    const c = applyOTF(src, W, H, otf.re, otf.im, W, H, false);
    // r = y / max(c, eps)
    const r = new Float64Array(W * H);
    for (let i = 0; i < r.length; i++) r[i] = y[i] / Math.max(c[i], eps);
    // g = H^T * r(与翻转 PSF 相关)
    const g = applyOTF(r, W, H, otf.re, otf.im, W, H, true);
    const out = new Float64Array(src.length);
    // RL-TV 耦合更新(Dey et al.):x *= g / (1 - λ·div(∇x/|∇x|))
    if (tv > 0) {
      const div = tvDivergence(src, W, H);
      for (let i = 0; i < out.length; i++) {
        const denom = Math.max(1 - tv * div[i], 0.1);
        out[i] = Math.max((src[i] * Math.max(g[i], 0)) / denom, 0);
      }
    } else {
      for (let i = 0; i < out.length; i++) {
        out[i] = Math.max(src[i] * Math.max(g[i], 0), 0);
      }
    }
    return out;
  };
  for (let it = 0; it < opts.iterations; it++) {
    const yk = rlStep(x);
    if (accel > 0) {
      // Biggs-Andrews:α = <g_k, g_{k-1}> / <g_{k-1}, g_{k-1}>,g 为 RL 修正量
      const gk = new Float64Array(yk.length);
      let num = 0, den = 0;
      for (let i = 0; i < gk.length; i++) {
        gk[i] = yk[i] - x[i];
        if (gPrev) { num += gk[i] * gPrev[i]; den += gPrev[i] * gPrev[i]; }
      }
      const alpha = gPrev && den > 0 ? Math.min(Math.max(num / den, 0), accel) : 0;
      for (let i = 0; i < x.length; i++) {
        x[i] = Math.max(yk[i] + alpha * gk[i], 0);
      }
      gPrev = gk;
    } else {
      x = yk;
    }
    if (opts.onIteration) {
      const cropped = new Float64Array(w * h);
      for (let j = 0; j < h; j++) {
        for (let i = 0; i < w; i++) cropped[j * w + i] = x[j * W + i];
      }
      opts.onIteration(cropped, it + 1);
    }
  }
  const out = new Float64Array(w * h);
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) out[j * w + i] = x[j * W + i];
  }
  return out;
}
