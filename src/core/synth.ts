/**
 * 合成测试图:模拟体视显微镜标本照中的典型结构——
 * 刚毛(细线)、触角节(串珠)、点阵、星形分辨率靶、平滑体壁渐变。
 * 确定性生成,作为反卷积验证的已知真值。
 */
import { mulberry32 } from './noise';
import { nextPow2, makeOTF, applyOTF } from './fft';

export function synthSpecimen(size: number, seed = 7): Float64Array {
  const img = new Float64Array(size * size).fill(0.08);
  const rand = mulberry32(seed);
  // 特征宽度单位:256 像素图下 1 单位 = 1 像素,随图尺寸缩放,
  // 模拟高像素照片里刚毛/触角节占数个像素的实际情况
  const u = size / 256;

  // 平滑背景起伏(体壁反光)
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      img[y * size + x] += 0.06 * Math.sin((x / size) * Math.PI * 2) * Math.cos((y / size) * Math.PI * 3);
    }
  }

  const drawDisk = (cx: number, cy: number, r: number, v: number) => {
    for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(size - 1, Math.ceil(cy + r)); y++) {
      for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(size - 1, Math.ceil(cx + r)); x++) {
        const d = Math.hypot(x - cx, y - cy);
        if (d <= r) {
          const i = y * size + x;
          img[i] = Math.max(img[i], v * (d > r - 1 ? r - d : 1));
        }
      }
    }
  };

  const drawLine = (x0: number, y0: number, x1: number, y1: number, width: number, v: number) => {
    const steps = Math.ceil(Math.hypot(x1 - x0, y1 - y0) * 2);
    for (let s = 0; s <= steps; s++) {
      const t = s / steps;
      drawDisk(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, width / 2, v);
    }
  };

  // 中央椭圆「虫体」
  const cx = size / 2, cy = size / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const e = ((x - cx) / (size * 0.3)) ** 2 + ((y - cy) / (size * 0.2)) ** 2;
      if (e < 1) img[y * size + x] = Math.max(img[y * size + x], 0.45 + 0.1 * (1 - e));
    }
  }

  // 刚毛:从虫体边缘向外辐射的细线(高对比)
  for (let k = 0; k < 28; k++) {
    const a = (k / 28) * Math.PI * 2;
    const r0x = Math.cos(a) * size * 0.28, r0y = Math.sin(a) * size * 0.18;
    const len = size * (0.08 + 0.06 * rand());
    drawLine(cx + r0x, cy + r0y, cx + r0x + Math.cos(a) * len, cy + r0y + Math.sin(a) * len, 1.5 * u, 0.95);
  }

  // 触角:一串递减的圆节
  let ax = cx + size * 0.28, ay = cy - size * 0.05;
  let ang = -0.5;
  for (let seg = 0; seg < 9; seg++) {
    const r = size * 0.02 * (1 - seg * 0.07);
    drawDisk(ax, ay, r, 0.7);
    ang += 0.12;
    ax += Math.cos(ang) * r * 2.4;
    ay += Math.sin(ang) * r * 2.4;
  }

  // 点阵(不同间距,检验分辨率)
  for (const [pitch, v] of [[4, 0.9], [7, 0.8], [11, 0.7]] as const) {
    const ox = size * 0.06, oy = size * (0.06 + pitch * 0.02);
    for (let y = oy; y < oy + size * 0.2; y += pitch * u) {
      for (let x = ox; x < ox + size * 0.2; x += pitch * u) {
        drawDisk(x, y, 1.2 * u, v);
      }
    }
  }

  // 星形分辨率靶(右下)
  const sx = size * 0.78, sy = size * 0.78, sr = size * 0.16;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const r = Math.hypot(x - sx, y - sy);
      if (r < sr) {
        const theta = Math.atan2(y - sy, x - sx);
        if (Math.sin(theta * 18) > 0) img[y * size + x] = Math.max(img[y * size + x], 0.85);
      }
    }
  }

  for (let i = 0; i < img.length; i++) img[i] = Math.min(1, Math.max(0, img[i]));
  return img;
}

/** 把真值按指定 PSF 模糊成观测图(测试用,边界反射延拓减小截断影响) */
export function blurImage(
  img: Float64Array, w: number, h: number,
  psf: Float64Array, pw: number, ph: number,
): Float64Array {
  const W = nextPow2(w + pw), H = nextPow2(h + ph);
  const refl = (i: number, n: number) => (i < n ? i : Math.max(0, 2 * n - 2 - i));
  const padded = new Float64Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      padded[y * W + x] = img[refl(y, h) * w + refl(x, w)];
    }
  }
  const otf = makeOTF(psf, pw, ph, W, H);
  const full = applyOTF(padded, W, H, otf.re, otf.im, W, H, false);
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) out[y * w + x] = full[y * W + x];
  }
  return out;
}
