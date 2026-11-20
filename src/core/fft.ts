/**
 * 基-2 复数 FFT(行优先二维,就地)。仅供 CPU 参考实现与测试使用。
 * 约定:正变换符号为 exp(-2πi·k/N),逆变换 exp(+2πi·k/N) 并除以 N。
 */

export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

function fft1d(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length;
  // 位反转置换
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (sign * 2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cwr = 1, cwi = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = i + k + half;
        const br = re[b] * cwr - im[b] * cwi;
        const bi = re[b] * cwi + im[b] * cwr;
        re[b] = re[a] - br; im[b] = im[a] - bi;
        re[a] += br; im[a] += bi;
        const nwr = cwr * wr - cwi * wi;
        cwi = cwr * wi + cwi * wr;
        cwr = nwr;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }
}

/** 二维 FFT,re/im 为 w*h 行优先数组,w、h 均须为 2 的幂。 */
export function fft2(re: Float64Array, im: Float64Array, w: number, h: number, inverse: boolean): void {
  const rowRe = new Float64Array(w);
  const rowIm = new Float64Array(w);
  for (let y = 0; y < h; y++) {
    const off = y * w;
    for (let x = 0; x < w; x++) { rowRe[x] = re[off + x]; rowIm[x] = im[off + x]; }
    fft1d(rowRe, rowIm, inverse);
    for (let x = 0; x < w; x++) { re[off + x] = rowRe[x]; im[off + x] = rowIm[x]; }
  }
  const colRe = new Float64Array(h);
  const colIm = new Float64Array(h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) { colRe[y] = re[y * w + x]; colIm[y] = im[y * w + x]; }
    fft1d(colRe, colIm, inverse);
    for (let y = 0; y < h; y++) { re[y * w + x] = colRe[y]; im[y * w + x] = colIm[y]; }
  }
}

/**
 * 由 PSF 构造 OTF(光学传递函数),尺寸 W×H(2 的幂)。
 * PSF 以其中心为原点,按 ifftshift 方式环绕放置。
 */
export function makeOTF(
  psf: Float64Array | Float32Array, pw: number, ph: number, W: number, H: number,
): { re: Float64Array; im: Float64Array } {
  const re = new Float64Array(W * H);
  const im = new Float64Array(W * H);
  const cx = Math.floor(pw / 2), cy = Math.floor(ph / 2);
  for (let y = 0; y < ph; y++) {
    for (let x = 0; x < pw; x++) {
      const v = psf[y * pw + x];
      if (v === 0) continue;
      const dx = (x - cx + W) % W;
      const dy = (y - cy + H) % H;
      re[dy * W + dx] = v;
    }
  }
  fft2(re, im, W, H, false);
  return { re, im };
}

/** 用 OTF 对实图像做循环卷积(correlate=true 时为相关,即与翻转 PSF 卷积)。 */
export function applyOTF(
  img: Float64Array, w: number, h: number,
  otfRe: Float64Array, otfIm: Float64Array,
  W: number, H: number, correlate: boolean,
): Float64Array {
  const re = new Float64Array(W * H);
  const im = new Float64Array(W * H);
  re.set(img);
  fft2(re, im, W, H, false);
  for (let i = 0; i < re.length; i++) {
    const ar = re[i], ai = im[i];
    const br = otfRe[i], bi = correlate ? -otfIm[i] : otfIm[i];
    re[i] = ar * br - ai * bi;
    im[i] = ar * bi + ai * br;
  }
  fft2(re, im, W, H, true);
  return re;
}
