/** 图像读写:文件 → 灰度浮点图;浮点图 → PNG 下载 */

export interface GrayImage {
  data: Float32Array;
  w: number;
  h: number;
}

/** 解码图像文件并转为 0..1 灰度(Rec.709 亮度) */
export async function loadGrayImage(file: File | Blob): Promise<GrayImage> {
  const bmp = await createImageBitmap(file);
  const canvas = document.createElement('canvas');
  canvas.width = bmp.width;
  canvas.height = bmp.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const n = canvas.width * canvas.height;
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    data[i] = (0.2126 * rgba[i * 4] + 0.7152 * rgba[i * 4 + 1] + 0.0722 * rgba[i * 4 + 2]) / 255;
  }
  return { data, w: canvas.width, h: canvas.height };
}

/** 浮点灰度图 → canvas(可显示/导出) */
export function grayToCanvas(img: GrayImage): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = img.w;
  canvas.height = img.h;
  const ctx = canvas.getContext('2d')!;
  const out = new ImageData(img.w, img.h);
  for (let i = 0; i < img.w * img.h; i++) {
    const v = Math.max(0, Math.min(255, Math.round(img.data[i] * 255)));
    out.data[i * 4] = v;
    out.data[i * 4 + 1] = v;
    out.data[i * 4 + 2] = v;
    out.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
  return canvas;
}

/** 浮点灰度图 → 触发 PNG 下载 */
export async function downloadPng(img: GrayImage, filename: string): Promise<void> {
  const canvas = grayToCanvas(img);
  const blob = await new Promise<Blob | null>((r) => canvas.toBlob(r, 'image/png'));
  if (!blob) throw new Error('PNG 编码失败');
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
