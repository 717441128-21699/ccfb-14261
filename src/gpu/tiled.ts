/**
 * 分块反卷积驱动:把任意大小的图像切成带重叠边缘的块,
 * 每块在 2 的幂画布上做 GPU RL-TV,中心区域拼回。
 * 预览模式在较小画布上运行并保存每轮迭代快照供回看。
 */
import { GL, Tex } from './gl';
import { GpuDeconvolver } from './deconvGpu';

export interface TiledOptions {
  iterations: number;
  tv: number;
  accel: number;
  /** FFT 画布尺寸(2 的幂),默认 4096,显存不足时调用方降到 2048 */
  canvasSize?: number;
  /** 进度回调:(已完成迭代数, 总迭代数, 当前块/总块数) */
  onProgress?: (done: number, total: number, tile: number, tiles: number) => void;
  /** 每轮迭代后让出主线程的机会 */
  yieldEvery?: number;
}

function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/** 把 img 的 [x0,x0+coreW)×[y0,y0+coreH) 区域连同 margin 复制边缘填充进 n×n 画布 */
function extractTile(
  img: Float32Array, w: number, h: number,
  x0: number, y0: number, coreW: number, coreH: number,
  margin: number, n: number,
): Float32Array {
  const out = new Float32Array(n * n);
  const at = (x: number, y: number) =>
    img[Math.min(Math.max(y, 0), h - 1) * w + Math.min(Math.max(x, 0), w - 1)];
  for (let y = 0; y < n; y++) {
    const sy = y0 - margin + y;
    for (let x = 0; x < n; x++) {
      out[y * n + x] = at(x0 - margin + x, sy);
    }
  }
  return out;
}

/**
 * 全分辨率分块反卷积。
 * @returns 与输入同尺寸的复原结果
 */
export async function deconvolveTiled(
  gl: GL,
  img: Float32Array, w: number, h: number,
  psf: Float32Array, ps: number,
  opts: TiledOptions,
): Promise<Float32Array> {
  const margin = ps + 64; // 重叠边缘:容纳 PSF 支撑区 + 振铃余量
  const needed = Math.max(w, h) + 2 * margin;
  const n = Math.min(opts.canvasSize ?? 4096, nextPow2(needed));
  if (n < 256) throw new Error('画布尺寸过小');
  const core = n - 2 * margin;

  const tilesX = Math.ceil(w / core);
  const tilesY = Math.ceil(h / core);
  const tiles = tilesX * tilesY;
  const total = tiles * opts.iterations;
  let done = 0;

  const out = new Float32Array(w * h);
  const dc = new GpuDeconvolver(gl, n);
  try {
    dc.setPSF(psf, ps);
    let tileIdx = 0;
    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        tileIdx++;
        const x0 = tx * core, y0 = ty * core;
        const coreW = Math.min(core, w - x0);
        const coreH = Math.min(core, h - y0);
        const tile = extractTile(img, w, h, x0, y0, coreW, coreH, margin, n);
        dc.setImage(tile);
        await dc.run({
          iterations: opts.iterations,
          tv: opts.tv,
          accel: opts.accel,
          onIteration: () => {
            done++;
            opts.onProgress?.(done, total, tileIdx, tiles);
          },
        });
        const res = dc.readResult();
        // 中心区域写回(纹理行序与图像行序一致,无需翻转)
        for (let y = 0; y < coreH; y++) {
          const srcRow = margin + y;
          for (let x = 0; x < coreW; x++) {
            out[(y0 + y) * w + (x0 + x)] = res[srcRow * n + margin + x];
          }
        }
        if (opts.yieldEvery !== 0) {
          await new Promise((r) => setTimeout(r, 0));
        }
      }
    }
  } finally {
    dc.dispose();
  }
  return out;
}

export interface PreviewResult {
  /** 每轮迭代的快照(RGBA8 纹理),下标即迭代序号-1 */
  history: Tex[];
  /** 预览画布尺寸 */
  n: number;
  deconvolver: GpuDeconvolver;
}

/**
 * 预览反卷积:图像缩放到 ≤maxN 的画布,保存每轮迭代快照。
 * 调用方负责 dispose 返回的 deconvolver 与 history 纹理。
 */
export async function deconvolvePreview(
  gl: GL,
  img: Float32Array, w: number, h: number,
  psf: Float32Array, ps: number,
  opts: { iterations: number; tv: number; accel: number; maxN?: number; onIteration?: (iter: number) => void | Promise<void> },
): Promise<PreviewResult> {
  const maxN = opts.maxN ?? 1024;
  const scale = Math.min(1, maxN / Math.max(w, h));
  const pw = Math.max(1, Math.round(w * scale));
  const ph = Math.max(1, Math.round(h * scale));
  const n = nextPow2(Math.max(pw, ph));

  // 简单盒式缩小到 n×n 画布(图像置于中心,边缘复制填充)
  const canvas = new Float32Array(n * n);
  for (let y = 0; y < n; y++) {
    const sy = Math.min(h - 1, Math.floor((y / n) * ph / scale)) ;
    for (let x = 0; x < n; x++) {
      const sx = Math.min(w - 1, Math.floor((x / n) * pw / scale));
      canvas[y * n + x] = img[Math.max(0, sy) * w + Math.max(0, sx)];
    }
  }

  const dc = new GpuDeconvolver(gl, n);
  dc.setPSF(psf, ps);
  dc.setImage(canvas);
  const history: Tex[] = [];
  await dc.run({
    iterations: opts.iterations,
    tv: opts.tv,
    accel: opts.accel,
    onIteration: async (iter) => {
      const snap = gl.tex(n, n, gl.gl.RGBA8);
      dc.snapshotTo(snap);
      history.push(snap);
      if (opts.onIteration) await opts.onIteration(iter);
    },
  });
  return { history, n, deconvolver: dc };
}
