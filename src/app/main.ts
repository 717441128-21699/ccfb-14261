/** 主界面逻辑 */
import { GL } from '../gpu/gl';
import { GpuDeconvolver } from '../gpu/deconvGpu';
import { deconvolveTiled, deconvolvePreview, type PreviewResult } from '../gpu/tiled';
import { calibratePSF, type CalibrateResult } from '../core/psfCalibrate';
import { richardsonLucy } from '../core/deconv';
import { gaussianPSF } from '../core/psfModels';
import { psnr } from '../core/metrics';
import { savePsf, listPsfs, deletePsf, findPsf, type PsfRecord } from './psfStore';
import { loadGrayImage, grayToCanvas, downloadPng, type GrayImage } from './imageIO';
import { ProcessQueue } from './queue';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ---------- 页签 ----------
for (const btn of document.querySelectorAll<HTMLButtonElement>('nav .tab')) {
  btn.onclick = () => {
    document.querySelectorAll('nav .tab').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    btn.classList.add('active');
    $('tab-' + btn.dataset.tab).classList.add('active');
  };
}

// ---------- GL 惰性初始化 ----------
let glCtx: GL | null = null;
function getGL(): GL {
  if (!glCtx) glCtx = new GL();
  return glCtx;
}

// ---------- PSF 库 ----------
let psfLib: PsfRecord[] = [];

async function refreshLibrary(): Promise<void> {
  psfLib = await listPsfs();
  // 设备下拉
  const microscopes = [...new Set(psfLib.map((r) => r.microscope))];
  const selM = $<HTMLSelectElement>('sel-microscope');
  selM.innerHTML = microscopes.map((m) => `<option>${m}</option>`).join('');
  refreshMagnifications();
  // 库表格
  const tbody = $('psf-table').querySelector('tbody')!;
  tbody.innerHTML = '';
  for (const r of psfLib) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${r.microscope}</td><td>${r.magnification}</td><td>${r.size}×${r.size}</td>` +
      `<td>${new Date(r.createdAt).toLocaleString()}</td><td>${r.notes ?? ''}</td>`;
    const td = document.createElement('td');
    const del = document.createElement('button');
    del.textContent = '删除';
    del.onclick = async () => {
      await deletePsf(r.id!);
      await refreshLibrary();
    };
    td.appendChild(del);
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
  updatePsfMatchStatus();
}

function refreshMagnifications(): void {
  const m = $<HTMLSelectElement>('sel-microscope').value;
  const mags = psfLib.filter((r) => r.microscope === m).map((r) => r.magnification);
  $<HTMLSelectElement>('sel-magnification').innerHTML = mags.map((x) => `<option>${x}</option>`).join('');
}

function currentPsf(): PsfRecord | undefined {
  const m = $<HTMLSelectElement>('sel-microscope').value;
  const mag = $<HTMLSelectElement>('sel-magnification').value;
  return psfLib.find((r) => r.microscope === m && r.magnification === mag);
}

function updatePsfMatchStatus(): void {
  const el = $('psf-match-status');
  const rec = currentPsf();
  if (rec) {
    el.textContent = `✓ 已匹配 PSF(${rec.size}×${rec.size},${new Date(rec.createdAt).toLocaleDateString()} 建档)`;
    el.className = 'hint ok';
  } else {
    el.textContent = '✗ 该设备/倍率尚未校准,请先到「PSF 校准」建档';
    el.className = 'hint bad';
  }
}

$('sel-microscope').addEventListener('change', () => {
  refreshMagnifications();
  updatePsfMatchStatus();
});
$('sel-magnification').addEventListener('change', updatePsfMatchStatus);

// ---------- PSF 校准 ----------
let calSource: GrayImage | null = null;
let calResult: CalibrateResult | null = null;

$('cal-file').addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  calSource = await loadGrayImage(file);
  calResult = null;
  $('btn-save-psf').toggleAttribute('disabled', true);
  const canvas = $<HTMLCanvasElement>('cal-source-canvas');
  const src = grayToCanvas(calSource);
  canvas.width = src.width;
  canvas.height = src.height;
  canvas.getContext('2d')!.drawImage(src, 0, 0);
  $('cal-status').textContent = `已载入 ${calSource.w}×${calSource.h}`;
});

$('btn-extract-psf').addEventListener('click', () => {
  if (!calSource) return;
  const size = Math.max(15, Number($<HTMLInputElement>('cal-size').value) | 1);
  try {
    calResult = calibratePSF(calSource.data, calSource.w, calSource.h, size);
  } catch (err) {
    $('cal-status').textContent = String(err);
    return;
  }
  // 画 PSF 热图(对数拉伸)
  const canvas = $<HTMLCanvasElement>('cal-psf-canvas');
  canvas.width = size;
  canvas.height = size;
  const img = new ImageData(size, size);
  let peak = 0;
  for (const v of calResult.psf) peak = Math.max(peak, v);
  for (let i = 0; i < size * size; i++) {
    const v = Math.log10(1 + (calResult.psf[i] / peak) * 999) / 3; // 0..1
    img.data[i * 4] = Math.round(255 * Math.min(1, v * 2));
    img.data[i * 4 + 1] = Math.round(255 * Math.min(1, v * 1.2));
    img.data[i * 4 + 2] = Math.round(255 * v);
    img.data[i * 4 + 3] = 255;
  }
  canvas.getContext('2d')!.putImageData(img, 0, 0);
  $('cal-status').textContent =
    `提取成功:质心 (${calResult.cx.toFixed(1)}, ${calResult.cy.toFixed(1)}),峰值 ${calResult.peak.toFixed(1)},背景 ${calResult.background.toFixed(1)}`;
  $('btn-save-psf').toggleAttribute('disabled', false);
});

$('btn-save-psf').addEventListener('click', async () => {
  if (!calResult) return;
  const microscope = $<HTMLInputElement>('cal-microscope').value.trim();
  const magnification = $<HTMLInputElement>('cal-magnification').value.trim();
  if (!microscope || !magnification) {
    $('cal-status').textContent = '请填写显微镜与倍率';
    return;
  }
  await savePsf({
    microscope,
    magnification,
    size: calResult.size,
    data: calResult.psf,
    createdAt: Date.now(),
    notes: $<HTMLInputElement>('cal-notes').value.trim(),
  });
  $('cal-status').textContent = `已保存:${microscope} @ ${magnification}`;
  await refreshLibrary();
});

// ---------- 批量队列 ----------
const queue = new ProcessQueue();
const queueTbody = $('queue-table').querySelector('tbody')!;

queue.onChange((items) => {
  queueTbody.innerHTML = '';
  for (const it of items) {
    const tr = document.createElement('tr');
    const statusText = it.status === '失败' ? `失败:${it.error}` : it.status;
    tr.innerHTML =
      `<td>${it.name}</td><td>${it.microscope} @ ${it.magnification}</td>` +
      `<td>${statusText}</td>`;
    const td = document.createElement('td');
    const bar = document.createElement('progress');
    bar.max = 1;
    bar.value = it.progress;
    td.appendChild(bar);
    td.append(` ${it.detail}`);
    tr.appendChild(td);
    queueTbody.appendChild(tr);
  }
});

$('file-input').addEventListener('change', (e) => {
  const files = (e.target as HTMLInputElement).files;
  if (!files) return;
  const m = $<HTMLSelectElement>('sel-microscope').value;
  const mag = $<HTMLSelectElement>('sel-magnification').value;
  for (const f of files) queue.add(f, m, mag);
  (e.target as HTMLInputElement).value = '';
});

$('btn-clear-queue').addEventListener('click', () => queue.clear());

$('btn-run-queue').addEventListener('click', async () => {
  const iterations = Number($<HTMLInputElement>('inp-iterations').value);
  const tv = Number($<HTMLInputElement>('inp-tv').value);
  const accel = $<HTMLInputElement>('inp-accel').checked ? 0.9 : 0;
  await queue.run(async (item, onProgress) => {
    const rec = await findPsf(item.microscope, item.magnification);
    if (!rec) throw new Error(`未找到 ${item.microscope} @ ${item.magnification} 的 PSF`);
    onProgress(0.02, '解码图像');
    const img = await loadGrayImage(item.file);
    const result = await deconvolveTiled(getGL(), img.data, img.w, img.h, rec.data, rec.size, {
      iterations,
      tv,
      accel,
      onProgress: (done, total, tile, tiles) =>
        onProgress(0.05 + 0.9 * (done / total), `块 ${tile}/${tiles},迭代 ${done % iterations || iterations}/${iterations}`),
    });
    onProgress(0.97, '编码 PNG');
    await downloadPng({ data: result, w: img.w, h: img.h }, item.name.replace(/\.[^.]+$/, '') + '_deconv.png');
  });
});

// ---------- 预览 + 迭代回看 ----------
let preview: PreviewResult | null = null;
let previewSourceCanvas: HTMLCanvasElement | null = null;

$('btn-preview').addEventListener('click', async () => {
  const rec = currentPsf();
  if (!rec) {
    $('preview-status').textContent = '请先选择已校准的显微镜/倍率';
    return;
  }
  const item = queue.pending[0] ?? null;
  if (!item) {
    $('preview-status').textContent = '请先向队列添加图像';
    return;
  }
  // 释放上一次预览
  if (preview) {
    for (const t of preview.history) getGL().deleteTex(t);
    preview.deconvolver.dispose();
    preview = null;
  }
  const iterations = Number($<HTMLInputElement>('inp-iterations').value);
  const tv = Number($<HTMLInputElement>('inp-tv').value);
  const accel = $<HTMLInputElement>('inp-accel').checked ? 0.9 : 0;
  $('preview-status').textContent = '解码图像…';
  const img = await loadGrayImage(item.file);
  previewSourceCanvas = grayToCanvas(img);
  $('preview-status').textContent = '预览计算中…';
  preview = await deconvolvePreview(getGL(), img.data, img.w, img.h, rec.data, rec.size, {
    iterations,
    tv,
    accel,
    onIteration: (iter) => {
      $('preview-status').textContent = `预览计算中… 迭代 ${iter}/${iterations}`;
    },
  });
  $('preview-status').textContent = `预览完成(${preview.n}×${preview.n} 画布)`;
  // 胶片条
  const strip = $('filmstrip');
  strip.innerHTML = '';
  const slider = $<HTMLInputElement>('iteration-slider');
  slider.min = '1';
  slider.max = String(preview.history.length);
  slider.value = String(preview.history.length);
  slider.disabled = false;
  preview.history.forEach((tex, i) => {
    const thumb = document.createElement('canvas');
    thumb.width = 72;
    thumb.height = 72;
    const full = document.createElement('canvas');
    preview!.deconvolver.display(tex, full);
    thumb.getContext('2d')!.drawImage(full, 0, 0, 72, 72);
    const el = document.createElement('img');
    el.src = thumb.toDataURL();
    el.title = `迭代 ${i + 1}`;
    el.onclick = () => {
      slider.value = String(i + 1);
      showIteration(i + 1);
    };
    strip.appendChild(el);
  });
  showIteration(preview.history.length);
  $('btn-use-iterations').toggleAttribute('disabled', false);
});

function showIteration(k: number): void {
  if (!preview) return;
  $('iteration-label').textContent = `第 ${k} / ${preview.history.length} 次迭代`;
  document.querySelectorAll('#filmstrip img').forEach((el, i) => {
    el.classList.toggle('selected', i === k - 1);
  });
  const canvas = $<HTMLCanvasElement>('preview-canvas');
  const compare = $<HTMLInputElement>('chk-compare').checked;
  if (compare && previewSourceCanvas) {
    canvas.width = preview.n;
    canvas.height = preview.n;
    canvas.getContext('2d')!.drawImage(previewSourceCanvas, 0, 0, preview.n, preview.n);
  } else {
    preview.deconvolver.display(preview.history[k - 1], canvas);
  }
}

$('iteration-slider').addEventListener('input', (e) => {
  showIteration(Number((e.target as HTMLInputElement).value));
});
$('chk-compare').addEventListener('change', () => {
  const slider = $<HTMLInputElement>('iteration-slider');
  if (slider.value !== '0') showIteration(Number(slider.value));
});
$('btn-use-iterations').addEventListener('click', () => {
  $<HTMLInputElement>('inp-iterations').value = $<HTMLInputElement>('iteration-slider').value;
});

// ---------- 验证与基准 ----------
$('btn-selftest').addEventListener('click', async () => {
  const el = $('selftest-result');
  el.textContent = '运行中…';
  el.className = 'hint';
  try {
    const n = 256;
    const { psf, size: ps } = gaussianPSF(1.5, 15);
    // 确定性伪随机观测图
    const obs = new Float64Array(n * n);
    let seed = 12345;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    for (let i = 0; i < obs.length; i++) obs[i] = 0.1 + 0.5 * rand();
    const psf32 = Float32Array.from(psf);
    const obs32 = Float32Array.from(obs);
    // CPU 参考
    const cpu = richardsonLucy(obs, n, n, psf, ps, ps, { iterations: 5, tv: 0.002, accel: 0.9 });
    // GPU
    const dc = new GpuDeconvolver(getGL(), n);
    dc.setPSF(psf32, ps);
    dc.setImage(obs32);
    await dc.run({ iterations: 5, tv: 0.002, accel: 0.9 });
    const gpu = dc.readResult();
    dc.dispose();
    const gpu64 = Float64Array.from(gpu);
    const p = psnr(gpu64, cpu);
    if (p > 30) {
      el.textContent = `✓ 通过(GPU 与 CPU 结果 PSNR = ${p.toFixed(1)} dB)`;
      el.className = 'hint ok';
    } else {
      el.textContent = `✗ 偏差过大(PSNR = ${p.toFixed(1)} dB),GPU 实现可能有问题`;
      el.className = 'hint bad';
    }
  } catch (err) {
    el.textContent = '自检失败:' + String(err);
    el.className = 'hint bad';
  }
});

$('btn-benchmark').addEventListener('click', async () => {
  const el = $('bench-result');
  el.className = 'hint';
  try {
    const w = 5472, h = 3648; // ≈2000 万像素
    el.textContent = '生成 2000 万像素测试图…';
    await new Promise((r) => setTimeout(r, 10));
    const img = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        img[y * w + x] =
          0.3 +
          0.2 * Math.sin(x * 0.13) * Math.cos(y * 0.11) +
          0.2 * Math.sin((x + y) * 0.031) +
          0.1 * Math.sin(x * 0.007) * Math.sin(y * 0.009);
      }
    }
    const { psf, size: ps } = gaussianPSF(1.2, 15);
    const psf32 = Float32Array.from(psf);
    const canvasSize = Number($<HTMLSelectElement>('bench-canvas').value);
    el.textContent = '反卷积计算中(30 次迭代)…';
    const t0 = performance.now();
    let lastReport = 0;
    await deconvolveTiled(getGL(), img, w, h, psf32, ps, {
      iterations: 30,
      tv: 0.002,
      accel: 0.9,
      canvasSize,
      onProgress: (done, total, tile, tiles) => {
        if (done - lastReport >= 30) {
          lastReport = done;
          el.textContent = `反卷积计算中… 迭代 ${done}/${total}(块 ${tile}/${tiles})`;
        }
      },
    });
    const dt = (performance.now() - t0) / 1000;
    const mp = (w * h) / 1e6;
    el.textContent =
      `${mp.toFixed(1)} MP × 30 次迭代耗时 ${dt.toFixed(1)} 秒(${(mp / dt).toFixed(2)} MP/s)` +
      (dt < 20 ? ' ✓ 达到 20 秒目标' : ' ✗ 超过 20 秒目标(可降低 FFT 画布或迭代数)');
    el.className = dt < 20 ? 'hint ok' : 'hint bad';
  } catch (err) {
    el.textContent = '基准失败:' + String(err);
    el.className = 'hint bad';
  }
});

// ---------- 启动 ----------
refreshLibrary();
