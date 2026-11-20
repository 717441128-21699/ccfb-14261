import { describe, it, expect } from 'vitest';
import { fft2, nextPow2 } from '../src/core/fft';
import { richardsonLucy } from '../src/core/deconv';
import { gaussianPSF, airyPSF, motionPSF, diskPSF } from '../src/core/psfModels';
import { synthSpecimen, blurImage } from '../src/core/synth';
import { addNoise, type NoiseOptions } from '../src/core/noise';
import { psnr, ncc } from '../src/core/metrics';
import { calibratePSF } from '../src/core/psfCalibrate';

describe('FFT', () => {
  it('正逆变换往返还原', () => {
    const n = 64;
    const re = new Float64Array(n * n);
    const im = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) re[i] = Math.sin(i * 0.37) * Math.cos(i * 0.11);
    const orig = Float64Array.from(re);
    fft2(re, im, n, n, false);
    fft2(re, im, n, n, true);
    let maxErr = 0;
    for (let i = 0; i < n * n; i++) maxErr = Math.max(maxErr, Math.abs(re[i] - orig[i]));
    expect(maxErr).toBeLessThan(1e-10);
  });

  it('nextPow2', () => {
    expect(nextPow2(1)).toBe(1);
    expect(nextPow2(300)).toBe(512);
    expect(nextPow2(1024)).toBe(1024);
  });
});

describe('PSF 校准', () => {
  it('从模拟微孔光源图中提取的 PSF 与真值形状一致', () => {
    const size = 256;
    // 真值 PSF:高斯 σ=3;微孔光源图 = 点光源经镜头模糊 ≈ PSF 本身(加背景与噪声)
    const { psf: truth, size: ps } = gaussianPSF(3, 31);
    const img = new Float64Array(size * size).fill(100); // 背景
    const cx = 128.4, cy = 127.6; // 故意偏离整数像素
    for (let y = 0; y < ps; y++) {
      for (let x = 0; x < ps; x++) {
        const gx = Math.round(cx + x - (ps - 1) / 2);
        const gy = Math.round(cy + y - (ps - 1) / 2);
        img[gy * size + gx] += truth[y * ps + x] * 50000;
      }
    }
    const noisy = addNoise(img, { gaussianSigma: 3, seed: 1 });
    const res = calibratePSF(noisy, size, size, 31);
    expect(res.size).toBe(31);
    // 质心定位误差 < 1.5 像素
    expect(Math.hypot(res.cx - cx, res.cy - cy)).toBeLessThan(1.5);
    // 形状一致性
    expect(ncc(res.psf, truth)).toBeGreaterThan(0.98);
    // 和为 1
    let s = 0;
    for (const v of res.psf) s += v;
    expect(s).toBeCloseTo(1, 5);
  });
});

/**
 * 反卷积验证矩阵:4 种代表性 PSF × 3 种噪声水平。
 * 测试图 512×512,刚毛/触角节等特征宽 3 像素(模拟 20MP 实拍中特征占数像素的情况)。
 * 验收标准:复原 PSNR 相对模糊观测提升 ≥ 6 dB。
 *
 * 注:更强的模糊(如高斯 σ≥1.5)或更重噪声下,高频信息被物理性截断,
 * 6 dB 不可达——这正是需要按显微镜/倍率实测 PSF 并回看迭代过程的原因。
 */
describe('Richardson-Lucy 反卷积验证(已知真值)', () => {
  const SIZE = 512;
  const truth = synthSpecimen(SIZE);

  const psfs: Array<[string, { psf: Float64Array; size: number }]> = [
    ['高斯 σ=1.0', gaussianPSF(1.0)],
    ['艾里斑 第一暗环2.5px', airyPSF(2.5)],
    ['运动模糊 7px/30°', motionPSF(7, 30)],
    ['离焦圆盘 r=2.5', diskPSF(2.5)],
  ];

  const noises: Array<[string, NoiseOptions, number]> = [
    // [名称, 噪声参数, TV 强度]
    ['无噪声', {}, 0.001],
    ['轻噪声 σ=0.003', { gaussianSigma: 0.003, seed: 11 }, 0.002],
    ['中噪声 σ=0.006+光子', { gaussianSigma: 0.006, photonPeak: 5000, seed: 13 }, 0.003],
  ];

  for (const [psfName, { psf, size: ps }] of psfs) {
    for (const [noiseName, noiseOpt, tv] of noises) {
      it(`${psfName} × ${noiseName}:PSNR 提升 ≥ 6 dB`, () => {
        const blurred = blurImage(truth, SIZE,SIZE, psf, ps, ps);
        const observed = addNoise(blurred, noiseOpt);
        const psnrBefore = psnr(observed, truth);

        const restored = richardsonLucy(observed, SIZE, SIZE, psf, ps, ps, {
          iterations: 80,
          tv,
          accel: 0.9,
        });
        const psnrAfter = psnr(restored, truth);
        const gain = psnrAfter - psnrBefore;
        console.log(
          `  ${psfName} × ${noiseName}: 模糊 ${psnrBefore.toFixed(2)} dB → 复原 ${psnrAfter.toFixed(2)} dB(+${gain.toFixed(2)} dB)`,
        );
        expect(gain).toBeGreaterThanOrEqual(6);
      }, 180_000);
    }
  }
});
