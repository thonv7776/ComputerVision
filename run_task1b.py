"""Task 1B: python run_task1b.py

Input : three photos in data/task1b/
          scene1.jpg - base image (also the clean reference for noise, FFT and edges)
          scene2.jpg - the SAME scene from a slightly different angle or distance
          other.jpg  - an UNRELATED scene (control for the number of matches)
Output: results/task1b/ (PNG figures, filter_metrics.csv, summary.txt)
"""
import csv
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.filtering import (load_image, to_gray, add_gaussian_noise, add_salt_pepper_noise,
                           apply_filters, mse, psnr, fft_log_spectrum, fft_filter,
                           sobel_edges, canny_edges)
from src.features import sift_match, draw_matches

DATA = Path('data/task1b')
OUT = Path('results/task1b')
OUT.mkdir(parents=True, exist_ok=True)

scene1 = load_image(DATA / 'scene1.jpg')
scene2 = load_image(DATA / 'scene2.jpg')
other = load_image(DATA / 'other.jpg')
clean = to_gray(scene1)                      # ground truth for MSE / PSNR
summary = [f'Image size (w x h): {clean.shape[1]} x {clean.shape[0]}']

# Spatial filtering: 2 noise types x 3 filters
KERNEL, SIGMA = 5, 1.2
noisy_images = {'Gaussian noise (sigma=20)': add_gaussian_noise(clean, sigma=20),
                'Salt & pepper (5%)': add_salt_pepper_noise(clean, amount=0.05)}
rows = []
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for r, (noise_name, noisy) in enumerate(noisy_images.items()):
    outputs = {'Original': clean, 'Noisy': noisy, **apply_filters(noisy, KERNEL, SIGMA)}
    for ax, (name, img) in zip(axes[r], outputs.items()):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'{name}\n({noise_name})' if name == 'Noisy' else name, fontsize=9)
        ax.axis('off')
        if name != 'Original':
            rows.append([noise_name, name, round(mse(clean, img), 2), round(psnr(clean, img), 2)])
fig.tight_layout(); fig.savefig(OUT / '01_filtering.png', dpi=110); plt.close(fig)
with (OUT / 'filter_metrics.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['noise', 'method', 'MSE', 'PSNR_dB'])
    writer.writerows(rows)
summary.append(f'Filters: kernel={KERNEL}, gaussian sigma={SIGMA}')
summary += [f'  {r[0]:26s} {r[1]:9s} MSE={r[2]:9.2f}  PSNR={r[3]:6.2f} dB' for r in rows]

# FFT: spectrum, low-pass with two cutoffs, high-pass
low_small, _ = fft_filter(clean, cutoff=10, kind='low')
low_large, mask = fft_filter(clean, cutoff=40, kind='low')
high, _ = fft_filter(clean, cutoff=40, kind='high')
panels = [('Original', clean, 'gray', (0, 255)), ('Log magnitude spectrum', fft_log_spectrum(clean), 'magma', None),
          ('Low-pass mask (cutoff=40)', mask, 'gray', None), ('Low-pass cutoff=10', low_small, 'gray', (0, 255)),
          ('Low-pass cutoff=40', low_large, 'gray', (0, 255)), ('High-pass cutoff=40 (signed)', high, 'gray', (-100, 100))]
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, (title, img, cmap, lim) in zip(axes.flat, panels):
    ax.imshow(img, cmap=cmap, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
    ax.set_title(title, fontsize=10); ax.axis('off')
fig.tight_layout(); fig.savefig(OUT / '02_fft.png', dpi=110); plt.close(fig)

# Sobel gradients and Canny edges with two threshold pairs
gx, gy, magnitude = sobel_edges(clean)
canny_a, canny_b = canny_edges(clean, 80, 160), canny_edges(clean, 30, 90)
panels = [('Original', clean, 'gray'), ('Sobel Gx (signed)', gx, 'coolwarm'), ('Sobel Gy (signed)', gy, 'coolwarm'),
          ('Gradient magnitude', magnitude, 'gray'), ('Canny (80, 160)', canny_a, 'gray'), ('Canny (30, 90)', canny_b, 'gray')]
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, (title, img, cmap) in zip(axes.flat, panels):
    ax.imshow(img, cmap=cmap); ax.set_title(title, fontsize=10); ax.axis('off')
fig.tight_layout(); fig.savefig(OUT / '03_edges.png', dpi=110); plt.close(fig)
summary.append(f'Canny edge pixels: (80,160)={(canny_a > 0).mean():.3%}, (30,90)={(canny_b > 0).mean():.3%}')

# SIFT with ratio test: same scene vs unrelated scene
RATIO = 0.75
kp1, kp2, good_same = sift_match(clean, to_gray(scene2), RATIO)
_, kp3, good_other = sift_match(clean, to_gray(other), RATIO)
for tag, img2, kp, good in [('same_scene', scene2, kp2, good_same), ('unrelated', other, kp3, good_other)]:
    cv2.imwrite(str(OUT / f'04_sift_{tag}.png'), draw_matches(scene1, kp1, img2, kp, good))
summary += [f'SIFT ratio test = {RATIO}',
            f'  keypoints: scene1={len(kp1)}, scene2={len(kp2)}, other={len(kp3)}',
            f'  good matches (same scene)  = {len(good_same)}',
            f'  good matches (unrelated)   = {len(good_other)}']

(OUT / 'summary.txt').write_text('\n'.join(summary), encoding='utf-8')
print('\n'.join(summary))
print('Saved figures and metrics to', OUT.resolve())
