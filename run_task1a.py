"""Task 1A: python run_task1a.py

Input : checkerboard photos in data/calibration/ (same camera, same resolution)
Output: results/task1a/ (corner overlays, calibration.npz, metrics.txt, before_after.jpg, per_view_error.png)
"""
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.calibration import (COLS, ROWS, SQUARE_MM, find_corners, calibrate,
                             undistort, save_calibration)

INPUT = Path('data/calibration')
OUT = Path('results/task1a')
SAMPLE = 'data/calibration_test/take1.jpg'      # image to undistort (same camera); None = first calibration image

OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'corners').mkdir(exist_ok=True)

names, points, size, overlays, rejected = find_corners(INPUT)
for name, img in overlays.items():
    cv2.imwrite(str(OUT / 'corners' / f'{Path(name).stem}_corners.jpg'), img)
print(f'Usable images: {len(names)} / {len(names) + len(rejected)}')
for name, reason in rejected:
    print('  Rejected:', name, '-', reason)

rms, K, dist, view_errors = calibrate(points, size)
mean_error = float(np.mean(view_errors))
save_calibration(OUT / 'calibration.npz', K, dist, size)

before = cv2.imread(SAMPLE or str(INPUT / names[0]))
if (before.shape[1], before.shape[0]) != size:
    raise ValueError('SAMPLE must have the same resolution as the calibration images.')
after = undistort(before, K, dist)
both = np.concatenate([before, after], axis=1)
cv2.putText(both, 'BEFORE', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
cv2.putText(both, 'AFTER', (size[0] + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 160, 0), 3)
cv2.imwrite(str(OUT / 'before_after.jpg'), both)
cv2.imwrite(str(OUT / 'undistorted.jpg'), after)

plt.figure(figsize=(7, 3))
plt.bar(range(1, len(view_errors) + 1), view_errors)
plt.xlabel('Image #'); plt.ylabel('Mean reprojection error (px)'); plt.tight_layout()
plt.savefig(OUT / 'per_view_error.png', dpi=120)
plt.close()

report = (
    f'Accepted/total images: {len(names)}/{len(names) + len(rejected)}\n'
    f'Pattern (inner corners): {COLS} x {ROWS}\n'
    f'Square size: {SQUARE_MM} mm\n'
    f'Image size (w x h): {size[0]} x {size[1]} px\n'
    f'OpenCV calibration RMS: {rms:.4f} px\n'
    f'Mean reprojection error: {mean_error:.4f} px\n'
    f'Per-view errors (px): {[round(e, 3) for e in view_errors]}\n'
    f'K =\n{K}\n'
    f'dist [k1, k2, p1, p2, k3] = {dist.ravel()}\n')
(OUT / 'metrics.txt').write_text(report, encoding='utf-8')
print(report)
