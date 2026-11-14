"""Task 1A: camera calibration from checkerboard images, and undistortion."""
from pathlib import Path
import cv2
import numpy as np

COLS, ROWS = 9, 6      # INNER corners (a 10x7-square board has 9x6 inner corners)
SQUARE_MM = 25.0       # side of one square; measure it with a ruler after printing


def find_corners(folder, cols=COLS, rows=ROWS):
    """Detect the inner corners in every image of a folder.

    Returns accepted image names, their corner arrays, the image size (w, h),
    overlay images with the corners drawn, and a list of rejected images.
    """
    names, points, overlays, rejected = [], [], {}, []
    size = None
    for path in sorted(Path(folder).glob('*')):
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        img = cv2.imread(str(path))
        if img is None:
            rejected.append((path.name, 'cannot read image'))
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        this_size = (gray.shape[1], gray.shape[0])
        if size is None:
            size = this_size
        if this_size != size:
            raise ValueError(f'{path.name} has size {this_size} but expected {size}. '
                             'All calibration images must come from the same camera and resolution.')
        found, corners = cv2.findChessboardCorners(gray, (cols, rows))
        if not found:
            rejected.append((path.name, 'chessboard corners not found'))
            continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)   # sub-pixel refinement
        cv2.drawChessboardCorners(img, (cols, rows), corners, True)
        names.append(path.name)
        points.append(corners)
        overlays[path.name] = img
    return names, points, size, overlays, rejected


def board_points(cols=COLS, rows=ROWS, square_mm=SQUARE_MM):
    """3D corner coordinates in metres in the board frame (Z = 0 because the board is flat)."""
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * (square_mm / 1000.0)
    return pts


def calibrate(image_points, size, cols=COLS, rows=ROWS, square_mm=SQUARE_MM):
    """Run cv2.calibrateCamera. Returns RMS, K, dist and the per-view reprojection error (px)."""
    if len(image_points) < 3:
        raise RuntimeError('At least 3 usable images are needed (10-20 recommended).')
    obj = board_points(cols, rows, square_mm)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj] * len(image_points), image_points, size, None, None)

    # Reproject the 3D corners with the estimated camera and compare with the detected corners
    view_errors = []
    for pts, rvec, tvec in zip(image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        distance = np.linalg.norm(pts.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
        view_errors.append(float(distance.mean()))
    return rms, K, dist, view_errors


def undistort(img, K, dist):
    """Remove lens distortion; the image size is kept so before/after are comparable."""
    return cv2.undistort(img, K, dist)


def save_calibration(path, K, dist, size):
    np.savez(path, K=K, dist=dist, width=size[0], height=size[1])


def load_calibration(path):
    """Return K, dist and (width, height) of the calibration images."""
    if not Path(path).exists():
        raise FileNotFoundError(f'Calibration file not found: {path}. Run run_task1a.py first.')
    data = np.load(path)
    return data['K'], data['dist'], (int(data['width']), int(data['height']))
