"""Task 1B: noise, spatial filtering, FFT filtering, Sobel and Canny edges."""
import cv2
import numpy as np


def load_image(path, max_side=800):
    """Read a BGR image and shrink it if its longest side exceeds max_side (keeps SIFT fast)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    scale = max_side / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def add_gaussian_noise(gray, sigma=20, seed=42):
    rng = np.random.default_rng(seed)                      # fixed seed: reproducible noise
    noisy = gray.astype(np.float32) + rng.normal(0, sigma, gray.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(gray, amount=0.05, seed=43):
    rng = np.random.default_rng(seed)
    out = gray.copy()
    r = rng.random(gray.shape)
    out[r < amount / 2] = 0                                # pepper (black)
    out[(r >= amount / 2) & (r < amount)] = 255            # salt (white)
    return out


def apply_filters(gray, kernel=5, sigma=1.2):
    """Box, Gaussian and Median filters with the same kernel size for a fair comparison."""
    return {'Box': cv2.blur(gray, (kernel, kernel)),
            'Gaussian': cv2.GaussianBlur(gray, (kernel, kernel), sigma),
            'Median': cv2.medianBlur(gray, kernel)}


def mse(clean, other):
    diff = clean.astype(np.float64) - other.astype(np.float64)
    return float(np.mean(diff ** 2))


def psnr(clean, other):
    return float(10 * np.log10(255 ** 2 / mse(clean, other)))


def fft_log_spectrum(gray):
    """Log magnitude spectrum log(1 + |F|) with the zero frequency at the image centre."""
    shifted = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    return np.log1p(np.abs(shifted))


def fft_filter(gray, cutoff, kind='low'):
    """Ideal low-pass or high-pass filter: multiply the spectrum by a circular mask, then inverse FFT.

    Returns (filtered image, mask). A high-pass result contains negative values, so it is not clipped.
    """
    shifted = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - h // 2) ** 2 + (xx - w // 2) ** 2        # squared distance to the spectrum centre
    mask = (dist2 <= cutoff ** 2).astype(np.float64)       # 1 inside the circle = low-pass
    if kind == 'high':
        mask = 1 - mask
    result = np.fft.ifft2(np.fft.ifftshift(shifted * mask)).real
    return result, mask


def sobel_edges(gray):
    """Smooth with a Gaussian (gradients amplify noise), then return Gx, Gy and the gradient magnitude."""
    smooth = cv2.GaussianBlur(gray, (5, 5), 1.2)
    gx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)      # CV_64F keeps negative gradients
    gy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    return gx, gy, np.hypot(gx, gy)


def canny_edges(gray, low=80, high=160):
    smooth = cv2.GaussianBlur(gray, (5, 5), 1.2)
    return cv2.Canny(smooth, low, high)
