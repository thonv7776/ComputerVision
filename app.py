"""Task 3: assistant app. Run: streamlit run app.py

The user uploads an image and picks a function; the app calls the matching Task 1 module or the
Task 2 model and shows the result. app.py only handles the interface; the algorithms live in src/.
"""
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from src import filtering, features, calibration
from src.inference import load_model, predict_pil

st.set_page_config(page_title='FCVG101 FruitSort Assistant', layout='wide')
st.title('FCVG101 - FruitSort Vision Assistant')

MODES = ['Filtering', 'Edges', 'FFT', 'Features / Matching', 'Calibration / Undistort', 'Classify']
mode = st.sidebar.radio('Choose a function', MODES)


@st.cache_resource                       # load the model once, not on every interaction
def get_model():
    return load_model()


def read_upload(file):
    """Decode an uploaded file to a BGR image; show an error and return None if it is not a valid image."""
    img = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error(f'Cannot read "{file.name}". Please choose a valid JPG/PNG image.')
    return img


def show(images, captions):
    """Show images side by side. OpenCV colour images are BGR and must be converted to RGB for display."""
    cols = st.columns(len(images))
    for col, img, caption in zip(cols, images, captions):
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        col.image(img, caption=caption, clamp=True)


def normalize(img):
    """Scale a float image (gradient, spectrum, FFT result) to 0..255 for display."""
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


file = st.file_uploader('Upload an image (jpg/png)', type=['jpg', 'jpeg', 'png'])
file2 = None
if mode == 'Features / Matching':
    file2 = st.file_uploader('Upload a second image', type=['jpg', 'jpeg', 'png'])

if file is None:
    st.info('Upload an image to get started.')
    st.stop()
bgr = read_upload(file)
if bgr is None:
    st.stop()
gray = filtering.to_gray(bgr)

# Task 1 modes
if mode == 'Filtering':
    noise = st.selectbox('Add noise', ['None', 'Gaussian', 'Salt & pepper'])
    kernel = st.slider('Kernel size', 3, 15, 5, step=2)
    noisy = {'None': gray, 'Gaussian': filtering.add_gaussian_noise(gray),
             'Salt & pepper': filtering.add_salt_pepper_noise(gray)}[noise]
    results = filtering.apply_filters(noisy, kernel)
    show([gray, noisy, *results.values()], ['Original', 'Noisy', *results.keys()])
    if noise != 'None':                  # MSE / PSNR need a clean reference image
        st.table({name: {'MSE': round(filtering.mse(gray, img), 2), 'PSNR (dB)': round(filtering.psnr(gray, img), 2)}
                  for name, img in [('Noisy', noisy), *results.items()]})

elif mode == 'Edges':
    low = st.slider('Canny low threshold', 0, 255, 80)
    high = st.slider('Canny high threshold', 0, 255, 160)
    gx, gy, magnitude = filtering.sobel_edges(gray)
    canny = filtering.canny_edges(gray, low, high)
    show([gray, normalize(magnitude), canny], ['Original', 'Sobel gradient magnitude', f'Canny ({low}, {high})'])
    st.write(f'Edge pixel ratio (Canny): {(canny > 0).mean():.2%}')

elif mode == 'FFT':
    cutoff = st.slider('Cutoff radius (pixels)', 5, 100, 30)
    low_img, _ = filtering.fft_filter(gray, cutoff, 'low')
    high_img, _ = filtering.fft_filter(gray, cutoff, 'high')
    show([gray, normalize(filtering.fft_log_spectrum(gray)), normalize(low_img), normalize(high_img)],
         ['Original', 'Log spectrum', f'Low-pass (r={cutoff})', f'High-pass (r={cutoff})'])

elif mode == 'Features / Matching':
    if file2 is None:
        st.info('Upload a second image to match against.')
        st.stop()
    bgr2 = read_upload(file2)
    if bgr2 is None:
        st.stop()
    ratio = st.slider('Ratio test', 0.5, 0.95, 0.75)
    kp1, kp2, good = features.sift_match(gray, filtering.to_gray(bgr2), ratio)
    show([features.draw_matches(bgr, kp1, bgr2, kp2, good)], ['Best matches'])
    st.write(f'Keypoints: {len(kp1)} / {len(kp2)}  |  Good matches (ratio test): {len(good)}')

elif mode == 'Calibration / Undistort':
    try:
        K, dist, size = calibration.load_calibration('results/task1a/calibration.npz')
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
    if (bgr.shape[1], bgr.shape[0]) != size:
        st.error(f'The image must have the same resolution as the calibration images ({size[0]}x{size[1]}). '
                 'K and dist are only valid for the camera and settings that were calibrated.')
        st.stop()
    st.warning('Only upload images taken with the SAME camera that was calibrated.')
    show([bgr, calibration.undistort(bgr, K, dist)], ['Before', 'After (undistorted)'])
    st.write('K =', K.round(2).tolist())
    st.write('dist =', dist.ravel().round(4).tolist())

# Task 2 mode
elif mode == 'Classify':
    try:
        model, idx_to_class, transform = get_model()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
    pil = Image.open(file).convert('RGB')
    label, score = predict_pil(model, idx_to_class, transform, pil)
    st.image(pil, width=300)
    st.success(f'Predicted class: **{label}**  |  model score: {score:.3f}')
    st.caption("The score is the model's softmax output, not a calibrated probability.")
