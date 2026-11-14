# FCVG101 Coursework - FruitSort Vision Assistant

One project, three parts: **Task 1** (camera calibration, filtering, FFT, edges, SIFT), **Task 2** (data pipeline + CNN) and **Task 3** (Streamlit app that calls both).

## 1. Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```
Run every command from this folder (the one containing `app.py`). Tested with Python 3.12.

## 2. Project structure
```
app.py                  Task 3: Streamlit interface (calls the modules in src/)
run_task1a.py           Task 1A: run calibration, save results to results/task1a/
run_task1b.py           Task 1B: run filtering / FFT / edges / SIFT, save results to results/task1b/
prepare_data.py         Task 2: clean the data and split it into train/val/test by group
train.py                Task 2: train the CNN, keep the best checkpoint (lowest validation loss)
evaluate.py             Task 2: evaluate on the test set
predict.py              Task 2: classify a single image
src/calibration.py      corner detection, calibrateCamera, reprojection error, undistortion
src/filtering.py        noise, Box/Gaussian/Median filters, MSE/PSNR, FFT, Sobel, Canny
src/features.py         SIFT matching with Lowe's ratio test
src/data.py             preprocessing and augmentation (shared by train/val/test/inference)
src/model.py            FruitCNN architecture
src/inference.py        load the model and predict (used by app.py)
data/raw/               Task 2 images, one folder per class: <group_id>__<view>.jpg
data/calibration/       checkerboard photos (Task 1A)
data/task1b/            scene1.jpg, scene2.jpg, other.jpg (Task 1B)
artifacts/              model.pth and classes.json (written by train.py)
```

## 3. How to run
| Step | Command | Output |
|---|---|---|
| Task 1A (10-20 photos of a 9x6 inner-corner checkerboard, 18 mm squares, same camera, in `data/calibration/`) | `python run_task1a.py` | `results/task1a/` (corner overlays, `metrics.txt`, `before_after.jpg`, `calibration.npz`) |
| Task 1B (three photos in `data/task1b/`: `scene1.jpg` base image, `scene2.jpg` same scene from another angle, `other.jpg` unrelated scene) | `python run_task1b.py` | `results/task1b/` (figures, `filter_metrics.csv`, `summary.txt`) |
| Task 2: prepare data | `python prepare_data.py` | `data/splits/`, `artifacts/data_manifest.csv`, `artifacts/rejected_images.csv` |
| Task 2: train | `python train.py` | `artifacts/model.pth`, `artifacts/classes.json`, `results/learning_curves.png`, `results/train_history.csv` |
| Task 2: evaluate (only after the model is final) | `python evaluate.py` | `results/test_metrics.json`, `confusion_matrix.png`, `sample_predictions.png`, `test_predictions.csv` |
| Task 2: predict one image | `python predict.py data/splits/test/apple/<file>.jpg` | class + model score |
| Task 3 | `streamlit run app.py` | app at http://localhost:8501 |

The **Calibration / Undistort** mode of the app needs `run_task1a.py` to be run first. The **Classify** mode needs `artifacts/model.pth` (included, or re-create it with `train.py`).

## 4. Task 2 settings
Train/val/test split of about 70/15/15 by group (images sharing a `group_id` stay together), made before augmentation. Augmentation (rotation of 12 degrees, horizontal flip, brightness/contrast jitter) is applied to the training set only. Images are resized to 128x128 RGB and normalized with mean = std = 0.5. The CNN has 3 conv layers, global average pooling and a linear classifier, trained with Adam (lr 1e-3, batch size 32, 40 epochs, seed 42). The checkpoint with the lowest validation loss is saved, and the test set is evaluated once.

The score shown in the app is the model's softmax output, not a calibrated probability. K and dist are only valid for the camera and resolution that were calibrated.
