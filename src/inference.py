"""Load the trained model and classify one image (also used by app.py)."""
from pathlib import Path
import torch
from PIL import ImageOps
from src.data import make_transform
from src.model import FruitCNN

MODEL_PATH = 'artifacts/model.pth'


def load_model(path=MODEL_PATH):
    """Return (model, idx_to_class, transform) built from the settings saved at training time."""
    if not Path(path).exists():
        raise FileNotFoundError(f'Model not found: {path}. Run train.py first.')
    saved = torch.load(path, map_location='cpu')
    idx_to_class = {i: name for name, i in saved['class_to_idx'].items()}
    model = FruitCNN(n_classes=len(idx_to_class))
    model.load_state_dict(saved['model_state'])
    model.eval()
    transform = make_transform(False, saved['image_size'], saved['mean'], saved['std'])
    return model, idx_to_class, transform


def predict_pil(model, idx_to_class, transform, image):
    """Classify a PIL image. Returns (class name, model score).

    The score is the softmax output, not a calibrated probability.
    """
    rgb = ImageOps.exif_transpose(image).convert('RGB')    # fix EXIF rotation, force 3 channels
    x = transform(rgb).unsqueeze(0)                        # add batch dimension: (1, 3, H, W)
    with torch.no_grad():
        scores = torch.softmax(model(x), dim=1)[0]
    best = int(scores.argmax())
    return idx_to_class[best], float(scores[best])
