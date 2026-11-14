"""Classify one image: python predict.py path/to/image.jpg"""
import sys
from PIL import Image
from src.inference import load_model, predict_pil

model, idx_to_class, transform = load_model()
with Image.open(sys.argv[1]) as image:
    label, score = predict_pil(model, idx_to_class, transform, image)
print(f'Predicted: {label} | model score: {score:.3f}')
