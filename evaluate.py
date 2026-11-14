"""Task 2, step 3: evaluate the chosen model on the TEST set (run only after the model is final).

Run: python evaluate.py
Output: results/test_metrics.json, test_predictions.csv, confusion_matrix.png, sample_predictions.png
"""
import csv
import json
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.inference import load_model

model, idx_to_class, transform = load_model()
test_set = ImageFolder('data/splits/test', transform=transform)       # same preprocessing as inference
assert test_set.class_to_idx == {n: i for i, n in idx_to_class.items()}, 'Class order differs from training!'

actual, predicted = [], []
with torch.no_grad():
    for images, labels in DataLoader(test_set, batch_size=32):
        predicted += model(images).argmax(1).tolist()
        actual += labels.tolist()

ids = list(range(len(idx_to_class)))
names = [idx_to_class[i] for i in ids]
accuracy = accuracy_score(actual, predicted)
macro_f1 = f1_score(actual, predicted, average='macro', labels=ids, zero_division=0)
print(f'TEST n={len(test_set)}  accuracy={accuracy:.4f}  macro_F1={macro_f1:.4f}')
print(classification_report(actual, predicted, labels=ids, target_names=names, zero_division=0))

report = classification_report(actual, predicted, labels=ids, target_names=names, output_dict=True, zero_division=0)
with open('results/test_metrics.json', 'w') as f:
    json.dump({'test_samples': len(test_set), 'accuracy': accuracy, 'macro_f1': macro_f1, 'per_class': report}, f, indent=2)

with open('results/test_predictions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file', 'actual', 'predicted', 'correct'])
    for (path, _), a, p in zip(test_set.samples, actual, predicted):
        writer.writerow([path, names[a], names[p], int(a == p)])

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ConfusionMatrixDisplay(confusion_matrix(actual, predicted, labels=ids), display_labels=names).plot(ax=ax, colorbar=False, cmap='Blues')
fig.tight_layout(); fig.savefig('results/confusion_matrix.png', dpi=130); plt.close(fig)

# Up to 8 test images, misclassified ones first. Green title = correct, red = wrong
rows = sorted(zip(test_set.samples, actual, predicted), key=lambda r: r[1] == r[2])[:8]
fig, axes = plt.subplots(1, len(rows), figsize=(2.2 * len(rows), 2.8), squeeze=False)
for ax, ((path, _), a, p) in zip(axes[0], rows):
    ax.imshow(Image.open(path)); ax.axis('off')
    ax.set_title(f'true: {names[a]}\npred: {names[p]}', fontsize=8, color='green' if a == p else 'red')
fig.tight_layout(); fig.savefig('results/sample_predictions.png', dpi=130); plt.close(fig)
print(f'Misclassified images: {sum(a != p for a, p in zip(actual, predicted))}. Results saved to results/')
