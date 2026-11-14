"""Task 2, step 2: train the CNN and keep the checkpoint with the lowest validation loss.

Run: python train.py
Output: artifacts/model.pth, artifacts/classes.json, results/train_history.csv, results/learning_curves.png
"""
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import IMAGE_SIZE, MEAN, STD, SEED, make_transform
from src.model import FruitCNN

EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)     # reproducible runs
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Augmentation is applied to the training set only
train_set = ImageFolder('data/splits/train', transform=make_transform(train=True))
val_set = ImageFolder('data/splits/val', transform=make_transform(train=False))
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

model = FruitCNN(n_classes=len(train_set.classes)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
Path('artifacts').mkdir(exist_ok=True); Path('results').mkdir(exist_ok=True)
Path('artifacts/classes.json').write_text(json.dumps(train_set.class_to_idx, indent=2))


def train_one_epoch():
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
    return total_loss / len(train_set), correct / len(train_set)


def validate():
    model.eval()                                 # no weight updates and no gradients during validation
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
    return total_loss / len(val_set), correct / len(val_set)


print(f'device={device} classes={train_set.classes} train={len(train_set)} val={len(val_set)}')
history, best_val_loss = [], float('inf')
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = validate()
    history.append([epoch, train_loss, train_acc, val_loss, val_acc])
    if val_loss < best_val_loss:                 # best checkpoint = lowest validation loss
        best_val_loss = val_loss
        torch.save({'model_state': model.state_dict(), 'class_to_idx': train_set.class_to_idx,
                    'image_size': IMAGE_SIZE, 'mean': MEAN, 'std': STD,
                    'best_epoch': epoch, 'best_val_loss': val_loss, 'best_val_accuracy': val_acc,
                    'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE, 'seed': SEED},
                   'artifacts/model.pth')
    print(f'epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  '
          f'val_loss={val_loss:.4f} val_acc={val_acc:.3f}')

with open('results/train_history.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])
    writer.writerows(history)

epochs_axis = [h[0] for h in history]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
ax1.plot(epochs_axis, [h[1] for h in history], label='train'); ax1.plot(epochs_axis, [h[3] for h in history], label='validation')
ax1.set_title('Loss'); ax1.set_xlabel('epoch'); ax1.legend()
ax2.plot(epochs_axis, [h[2] for h in history], label='train'); ax2.plot(epochs_axis, [h[4] for h in history], label='validation')
ax2.set_title('Accuracy'); ax2.set_xlabel('epoch'); ax2.legend()
fig.tight_layout(); fig.savefig('results/learning_curves.png', dpi=130)
saved = torch.load('artifacts/model.pth', map_location='cpu')
print(f'Best epoch = {saved["best_epoch"]} (val_loss={saved["best_val_loss"]:.4f}). Saved artifacts/model.pth')
