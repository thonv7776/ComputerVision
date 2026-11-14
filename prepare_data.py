"""Task 2, step 1: clean the data and split it into train / val / test BY GROUP.

Run: python prepare_data.py
Input : data/raw/<class>/<group_id>__<view>.jpg   (e.g. item001__view1.jpg)
Output: data/splits/{train,val,test}/<class>/, artifacts/data_manifest.csv, artifacts/rejected_images.csv

Images with the same group_id (the part of the file name before '__') always go to the same split,
so near-identical views of one object cannot leak from train into test. The split is made BEFORE
augmentation, which is applied later and only to the training set (train.py).
"""
import csv
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path
from PIL import Image

RAW, SPLITS, ART = Path('data/raw'), Path('data/splits'), Path('artifacts')
SEED = 42
ART.mkdir(exist_ok=True)

classes = sorted(p.name for p in RAW.iterdir() if p.is_dir())
groups = {c: defaultdict(list) for c in classes}      # groups[class][group_id] = [files]
rejected, seen = [], {}

# Cleaning: drop unreadable, tiny and byte-identical images
for label in classes:
    for path in sorted((RAW / label).glob('*')):
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        try:
            with Image.open(path) as im:
                im.convert('RGB').load()                         # raises on corrupt files
                if min(im.size) < 32:
                    raise ValueError('smaller than 32 px')
        except Exception as error:
            rejected.append((path.name, label, f'unreadable or too small: {error}'))
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()   # same hash = same bytes
        if digest in seen:
            rejected.append((path.name, label, f'duplicate of {seen[digest]}'))
            continue
        seen[digest] = f'{label}/{path.name}'
        groups[label][path.stem.split('__')[0]].append(path)

# Split each class by group: about 70% train, 15% val, 15% test
rng = random.Random(SEED)
if SPLITS.exists():
    shutil.rmtree(SPLITS)                                        # rebuild from scratch
manifest = []
for label in classes:
    ids = sorted(groups[label])
    if len(ids) < 3:
        raise SystemExit(f'Class {label} has only {len(ids)} groups; at least 3 are needed for train/val/test.')
    rng.shuffle(ids)
    n = max(1, round(len(ids) * 0.15))                           # groups per val / test split
    parts = {'test': ids[:n], 'val': ids[n:2 * n], 'train': ids[2 * n:]}
    for split, id_list in parts.items():
        folder = SPLITS / split / label
        folder.mkdir(parents=True, exist_ok=True)
        for gid in id_list:
            for path in groups[label][gid]:
                shutil.copy2(path, folder / path.name)
                manifest.append([path.name, label, gid, split])

with (ART / 'data_manifest.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['file_name', 'label', 'group_id', 'split'])
    writer.writerows(manifest)
with (ART / 'rejected_images.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['file_name', 'label', 'reason'])
    writer.writerows(rejected)

print(f'Valid images: {len(manifest)} | rejected: {len(rejected)}')
print(f'{"split":6s}' + ''.join(f'{c:>8s}' for c in classes))
for split in ['train', 'val', 'test']:
    counts = [sum(1 for m in manifest if m[3] == split and m[1] == c) for c in classes]
    print(f'{split:6s}' + ''.join(f'{k:8d}' for k in counts))
