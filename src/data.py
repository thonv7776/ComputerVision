"""Image preprocessing shared by training, validation, test and inference."""
from torchvision import transforms

IMAGE_SIZE = 128
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
SEED = 42


def make_transform(train, size=IMAGE_SIZE, mean=MEAN, std=STD):
    """Random augmentation is added only when train=True; validation, test and
    inference use the same deterministic resize + normalize."""
    steps = [transforms.Resize((size, size))]
    if train:
        steps += [transforms.RandomRotation(12),
                  transforms.RandomHorizontalFlip(0.5),
                  transforms.ColorJitter(brightness=0.15, contrast=0.15)]
    steps += [transforms.ToTensor(),                  # (3, H, W) tensor with values in 0..1
              transforms.Normalize(mean, std)]        # values in -1..1
    return transforms.Compose(steps)
