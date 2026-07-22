# -*- coding: utf-8 -*-
"""Image loading + combined (image, tabular) PyTorch Dataset.

Image-side functions (``preprocess_data``, ``PetSurvivalDataset``,
``create_dataset``, mean/std computation, tabular-to-sample alignment) are
copied verbatim from
``clinical+image/SCLC_simple_CNN-main/dataset.py`` — the image backbone and
its preprocessing must stay unchanged per the fusion instructions, so this is
a straight copy, not a rewrite. The ``tabular`` slot now carries the combined
[clinical | tfidf] vector built by ``features.build_fold_multimodal_tabular``;
the fusion model splits it back into two branches in ``forward()``.
"""
import os

import numpy as np
import pandas as pd
import scipy.integrate
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson

np.random.seed(1234)
_ = torch.manual_seed(123)


def preprocess_data(image_dir_path, df, target, gray_scale=True):
    """Loads CT-slice PNGs paired with survival labels for the given rows.

    Returns list of dicts: ``{'image': PIL.Image, 'duration': float,
    'event': int, 'source_position': int}``. ``source_position`` lets
    ``_align_tabular_to_samples`` drop tabular rows for any image that failed
    to load, keeping tabular/image alignment even when files are missing.
    research_ids ``[9, 223, 233, 293, 298]`` are pixel-inverted (acquisition
    artifact fix, copied from the source project).
    """
    samples = []
    image_not_found = []

    for source_position, (_, row) in enumerate(df.iterrows()):
        research_id = row["research_id"]
        duration = row[target + "_days"]
        event = row[target + "_event"]

        image_path = os.path.join(image_dir_path, str(int(research_id)) + ".png")

        try:
            if gray_scale:
                img = Image.open(image_path).convert("L")
            else:
                img = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            image_not_found.append(int(research_id))
            continue

        if research_id in [9, 223, 233, 293, 298]:
            img = ImageOps.invert(img)

        samples.append({
            "image": img,
            "duration": duration,
            "event": event,
            "source_position": source_position,
        })

    if image_not_found:
        print(f"Image not found: {image_not_found}")
    return samples


class PetSurvivalDataset(Dataset):
    """(image, tabular, duration, event) survival dataset.

    ``tabular`` is the combined [clinical | tfidf] vector when provided,
    otherwise the item is a plain (image, duration, event) 3-tuple.
    """

    def __init__(self, samples, transform=None, tabular=None):
        self.samples = samples
        self.transform = transform
        self.tabular = tabular

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = self.samples[index]["image"]
        duration = self.samples[index]["duration"]
        event = self.samples[index]["event"]

        if self.transform is not None:
            image = self.transform(image)

        if self.tabular is not None:
            tabular = torch.from_numpy(self.tabular[index]).float()
            return image, tabular, duration, event

        return image, duration, event


def _compute_mean_std_from_samples(samples, gray_scale=True):
    """Per-channel pixel mean/std from PIL samples (train fold only, to avoid
    normalization leakage)."""
    pixel_sum = None
    pixel_sq_sum = None
    num_pixels = 0

    for sample in samples:
        img = sample["image"]
        img = np.array(img, dtype=np.float32) / 255.0
        if gray_scale:
            img = img[:, :, np.newaxis]
        h, w, c = img.shape
        if pixel_sum is None:
            pixel_sum = np.zeros(c)
            pixel_sq_sum = np.zeros(c)
        pixel_sum += img.sum(axis=(0, 1))
        pixel_sq_sum += (img**2).sum(axis=(0, 1))
        num_pixels += h * w

    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_sq_sum / num_pixels - mean**2)
    print(f"mean: {mean.tolist()}, std: {std.tolist()}")
    return mean.tolist(), std.tolist()


def _align_tabular_to_samples(tabular, samples):
    """Realigns tabular rows to the samples that survived image loading."""
    if tabular is None:
        return None
    if len(tabular) == len(samples):
        return tabular
    positions = [sample.get("source_position") for sample in samples]
    if any(pos is None for pos in positions):
        raise ValueError("Cannot align tabular data because samples do not include source_position.")
    return tabular[np.asarray(positions, dtype=np.int64)]


def create_dataset(
    train_samples, test_samples, resize, transform, gray_scale, pretrained,
    train_tabular=None, test_tabular=None,
):
    """Builds train/test ``PetSurvivalDataset`` with train-fold-only
    normalization stats and (optionally) horizontal-flip augmentation."""
    train_tabular = _align_tabular_to_samples(train_tabular, train_samples)
    test_tabular = _align_tabular_to_samples(test_tabular, test_samples)
    if pretrained and not gray_scale:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean, std = _compute_mean_std_from_samples(train_samples, gray_scale=gray_scale)

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize, resize)),
        transforms.Normalize(mean, std),
    ])
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize, resize)),
        transforms.Normalize(mean, std),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    if transform:
        train_dataset = PetSurvivalDataset(train_samples, train_transforms, tabular=train_tabular)
    else:
        train_dataset = PetSurvivalDataset(train_samples, test_transforms, tabular=train_tabular)
    test_dataset = PetSurvivalDataset(test_samples, test_transforms, tabular=test_tabular)
    print(f"image shape = {train_dataset[0][0].shape}")

    return train_dataset, test_dataset
