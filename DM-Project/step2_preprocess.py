"""
step2_preprocess.py
───────────────────
Cleans the training, validation, and test splits by:
  1. Removing corrupt / unreadable images.
  2. Removing images that have no corresponding label file.
  3. Removing images whose label file is empty (no annotations).

Run AFTER step1_download_dataset.py.

Usage:
    python step2_preprocess.py
"""

import os
import cv2
from config import (
    TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR,
    VALID_IMAGE_DIR, VALID_LABEL_DIR,
    TEST_IMAGE_DIR,  TEST_LABEL_DIR,
)


# ─── Cleaning functions ────────────────────────────────────────────────────────

def remove_corrupt_images(image_dir: str) -> None:
    """Remove images that OpenCV cannot decode."""
    removed = 0
    for img_file in os.listdir(image_dir):
        path = os.path.join(image_dir, img_file)
        try:
            img = cv2.imread(path)
            if img is None:
                os.remove(path)
                removed += 1
        except Exception:
            os.remove(path)
            removed += 1
    print(f"[{image_dir}] Removed corrupt images: {removed}")


def remove_unlabeled_images(image_dir: str, label_dir: str) -> None:
    """Remove images that have no corresponding label (.txt) file."""
    removed = 0
    for img_file in os.listdir(image_dir):
        label_file = img_file.replace(".jpg", ".txt").replace(".png", ".txt")
        if not os.path.exists(os.path.join(label_dir, label_file)):
            os.remove(os.path.join(image_dir, img_file))
            removed += 1
    print(f"[{image_dir}] Removed unlabeled images: {removed}")


def remove_empty_labels(image_dir: str, label_dir: str) -> None:
    """Remove label files that are empty, together with their images."""
    removed = 0
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        if os.path.getsize(label_path) == 0:
            for ext in (".jpg", ".png"):
                img_path = os.path.join(image_dir, label_file.replace(".txt", ext))
                if os.path.exists(img_path):
                    os.remove(img_path)
            os.remove(label_path)
            removed += 1
    print(f"[{label_dir}] Removed empty annotation samples: {removed}")


def clean_split(image_dir: str, label_dir: str) -> None:
    """Run all three cleaning passes on one split."""
    print(f"\n── Cleaning split: {image_dir} ──")
    remove_corrupt_images(image_dir)
    remove_unlabeled_images(image_dir, label_dir)
    remove_empty_labels(image_dir, label_dir)
    print(f"Remaining images : {len(os.listdir(image_dir))}")
    print(f"Remaining labels : {len(os.listdir(label_dir))}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    clean_split(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)
    clean_split(VALID_IMAGE_DIR, VALID_LABEL_DIR)
    clean_split(TEST_IMAGE_DIR,  TEST_LABEL_DIR)
    print("\n✅ Preprocessing complete!")
