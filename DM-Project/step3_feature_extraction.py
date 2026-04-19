"""
step3_feature_extraction.py
───────────────────────────
Extracts hand-crafted features from coral images:
  - avg_B, avg_G, avg_R  (mean BGR channel values)
  - brightness           (overall mean pixel intensity)
  - texture              (pixel standard deviation)
  - coral_count          (number of annotated bounding boxes)
  - label                (majority class from YOLO annotations)

Saves each split as a CSV so later steps don't re-process images.

Output files:
  features_train.csv
  features_valid.csv
  features_test.csv

Usage:
    python step3_feature_extraction.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR,
    VALID_IMAGE_DIR, VALID_LABEL_DIR,
    TEST_IMAGE_DIR,  TEST_LABEL_DIR,
    CLASS_MAP,
)


# ─── Core function ─────────────────────────────────────────────────────────────

def extract_features(image_dir: str, label_dir: str) -> pd.DataFrame:
    """
    Iterate over every image in *image_dir*, read its YOLO label, and
    return a DataFrame with one row per image.
    """
    data = []

    for img_file in sorted(os.listdir(image_dir)):
        if not img_file.endswith((".jpg", ".png")):
            continue

        img_path   = os.path.join(image_dir, img_file)
        label_path = os.path.join(
            label_dir,
            img_file.replace(".jpg", ".txt").replace(".png", ".txt")
        )

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: {img_file} could not be read, skipping.")
            continue

        # Resize to fixed size for consistent features
        img_resized = cv2.resize(img, (224, 224))

        # Feature extraction
        avg_color  = img_resized.mean(axis=(0, 1))   # BGR
        brightness = float(img_resized.mean())
        texture    = float(img_resized.std())

        # Label processing
        if not os.path.exists(label_path):
            print(f"  Warning: label file missing for {img_file}, skipping.")
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            print(f"  Warning: {label_path} is empty, skipping.")
            continue

        class_ids      = [int(line.split()[0]) for line in lines]
        majority_class = max(set(class_ids), key=class_ids.count)
        label_name     = CLASS_MAP[majority_class]
        coral_count    = len(class_ids)

        data.append([
            avg_color[0], avg_color[1], avg_color[2],
            brightness, texture, coral_count, label_name
        ])

    columns = ["avg_B", "avg_G", "avg_R", "brightness", "texture", "coral_count", "label"]
    return pd.DataFrame(data, columns=columns)


# ─── Quick visualisation helper ────────────────────────────────────────────────

def visualise_samples(image_dir: str, label_dir: str, n: int = 5) -> None:
    """Display the first *n* images with their majority label."""
    for img_file in list(sorted(os.listdir(image_dir)))[:n]:
        if not img_file.endswith((".jpg", ".png")):
            continue

        img_path   = os.path.join(image_dir, img_file)
        label_path = os.path.join(
            label_dir,
            img_file.replace(".jpg", ".txt").replace(".png", ".txt")
        )

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with open(label_path) as f:
            lines = f.readlines()
        class_ids      = [int(line.split()[0]) for line in lines]
        majority_class = max(set(class_ids), key=class_ids.count)
        label_name     = CLASS_MAP[majority_class]

        plt.imshow(img_rgb)
        plt.title(f"Label: {label_name}  |  Coral Count: {len(class_ids)}")
        plt.axis("off")
        plt.show()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Extracting training features …")
    df_train = extract_features(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)
    print(f"  Training samples : {len(df_train)}")

    print("Extracting validation features …")
    df_valid = extract_features(VALID_IMAGE_DIR, VALID_LABEL_DIR)
    print(f"  Validation samples : {len(df_valid)}")

    print("Extracting test features …")
    df_test = extract_features(TEST_IMAGE_DIR, TEST_LABEL_DIR)
    print(f"  Test samples : {len(df_test)}")

    # Save to CSV
    df_train.to_csv("features_train.csv", index=False)
    df_valid.to_csv("features_valid.csv", index=False)
    df_test.to_csv("features_test.csv",  index=False)
    print("\n✅ Features saved to features_train/valid/test.csv")

    # Visualise a few training samples
    print("\nVisualising 5 training samples …")
    visualise_samples(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, n=5)
