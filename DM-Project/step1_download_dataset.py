"""
step1_download_dataset.py
─────────────────────────
Downloads the CoralCare dataset from Roboflow in YOLOv8 format.
Run this ONCE before any other step.

Usage:
    python step1_download_dataset.py
"""

from roboflow import Roboflow
from config import (
    ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE,
    ROBOFLOW_PROJECT, ROBOFLOW_VERSION, ROBOFLOW_FORMAT
)


def download_dataset():
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download(ROBOFLOW_FORMAT)
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset


if __name__ == "__main__":
    import os
    from config import BASE_DIR, TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR
    from config import VALID_IMAGE_DIR, VALID_LABEL_DIR, TEST_IMAGE_DIR, TEST_LABEL_DIR

    dataset = download_dataset()

    print("\nFolders inside dataset:")
    print(os.listdir(BASE_DIR))

    print("Train Images :", len(os.listdir(TRAIN_IMAGE_DIR)))
    print("Train Labels :", len(os.listdir(TRAIN_LABEL_DIR)))
    print("Valid Images :", len(os.listdir(VALID_IMAGE_DIR)))
    print("Valid Labels :", len(os.listdir(VALID_LABEL_DIR)))
    print("Test Images  :", len(os.listdir(TEST_IMAGE_DIR)))
    print("Test Labels  :", len(os.listdir(TEST_LABEL_DIR)))
