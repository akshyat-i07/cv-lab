"""
generate_dataset.py
--------------------
Creates a small synthetic grayscale image dataset with two classes:
  - normal   : smooth, uniform circular region (no irregularities)
  - abnormal : same base image + a random irregular blob (simulated anomaly)

Run once to populate dataset/normal and dataset/abnormal.
"""

import os
import cv2
import numpy as np

np.random.seed(42)

IMG_SIZE = 128
N_PER_CLASS = 60
OUT_DIR = "dataset"


def base_image():
    """Background + one large well-defined circle (like a normal scan region).
    Hard-ish edges (just anti-aliased) so Canny can actually detect them."""
    img = np.full((IMG_SIZE, IMG_SIZE), 60, dtype=np.uint8)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    cv2.circle(img, (cx, cy), 40, 190, -1, lineType=cv2.LINE_AA)
    noise = np.random.normal(0, 3, (IMG_SIZE, IMG_SIZE))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def make_normal():
    return base_image()


def make_abnormal():
    img = base_image()
    # add 1-3 small well-defined circular "lesions" = simulated anomaly
    for _ in range(np.random.randint(1, 4)):
        bx = np.random.randint(15, IMG_SIZE - 15)
        by = np.random.randint(15, IMG_SIZE - 15)
        br = np.random.randint(6, 12)
        intensity = int(np.random.choice([30, 230]))  # clearly darker or brighter
        cv2.circle(img, (bx, by), br, intensity, -1, lineType=cv2.LINE_AA)
    noise = np.random.normal(0, 3, (IMG_SIZE, IMG_SIZE))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def main():
    for label, fn in [("normal", make_normal), ("abnormal", make_abnormal)]:
        out_path = os.path.join(OUT_DIR, label)
        os.makedirs(out_path, exist_ok=True)
        for i in range(N_PER_CLASS):
            img = fn()
            cv2.imwrite(os.path.join(out_path, f"{label}_{i:03d}.png"), img)
    print(f"Dataset created: {N_PER_CLASS} normal + {N_PER_CLASS} abnormal images in '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
