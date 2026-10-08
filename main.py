"""
main.py
-------
Binary image classification pipeline: Normal vs Abnormal

Steps:
  1. Load dataset (two classes: normal, abnormal)
  2. Preprocess        -> Gaussian Blur
  3. Edge detection    -> Canny
  4. Feature expansion -> HOG (Histogram of Oriented Gradients)
  5. Train/Test split  -> 60% train, 40% test
  6. Classifier        -> SVM (Support Vector Machine)
  7. Performance       -> Accuracy, Precision, Recall, F1, Confusion Matrix
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, ConfusionMatrixDisplay)

DATASET_DIR = "dataset"
IMG_SIZE = 128


# ---------- 1. Load dataset ----------
def load_dataset():
    images, labels = [], []
    for label, class_name in enumerate(["normal", "abnormal"]):  # 0=normal, 1=abnormal
        folder = os.path.join(DATASET_DIR, class_name)
        for fname in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
    return images, np.array(labels)


# ---------- 2. Preprocessing ----------
def preprocess(img):
    return cv2.GaussianBlur(img, (5, 5), 0)


# ---------- 3. Edge detection ----------
def detect_edges(img):
    return cv2.Canny(img, 50, 150)


# ---------- 4. Feature expansion (HOG) ----------
def extract_features(edge_img):
    features = hog(edge_img, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), feature_vector=True)
    return features


def build_feature_matrix(images):
    feats = []
    for img in images:
        pre = preprocess(img)
        edges = detect_edges(pre)
        feats.append(extract_features(edges))
    return np.array(feats)


# ---------- Visualize the pipeline stages (saved as an image) ----------
def save_pipeline_visualization(images, labels):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    col_titles = ["Original", "Preprocessed\n(Gaussian Blur)", "Edges\n(Canny)", "Features\n(HOG)"]

    for row, class_name in enumerate(["normal", "abnormal"]):
        label = 0 if class_name == "normal" else 1
        idx = int(np.where(labels == label)[0][0])
        img = images[idx]
        pre = preprocess(img)
        edges = detect_edges(pre)
        _, hog_img = hog(edges, orientations=9, pixels_per_cell=(16, 16),
                          cells_per_block=(2, 2), visualize=True)

        for col, stage_img in enumerate([img, pre, edges, hog_img]):
            ax = axes[row, col]
            ax.imshow(stage_img, cmap="gray")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11)
        axes[row, 0].set_ylabel(class_name.capitalize(), fontsize=12)

    fig.suptitle("Pipeline Stages: Preprocessing -> Edge Detection -> Feature Extraction", fontsize=13)
    plt.tight_layout()
    plt.savefig("pipeline_stages.png", bbox_inches="tight", dpi=150)
    print("Saved pipeline stages image -> pipeline_stages.png")


def main():
    print("Loading dataset...")
    images, labels = load_dataset()
    print(f"Total images: {len(images)}  (normal={np.sum(labels==0)}, abnormal={np.sum(labels==1)})")

    print("Preprocessing + edge detection + feature extraction...")
    save_pipeline_visualization(images, labels)
    X = build_feature_matrix(images)
    y = labels

    # ---------- 5. Train/Test split (60% train, 40% test) ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y)
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # ---------- 6. Classifier (SVM) ----------
    clf = SVC(kernel="linear", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # ---------- 7. Performance measures ----------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Sensitivity (SC) and Specificity (SP)
    tn, fp, fn, tp = cm.ravel()

    sc = tp / (tp + fn)   # Sensitivity
    sp = tn / (tn + fp)   # Specificity

    print("\n--- Performance Measures ---")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {rec:.4f}")
    print(f"F1-score    : {f1:.4f}")
    print(f"SC (Sensitivity) : {sc:.4f}")
    print(f"SP (Specificity) : {sp:.4f}")
    print("Confusion Matrix:\n", cm)

    # ---------- Final output image (confusion matrix + metrics) ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Abnormal"])
    disp.plot(cmap="Blues", ax=ax1, colorbar=False)
    ax1.set_title("Confusion Matrix")

    ax2.axis("off")
    metrics_text = (
        f"Performance Measures\n\n"
        f"Accuracy      : {acc:.4f}\n"
        f"Precision     : {prec:.4f}\n"
        f"Recall        : {rec:.4f}\n"
        f"F1-score      : {f1:.4f}\n"
        f"SC (Sensitivity): {sc:.4f}\n"
        f"SP (Specificity): {sp:.4f}\n\n"
        f"Train samples : {len(X_train)}\n"
        f"Test samples  : {len(X_test)}"
    )
    ax2.text(0.05, 0.5, metrics_text, fontsize=13, va="center", family="monospace")

    fig.suptitle("Normal vs Abnormal Classification - Final Results", fontsize=14)
    plt.tight_layout()
    plt.savefig("final_output.png", bbox_inches="tight", dpi=150)
    print("\nSaved final results image -> final_output.png")


if __name__ == "__main__":
    main()