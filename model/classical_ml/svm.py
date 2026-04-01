import os
import cv2
import numpy as np
from sklearn import svm
import joblib

MODEL_PATH = "model/classical_ml/svm_model.pkl"

def train_svm():
    # 🔥 LOAD IF ALREADY TRAINED
    if os.path.exists(MODEL_PATH):
        print("⚡ Loading saved SVM...")
        return joblib.load(MODEL_PATH)

    print("🚀 Training SVM (first time only)...")

    X, y = [], []
    base_path = "data/spectrograms/train"

    classes = [c for c in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, c))]

    for label, cls in enumerate(classes):
        cls_path = os.path.join(base_path, cls)

        for root, dirs, files in os.walk(cls_path):
            for img in files:
                if not img.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                img_path = os.path.join(root, img)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                image = cv2.resize(image, (64,64)).flatten()
                X.append(image)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    model = svm.SVC(kernel='rbf')
    model.fit(X, y)

    # 🔥 SAVE MODEL
    joblib.dump((model, classes), MODEL_PATH)

    print("✅ SVM Saved")

    return model, classes