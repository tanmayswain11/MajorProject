import os
import cv2
import numpy as np
from sklearn import svm

def train_svm():
    X, y = [], []
    base_path = "data/spectrograms/train"

    classes = [c for c in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, c))]
    print("Classes:", classes)

    for label, cls in enumerate(classes):
        cls_path = os.path.join(base_path, cls)

        # 🔥 RECURSIVE SEARCH (IMPORTANT)
        for root, dirs, files in os.walk(cls_path):
            for img in files:
                img_path = os.path.join(root, img)

                if not img.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                image = cv2.imread(img_path)

                if image is None:
                    print("Skipping:", img_path)
                    continue

                image = cv2.resize(image, (64,64))
                image = image.flatten()

                X.append(image)
                y.append(label)

    print("Total samples:", len(X))

    if len(X) == 0:
        raise ValueError("❌ No images found! Check dataset.")

    X = np.array(X)
    y = np.array(y)

    model = svm.SVC(kernel='rbf')
    model.fit(X, y)

    return model, classes