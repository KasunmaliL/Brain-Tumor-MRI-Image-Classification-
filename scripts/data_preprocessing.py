import numpy as np
import cv2
import os

IMAGE_SIZE = 224

def load_images(folder_path):
    images, labels = [], []
    classes = os.listdir(folder_path)
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(folder_path, cls)
        for img_file in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))/255.0
            images.append(img)
            labels.append(idx)
    return np.array(images), np.array(labels)
