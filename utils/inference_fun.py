import cv2
import numpy as np
import torch

def get_img_tensor(img_path):
    img_arr = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    img_resize = cv2.resize(img_gray, dsize=(60, 60))
    img_norm = img_resize / 255.0
    img = np.expand_dims(img_norm, 0)
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(dim=0)
    return img_tensor