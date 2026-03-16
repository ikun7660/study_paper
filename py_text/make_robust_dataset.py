import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

# ========= 路径配置（按你自己情况改） =========
SRC_VALID = r"E:\User\ultralytics-8.3.241\datasets\Knife\valid"
DST_ROOT  = r"E:\User\ultralytics-8.3.241\datasets\Knife_robust"


IMG_EXTS = (".jpg", ".png", ".jpeg")

# ========= 干扰函数 =========
def add_gaussian_noise(img, std=15):
    noise = np.random.normal(0, std, img.shape)
    out = img + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def add_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def adjust_brightness(img, factor=0.6):
    return np.clip(img * factor, 0, 255).astype(np.uint8)

# ========= 创建目录 =========
def prepare_dirs():
    for name in ["clean", "noise", "blur", "dark"]:
        os.makedirs(os.path.join(DST_ROOT, name, "images"), exist_ok=True)
        os.makedirs(os.path.join(DST_ROOT, name, "labels"), exist_ok=True)

def main():
    prepare_dirs()

    img_dir = os.path.join(SRC_VALID, "images")
    label_dir = os.path.join(SRC_VALID, "labels")

    for img_name in tqdm(os.listdir(img_dir)):
        if not img_name.lower().endswith(IMG_EXTS):
            continue

        img_path = os.path.join(img_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # clean
        cv2.imwrite(os.path.join(DST_ROOT, "clean", "images", img_name), img)

        # noise
        noisy = add_gaussian_noise(img)
        cv2.imwrite(os.path.join(DST_ROOT, "noise", "images", img_name), noisy)

        # blur
        blurred = add_blur(img)
        cv2.imwrite(os.path.join(DST_ROOT, "blur", "images", img_name), blurred)

        # dark
        dark = adjust_brightness(img)
        cv2.imwrite(os.path.join(DST_ROOT, "dark", "images", img_name), dark)

        # labels 原样复制
        for name in ["clean", "noise", "blur", "dark"]:
            shutil.copy(label_path, os.path.join(DST_ROOT, name, "labels"))

    print("鲁棒性验证集生成完成")

if __name__ == "__main__":
    main()
