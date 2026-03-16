from pathlib import Path

# =========================
# 配置区（一般不用改）
# =========================

BASE = Path(r"E:\User\ultralytics-8.3.241\datasets\Knife")

TRAIN_LABELS = BASE / "train" / "labels"
TRAIN_IMAGES = BASE / "train" / "images"
VALID_LABELS = BASE / "valid" / "labels"
VALID_IMAGES = BASE / "valid" / "images"

# 你刚才查出来的异常 label 文件名
BAD_LABELS = [
    # train
    "opencv_frame_80_png.rf.5b62b4a171f8d0f4791da13644d598b3.txt",
    "opencv_frame_80_png_jpg.rf.0a7c0ad6834813fd8ea8efef24832b3a.txt",
    "opencv_frame_8_png.rf.90dbe96b5610178f92ef56f2e629809e.txt",
    "opencv_frame_8_png_jpg.rf.f73f5bd0bf7c28b709ed6cb7df17fdae.txt",
    # valid
    "knife-8-_jpg.rf.ad21893daabb1b034f81b17223b73145.txt",
]

IMAGE_EXTS = [".jpg", ".png", ".jpeg", ".bmp", ".webp"]


def delete_one(label_path: Path, image_dir: Path):
    """删除 label 及其对应 image"""
    stem = label_path.stem  # 不带后缀的文件名

    # 删除 label
    if label_path.exists():
        label_path.unlink()
        print(f"[DEL] label: {label_path}")
    else:
        print(f"[MISS] label 不存在: {label_path}")

    # 删除 image（可能是 jpg / png 等）
    found_image = False
    for ext in IMAGE_EXTS:
        img = image_dir / (stem + ext)
        if img.exists():
            img.unlink()
            print(f"[DEL] image: {img}")
            found_image = True
            break

    if not found_image:
        print(f"[WARN] 未找到对应 image: {stem}.*")


def main():
    print("=== 开始删除异常标注及图片 ===\n")

    for name in BAD_LABELS:
        # 先在 train 查
        p = TRAIN_LABELS / name
        if p.exists():
            delete_one(p, TRAIN_IMAGES)
            continue

        # 再在 valid 查
        p = VALID_LABELS / name
        if p.exists():
            delete_one(p, VALID_IMAGES)
            continue

        print(f"[ERROR] 未在 train/valid labels 中找到: {name}")

    print("\n=== 删除完成 ===")


if __name__ == "__main__":
    main()
