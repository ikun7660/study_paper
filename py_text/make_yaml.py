import os

# 你的 knife_robust 根目录（按你实际生成的位置改）
ROBUST_ROOT = r"E:\User\ultralytics-8.3.241\datasets\knife_robust"

# 写进 yaml 的 path 字段（建议用 /，Windows 更稳）
# 注意：这里要指向每个子集目录：.../clean  .../noise ...
ROBUST_ROOT_FOR_YAML = "E:/User/ultralytics-8.3.241/datasets/knife_robust"

SUBSETS = ["clean", "noise", "blur", "dark"]

YAML_TEMPLATE = """path: {path}
train: images
val: images

names:
  0: knife
"""


def main():
    for s in SUBSETS:
        subset_dir = os.path.join(ROBUST_ROOT, s)
        os.makedirs(subset_dir, exist_ok=True)

        yaml_path = os.path.join(subset_dir, "data.yaml")
        yaml_content = YAML_TEMPLATE.format(path=f"{ROBUST_ROOT_FOR_YAML}/{s}")

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        print(f"Written: {yaml_path}")

    print("All data.yaml generated.")


if __name__ == "__main__":
    main()
