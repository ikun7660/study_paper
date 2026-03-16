from pathlib import Path

LABEL_DIRS = [
    Path(r"E:\User\ultralytics-8.3.241\datasets\Knife\train\labels"),
    Path(r"E:\User\ultralytics-8.3.241\datasets\Knife\valid\labels"),
]


def check_one_file(p: Path):
    bad_lines = []
    try:
        text = p.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        text = p.read_text(encoding="gbk", errors="ignore").strip()

    if not text:
        return []  # 空文件（负样本）是允许的

    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # detect 的 bbox 标注必须是 5 个数：cls x y w h
        if len(parts) != 5:
            bad_lines.append((i, len(parts), line[:120]))
    return bad_lines


def main():
    total_bad_files = 0
    for d in LABEL_DIRS:
        if not d.exists():
            print(f"[SKIP] 目录不存在: {d}")
            continue
        bad_files = []
        for p in d.glob("*.txt"):
            bad = check_one_file(p)
            if bad:
                bad_files.append((p, bad))

        print(f"\n=== 扫描目录: {d} ===")
        if not bad_files:
            print("未发现异常 label（全部是 5 列 bbox 或空文件）")
        else:
            total_bad_files += len(bad_files)
            print(f"发现异常文件数: {len(bad_files)}")
            for p, bad in bad_files:
                print(f"\n[异常] {p.name}")
                for lineno, ncols, preview in bad[:3]:  # 每个文件最多预览 3 行
                    print(f"  行{lineno}: 列数={ncols} 内容预览: {preview}")

    print(f"\n总异常文件数: {total_bad_files}")
    print("提示：列数>5 通常是分割 polygon（segment）格式；列数<5 通常是损坏/空格问题。")


if __name__ == "__main__":
    main()
