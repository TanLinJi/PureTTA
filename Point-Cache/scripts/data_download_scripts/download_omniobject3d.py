import json
import os
import subprocess
import urllib.request
from urllib.parse import quote


# 1) hf-mirror 的目录树 API 根路径（只负责列目录）
API_TREE_BASE = "https://hf-mirror.com/api/datasets/auniquesun/Point-PRC/tree/main"
# 2) 文件下载根路径（配合 resolve/main）
DOWNLOAD_BASE = "https://hf-mirror.com/datasets/auniquesun/Point-PRC/resolve/main"

# omniobject3d 在数据集仓库中的相对路径
ROOT_PATH = "new-3ddg-benchmarks/xset/dg/omniobject3d"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SAVE_DIR = os.path.join(DATA_DIR, "omniobject3d")


def fetch_entries(tree_path):
    clean_tree_path = normalize_repo_path(tree_path)
    api_url = f"{API_TREE_BASE}/{quote(clean_tree_path, safe='/')}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_repo_path(path):
    return str(path).replace("\\", "/").strip().lstrip("/")


def to_local_relative_path(full_path, root_path):
    clean_full = normalize_repo_path(full_path)
    clean_root = normalize_repo_path(root_path).rstrip("/")
    prefix = clean_root + "/"

    if clean_full.startswith(prefix):
        rel_path = clean_full[len(prefix) :]
    elif prefix in clean_full:
        rel_path = clean_full.split(prefix, 1)[1]
    elif "/" in clean_full:
        # 兜底：保留除顶层外的相对层级，避免多层目录被扁平化。
        rel_path = clean_full.split("/", 1)[1]
    else:
        rel_path = clean_full

    rel_path = rel_path.lstrip("/")
    parts = rel_path.split("/") if rel_path else []
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"非法相对路径: full_path={full_path}, root_path={root_path}")

    return rel_path


def build_download_url(full_path):
    clean_full = normalize_repo_path(full_path)
    return f"{DOWNLOAD_BASE}/{quote(clean_full, safe='/')}"


def download_one(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cmd = ["wget", "-c", "--show-progress", url, "-O", save_path]
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def walk_and_download(root_path, save_dir):
    stack = [normalize_repo_path(root_path)]
    file_count = 0
    success_count = 0
    failed = []

    while stack:
        current = stack.pop()
        print(f"扫描目录: {current}")

        try:
            entries = fetch_entries(current)
        except Exception as exc:
            failed.append((current, f"目录扫描失败: {exc}"))
            continue

        for entry in entries:
            entry_type = entry.get("type")
            entry_path = entry.get("path")
            if not entry_path:
                continue

            if entry_type in ("directory", "dir"):
                stack.append(normalize_repo_path(entry_path))
                continue

            if entry_type != "file":
                continue

            file_count += 1
            try:
                rel_path = to_local_relative_path(entry_path, root_path)
            except ValueError as exc:
                failed.append((entry_path, str(exc)))
                continue

            file_url = build_download_url(entry_path)
            save_path = os.path.join(save_dir, rel_path)

            print(f"\n>>> 正在处理: {rel_path}")
            ok = download_one(file_url, save_path)
            if ok:
                success_count += 1
            else:
                failed.append((rel_path, file_url))

            if file_count % 200 == 0:
                print(f"进度: 已处理 {file_count} 个文件，成功 {success_count} 个")

    return file_count, success_count, failed


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"数据将下载到: {SAVE_DIR}")

    print(f"开始扫描并下载 omniobject3d 文件树: {ROOT_PATH}")
    total, success, failed = walk_and_download(ROOT_PATH, SAVE_DIR)

    print(f"\n处理完成: 共处理 {total} 个文件，成功 {success} 个")

    if failed:
        print("\n以下文件下载失败：")
        for item in failed:
            print(f"- {item[0]} -> {item[1]}")
        raise SystemExit(1)

    print("\nomniobject3d 文件夹全部下载完成！")


if __name__ == "__main__":
    main()
