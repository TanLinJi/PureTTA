import json
import os
import subprocess
import urllib.request


# 1) hf-mirror 的目录树 API 根路径（只负责列目录）
API_TREE_BASE = "https://hf-mirror.com/api/datasets/auniquesun/Point-PRC/tree/main"
# 2) 文件下载根路径（配合 resolve/main）
DOWNLOAD_BASE = "https://hf-mirror.com/datasets/auniquesun/Point-PRC/resolve/main"

# omniobject3d 在数据集仓库中的相对路径
ROOT_PATH = "new-3ddg-benchmarks/xset/dg/omniobject3d"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(SCRIPT_DIR)
SAVE_DIR = os.path.join(DATA_DIR, "omniobject3d")


def fetch_entries(tree_path):
    api_url = f"{API_TREE_BASE}/{tree_path}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def collect_all_files(root_path):
    file_paths = []
    stack = [root_path]

    while stack:
        current = stack.pop()
        print(f"扫描目录: {current}")
        entries = fetch_entries(current)

        for entry in entries:
            entry_type = entry.get("type")
            entry_path = entry.get("path")
            if not entry_path:
                continue

            if entry_type == "file":
                file_paths.append(entry_path)
            elif entry_type in ("directory", "dir"):
                stack.append(entry_path)

    return sorted(file_paths)


def to_local_relative_path(full_path, root_path):
    prefix = root_path + "/"
    if full_path.startswith(prefix):
        return full_path[len(prefix) :]

    # 兜底：如果返回路径不含上面的前缀，就退化到文件名。
    return full_path.split("/")[-1]


def download_one(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cmd = ["wget", "-c", "--show-progress", url, "-O", save_path]
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"开始扫描 omniobject3d 文件树: {ROOT_PATH}")
    try:
        all_files = collect_all_files(ROOT_PATH)
    except Exception as exc:
        print(f"获取目录树失败，请检查网络或地址: {exc}")
        raise SystemExit(1)

    print(f"共发现 {len(all_files)} 个文件，开始下载...")

    failed = []
    for full_path in all_files:
        rel_path = to_local_relative_path(full_path, ROOT_PATH)
        file_url = f"{DOWNLOAD_BASE}/{full_path}"
        save_path = os.path.join(SAVE_DIR, rel_path)

        print(f"\n>>> 正在处理: {rel_path}")
        ok = download_one(file_url, save_path)
        if not ok:
            failed.append(rel_path)

    if failed:
        print("\n以下文件下载失败：")
        for item in failed:
            print(f"- {item}")
        raise SystemExit(1)

    print("\nomniobject3d 文件夹全部下载完成！")


if __name__ == "__main__":
    main()
