import json
import os
import subprocess
import urllib.request


API_URL = "https://hf-mirror.com/api/datasets/auniquesun/Point-Cache/tree/main/modelnet40"
DOWNLOAD_BASE = "https://hf-mirror.com/datasets/auniquesun/Point-Cache/resolve/main/modelnet40"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(SCRIPT_DIR)
SAVE_DIR = os.path.join(DATA_DIR, "modelnet40")
PATH_PREFIX = "modelnet40/"


def fetch_file_entries(api_url):
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as response:
        entries = json.loads(response.read().decode("utf-8"))

    return [entry for entry in entries if entry.get("type") == "file"]


def to_relative_path(full_path, prefix):
    if full_path.startswith(prefix):
        return full_path[len(prefix) :]

    if "/" in full_path:
        return full_path.split("/", 1)[1]

    return full_path


def download_file(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cmd = ["wget", "-c", "--show-progress", url, "-O", save_path]
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Requesting dataset tree: {API_URL}")
    try:
        file_entries = fetch_file_entries(API_URL)
    except Exception as exc:
        print(f"Failed to fetch file list: {exc}")
        raise SystemExit(1)

    print(f"Found {len(file_entries)} files. Starting download...")

    failed = []
    for entry in file_entries:
        rel_path = to_relative_path(entry.get("path", ""), PATH_PREFIX)
        file_url = f"{DOWNLOAD_BASE}/{rel_path}"
        save_path = os.path.join(SAVE_DIR, rel_path)

        print(f"\n>>> Downloading: {rel_path}")
        ok = download_file(file_url, save_path)
        if not ok:
            failed.append(rel_path)

    if failed:
        print("\nSome files failed to download:")
        for item in failed:
            print(f"- {item}")
        raise SystemExit(1)

    print("\nmodelnet40 download completed.")


if __name__ == "__main__":
    main()
