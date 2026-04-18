import os
import shutil
import subprocess
import zipfile


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(SCRIPT_DIR)
ZIP_PATH = os.path.join(DATA_DIR, "objaverse_lvis.zip")
TARGET_DIR = os.path.join(DATA_DIR, "objaverse_lvis")

DOWNLOAD_URLS = [
    "https://hf-mirror.com/datasets/auniquesun/Point-Cache/resolve/main/objaverse_lvis.zip",
    "https://huggingface.co/datasets/auniquesun/Point-Cache/resolve/main/objaverse_lvis.zip",
]


def download_zip(zip_path):
    for url in DOWNLOAD_URLS:
        print(f"Trying to download from: {url}")
        cmd = ["wget", "-c", "--show-progress", url, "-O", zip_path]
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print("Download succeeded.")
            return True
        print("Download failed on this source, trying next source...")

    return False


def normalize_member_path(member_name):
    name = member_name.replace("\\", "/").lstrip("/")
    while name.startswith("./"):
        name = name[2:]

    if not name or name.endswith("/"):
        return None

    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts:
        return None

    # If the archive is packed with a top-level folder named objaverse_lvis,
    # strip it to avoid creating objaverse_lvis/objaverse_lvis.
    if parts[0] == "objaverse_lvis":
        parts = parts[1:]

    if not parts:
        return None

    rel_path = os.path.normpath(os.path.join(*parts))
    if rel_path == ".." or rel_path.startswith(".." + os.sep):
        raise ValueError(f"Unsafe path in zip: {member_name}")

    return rel_path


def extract_zip(zip_path, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = []
        for info in zf.infolist():
            rel_path = normalize_member_path(info.filename)
            if rel_path is not None:
                members.append((info, rel_path))

        total = len(members)
        print(f"Extracting {total} files to: {target_dir}")

        for idx, (info, rel_path) in enumerate(members, start=1):
            dst_path = os.path.join(target_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            with zf.open(info, "r") as src, open(dst_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            if idx % 500 == 0 or idx == total:
                print(f"Extracted {idx}/{total}")


def main():
    marker_file = os.path.join(TARGET_DIR, "lvis_testset.txt")

    if os.path.isfile(marker_file):
        print(f"Found existing extracted data: {marker_file}")
        print("Skip extraction. If you want to re-extract, delete objaverse_lvis directory first.")
        return

    ok = download_zip(ZIP_PATH)
    if not ok:
        print("All download sources failed.")
        raise SystemExit(1)

    if not zipfile.is_zipfile(ZIP_PATH):
        print(f"Downloaded file is not a valid zip archive: {ZIP_PATH}")
        raise SystemExit(1)

    extract_zip(ZIP_PATH, TARGET_DIR)

    if os.path.isfile(marker_file):
        print("objaverse_lvis download and extraction completed.")
    else:
        print("Extraction completed, but lvis_testset.txt is not found. Please verify archive contents.")


if __name__ == "__main__":
    main()
