import urllib.request
import json
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 1. 设置镜像站 API 路径，指向 shapenet_c 文件夹
API_URL = "https://hf-mirror.com/api/datasets/auniquesun/Point-PRC/tree/main/new-3ddg-benchmarks/xset/corruption/shapenet_c"
DOWNLOAD_BASE = "https://hf-mirror.com/datasets/auniquesun/Point-PRC/resolve/main/new-3ddg-benchmarks/xset/corruption/shapenet_c"

# 2. 准备存数据的目录（Point-Cache/data/shapenet_c）
save_dir = os.path.join(DATA_DIR, "shapenet_c")
os.makedirs(save_dir, exist_ok=True)

print(f"数据将下载到: {save_dir}")

print(f"正在向 hf-mirror 请求目录树: {API_URL}")
try:
    # 使用系统自带的 urllib，伪装一下浏览器头防止被墙
    req = urllib.request.Request(API_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as response:
        files = json.loads(response.read().decode('utf-8'))
except Exception as e:
    print(f"获取目录失败，请检查网络: {e}")
    exit(1)

# 3. 过滤出文件并逐个下载
file_list = [f for f in files if f.get("type") == "file"]
print(f"成功获取目录，共发现 {len(file_list)} 个文件。开始下载...")

for f in file_list:
    filename = f["path"].split("/")[-1]
    file_url = f"{DOWNLOAD_BASE}/{filename}"
    save_path = os.path.join(save_dir, filename)
    
    print(f"\n>>> 正在处理: {filename}")
    # 使用系统的 wget 命令，-c 参数表示断点续传
    cmd = f"wget -c -q --show-progress '{file_url}' -O '{save_path}'"
    os.system(cmd)

print("\n🎉 shapenet_c 文件夹全部下载完成！")
