"""Download a small random sample of the public HaGRID 30k sample set (CC BY-SA 4.0) via HTTP range requests. Images are not committed."""
import os, random
from concurrent.futures import ThreadPoolExecutor
from remotezip import RemoteZip
URL = "https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p/resolve/main/hagrid-sample-30k-384p.zip"
CLASSES = ["fist", "one", "peace", "three", "three2", "palm"]
N = 60
names = RemoteZip(URL).namelist(); random.seed(0); jobs = []
for c in CLASSES:
    fs = sorted(x for x in names if f"hagrid_30k/train_val_{c}/" in x and x.endswith(".jpg")); os.makedirs(f"data/{c}", exist_ok=True)
    jobs += [(c, x) for x in random.sample(fs, N)]
def get(j):
    c, x = j; p = f"data/{c}/{os.path.basename(x)}"
    if not os.path.exists(p): open(p, "wb").write(RemoteZip(URL).read(x))
with ThreadPoolExecutor(8) as ex: list(ex.map(get, jobs))
print("downloaded", len(jobs))
