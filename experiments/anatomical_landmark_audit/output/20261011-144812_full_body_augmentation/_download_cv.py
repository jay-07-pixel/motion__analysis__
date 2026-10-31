import json
import urllib.request
from pathlib import Path

dest = Path(__file__).resolve().parent / "_wheels"
dest.mkdir(exist_ok=True)
d = json.load(urllib.request.urlopen("https://pypi.org/pypi/opencv-python/5.0.0.93/json"))
url = None
name = None
for u in d["urls"]:
    if (
        u["packagetype"] == "bdist_wheel"
        and "win_amd64" in u["filename"]
        and "cp37-abi3" in u["filename"]
    ):
        url = u["url"]
        name = u["filename"]
        print(name, u["size"])
        break
out = dest / name
print("downloading", url)
urllib.request.urlretrieve(url, out)
print("bytes", out.stat().st_size)
print(str(out))
