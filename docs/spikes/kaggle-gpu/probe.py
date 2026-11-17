"""Where Kaggle mounts a kernel's output and a dataset when a kernel lists them as sources."""

import json
import os

out = {"input_tree": [], "sizes": {}}
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.count(os.sep) - "/kaggle/input".count(os.sep)
    if depth > 7:
        dirs[:] = []
        continue
    for name in sorted(files)[:40]:
        path = os.path.join(root, name)
        out["input_tree"].append(path)
        out["sizes"][path] = os.path.getsize(path)
    out["input_tree"].append(root + "/")
out["input_tree"].sort()
print(json.dumps(out, indent=1))
with open("/kaggle/working/mount-probe.json", "w") as f:
    json.dump(out, f, indent=1)
