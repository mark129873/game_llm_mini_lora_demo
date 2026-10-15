#!/usr/bin/env python3
"""Download and verify the pinned, official Qwen3.5 checkpoint."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
REPO = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
DEST = ROOT / "models" / "Qwen3.5-0.8B"
DEST.mkdir(parents=True, exist_ok=True)
source_path = DEST / "source.json"
if source_path.exists():
    source = json.loads(source_path.read_text())
    if source["repo"] != REPO or source["revision"] != REVISION:
        raise RuntimeError("Existing source.json describes a different model/revision.")
else:
    url = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=120) as response:
        metadata = json.load(response)
    source = {"repo": REPO, "revision": REVISION, "files": [
        entry for entry in metadata["siblings"]
        if entry["rfilename"].endswith((".json", ".jinja", ".safetensors"))
    ]}
    source_path.write_text(json.dumps(source, indent=2) + "\n")

def valid(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    if "lfs" in entry:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == entry["lfs"]["sha256"]
    with path.open("rb") as stream:
        digest = hashlib.sha1(f"blob {entry['size']}\0".encode())
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == entry["blobId"]

for entry in source["files"]:
    name = entry["rfilename"]
    path = DEST / name
    if not path.resolve().is_relative_to(DEST.resolve()):
        raise RuntimeError(f"Invalid model filename: {name}")
    if valid(path, entry):
        print(f"Verified: {name}", flush=True)
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{urllib.parse.quote(name)}?download=true"
    part = path.with_name(path.name + ".part")
    if "lfs" in entry:
        subprocess.run([sys.executable, str(ROOT / "download_file.py"), url, str(part),
                        "--sha256", entry["lfs"]["sha256"], "--workers", "8"], check=True)
    else:
        with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
    if not valid(part, entry):
        raise RuntimeError(f"Downloaded file failed integrity check: {name}")
    part.replace(path)
    print(f"Downloaded and verified: {name}", flush=True)
print(f"Model ready: {DEST}")
