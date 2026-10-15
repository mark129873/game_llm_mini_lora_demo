#!/usr/bin/env python3
"""Optional parallel APT download; package metadata still comes from signed APT lists."""
import concurrent.futures
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PACKAGES = "miopen-hip hipblas hipfft hiprand hipsparse hipsolver rocsolver hipsparselt rccl rocblas hipblaslt roctracer".split()
if os.geteuid() != 0:
    raise SystemExit("Run as root.")
plan = subprocess.check_output(["apt-get", "--print-uris", "-y", "--no-install-recommends", "install", *PACKAGES], text=True)
entries = []
for line in plan.splitlines():
    if not line.startswith("'https://"):
        continue
    url, filename, size, _ = shlex.split(line)
    if not url.startswith("https://repo.radeon.com/") or "/" in filename:
        raise RuntimeError(f"Unexpected APT package: {url}")
    package, version, arch = filename.rsplit("_", 2)
    metadata = subprocess.check_output(["apt-cache", "show", f"{package}={version}"], text=True)
    sha = re.search(r"^SHA256: ([0-9a-f]{64})$", metadata, re.M).group(1)
    entries.append(dict(url=url, filename=filename, bytes=int(size), sha256=sha))
(ROOT / "logs" / "rocm-package-downloads.json").write_text(json.dumps(entries, indent=2) + "\n")
def download(entry):
    target = Path("/var/cache/apt/archives") / entry["filename"]
    part = target.with_name(target.name + ".part")
    subprocess.run([sys.executable, str(ROOT / "download_file.py"), entry["url"], str(part),
                    "--sha256", entry["sha256"], "--workers", "8"], check=True)
    part.replace(target)
    return entry["filename"]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for name in pool.map(download, entries):
        print("APT package verified:", name, flush=True)
print("APT packages ready.")
