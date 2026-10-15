#!/usr/bin/env python3
"""Download a public file in verified byte ranges, preserving a partial prefix."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("url")
p.add_argument("target", type=Path)
p.add_argument("--sha256")
p.add_argument("--workers", type=int, default=8)
args = p.parse_args()
args.target.parent.mkdir(parents=True, exist_ok=True)
req = urllib.request.Request(args.url, headers={"Range": "bytes=0-0"})
with urllib.request.urlopen(req, timeout=45) as response:
    if response.status != 206:
        raise RuntimeError("Server must support byte ranges.")
    size = int(response.headers["Content-Range"].split("/")[-1])
state_path = Path(str(args.target) + ".download.json")
if state_path.exists():
    state = json.loads(state_path.read_text())
    if state["url"] != args.url or state["size"] != size:
        raise RuntimeError("Download state belongs to a different file.")
    prefix = state["prefix"]
    done = set(state["done"])
else:
    prefix = args.target.stat().st_size if args.target.exists() else 0
    if prefix > size:
        raise RuntimeError("Existing partial file is too large.")
    done = set()
chunk_size = 16 * 2**20
chunks = [(i, start, min(start + chunk_size, size)) for i, start in enumerate(range(prefix, size, chunk_size))]
lock = threading.Lock()
fd = os.open(args.target, os.O_RDWR | os.O_CREAT, 0o644)
os.ftruncate(fd, size)
def save_state():
    state = {"url": args.url, "size": size, "prefix": prefix, "done": sorted(done)}
    tmp = Path(str(state_path) + ".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(state_path)
save_state()
def fetch(chunk):
    index, start, end = chunk
    if index in done:
        return 0
    for attempt in range(4):
        try:
            request = urllib.request.Request(args.url, headers={"Range": f"bytes={start}-{end - 1}"})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end - 1}/{size}":
                    raise RuntimeError("Unexpected range response.")
                offset = start
                while offset < end:
                    data = response.read(min(2**20, end - offset))
                    if not data:
                        raise RuntimeError("Incomplete range.")
                    os.pwrite(fd, data, offset)
                    offset += len(data)
            with lock:
                done.add(index)
                save_state()
            return end - start
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
start_time = time.monotonic()
last_print = start_time
completed = prefix + sum(end - start for index, start, end in chunks if index in done)
print(f"{args.target.name}: resume {completed / 2**20:.0f}/{size / 2**20:.0f} MiB", flush=True)
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, chunk) for chunk in chunks if chunk[0] not in done]
        for future in concurrent.futures.as_completed(futures):
            completed += future.result()
            now = time.monotonic()
            if now - last_print >= 10 or completed == size:
                print(f"{args.target.name}: {completed / 2**20:.0f}/{size / 2**20:.0f} MiB", flush=True)
                last_print = now
finally:
    os.fsync(fd)
    os.close(fd)
if completed != size:
    raise RuntimeError("Download did not complete.")
if args.sha256:
    digest = hashlib.sha256()
    with args.target.open("rb") as source:
        for block in iter(lambda: source.read(8 * 2**20), b""):
            digest.update(block)
    if digest.hexdigest() != args.sha256:
        raise RuntimeError("SHA256 mismatch.")
    print("SHA256 verified: " + digest.hexdigest(), flush=True)
state_path.unlink()
print("Completed: " + str(args.target), flush=True)
