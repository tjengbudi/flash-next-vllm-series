#!/usr/bin/env python3
"""Build the INT8 PLE n-gram table (ngram_int8.safetensors) from the BF16 shards.

The Flash-Next PLE n-gram embedding is 320,001,536 rows x 160 BF16 values
(model-extra-ngram-*.safetensors, 128 tensors ...ngram_embedding.shard_N.weight).
This writes each row as 160 int8 values followed by one fp16 scale:

    scale = fp16(max(|x|) / 127)        per row, symmetric
    q     = clip(round(x / scale), -127, 127)
    x    ~= q * scale

into one safetensors file with 128 U8 tensors ...ngram_q8.shard_N of shape
[rows, 162], stored contiguously in shard order. That is the layout the local-49
vLLM patch maps (VLLM_PLE_INT8_PATH). 52 GB instead of 100 GB; mean relative
error per row ~0.7%.

Usage:
    python3 make_ple_int8.py <model-dir> <out.safetensors>
    python3 make_ple_int8.py <model-dir> <out.safetensors> --check <other-int8-file>

Needs numpy only. Streams the input (a few hundred MB of RAM) and drops the page
cache of what it has read, so it can run next to a serving engine.
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

SRC_KEY = ".ngram_embedding.shard_"
OUT_NAME = "model.language_model.layers.1.ple.ple_embedding.ngram_q8.shard_{}"
CHUNK_ROWS = 1 << 18


def read_header(path):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    return 8 + size, header


def source_shards(model_dir):
    """[(index, path, data_start, rows, dim)] for every BF16 n-gram shard, in order."""
    shards = []
    for path in sorted(glob.glob(os.path.join(model_dir, "model-extra-ngram-*.safetensors"))):
        base, header = read_header(path)
        for name, info in header.items():
            if SRC_KEY not in name:
                continue
            if info["dtype"] != "BF16" or len(info["shape"]) != 2:
                sys.exit(f"{path}: {name} is {info['dtype']} {info['shape']}, expected BF16 2-D")
            index = int(name.split(SRC_KEY, 1)[1].split(".", 1)[0])
            rows, dim = info["shape"]
            shards.append((index, path, base + info["data_offsets"][0], rows, dim))
    shards.sort()
    if not shards:
        sys.exit(f"no model-extra-ngram-*.safetensors with {SRC_KEY}* tensors in {model_dir}")
    if [s[0] for s in shards] != list(range(len(shards))):
        sys.exit("n-gram shard indices are not 0..N-1")
    if len({s[4] for s in shards}) != 1:
        sys.exit("n-gram shards disagree on the embedding width")
    return shards


def quantize(bf16_rows):
    """uint16 BF16 bits [n, dim] -> uint8 rows [n, dim + 2] (int8 values, fp16 scale)."""
    x = (bf16_rows.astype(np.uint32) << 16).view(np.float32)
    absmax = np.abs(x).max(axis=1)
    scale = (absmax / np.float32(127)).astype(np.float16)
    s32 = scale.astype(np.float32)
    safe = np.where(s32 == 0, np.float32(1), s32)
    q = np.clip(np.rint(x / safe[:, None]), -127, 127).astype(np.int8)
    q[s32 == 0] = 0
    out = np.empty((x.shape[0], x.shape[1] + 2), dtype=np.uint8)
    out[:, :-2] = q.view(np.uint8)
    out[:, -2:] = scale.view(np.uint8).reshape(-1, 2)
    return out


def drop_cache(handle, offset, length):
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(handle.fileno(), offset, length, os.POSIX_FADV_DONTNEED)


def build(model_dir, out_path):
    shards = source_shards(model_dir)
    dim = shards[0][4]
    row_bytes = dim + 2
    header, offset = {}, 0
    for index, _, _, rows, _ in shards:
        size = rows * row_bytes
        header[OUT_NAME.format(index)] = {
            "dtype": "U8", "shape": [rows, row_bytes], "data_offsets": [offset, offset + size],
        }
        offset += size
    header["__metadata__"] = {
        "format": "pt",
        "layout": f"per row: {dim} x int8 then fp16 scale (x = q * scale)",
        "quantization": "per-row symmetric absmax/127, round to nearest even",
        "source": "model-extra-ngram-*.safetensors (BF16 PLE n-gram embedding)",
        "tool": "make_ple_int8.py",
    }
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    total_rows = sum(s[3] for s in shards)
    tmp = out_path + ".partial"
    done = 0
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(blob)) + blob)
        for index, path, start, rows, _ in shards:
            with open(path, "rb") as src:
                for first in range(0, rows, CHUNK_ROWS):
                    count = min(CHUNK_ROWS, rows - first)
                    pos = start + first * dim * 2
                    src.seek(pos)
                    raw = np.frombuffer(src.read(count * dim * 2), dtype=np.uint16)
                    written = out.tell()
                    out.write(quantize(raw.reshape(count, dim)).tobytes())
                    drop_cache(src, pos, count * dim * 2)
                    out.flush()
                    os.fdatasync(out.fileno())
                    drop_cache(out, written, count * row_bytes)
                    done += count
            print(f"shard {index + 1}/{len(shards)}  rows {done}/{total_rows}", flush=True)
    os.replace(tmp, out_path)
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e9:.1f} GB)")


def check(path, other, samples=20000):
    """Compare random rows of two INT8 tables byte for byte."""
    tables = []
    for p in (path, other):
        base, header = read_header(p)
        names = sorted(header, key=lambda n: int(n.rsplit("_", 1)[1]))
        start = base + header[names[0]]["data_offsets"][0]
        rows = sum(header[n]["shape"][0] for n in names)
        width = header[names[0]]["shape"][1]
        tables.append(np.memmap(p, dtype=np.uint8, mode="r", offset=start, shape=(rows, width)))
    a, b = tables
    if a.shape != b.shape:
        sys.exit(f"shapes differ: {a.shape} vs {b.shape}")
    idx = np.sort(np.random.default_rng(0).choice(a.shape[0], samples, replace=False))
    diff = int((a[idx] != b[idx]).any(axis=1).sum())
    print(f"{samples} random rows: {samples - diff} identical, {diff} differ")
    return diff == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--check", metavar="INT8_FILE", help="compare the result with another INT8 table")
    args = ap.parse_args()
    if not os.path.exists(args.out):
        build(args.model_dir, args.out)
    if args.check:
        sys.exit(0 if check(args.out, args.check) else 1)


if __name__ == "__main__":
    main()
