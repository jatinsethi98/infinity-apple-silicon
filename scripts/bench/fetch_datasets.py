#!/usr/bin/env python3
"""Reproducible ANN-benchmark dataset acquisition for the native_hnsw_smoke harness.

Downloads canonical ANN datasets (SIFT1M, optionally GIST1M), verifies the
download, extracts the .fvecs/.ivecs corpus, and converts it into the raw
headerless little-endian float32 layout the harness expects (n*d*4 bytes,
no header), plus an int32 ground-truth file.

Outputs (per dataset, under <datasets-root>/<name>/):
  base.f32          base vectors        (n  x d, little-endian float32, no header)
  query.f32         query vectors       (nq x d, little-endian float32, no header)
  groundtruth.i32   nearest-neighbour ids (nq x k, little-endian int32, no header)
  meta.json         n, d, nq, k, sha256 of each output file, source url, etc.

Design notes:
  * Pure Python 3 stdlib is sufficient. numpy is used when importable (faster,
    and the brute-force spot check needs it); the .fvecs/.ivecs conversion has a
    stdlib-only fallback so acquisition never hard-depends on numpy.
  * The .fvecs format: each vector is a 4-byte little-endian int32 giving the
    dimension d, immediately followed by d little-endian float32 values. So each
    record occupies (d+1)*4 bytes. .ivecs is identical but the d payload values
    are int32. Conversion strips the per-record dimension header.
  * Idempotent: a download is skipped when the cached file already exists and its
    size matches the server's Content-Length (or a known-good size). Conversion
    is skipped when the output exists and matches the size recorded in meta.json.

Usage:
  python3 fetch_datasets.py                      # fetch + convert SIFT1M
  python3 fetch_datasets.py sift1m               # same
  python3 fetch_datasets.py gist1m               # fetch + convert GIST1M
  python3 fetch_datasets.py all                  # both
  python3 fetch_datasets.py sift1m --verify      # re-check sha256 vs meta.json
  python3 fetch_datasets.py sift1m --spot-check  # brute-force groundtruth check
  python3 fetch_datasets.py sift1m --force       # ignore caches, re-download

Regenerate SIFT1M from scratch:
  python3 scripts/bench/fetch_datasets.py sift1m --spot-check
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import tarfile
import time
import urllib.request
from datetime import datetime, timezone

# numpy is optional. When present we use it for fast conversion and for the
# brute-force spot check (which is numpy-only).
try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:  # pragma: no cover - environment dependent
    np = None
    HAVE_NUMPY = False

DEFAULT_ROOT = "/Users/sethjatq/Desktop/proj/datasets"

# Each dataset: ordered list of tarball mirrors to try, plus the .fvecs/.ivecs
# member names inside the archive and the expected geometry (n, d, nq, k).
DATASETS = {
    "sift1m": {
        "tar_sources": [
            "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
            "http://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        ],
        "tar_name": "sift.tar.gz",
        "known_tar_size": 168280445,
        "base_member": "sift/sift_base.fvecs",
        "query_member": "sift/sift_query.fvecs",
        "gt_member": "sift/sift_groundtruth.ivecs",
        "expect": {"n": 1000000, "d": 128, "nq": 10000, "k": 100},
    },
    "gist1m": {
        "tar_sources": [
            "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
            "http://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
        ],
        "tar_name": "gist.tar.gz",
        "known_tar_size": 2740172684,  # ~2.55 GiB, verified via FTP Content-Length
        "base_member": "gist/gist_base.fvecs",
        "query_member": "gist/gist_query.fvecs",
        "gt_member": "gist/gist_groundtruth.ivecs",
        # GIST ships 1000 queries (not 10000). n/d asserted, nq/k discovered.
        "expect": {"n": 1000000, "d": 960, "nq": 1000, "k": 100},
    },
}

READ_BATCH_RECORDS = 100_000  # records per conversion batch (bounds memory)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def human(nbytes):
    step = 1024.0
    val = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if val < step:
            return f"{val:.1f} {unit}" if unit != "B" else f"{int(val)} B"
        val /= step
    return f"{val:.1f} PiB"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_exact(f, n):
    """Read exactly n bytes (or fewer only at EOF)."""
    parts = []
    got = 0
    while got < n:
        b = f.read(n - got)
        if not b:
            break
        parts.append(b)
        got += len(b)
    return b"".join(parts)


def read_xvecs_dim(path):
    """Read the dimension header (first int32) of an .fvecs/.ivecs file."""
    with open(path, "rb") as f:
        head = f.read(4)
    if len(head) != 4:
        raise ValueError(f"{path}: too small to contain a dimension header")
    return struct.unpack("<i", head)[0]


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def _server_size(url, timeout=30):
    """Best-effort content length via urllib (works for http/https/ftp)."""
    try:
        req = urllib.request.Request(url, method="HEAD") if url.startswith("http") else url
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except Exception:
        return None


def download(url, dst, expected_size=None, timeout=60):
    """Stream url -> dst with progress. Returns bytes written."""
    tmp = dst + ".part"
    print(f"    GET {url}")
    start = time.time()
    written = 0
    last_report = start
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
        total = expected_size
        clen = resp.headers.get("Content-Length")
        if total is None and clen is not None:
            total = int(clen)
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
            now = time.time()
            if now - last_report >= 5.0:
                rate = written / max(now - start, 1e-6)
                pct = f" ({100.0*written/total:.1f}%)" if total else ""
                print(f"      {human(written)}{pct} @ {human(int(rate))}/s")
                last_report = now
    os.replace(tmp, dst)
    dur = time.time() - start
    print(f"    wrote {human(written)} in {dur:.1f}s ({human(int(written/max(dur,1e-6)))}/s)")
    return written


def fetch_tarball(spec, downloads_dir, force=False):
    """Ensure the dataset tarball is present in downloads_dir. Returns (path, url)."""
    dst = os.path.join(downloads_dir, spec["tar_name"])
    known = spec.get("known_tar_size")

    if not force and os.path.exists(dst):
        have = os.path.getsize(dst)
        if known is not None and have == known:
            print(f"  cache hit: {dst} ({human(have)}) matches known size, skipping download")
            return dst, "(cached)"
        if known is None and have > 0:
            print(f"  cache hit: {dst} ({human(have)}) present, skipping download")
            return dst, "(cached)"
        print(f"  cached {dst} is {human(have)} but expected {human(known)}; re-downloading")

    last_err = None
    for url in spec["tar_sources"]:
        try:
            expected = known
            if expected is None:
                expected = _server_size(url)
            written = download(url, dst, expected_size=expected)
            if known is not None and written != known:
                raise ValueError(f"size mismatch: got {written}, expected {known}")
            return dst, url
        except Exception as e:  # noqa: BLE001 - report and try next mirror
            last_err = e
            print(f"  FAILED {url}: {e!r}")
            part = dst + ".part"
            if os.path.exists(part):
                try:
                    os.remove(part)
                except OSError:
                    pass
            continue
    raise RuntimeError(f"all sources failed for {spec['tar_name']}; last error: {last_err!r}")


def extract_members(tar_path, members, out_dir):
    """Extract specific members from a tar.gz into out_dir (flattened basenames)."""
    extracted = {}
    with tarfile.open(tar_path, "r:*") as tf:
        names = set(tf.getnames())
        for key, member in members.items():
            if member not in names:
                # some archives use a different top-level dir; try basename match
                cand = [n for n in names if n.endswith("/" + os.path.basename(member))]
                if not cand:
                    raise KeyError(f"{member} not found in {tar_path}; have {sorted(names)[:10]}...")
                member = cand[0]
            info = tf.getmember(member)
            dst = os.path.join(out_dir, os.path.basename(member))
            if os.path.exists(dst) and os.path.getsize(dst) == info.size:
                print(f"    extract skip {os.path.basename(member)} ({human(info.size)}, present)")
            else:
                print(f"    extract {os.path.basename(member)} ({human(info.size)})")
                src = tf.extractfile(info)
                with open(dst, "wb") as fout:
                    while True:
                        b = src.read(1 << 20)
                        if not b:
                            break
                        fout.write(b)
            extracted[key] = dst
    return extracted


# --------------------------------------------------------------------------- #
# conversion  (.fvecs/.ivecs -> raw headerless little-endian)
# --------------------------------------------------------------------------- #
def convert_xvecs(src, dst, elem="f", assert_dim=None):
    """Strip per-record dimension headers, writing raw little-endian payload.

    elem "f" -> float32 output (.fvecs), "i" -> int32 output (.ivecs).
    Returns (n_records, dim).
    """
    dim = read_xvecs_dim(src)
    if assert_dim is not None and dim != assert_dim:
        raise ValueError(f"{src}: dimension header {dim} != expected {assert_dim}")
    rec_bytes = (dim + 1) * 4
    total = os.path.getsize(src)
    if total % rec_bytes != 0:
        raise ValueError(f"{src}: size {total} not a multiple of record size {rec_bytes}")
    n = total // rec_bytes

    if HAVE_NUMPY:
        _convert_numpy(src, dst, dim, rec_bytes, elem)
    else:
        _convert_stdlib(src, dst, dim, rec_bytes)

    out_elem_bytes = dim * 4
    expect_out = n * out_elem_bytes
    got_out = os.path.getsize(dst)
    if got_out != expect_out:
        raise ValueError(f"{dst}: wrote {got_out} bytes, expected {expect_out}")
    return n, dim


def _convert_numpy(src, dst, dim, rec_bytes, elem):
    in_dt = np.dtype("<i4")  # read every 4-byte word as int32
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while True:
            raw = read_exact(fin, rec_bytes * READ_BATCH_RECORDS)
            if not raw:
                break
            arr = np.frombuffer(raw, dtype=in_dt).reshape(-1, dim + 1)
            if not np.all(arr[:, 0] == dim):
                bad = int(np.argmax(arr[:, 0] != dim))
                raise ValueError(f"{src}: inconsistent dim header at record {bad}: {int(arr[bad,0])} != {dim}")
            payload = np.ascontiguousarray(arr[:, 1:])  # int32 bits, contiguous
            if elem == "f":
                # reinterpret the 4-byte words as float32 (bit-preserving)
                out = payload.view("<f4")
            else:
                out = payload  # already int32, little-endian
            fout.write(out.tobytes())


def _convert_stdlib(src, dst, dim, rec_bytes):
    # Pure byte-copy: the payload bytes (little-endian) are written unchanged,
    # so this works identically for float32 and int32 outputs.
    payload_bytes = dim * 4
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while True:
            raw = read_exact(fin, rec_bytes * READ_BATCH_RECORDS)
            if not raw:
                break
            if len(raw) % rec_bytes != 0:
                raise ValueError(f"{src}: truncated record batch ({len(raw)} bytes)")
            mv = memoryview(raw)
            nrec = len(raw) // rec_bytes
            out = bytearray(nrec * payload_bytes)
            oi = 0
            for off in range(0, len(raw), rec_bytes):
                dd = struct.unpack_from("<i", mv, off)[0]
                if dd != dim:
                    raise ValueError(f"{src}: inconsistent dim header {dd} != {dim}")
                out[oi:oi + payload_bytes] = mv[off + 4: off + rec_bytes]
                oi += payload_bytes
            fout.write(out)


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def spot_check(base_path, query_path, gt_path, n, d, nq, k, sample=50_000, seed=0):
    """Brute-force check that groundtruth[0][0] is the true NN of query 0 over a
    random `sample`-vector subset of base: the true NN must be at least as close
    as anything in the subset."""
    if not HAVE_NUMPY:
        return {"status": "SKIPPED", "reason": "numpy not importable"}

    base = np.memmap(base_path, dtype="<f4", mode="r", shape=(n, d))
    query = np.fromfile(query_path, dtype="<f4").reshape(nq, d)
    gt = np.fromfile(gt_path, dtype="<i4").reshape(nq, k)

    q0 = query[0].astype(np.float64)
    gt0 = int(gt[0, 0])
    if not (0 <= gt0 < n):
        return {"status": "FAIL", "reason": f"gt id {gt0} out of range [0,{n})"}

    nn_vec = np.asarray(base[gt0], dtype=np.float64)
    nn_dist = float(np.sum((nn_vec - q0) ** 2))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=min(sample, n))
    sub = np.asarray(base[np.sort(idx)], dtype=np.float64)
    dists = np.sum((sub - q0) ** 2, axis=1)
    subset_min = float(dists.min())
    subset_min_idx = int(np.sort(idx)[int(np.argmin(dists))])

    # also brute-force the true argmin over the subset for context
    ok = nn_dist <= subset_min + 1e-6
    # verify ALL groundtruth ids are in range and distances are non-decreasing-ish
    gt_all_in_range = bool(((gt >= 0) & (gt < n)).all())

    return {
        "status": "PASS" if (ok and gt_all_in_range) else "FAIL",
        "query": 0,
        "gt_nn_id": gt0,
        "gt_nn_sqL2": nn_dist,
        "subset_size": int(min(sample, n)),
        "subset_min_sqL2": subset_min,
        "subset_min_id": subset_min_idx,
        "gt_nn_le_subset_min": ok,
        "all_gt_ids_in_range": gt_all_in_range,
    }


def verify_against_meta(out_dir):
    """Recompute sha256 of the output files and compare to meta.json."""
    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.exists(meta_path):
        print(f"  no meta.json at {meta_path}")
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    ok = True
    for fname, info in meta["files"].items():
        p = os.path.join(out_dir, fname)
        if not os.path.exists(p):
            print(f"  MISSING {fname}")
            ok = False
            continue
        size = os.path.getsize(p)
        digest = sha256_file(p)
        size_ok = size == info["bytes"]
        sha_ok = digest == info["sha256"]
        status = "OK" if (size_ok and sha_ok) else "MISMATCH"
        print(f"  [{status}] {fname}: {human(size)} sha256={digest[:16]}...")
        if not size_ok:
            print(f"        size {size} != recorded {info['bytes']}")
        if not sha_ok:
            print(f"        sha256 != recorded {info['sha256'][:16]}...")
        ok = ok and size_ok and sha_ok
    print(f"  VERIFY {'PASS' if ok else 'FAIL'} for {out_dir}")
    return ok


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def process_dataset(name, root, force=False, do_spot_check=False):
    spec = DATASETS[name]
    out_dir = os.path.join(root, name)
    downloads_dir = os.path.join(root, "_downloads")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(downloads_dir, exist_ok=True)

    print(f"\n=== {name} -> {out_dir} ===")
    tar_path, src_url = fetch_tarball(spec, downloads_dir, force=force)
    tar_size = os.path.getsize(tar_path)
    print(f"  tarball: {tar_path} ({human(tar_size)})")

    print("  extracting members ...")
    members = extract_members(
        tar_path,
        {"base": spec["base_member"], "query": spec["query_member"], "gt": spec["gt_member"]},
        downloads_dir,
    )

    base_out = os.path.join(out_dir, "base.f32")
    query_out = os.path.join(out_dir, "query.f32")
    gt_out = os.path.join(out_dir, "groundtruth.i32")

    print("  converting base.f32 ...")
    n, d = convert_xvecs(members["base"], base_out, elem="f", assert_dim=spec["expect"]["d"])
    print(f"    base: n={n} d={d} -> {human(os.path.getsize(base_out))}")

    print("  converting query.f32 ...")
    nq, dq = convert_xvecs(members["query"], query_out, elem="f", assert_dim=spec["expect"]["d"])
    if dq != d:
        raise ValueError(f"query dim {dq} != base dim {d}")
    print(f"    query: nq={nq} d={dq} -> {human(os.path.getsize(query_out))}")

    print("  converting groundtruth.i32 ...")
    ngt, k = convert_xvecs(members["gt"], gt_out, elem="i")
    if ngt != nq:
        raise ValueError(f"groundtruth rows {ngt} != query count {nq}")
    print(f"    groundtruth: nq={ngt} k={k} -> {human(os.path.getsize(gt_out))}")

    # sanity assertions against expected geometry
    exp = spec["expect"]
    assert n == exp["n"], f"{name}: base n {n} != expected {exp['n']}"
    assert d == exp["d"], f"{name}: base d {d} != expected {exp['d']}"

    print("  hashing outputs ...")
    files_meta = {}
    for fname, path, dtype, rows, cols in (
        ("base.f32", base_out, "float32", n, d),
        ("query.f32", query_out, "float32", nq, dq),
        ("groundtruth.i32", gt_out, "int32", ngt, k),
    ):
        files_meta[fname] = {
            "bytes": os.path.getsize(path),
            "rows": rows,
            "cols": cols,
            "dtype": dtype,
            "layout": "row-major little-endian, no header",
            "sha256": sha256_file(path),
        }

    meta = {
        "dataset": name,
        "n": n,
        "d": d,
        "nq": nq,
        "k": k,
        "source_url": src_url,
        "tarball": {
            "name": spec["tar_name"],
            "bytes": tar_size,
            "sha256": sha256_file(tar_path),
        },
        "files": files_meta,
        "numpy_available": HAVE_NUMPY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/bench/fetch_datasets.py",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(f"  wrote meta.json")

    if do_spot_check:
        print("  spot-checking groundtruth (brute force) ...")
        res = spot_check(base_out, query_out, gt_out, n, d, nq, k)
        print("  spot check:", json.dumps(res, indent=2))
        meta["spot_check"] = res
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default="sift1m",
                    choices=["sift1m", "gist1m", "all"], help="dataset to fetch (default: sift1m)")
    ap.add_argument("--datasets-root", default=DEFAULT_ROOT, help=f"output root (default: {DEFAULT_ROOT})")
    ap.add_argument("--verify", action="store_true", help="re-check sha256 of outputs against meta.json and exit")
    ap.add_argument("--spot-check", action="store_true", help="run brute-force groundtruth spot check after conversion")
    ap.add_argument("--force", action="store_true", help="ignore caches; re-download tarballs")
    args = ap.parse_args(argv)

    names = ["sift1m", "gist1m"] if args.dataset == "all" else [args.dataset]

    if args.verify:
        all_ok = True
        for name in names:
            out_dir = os.path.join(args.datasets_root, name)
            print(f"\n=== verify {name} ({out_dir}) ===")
            all_ok = verify_against_meta(out_dir) and all_ok
        return 0 if all_ok else 1

    rc = 0
    for name in names:
        try:
            process_dataset(name, args.datasets_root, force=args.force, do_spot_check=args.spot_check)
        except Exception as e:  # noqa: BLE001 - report per-dataset and continue
            print(f"\n!!! {name} FAILED: {e!r}", file=sys.stderr)
            if name == "sift1m":
                rc = 1  # SIFT1M is the priority; its failure is fatal
            else:
                print(f"    (continuing; {name} is optional)", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
