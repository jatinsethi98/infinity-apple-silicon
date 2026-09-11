#!/usr/bin/env python3
"""faiss_ab.py -- FAISS-vs-FAISS build-time fairness A/B (Task 1).

Both harness faiss_hnsw_d0 binaries are byte-for-byte the same harness; the only
difference is which libfaiss.dylib they link:
  * homebrew  -> /opt/homebrew/opt/faiss/lib/libfaiss.dylib   (OpenBLAS backend)
  * fromsrc   -> native-smoke-v1/.../faiss-d0-matched/...     (Accelerate backend)
We build the SAME dataset with each, alternating order, and compare the engine's
own faiss_cold_build_ns. Purpose: decide which FAISS is the fair (faster =
conservative) baseline, and report both so the choice is auditable.

Usage:
  python3 scripts/bench/faiss_ab.py --homebrew-bin A --fromsrc-bin B \
      --dataset D --n N --d D --m 32 --efc 200 --participants 12 --runs 5
"""

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign


def one_run(binary, args):
    # The harness refuses to overwrite an existing sidecar; use a fresh path.
    # Create the directory rather than assuming it: only run_baseline.py's dataset
    # generation made it, so running this script first failed with an unhelpful
    # ENOENT from inside the harness.
    sidecar = "/tmp/d0test/faiss_ab_sidecar_%d.json" % os.getpid()
    os.makedirs(os.path.dirname(sidecar), exist_ok=True)
    if os.path.exists(sidecar):
        os.remove(sidecar)
    argv = campaign.build_argv(
        binary, "faiss", args.dataset, args.n, args.d, args.m, args.efc,
        64, args.chunk_size, args.query_count, args.participants, args.build_grain,
        sidecar)
    res = campaign.run_campaign(argv, timeout=args.timeout)
    if os.path.exists(sidecar):
        os.remove(sidecar)
    if res["timed_out"]:
        return {"ok": False, "reason": "timeout", "wall_ns": res["wall_ns"]}
    kv = {}
    for line in res["stdout"].splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            kv[k.strip()] = v.strip()
    ok = res["returncode"] == 0 and kv.get("status") == "PASS" and kv.get("faiss_valid") == "1"
    return {
        "ok": ok,
        "cold_build_ns": int(kv["faiss_cold_build_ns"]) if "faiss_cold_build_ns" in kv else None,
        "threads": int(kv["faiss_threads"]) if "faiss_threads" in kv else None,
        "recall_ef64": kv.get("faiss_recall_at_10_ef_64"),
        "recall_ef128": kv.get("faiss_recall_at_10_ef_128"),
        "edges": kv.get("faiss_graph_directed_edges"),
        "wall_ns": res["wall_ns"],
        "returncode": res["returncode"],
        "stderr_tail": res["stderr"][-400:],
    }


def summarize(name, samples):
    vals = [r["cold_build_ns"] for r in samples if r["ok"] and r["cold_build_ns"]]
    if not vals:
        return None
    med = statistics.median(vals)
    return {"median_ns": med, "min_ns": min(vals), "max_ns": max(vals),
            "n": len(vals), "ms": med / 1e6}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--homebrew-bin", required=True)
    p.add_argument("--fromsrc-bin", required=True)
    p.add_argument("--dataset", default="/tmp/d0test/d0-f32le-n12288-d128-seed0.bin")
    p.add_argument("--n", type=int, default=12288)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--efc", type=int, default=200)
    p.add_argument("--participants", type=int, default=12)
    p.add_argument("--chunk-size", type=int, default=8192)
    p.add_argument("--query-count", type=int, default=256)
    p.add_argument("--build-grain", type=int, default=1)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    engines = {"homebrew": args.homebrew_bin, "fromsrc": args.fromsrc_bin}
    results = {"homebrew": [], "fromsrc": []}
    for i in range(args.runs):
        order = ["homebrew", "fromsrc"] if i % 2 == 0 else ["fromsrc", "homebrew"]
        for tag in order:
            r = one_run(engines[tag], args)
            results[tag].append(r)
            ms = (r["cold_build_ns"] / 1e6) if r.get("cold_build_ns") else None
            print(f"run {i} {tag:9s} ok={r['ok']} threads={r.get('threads')} "
                  f"build={('%.1f ms' % ms) if ms else 'n/a'} "
                  f"recall_ef128={r.get('recall_ef128')} edges={r.get('edges')}",
                  file=sys.stderr)
            if not r["ok"]:
                print(f"  stderr: {r.get('stderr_tail')}", file=sys.stderr)

    hb = summarize("homebrew", results["homebrew"])
    fs = summarize("fromsrc", results["fromsrc"])
    print("\n== FAISS A/B (build cold_build_ns, median of runs) ==")
    for tag, s in (("homebrew(OpenBLAS)", hb), ("fromsrc(Accelerate)", fs)):
        if s:
            print(f"  {tag:22s} median={s['ms']:.1f} ms  "
                  f"min={s['min_ns']/1e6:.1f} max={s['max_ns']/1e6:.1f}  n={s['n']}")
        else:
            print(f"  {tag:22s} NO VALID RUNS")
    if hb and fs:
        faster = "fromsrc(Accelerate)" if fs["median_ns"] < hb["median_ns"] else "homebrew(OpenBLAS)"
        ratio = hb["median_ns"] / fs["median_ns"]
        print(f"  homebrew/fromsrc build-time ratio = {ratio:.3f}x  "
              f"(>1 => homebrew slower)")
        print(f"  FASTER (=> conservative baseline) = {faster}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
