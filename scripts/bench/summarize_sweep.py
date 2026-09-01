#!/usr/bin/env python3
"""Summarize a buckets-per-worker sweep into one comparison table.

Reads the results.json written by each run_baseline.py invocation in the sweep and
prints build time, QPS and recall side by side, with speedups measured against the
buckets-per-worker=1 arm (the old one-bucket-per-worker behaviour).

Usage: summarize_sweep.py <results-dir>
"""
import glob
import json
import os
import re
import sys


def load(results_dir):
    """Map buckets-per-worker -> results.json dict, read from each arm's log."""
    arms = {}
    for log_path in sorted(glob.glob(os.path.join(results_dir, "bpw-*.log"))):
        bpw = int(re.search(r"bpw-(\d+)\.log$", log_path).group(1))
        # run_baseline.py prints the artifact directory on its last lines
        match = None
        with open(log_path) as fh:
            for line in fh:
                found = re.search(r"Artifacts: (\S+)", line)
                if found:
                    match = found.group(1)
        if not match:
            print(f"  (no artifact dir recorded in {log_path}; arm skipped)")
            continue
        json_path = os.path.join(match, "results.json")
        if not os.path.exists(json_path):
            print(f"  (missing {json_path}; arm skipped)")
            continue
        with open(json_path) as fh:
            arms[bpw] = json.load(fh)
    return arms


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    arms = load(sys.argv[1])
    if not arms:
        print("no sweep arms found")
        return 1

    control = arms.get(1)
    ctl_build = control["aggregate"]["infinity"]["build_ns"]["median"] if control else None

    print("\n### Index build (Infinity), by build-task granularity\n")
    print("| buckets/worker | tasks | build (median) | rel MAD | vectors/sec | "
          "vs bpw=1 | FAISS build | ratio Inf/FAISS |")
    print("|---------------:|------:|---------------:|--------:|------------:|"
          "---------:|------------:|----------------:|")
    for bpw in sorted(arms):
        inf = arms[bpw]["aggregate"]["infinity"]
        fai = arms[bpw]["aggregate"]["faiss"]
        b, fb = inf["build_ns"]["median"], fai["build_ns"]["median"]
        speedup = (ctl_build / b) if (ctl_build and b) else None
        ratio = arms[bpw].get("build_time_ratio_infinity_over_faiss")
        print(f"| {bpw} | {inf.get('submitted_tasks')} | {b/1e9:.3f} s | "
              f"{inf['build_ns']['rel_mad']*100:.1f}% | {inf['vectors_per_sec']:.0f} | "
              f"{('%.3fx' % speedup) if speedup else 'control'} | {fb/1e9:.3f} s | "
              f"{('%.3fx' % ratio) if ratio else 'n/a'} |")

    print("\n### Query performance (k=10, efSearch=256)\n")
    print("| buckets/worker | Infinity QPS | FAISS QPS | QPS ratio | Inf p50 | FAISS p50 |")
    print("|---------------:|-------------:|----------:|----------:|--------:|----------:|")
    for bpw in sorted(arms):
        inf = arms[bpw]["aggregate"]["infinity"]
        fai = arms[bpw]["aggregate"]["faiss"]
        iq, fq = inf["qps"]["median"], fai["qps"]["median"]
        qr = arms[bpw].get("qps_ratio_infinity_over_faiss")
        print(f"| {bpw} | {iq:.0f} | {fq:.0f} | {('%.3fx' % qr) if qr else 'n/a'} | "
              f"{inf['query_latency_p50_ns']/1e3:.0f} us | "
              f"{fai['query_latency_p50_ns']/1e3:.0f} us |")

    print("\n### recall@10 -- must not regress vs the bpw=1 control\n")
    efs = sorted(int(k) for k in arms[sorted(arms)[0]]["aggregate"]["infinity"]["recall_at_10"])
    header = " | ".join(f"ef{e}" for e in efs)
    print(f"| buckets/worker | {header} |")
    print("|---------------:|" + "|".join(["------:"] * len(efs)) + "|")
    for bpw in sorted(arms):
        r = arms[bpw]["aggregate"]["infinity"]["recall_at_10"]
        cells = " | ".join(f"{r[str(e)]:.4f}" if r.get(str(e)) is not None else "n/a" for e in efs)
        print(f"| {bpw} | {cells} |")
    fr = arms[sorted(arms)[0]]["aggregate"]["faiss"]["recall_at_10"]
    cells = " | ".join(f"{fr[str(e)]:.4f}" if fr.get(str(e)) is not None else "n/a" for e in efs)
    print(f"| FAISS (reference) | {cells} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
