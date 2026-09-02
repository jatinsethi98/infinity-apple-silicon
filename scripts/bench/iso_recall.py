#!/usr/bin/env python3
"""iso_recall.py -- Infinity iso-recall build-time comparison against FAISS (Task 3).

The equal-parameter build-time ratio is NOT apples-to-apples when the two engines
land at different recall at equal efConstruction. A faster build at lower recall is
not a win, so this driver equalises recall first and reports the ratio there.

Why the recall differs is NOT established. An earlier version of this docstring
attributed it to Infinity's forward budget of M per layer vs FAISS's 2*M at level 0
producing a sparser graph. Measurement refutes that: at n=100,000 the two engines
build 2,586,903 vs 2,595,348 directed edges (0.33% apart), Infinity's level-0 mean
degree is 25.3 against an M=32 budget and 64-slot capacity -- so the budget is not
binding, the diversity rule in SelectNeighborsHeuristic is -- and forcing the
FAISS convention adds 941 edges out of 2.5M. See docs/apple_silicon/README.md.

This does not affect what this driver does or what it reports. It equalises an
OBSERVED recall difference; it never depended on a mechanism for it.

This driver fixes FAISS at the reference efConstruction (default 200) and sweeps
Infinity's efConstruction upward until Infinity's recall@10 at BOTH ef=64 and ef=128
is >= FAISS's recall@10 at the same efSearch. It then reports the build-time ratio
at that iso-recall point. FAISS is built once (the reference); only Infinity is swept,
so this costs far less than re-running the full paired driver per efC.

It reuses campaign.py (the minimal 17-arg + SIGCONT shim the harness binaries need)
and the engine's own <engine>_cold_build_ns, exactly like run_baseline.py. Every
number here is independently reproducible with run_baseline.py:
  * FAISS@ref  == run_baseline.py --efc <ref>  (FAISS row)
  * Infinity@X == run_baseline.py --efc X       (Infinity row)

Usage:
  python3 scripts/bench/iso_recall.py \
    --dataset DS --n N --d 128 --m 32 --faiss-efc 200 --infinity-efc 200,250,300,400 \
    --participants 12 --pairs 2 --query-count 1000 \
    --infinity-bin ... --faiss-bin ...
"""

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign

ISO_EF_POINTS = (64, 128)          # efSearch points that must reach parity
REPORT_EF_POINTS = (32, 64, 128, 256, 512)


def one_build(binary, engine, args, efc):
    sidecar = "/tmp/d0test/iso_sidecar_%s_%d.json" % (engine, os.getpid())
    if os.path.exists(sidecar):
        os.remove(sidecar)
    argv = campaign.build_argv(
        binary, engine, args.dataset, args.n, args.d, args.m, efc,
        64, args.chunk_size, args.query_count, args.participants, args.build_grain,
        sidecar)
    res = campaign.run_campaign(argv, timeout=args.timeout)
    if os.path.exists(sidecar):
        os.remove(sidecar)
    kv = {}
    for line in res["stdout"].splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            kv[k.strip()] = v.strip()
    prefix = engine + "_"
    ok = (res["returncode"] == 0 and kv.get("status") == "PASS"
          and kv.get(prefix + "valid") == "1" and not res["timed_out"])
    recall = {}
    for ef in REPORT_EF_POINTS:
        key = "%srecall_at_10_ef_%d" % (prefix, ef)
        if key in kv:
            recall[ef] = float(kv[key])
    return {
        "ok": ok,
        "efc": efc,
        "cold_build_ns": int(kv[prefix + "cold_build_ns"]) if (prefix + "cold_build_ns") in kv else None,
        "threads": int(kv[prefix + "threads"]) if (prefix + "threads") in kv else None,
        "edges": int(kv[prefix + "graph_directed_edges"]) if (prefix + "graph_directed_edges") in kv else None,
        "recall": recall,
        "wall_ns": res["wall_ns"],
        "stderr_tail": res["stderr"][-300:],
    }


def agg(samples):
    good = [s for s in samples if s["ok"] and s["cold_build_ns"]]
    if not good:
        return None
    builds = [s["cold_build_ns"] for s in good]
    rec = {}
    for ef in REPORT_EF_POINTS:
        vals = [s["recall"][ef] for s in good if ef in s["recall"]]
        rec[ef] = statistics.median(vals) if vals else None
    return {
        "n": len(good),
        "build_median_ns": statistics.median(builds),
        "build_min_ns": min(builds), "build_max_ns": max(builds),
        "recall": rec,
        "edges": good[0]["edges"],
        "threads": good[0]["threads"],
    }


def ms(ns):
    return None if ns is None else ns / 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--faiss-efc", type=int, default=200)
    p.add_argument("--infinity-efc", default="200,250,300,400")
    p.add_argument("--participants", type=int, default=12)
    p.add_argument("--pairs", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=8192)
    p.add_argument("--query-count", type=int, default=1000)
    p.add_argument("--build-grain", type=int, default=1)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--infinity-bin", required=True)
    p.add_argument("--faiss-bin", required=True)
    args = p.parse_args()

    inf_efcs = [int(x) for x in args.infinity_efc.split(",") if x.strip()]

    # FAISS reference (built once at faiss-efc), alternating with a matched Infinity
    # efc=faiss_efc build so both engines see identical thermal/order conditions.
    faiss_runs, inf_by_efc = [], {e: [] for e in inf_efcs}
    for i in range(args.pairs):
        r = one_build(args.faiss_bin, "faiss", args, args.faiss_efc)
        faiss_runs.append(r)
        print("faiss    efc=%d ok=%s build=%s recall64=%s recall128=%s edges=%s"
              % (args.faiss_efc, r["ok"], ("%.1f ms" % ms(r["cold_build_ns"])) if r["cold_build_ns"] else "n/a",
                 r["recall"].get(64), r["recall"].get(128), r["edges"]), file=sys.stderr)
        if not r["ok"]:
            print("  stderr:", r["stderr_tail"], file=sys.stderr)
    for efc in inf_efcs:
        for i in range(args.pairs):
            r = one_build(args.infinity_bin, "infinity", args, efc)
            inf_by_efc[efc].append(r)
            print("infinity efc=%d ok=%s build=%s recall64=%s recall128=%s edges=%s"
                  % (efc, r["ok"], ("%.1f ms" % ms(r["cold_build_ns"])) if r["cold_build_ns"] else "n/a",
                     r["recall"].get(64), r["recall"].get(128), r["edges"]), file=sys.stderr)
            if not r["ok"]:
                print("  stderr:", r["stderr_tail"], file=sys.stderr)

    fa = agg(faiss_runs)
    print("\n== ISO-RECALL SWEEP (n=%d d=%d M=%d, %d pairs, cold_build_ns medians) ==" % (args.n, args.d, args.m, args.pairs))
    if not fa:
        print("FAISS reference produced no valid runs; aborting."); return 1
    print("FAISS reference efc=%d: build=%.1f ms  recall@10 ef64=%.4f ef128=%.4f  edges=%d threads=%d"
          % (args.faiss_efc, ms(fa["build_median_ns"]), fa["recall"][64], fa["recall"][128], fa["edges"], fa["threads"]))
    target64, target128 = fa["recall"][64], fa["recall"][128]

    print("\n| infinity efC | build (median) | ratio Inf/FAISS | recall ef32 | ef64 | ef128 | ef256 | ef512 | edges | ef64>=F & ef128>=F |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")
    iso = None
    for efc in inf_efcs:
        a = agg(inf_by_efc[efc])
        if not a:
            print("| %d | NO VALID RUNS |" % efc); continue
        ratio = a["build_median_ns"] / fa["build_median_ns"]
        meets = (a["recall"][64] is not None and a["recall"][128] is not None
                 and a["recall"][64] >= target64 and a["recall"][128] >= target128)
        print("| %d | %.1f ms | %.3fx | %.4f | %.4f | %.4f | %.4f | %.4f | %d | %s |"
              % (efc, ms(a["build_median_ns"]), ratio, a["recall"][32], a["recall"][64],
                 a["recall"][128], a["recall"][256], a["recall"][512], a["edges"],
                 "YES" if meets else "no"))
        if meets and iso is None:
            iso = (efc, a, ratio)

    print()
    if iso:
        efc, a, ratio = iso
        print("ISO-RECALL POINT: Infinity efC=%d reaches parity (ef64 %.4f>=%.4f, ef128 %.4f>=%.4f)."
              % (efc, a["recall"][64], target64, a["recall"][128], target128))
        print("ISO-RECALL build-time ratio Infinity/FAISS = %.3fx  (Infinity %.1f ms vs FAISS@%d %.1f ms)"
              % (ratio, ms(a["build_median_ns"]), args.faiss_efc, ms(fa["build_median_ns"])))
    else:
        print("No swept Infinity efC reached parity at BOTH ef64 and ef128; widen --infinity-efc.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
