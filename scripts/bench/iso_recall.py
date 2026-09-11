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

# WHICH RECALL DECIDES PARITY.
#
# This driver originally equalised the harness SELF-audit's recall: 64 synthetic queries,
# uniform in [0,1) per coordinate, with truth derived by exhaustive search. That audit is
# exact and deterministic, but it is the wrong instrument for LOCATING an iso-recall point:
# its queries are out-of-distribution for SIFT (canonical descriptors are integer-valued on a
# scale of tens, not clustered near the origin), and recall@10 over 64 queries has only 640
# neighbour slots, so its finest step is 1/640 = 0.0016 and one query's whole result set is worth
# 0.0156. Measured at n=1,000,000 / efC=200, the self-audit puts Infinity's deficit at
# 0.0172 (ef=64) and 0.0266 (ef=128); the published 10,000-query SIFT ground truth puts the
# same two deficits at 0.00083 and 0.00045. Equalising the self-audit therefore overshoots
# efConstruction badly -- it is what placed this project's iso-recall point at efC=250 when
# the published-truth crossing is at efC 225-235.
#
# So: when both engines report a valid external-truth audit, that is what decides parity, and
# the self-audit is reported alongside for continuity. Turn it on by setting
# HNSW_D0_EXTERNAL_QUERIES and HNSW_D0_EXTERNAL_GROUNDTRUTH in the environment; they are
# inherited by the harness subprocesses. See hnsw_d0_external_truth.h.
#
# SATURATION. At ef>=256 both engines exceed 0.999 recall@10 on published truth, where a
# 0.00001 difference is ONE query in 10,000 and carries no information about graph quality.
# Requiring parity at a saturated point would make the crossing a coin flip, so points at or
# above this level are reported but excluded from the parity decision.
EXTERNAL_SATURATION_RECALL = 0.999


def one_build(binary, engine, args, efc):
    # Create the directory rather than assuming it: only run_baseline.py's dataset
    # generation made it, so running this script first failed with an unhelpful
    # ENOENT from inside the harness.
    sidecar = "/tmp/d0test/iso_sidecar_%s_%d.json" % (engine, os.getpid())
    os.makedirs(os.path.dirname(sidecar), exist_ok=True)
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
    external = {}
    external_valid = kv.get(prefix + "external_recall_valid") == "1"
    if external_valid:
        for ef in REPORT_EF_POINTS:
            key = "%sexternal_recall_at_10_ef_%d" % (prefix, ef)
            if key in kv:
                external[ef] = float(kv[key])
    return {
        "ok": ok,
        "efc": efc,
        "external_valid": external_valid,
        "external": external,
        "external_query_count": kv.get(prefix + "external_recall_query_count"),
        "external_skip_reason": kv.get(prefix + "external_recall_skip_reason", ""),
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
    ext = {}
    ext_good = [s for s in good if s["external_valid"]]
    for ef in REPORT_EF_POINTS:
        vals = [s["external"][ef] for s in ext_good if ef in s["external"]]
        ext[ef] = statistics.median(vals) if vals else None
    return {
        "n": len(good),
        "build_median_ns": statistics.median(builds),
        "build_min_ns": min(builds), "build_max_ns": max(builds),
        "recall": rec,
        "external": ext,
        "external_n": len(ext_good),
        "external_query_count": ext_good[0]["external_query_count"] if ext_good else None,
        "external_skip_reason": good[0]["external_skip_reason"],
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
    print("FAISS reference efc=%d: build=%.1f ms  self-audit recall@10 ef64=%.4f ef128=%.4f  edges=%d threads=%d"
          % (args.faiss_efc, ms(fa["build_median_ns"]), fa["recall"][64], fa["recall"][128], fa["edges"], fa["threads"]))

    # Pick the instrument. External truth when BOTH engines produced it, else the self-audit
    # with a loud warning -- a number equalised on the self-audit is not a defensible
    # iso-recall point (see the note at EXTERNAL_SATURATION_RECALL).
    use_external = fa["external_n"] > 0 and all(fa["external"].get(ef) is not None for ef in ISO_EF_POINTS)
    if use_external:
        for efc in inf_efcs:
            a = agg(inf_by_efc[efc])
            if a and a["external_n"] == 0:
                use_external = False
    if use_external:
        # Exclude saturated points from the decision; keep them in the report.
        decide_efs = [ef for ef in REPORT_EF_POINTS
                      if fa["external"].get(ef) is not None
                      and fa["external"][ef] < EXTERNAL_SATURATION_RECALL]
        if not decide_efs:
            decide_efs = [min(REPORT_EF_POINTS)]
        print("PARITY INSTRUMENT: published ground truth, %s queries. Deciding at efSearch %s "
              "(points at recall >= %.3f are saturated and excluded)."
              % (fa["external_query_count"], decide_efs, EXTERNAL_SATURATION_RECALL))
        print("FAISS external recall@10: "
              + "  ".join("ef%d=%.5f" % (ef, fa["external"][ef])
                          for ef in REPORT_EF_POINTS if fa["external"].get(ef) is not None))
    else:
        decide_efs = list(ISO_EF_POINTS)
        print("PARITY INSTRUMENT: 64-query SYNTHETIC self-audit -- set HNSW_D0_EXTERNAL_QUERIES and")
        print("  HNSW_D0_EXTERNAL_GROUNDTRUTH to decide on published truth instead. The self-audit")
        print("  overshoots efConstruction badly; see the note in this file. Reason external was")
        print("  unavailable: %r" % (fa.get("external_skip_reason") or "no external keys",))

    def parity_source(a):
        return a["external"] if use_external else a["recall"]

    targets = {ef: parity_source(fa)[ef] for ef in decide_efs}

    label = "external" if use_external else "self-audit"
    print("\n| infinity efC | build (median) | ratio Inf/FAISS | "
          + " | ".join("%s ef%d" % (label, ef) for ef in REPORT_EF_POINTS)
          + " | edges | parity |")
    print("|---:|---:|---:|" + "---:|" * len(REPORT_EF_POINTS) + "---:|:--:|")
    iso = None
    for efc in inf_efcs:
        a = agg(inf_by_efc[efc])
        if not a:
            print("| %d | NO VALID RUNS |" % efc); continue
        ratio = a["build_median_ns"] / fa["build_median_ns"]
        src = parity_source(a)
        meets = all(src.get(ef) is not None and src[ef] >= targets[ef] for ef in decide_efs)
        cells = []
        for ef in REPORT_EF_POINTS:
            v = src.get(ef)
            cells.append("n/a" if v is None else "%.5f" % v)
        print("| %d | %.1f ms | %.3fx | %s | %d | %s |"
              % (efc, ms(a["build_median_ns"]), ratio, " | ".join(cells), a["edges"],
                 "YES" if meets else "no"))
        if meets and iso is None:
            iso = (efc, a, ratio)

    print()
    if iso:
        efc, a, ratio = iso
        src = parity_source(a)
        print("ISO-RECALL POINT: Infinity efC=%d reaches parity on the %s instrument at every "
              "deciding efSearch (%s)."
              % (efc, label,
                 ", ".join("ef%d %.5f>=%.5f" % (ef, src[ef], targets[ef]) for ef in decide_efs)))
        print("ISO-RECALL build-time ratio Infinity/FAISS = %.3fx  (Infinity %.1f ms vs FAISS@%d %.1f ms)"
              % (ratio, ms(a["build_median_ns"]), args.faiss_efc, ms(fa["build_median_ns"])))
        print("Speedup FAISS/Infinity = %.3fx" % (1.0 / ratio))
    else:
        print("No swept Infinity efC reached parity at every deciding efSearch %s; widen --infinity-efc."
              % (decide_efs,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
