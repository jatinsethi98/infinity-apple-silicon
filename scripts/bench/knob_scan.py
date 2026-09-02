#!/usr/bin/env python3
"""
knob_scan.py -- paired multi-arm scan over arbitrary run CONFIGURATIONS.

An arm is a (binary, efConstruction, environment) triple. That covers three uses with one
statistic: sweeping a runtime knob on one binary, comparing two builds of the same engine, and
comparing two DIFFERENT ENGINES at different efConstruction -- which is what an iso-recall
build-time claim needs, and which iso_recall.py cannot do because it reports medians with no
confidence interval at all.

WHY, separately from ab_build.py:

ab_build.py answers "keep this change or not?" for exactly two arms and spends its
whole budget tightening one confidence interval. That is the right tool for a verdict
and the wrong tool for a SHAPE: when a knob has a half-dozen plausible settings and
nobody knows which is best, running C(k,2) two-arm campaigns wastes almost all of the
samples re-measuring the same reference arm.

This driver scans k arms in one campaign, keeping ab_build.py's two load-bearing
properties:

  * Paired blocks. Every arm runs once per block, so all arms share a block's thermal
    and background-load state, and the statistic is a per-block ratio against a chosen
    reference arm rather than a difference of independently drifting medians.
  * Randomized within-block order (seeded), so position-in-block effects cancel.

The statistic per arm is the same as ab_build.py's: a two-sided 95% t interval on
exp(mean(ln(t_arm / t_reference))) over blocks. Arms whose interval excludes 1.0 differ
from the reference; arms whose intervals overlap each other are not distinguished by
this campaign, and the report says so rather than ranking noise.

SCAN, NOT VERDICT. Use this to find which settings are worth a verdict, then confirm
the winner against the reference with ab_build.py at full block count. A scan that
tests k arms in b blocks spends b samples per arm, so its intervals are wide by
construction -- that is the intended trade.

Usage:
  python3 scripts/bench/knob_scan.py \
      --dataset /path/base.f32 --n 1000000 --d 128 --m 32 --efc 250 \
      --participants 12 --blocks 3 --require-ac \
      --bin build/bench/infinity_hnsw_d0 \
      --knob INFINITY_HNSW_PREFETCH_STEP \
      --values 64,32,16,8,4,2,1 --reference 64
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
from datetime import datetime, timezone

import campaign
from ab_build import t95
from run_baseline import ensure_dataset, fmt_ns, parse_kv, preflight, spread

here = os.path.dirname(os.path.abspath(__file__))


def run_one(binary, args, sidecar_path, env_extra, efc=None):
    argv = campaign.build_plain_argv(
        binary, args.dataset, args.n, args.d, args.m, efc if efc is not None else args.efc,
        args.ef_list[0], args.chunk_size, args.query_count,
        args.participants, args.build_grain, sidecar_path)
    res = campaign.run_plain(argv, timeout=args.timeout, env_extra=env_extra)
    if res["returncode"] == campaign.USAGE_EXIT_CODE and not res["timed_out"]:
        raise SystemExit(f"{binary} rejected the plain 12-arg invocation; rebuild it.")
    if res["timed_out"]:
        return {"ok": False, "reason": f"timeout after {args.timeout}s"}
    kv = parse_kv(res["stdout"])

    # The harness prefixes every key with its own compiled-in engine identity, so an arm may
    # emit infinity_* or faiss_*. Try both rather than assuming, or a cross-engine arm silently
    # reports no build time and gets dropped from the pairing.
    def g(key, cast=str, default=None):
        raw = kv.get("infinity_" + key, kv.get("faiss_" + key, kv.get(key)))
        if raw is None:
            return default
        try:
            return cast(raw)
        except (TypeError, ValueError):
            return default

    def gef(stem, ef):
        for pre in ("infinity_", "faiss_"):
            k = f"{pre}{stem}_{ef}"
            if k in kv:
                return float(kv[k])
        return None

    cold_ns = g("cold_build_ns", int)
    recall, external = {}, {}
    for ef in args.ef_list:
        v = gef("recall_at_10_ef", ef)
        if v is not None:
            recall[ef] = v
        v = gef("external_recall_at_10_ef", ef)
        if v is not None:
            external[ef] = v
    return {
        "ok": (res["returncode"] == 0 and kv.get("status") == "PASS"
               and g("valid", int) == 1 and cold_ns is not None),
        "build_ns": cold_ns,
        "wall_ns": res["wall_ns"],
        "recall_at_10": recall,
        "external_recall_at_10": external,
        "external_valid": g("external_recall_valid") == "1",
        "graph_sha256": g("graph_sha256"),
        "graph_directed_edges": g("graph_directed_edges", int),
        "echoed": {k: v for k, v in kv.items()
                   if k.startswith(("infinity_prefetch", "infinity_nearest"))
                   or k.endswith("_thread_count")},
        "stdout": res["stdout"],
        "stderr": res["stderr"],
    }


def ratio_ci(log_ratios):
    """95% t interval on exp(mean(log ratio)). Same estimator as ab_build.py so the two
    tools' numbers are directly comparable."""
    n = len(log_ratios)
    if n == 0:
        return None
    mean = statistics.fmean(log_ratios)
    if n == 1:
        return {"n": 1, "ratio": math.exp(mean), "lo": None, "hi": None,
                "half_width_pct": None}
    sd = statistics.stdev(log_ratios)
    tc = t95(n - 1)
    half = tc * sd / math.sqrt(n)
    return {"n": n, "ratio": math.exp(mean),
            "lo": math.exp(mean - half), "hi": math.exp(mean + half),
            "half_width_pct": 100.0 * (math.exp(half) - 1.0)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--efc", type=int, default=200)
    p.add_argument("--ef", default="64,128")
    p.add_argument("--participants", type=int, default=12)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--order-seed", type=int, default=20260902)
    p.add_argument("--chunk-size", type=int, default=8192)
    p.add_argument("--query-count", type=int, default=256)
    p.add_argument("--build-grain", type=int, default=1)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--require-ac", action="store_true")
    p.add_argument("--label", default="")
    p.add_argument("--results-root", default=os.path.join(here, "results-scan"))
    p.add_argument("--bin", dest="binary",
                   help="default harness binary, used for arms that do not name their own "
                        "with bin=PATH")
    p.add_argument("--knob", help="env var name to vary (single-knob form)")
    p.add_argument("--values",
                   help="comma-separated knob values, one arm each (single-knob form)")
    p.add_argument("--arm", action="append", metavar="LABEL:K=V[,K=V...]",
                   help="one arm, setting any number of env vars (repeatable). Use this "
                        "instead of --knob/--values when a shape needs two knobs set "
                        "together -- e.g. 'roll8:INFINITY_HNSW_PREFETCH_STEP=8,"
                        "INFINITY_HNSW_PREFETCH_PER_ITER=1'.")
    p.add_argument("--reference", default=None,
                   help="which arm label is the reference (default: the first)")
    args = p.parse_args()

    args.ef_list = [int(x) for x in args.ef.split(",") if x.strip()]
    # Two ways to name the arms. --knob/--values is the ergonomic single-knob form;
    # --arm is the general one. Mixing them would make the reference label ambiguous.
    if args.arm and (args.knob or args.values):
        raise SystemExit("use either --knob/--values or --arm, not both")
    arm_env, arm_bin, arm_efc = {}, {}, {}
    if args.arm:
        for spec in args.arm:
            if ":" not in spec:
                raise SystemExit(f"--arm expects LABEL:K=V[,K=V...], got {spec!r}")
            label, assigns = spec.split(":", 1)
            label = label.strip()
            if not label:
                raise SystemExit(f"--arm has an empty label: {spec!r}")
            if label in arm_env:
                raise SystemExit(f"duplicate --arm label {label!r}")
            env, spec_bin, spec_efc = {}, None, None
            for kv in assigns.split(","):
                if "=" not in kv:
                    raise SystemExit(f"--arm {label}: expected K=V, got {kv!r}")
                k, v = kv.split("=", 1)
                k = k.strip()
                # `bin` and `efc` are reserved: they select the executable and the
                # efConstruction for this arm rather than setting an environment variable.
                if k == "bin":
                    spec_bin = v
                elif k == "efc":
                    spec_efc = int(v)
                else:
                    env[k] = v
            if not env and spec_bin is None and spec_efc is None:
                raise SystemExit(f"--arm {label} sets nothing")
            arm_env[label] = env
            arm_bin[label] = spec_bin
            arm_efc[label] = spec_efc
        values = list(arm_env)
        knob_desc = "multi-knob arms"
    else:
        if not (args.knob and args.values):
            raise SystemExit("need --knob and --values, or one or more --arm")
        values = [v.strip() for v in args.values.split(",") if v.strip()]
        arm_env = {v: {args.knob: v} for v in values}
        arm_bin = {v: None for v in values}
        arm_efc = {v: None for v in values}
        knob_desc = args.knob
    args.knob = knob_desc
    if len(values) != len(set(values)):
        raise SystemExit("arm labels have duplicates")
    if len(values) < 2:
        raise SystemExit("need at least two arms")
    ref = args.reference if args.reference is not None else values[0]
    if ref not in values:
        raise SystemExit(f"--reference {ref!r} is not among the arms {values}")
    for label in values:
        b = arm_bin.get(label) or args.binary
        if b is None:
            raise SystemExit(f"arm {label} has no binary: pass --bin or bin=PATH in --arm")
        if not (os.path.exists(b) and os.access(b, os.X_OK)):
            raise SystemExit(f"arm {label} binary not found or not executable: {b}")
        arm_bin[label] = b

    ensure_dataset(args.dataset, args.n, args.d, args.seed)
    host = preflight()
    print("=" * 72)
    print("HOST PREFLIGHT")
    print(f"  power           : {'AC' if host['on_ac_power'] else 'BATTERY'} "
          f"(battery {host['battery_pct']}%)")
    print(f"  thermal nominal : {host['thermal_nominal']}")
    print(f"  cpu idle        : {host['cpu_idle_pct']}%")
    if host["competing_load_suspected"]:
        print("  !! COMPETING LOAD suspected. Top CPU procs: "
              + ", ".join(f"{x['name']}:{x['cpu_pct']}%" for x in host["top_cpu_procs"]))
    if not host["on_ac_power"]:
        print("\n!!! RUNNING ON BATTERY !!!\n")
        if args.require_ac:
            raise SystemExit("refusing to run on battery (--require-ac set)")
    print(f"  arms            : {args.knob}")
    for v in values:
        extra = []
        if arm_efc[v] is not None:
            extra.append(f"efc={arm_efc[v]}")
        if arm_bin[v] != args.binary:
            extra.append(os.path.basename(arm_bin[v]))
        tag = ("  [" + " ".join(extra) + "]") if extra else ""
        print(f"      {v:<12s} {arm_env[v]}{tag}"
              + ("   <-- reference" if v == ref else ""))
    print(f"  runs            : {len(values)} arms x {args.blocks} blocks = "
          f"{len(values) * args.blocks}")
    print("=" * 72)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.results_root, ts)
    os.makedirs(run_dir, exist_ok=True)

    rng = random.Random(args.order_seed)
    runs = []
    for block in range(args.blocks):
        order = list(values)
        rng.shuffle(order)
        print(f"\n--- block {block} (order: {' -> '.join(order)}) ---")
        for val in order:
            sidecar = os.path.join(run_dir, f"sidecar-{val}-b{block}.json")
            r = run_one(arm_bin[val], args, sidecar, arm_env[val], efc=arm_efc[val])
            r.update({"arm": val, "block": block, "order": "-".join(order)})
            raw = os.path.join(run_dir, f"raw-{val}-b{block}.txt")
            with open(raw, "w") as fh:
                fh.write(r.pop("stdout", "") + "\n### STDERR\n" + r.pop("stderr", ""))
            r["raw_path"] = raw
            status = "OK" if r["ok"] else f"FAIL({r.get('reason')})"
            print(f"  {val:<12s} {status:12s} build={fmt_ns(r.get('build_ns'))}"
                  f"  echoed={r.get('echoed')}")
            runs.append(r)
            if os.path.exists(sidecar):
                os.remove(sidecar)

    # Per-arm paired statistic against the reference arm. A block contributes only when
    # BOTH that arm and the reference succeeded in it -- a half-failed block must not
    # leak into the ratio.
    by = {(r["arm"], r["block"]): r for r in runs}
    stats = {}
    for val in values:
        lr, used = [], []
        for b in range(args.blocks):
            a, c = by.get((val, b)), by.get((ref, b))
            if a and c and a["ok"] and c["ok"]:
                lr.append(math.log(a["build_ns"] / c["build_ns"]))
                used.append(b)
        ok_runs = [r for r in runs if r["arm"] == val and r["ok"]]
        stats[val] = {
            "vs_reference": ratio_ci(lr),
            "paired_blocks": used,
            "build_ns": spread([r["build_ns"] for r in ok_runs]),
            "min_ns": min((r["build_ns"] for r in ok_runs), default=None),
            "ok_runs": len(ok_runs),
            "recall_at_10": {ef: statistics.median(
                                 [r["recall_at_10"][ef] for r in ok_runs
                                  if ef in r["recall_at_10"]] or [float("nan")])
                             for ef in args.ef_list},
            "external_recall_at_10": {
                ef: (statistics.median(vals) if vals else None)
                for ef in args.ef_list
                for vals in [[r["external_recall_at_10"][ef] for r in ok_runs
                              if ef in r["external_recall_at_10"]]]},
            "edges": ok_runs[0]["graph_directed_edges"] if ok_runs else None,
        }

    print("\n" + "#" * 72)
    print(f"# PAIRED KNOB SCAN  {args.knob}   (n={args.n} d={args.d} M={args.m} "
          f"efC={args.efc} threads={args.participants} blocks={args.blocks})")
    if args.label:
        print(f"# label={args.label}")
    print("#" * 72)
    print(f"\nReference arm: {ref}  {arm_env[ref]}\n")
    hdr = (f"| {'arm':>12} | {'median':>9} | {'min':>9} | {'relMAD':>7} | "
           f"{'ratio/ref':>9} | {'95% CI':>18} | {'verdict':<18} |")
    print(hdr)
    print("|" + "-" * (len(hdr) - 2) + "|")
    best = None
    for val in values:
        s = stats[val]
        vr = s["vs_reference"]
        sp = s["build_ns"]
        if vr and vr["lo"] is not None:
            ci = f"[{vr['lo']:.4f}, {vr['hi']:.4f}]"
            if vr["hi"] < 1.0:
                verdict = "FASTER than ref"
            elif vr["lo"] > 1.0:
                verdict = "SLOWER than ref"
            else:
                verdict = "indistinguishable"
        else:
            ci, verdict = "(needs >=2 blocks)", "no CI"
        mad = "  n/a " if sp["rel_mad"] is None else f"{100.0 * sp['rel_mad']:>6.2f}%"
        print(f"| {val:>12} | {fmt_ns(sp['median']):>9} | {fmt_ns(s['min_ns']):>9} | "
              f"{mad} | {vr['ratio']:>9.4f} | {ci:>18} | {verdict:<18} |")
        if best is None or (vr and vr["ratio"] < stats[best]["vs_reference"]["ratio"]):
            best = val
    print(f"\nLowest point estimate: {best}  {arm_env[best]} "
          f"({stats[best]['vs_reference']['ratio']:.4f} of reference).")
    print("A scan RANKS candidates; it does not license a claim. Confirm the winner "
          "against the reference with ab_build.py at full block count before keeping it.")

    print("\n## 64-query SYNTHETIC self-audit recall@10 "
          "(sanity only: a pure-hint knob must not move these)\n")
    print("| arm | " + " | ".join(f"ef{ef}" for ef in args.ef_list) + " | edges |")
    print("|---|" + "---|" * (len(args.ef_list) + 1))
    for val in values:
        st = stats[val]
        print(f"| {val} | "
              + " | ".join(f"{st['recall_at_10'][ef]:.4f}" for ef in args.ef_list)
              + f" | {st['edges']} |")
    if any(stats[v]["external_recall_at_10"] for v in values):
        print("\n## PUBLISHED-ground-truth recall@10 (the instrument that decides parity)\n")
        print("| arm | " + " | ".join(f"ef{ef}" for ef in args.ef_list) + " |")
        print("|---|" + "---|" * len(args.ef_list))
        for val in values:
            ext = stats[val]["external_recall_at_10"]
            print(f"| {val} | "
                  + " | ".join(("n/a" if ext.get(ef) is None else f"{ext[ef]:.5f}")
                               for ef in args.ef_list) + " |")

    all_ok = all(r["ok"] for r in runs)
    with open(os.path.join(run_dir, "results.json"), "w") as fh:
        json.dump({"timestamp": ts, "label": args.label, "kind": "paired-knob-scan",
                   "knob": args.knob, "values": values, "reference": ref,
                   "arm_env": arm_env, "arm_bin": arm_bin, "arm_efc": arm_efc,
                   "binary": args.binary, "host": host,
                   "config": {k: getattr(args, k) for k in
                              ("dataset", "n", "d", "seed", "m", "efc", "ef_list",
                               "participants", "blocks", "order_seed", "chunk_size",
                               "query_count", "build_grain", "timeout")},
                   "runs": runs, "stats": stats, "all_runs_ok": all_ok}, fh, indent=2)
    print(f"\nArtifacts: {run_dir}/")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
