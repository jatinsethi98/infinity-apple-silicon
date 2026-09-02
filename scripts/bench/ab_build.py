#!/usr/bin/env python3
"""
ab_build.py -- paired candidate-vs-control A/B for Infinity's HNSW index build.

WHY THIS EXISTS, separately from run_baseline.py:

run_baseline.py compares *two different engines* (Infinity vs FAISS) and alternates
them to cancel drift. It cannot be used for an old-vs-new Infinity comparison: the
harness binary derives its stdout key prefix from its own compiled-in engine identity
(hnsw_d0_runner.cpp), so an Infinity binary always emits `infinity_*`. Pointing
`--faiss-bin` at a second Infinity binary makes the driver miss `faiss_cold_build_ns`,
miss the unprefixed fallback, and silently downgrade that arm to `wall_clock_fallback`
with no recall values -- which also defeats its recall gate.

Worse, comparing two independent run_baseline.py campaigns is statistically weak. The
observed 6-run relative MAD for the 1M build is 0.6-1.4%, and one recorded run held a
57.8 s thermal outlier against a 49.0 s minimum. With SE(median) ~= 1.858*MAD/sqrt(n),
a 6-vs-6 difference of medians has a 95% half-width of 1.4-3.0% -- so a real 1-2% win
is indistinguishable from noise, and "the median improved" is not evidence.

This driver instead measures the two binaries as a PAIRED experiment:

  * Both arms run inside one campaign, interleaved, so they share thermal state.
  * Each block runs control and candidate once, in a RANDOMIZED order (seeded), so
    order effects cancel within the block rather than across the campaign.
  * The statistic is the per-block log ratio  r_i = ln(t_candidate_i / t_control_i).
    Pairing removes the between-block drift that dominates the unpaired variance;
    logs make the ratio symmetric and stabilise its variance.
  * The verdict is a two-sided 95% t confidence interval on exp(mean(r)). A change is
    kept only when the interval excludes 1.0. "Median moved" is never sufficient.
  * min-of-N is reported alongside: contamination here is one-sided (interference only
    ever makes a run slower), so the minimum is the more robust estimator of how fast
    the code can actually go.

Both binaries must share the argv/stdout contract of the smoke harness (see
scripts/bench/README.md). Recall is carried through so an A/B also catches a recall
change; the timing decision and the recall decision stay separate.

Usage:
  python3 scripts/bench/ab_build.py \
      --dataset /path/base.f32 --n 1000000 --d 128 --m 32 --efc 200 \
      --participants 12 --blocks 15 --require-ac \
      --control-bin  build/bench/infinity_hnsw_d0.control \
      --candidate-bin build/bench/infinity_hnsw_d0
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
from run_baseline import (
    EMITTED_EF_POINTS,
    ensure_dataset,
    fmt_ns,
    parse_kv,
    preflight,
    spread,
)

# Two-sided 95% Student-t critical values by degrees of freedom (n-1). Hardcoded so
# this script has no scipy dependency; falls through to the normal limit for df > 30.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t95(df):
    if df <= 0:
        return None
    return _T95.get(df, 1.960)


# --------------------------------------------------------------------------- #
# one invocation
# --------------------------------------------------------------------------- #
def parse_env_pairs(items, arm_name):
    """Turn a list of 'KEY=VALUE' strings into a dict, rejecting malformed entries loudly.
    An unnoticed typo here would silently measure control-vs-control, so this refuses
    rather than warns."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--{arm_name}-env expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise SystemExit(f"--{arm_name}-env has an empty key: {item!r}")
        out[k] = v
    return out


def run_one(binary, args, sidecar_path, env_extra=None):
    """Run one harness binary once in a fresh process. Both arms are Infinity, so the
    stdout key prefix is `infinity_` for both -- that is exactly why run_baseline.py's
    two-engine parser cannot be reused here.

    env_extra lets the two arms differ by a runtime knob rather than by binary, so an
    env-driven setting can be A/B'd with ONE executable and zero codegen difference."""
    argv = campaign.build_plain_argv(
        binary, args.dataset, args.n, args.d, args.m, args.efc,
        args.ef_list[0],                # argv efSearch only sets the echoed config line
        args.chunk_size, args.query_count, args.participants, args.build_grain,
        sidecar_path)
    res = campaign.run_plain(argv, timeout=args.timeout, env_extra=env_extra)
    if res["returncode"] == campaign.USAGE_EXIT_CODE and not res["timed_out"]:
        raise SystemExit(
            f"{binary} rejected the 12-arg plain invocation (exit {campaign.USAGE_EXIT_CODE}). "
            "ab_build.py requires binaries that accept the plain form; rebuild them.")
    if res["timed_out"]:
        return {"ok": False, "reason": f"timeout after {args.timeout}s", "argv": argv,
                "stdout": res["stdout"], "stderr": res["stderr"]}

    kv = parse_kv(res["stdout"])

    def g(key, cast=str, default=None):
        raw = kv.get("infinity_" + key, kv.get(key))
        if raw is None:
            return default
        try:
            return cast(raw)
        except (TypeError, ValueError):
            return default

    cold_ns = g("cold_build_ns", int)
    recall = {}
    for ef in args.ef_list:
        key = f"infinity_recall_at_10_ef_{ef}"
        if key in kv:
            recall[ef] = float(kv[key])

    throughput_ops = g("query_throughput_operations", int)
    throughput_wall_ns = g("query_throughput_wall_ns", int)
    return {
        # cold_build_ns is REQUIRED here: a wall-clock fallback would silently mix two
        # different measurement sources across arms, which is exactly the defect this
        # script exists to avoid.
        "ok": (res["returncode"] == 0 and kv.get("status") == "PASS"
               and g("valid", int) == 1 and cold_ns is not None),
        "argv": argv,
        "returncode": res["returncode"],
        "status": kv.get("status"),
        "valid": g("valid", int),
        "build_ns": cold_ns,
        "cold_build_ns_present": cold_ns is not None,
        "wall_ns": res["wall_ns"],
        "recall_at_10": recall,
        "env_extra": dict(env_extra or {}),
        "prefetch_step": g("prefetch_step", int),
        "graph_sha256": kv.get("infinity_graph_sha256") or kv.get("graph_sha256"),
        "graph_directed_edges": g("graph_directed_edges", int),
        "build_buckets_per_worker": g("build_buckets_per_worker", int),
        "submitted_tasks": g("submitted_tasks", int),
        "qps": (throughput_ops / (throughput_wall_ns / 1e9)
                if throughput_ops and throughput_wall_ns else None),
        "query_latency_p50_ns": g("query_latency_p50_ns", int),
        "query_latency_p99_ns": g("query_latency_p99_ns", int),
        "stdout": res["stdout"],
        "stderr": res["stderr"],
    }


# --------------------------------------------------------------------------- #
# paired statistics
# --------------------------------------------------------------------------- #
def paired_log_ratio(pairs):
    """pairs: list of (control_ns, candidate_ns). Returns the paired verdict.

    Statistic: r_i = ln(candidate_i / control_i). ratio = exp(mean(r)); ratio < 1 means
    the candidate is faster. speedup = 1/ratio. The CI is a two-sided 95% t interval on
    mean(r), exponentiated. n=1 yields no interval -- one pair is never a decision.
    """
    if not pairs:
        return {"n": 0, "verdict": "no data"}
    logs = [math.log(cand / ctrl) for ctrl, cand in pairs]
    n = len(logs)
    mean = statistics.fmean(logs)
    ratio = math.exp(mean)
    out = {
        "n": n,
        "ratio_candidate_over_control": ratio,
        "speedup_control_over_candidate": 1.0 / ratio,
        "pct_change": (ratio - 1.0) * 100.0,
        "log_ratios": logs,
    }
    if n < 2:
        out["verdict"] = "INDISTINGUISHABLE (need >= 2 pairs for an interval)"
        return out
    sd = statistics.stdev(logs)
    se = sd / math.sqrt(n)
    t = t95(n - 1)
    lo_log, hi_log = mean - t * se, mean + t * se
    # exp is monotonic, so the ratio interval is just the exponentiated log interval.
    lo, hi = math.exp(lo_log), math.exp(hi_log)
    out.update({
        "log_sd": sd, "log_se": se, "t_crit_95": t,
        "ratio_ci95_low": lo, "ratio_ci95_high": hi,
        # Half-width in percent, the number to compare against an expected effect size.
        "ci95_half_width_pct": (math.exp(t * se) - 1.0) * 100.0,
    })
    if hi < 1.0:
        out["verdict"] = "CANDIDATE FASTER (95% CI excludes 1.0)"
    elif lo > 1.0:
        out["verdict"] = "CANDIDATE SLOWER (95% CI excludes 1.0)"
    else:
        out["verdict"] = "INDISTINGUISHABLE (95% CI includes 1.0)"
    return out


def min_ratio(pairs):
    """Ratio of the two arms' minima -- one-sided-contamination-robust cross-check."""
    if not pairs:
        return None
    ctrl_min = min(c for c, _ in pairs)
    cand_min = min(c for _, c in pairs)
    return cand_min / ctrl_min if ctrl_min else None


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description="Paired candidate-vs-control A/B for the Infinity HNSW build")
    p.add_argument("--dataset", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--seed", type=int, default=0, help="dataset regen seed")
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--efc", type=int, default=200)
    p.add_argument("--ef", default="32,64,128,256,512",
                   help=f"efSearch list to report; subset of {EMITTED_EF_POINTS}")
    p.add_argument("--participants", type=int, default=12)
    p.add_argument("--blocks", type=int, default=15,
                   help="paired blocks; each runs control and candidate once in random "
                        "order. >=15 for effects under 3%%.")
    p.add_argument("--order-seed", type=int, default=20260901,
                   help="seed for the per-block order coin flips (reproducible)")
    p.add_argument("--chunk-size", type=int, default=8192)
    p.add_argument("--query-count", type=int, default=256)
    p.add_argument("--build-grain", type=int, default=1)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--require-ac", action="store_true")
    p.add_argument("--label", default="", help="short tag recorded in results.json")
    p.add_argument("--results-root", default=os.path.join(here, "results-ab"))
    p.add_argument("--control-bin", required=True, help="baseline Infinity binary")
    p.add_argument("--candidate-bin", required=True, help="Infinity binary under test")
    p.add_argument("--control-env", action="append", metavar="KEY=VALUE",
                   help="env var set for the control arm only (repeatable). Lets a runtime "
                        "knob be A/B'd with one binary on both arms.")
    p.add_argument("--candidate-env", action="append", metavar="KEY=VALUE",
                   help="env var set for the candidate arm only (repeatable)")
    args = p.parse_args()

    args.ef_list = [int(x) for x in args.ef.split(",") if x.strip()]
    for name, path in (("control-bin", args.control_bin),
                       ("candidate-bin", args.candidate_bin)):
        if not (os.path.exists(path) and os.access(path, os.X_OK)):
            raise SystemExit(f"{name} not found or not executable: {path}")
    env_arms = {"control": parse_env_pairs(args.control_env, "control"),
                "candidate": parse_env_pairs(args.candidate_env, "candidate")}
    same_bin = os.path.realpath(args.control_bin) == os.path.realpath(args.candidate_bin)
    if same_bin and env_arms["control"] == env_arms["candidate"]:
        print("WARNING: control and candidate are the SAME file with the SAME env. This "
              "measures the harness noise floor, which is a useful thing to do -- expect a "
              "verdict of INDISTINGUISHABLE and a CI half-width you can quote as the floor.",
              file=sys.stderr)
    elif same_bin:
        print(f"NOTE: one binary, two envs -- codegen is identical by construction.\n"
              f"      control  env: {env_arms['control'] or '(inherited only)'}\n"
              f"      candidate env: {env_arms['candidate'] or '(inherited only)'}",
              file=sys.stderr)

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
        print("\n!!! RUNNING ON BATTERY -- results may be throttled !!!\n")
        if args.require_ac:
            raise SystemExit("refusing to run on battery (--require-ac set)")
    print("=" * 72)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.results_root, ts)
    os.makedirs(run_dir, exist_ok=True)

    arms = {"control": args.control_bin, "candidate": args.candidate_bin}
    rng = random.Random(args.order_seed)
    runs = []
    for block in range(args.blocks):
        order = ["control", "candidate"]
        if rng.random() < 0.5:
            order.reverse()
        print(f"\n--- block {block} (order: {' -> '.join(order)}) ---")
        for arm in order:
            sidecar = os.path.join(run_dir, f"sidecar-{arm}-b{block}.json")
            r = run_one(arms[arm], args, sidecar, env_extra=env_arms[arm])
            r["arm"] = arm
            r["block"] = block
            r["order"] = "-".join(order)
            raw_path = os.path.join(run_dir, f"raw-{arm}-b{block}.txt")
            with open(raw_path, "w") as fh:
                fh.write("### ARGV\n" + " ".join(r.get("argv", [])) + "\n### STDOUT\n"
                         + r.get("stdout", "") + "\n### STDERR\n" + r.get("stderr", ""))
            r["raw_path"] = raw_path
            status = "OK" if r["ok"] else f"FAIL({r.get('reason', r.get('status'))})"
            print(f"  {arm:9s} {status:22s} build={fmt_ns(r.get('build_ns'))}")
            r.pop("stdout", None)
            r.pop("stderr", None)
            runs.append(r)
            if os.path.exists(sidecar):
                os.remove(sidecar)

    # Pair strictly by block, and only when BOTH arms of that block succeeded --
    # a half-failed block must not leak into the paired statistic.
    pairs, paired_blocks = [], []
    for block in range(args.blocks):
        ctrl = next((r for r in runs if r["block"] == block and r["arm"] == "control"), None)
        cand = next((r for r in runs if r["block"] == block and r["arm"] == "candidate"), None)
        if ctrl and cand and ctrl["ok"] and cand["ok"]:
            pairs.append((ctrl["build_ns"], cand["build_ns"]))
            paired_blocks.append(block)

    verdict = paired_log_ratio(pairs)
    verdict["min_ratio_candidate_over_control"] = min_ratio(pairs)
    verdict["paired_blocks"] = paired_blocks
    verdict["dropped_blocks"] = [b for b in range(args.blocks) if b not in paired_blocks]

    agg = {}
    for arm in ("control", "candidate"):
        ar = [r for r in runs if r["arm"] == arm and r["ok"]]
        sp = spread([r["build_ns"] for r in ar])
        recall_by_ef = {}
        for ef in args.ef_list:
            vals = [r["recall_at_10"][ef] for r in ar if ef in r["recall_at_10"]]
            recall_by_ef[ef] = statistics.median(vals) if vals else None
        graph_hashes = sorted({r["graph_sha256"] for r in ar if r.get("graph_sha256")})
        agg[arm] = {
            "ok_runs": len(ar),
            "build_ns": sp,
            "vectors_per_sec": (args.n / (sp["median"] / 1e9)) if sp["median"] else None,
            "recall_at_10": recall_by_ef,
            "qps": spread([r["qps"] for r in ar if r.get("qps")]),
            "graph_sha256_distinct": graph_hashes,
            "graph_directed_edges": ar[0]["graph_directed_edges"] if ar else None,
            "prefetch_step_distinct": sorted({r["prefetch_step"] for r in ar
                                              if r.get("prefetch_step") is not None}),
        }

    # Recall movement is reported but NOT folded into the timing verdict: a graph-neutral
    # change must show zero recall movement, and a graph-changing one needs the official
    # 10k truth, not this run's audit. Keeping them separate keeps both interpretable.
    recall_delta = {}
    for ef in args.ef_list:
        c, k = agg["control"]["recall_at_10"].get(ef), agg["candidate"]["recall_at_10"].get(ef)
        recall_delta[ef] = (k - c) if (c is not None and k is not None) else None

    all_ok = all(r["ok"] for r in runs)
    result = {
        "timestamp": ts,
        "label": args.label,
        "kind": "paired-ab-build",
        "config": {k: getattr(args, k) for k in
                   ("dataset", "n", "d", "seed", "m", "efc", "ef_list", "participants",
                    "blocks", "order_seed", "chunk_size", "query_count", "build_grain",
                    "timeout")},
        "binaries": {"control": args.control_bin, "candidate": args.candidate_bin},
        "env_arms": env_arms,
        "host": host,
        "runs": runs,
        "aggregate": agg,
        "paired_verdict": verdict,
        "recall_delta_candidate_minus_control": recall_delta,
        "all_runs_ok": all_ok,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    print_report(args, agg, verdict, recall_delta, all_ok, run_dir)
    # Exit 0 only when every run succeeded; the verdict itself is not a pass/fail.
    return 0 if all_ok else 1


def print_report(args, agg, verdict, recall_delta, all_ok, run_dir):
    ctl, cnd = agg["control"], agg["candidate"]
    print("\n" + "#" * 72)
    print(f"# PAIRED A/B  (n={args.n} d={args.d} M={args.m} efC={args.efc} "
          f"threads={args.participants} blocks={args.blocks})"
          + (f"  label={args.label}" if args.label else ""))
    print("#" * 72 + "\n")

    print("## Build timing per arm\n")
    print("| arm | ok runs | median | min | max | rel MAD | vectors/sec |")
    print("|-----|--------:|-------:|----:|----:|--------:|------------:|")
    for name, a in (("control", ctl), ("candidate", cnd)):
        b = a["build_ns"]
        print(f"| {name} | {a['ok_runs']} | {fmt_ns(b['median'])} | {fmt_ns(b['min'])} | "
              f"{fmt_ns(b['max'])} | "
              f"{('%.2f%%' % (b['rel_mad']*100)) if b['rel_mad'] is not None else 'n/a'} | "
              f"{('%.0f' % a['vectors_per_sec']) if a['vectors_per_sec'] else 'n/a'} |")

    print("\n## Paired verdict (the decision)\n")
    n = verdict.get("n", 0)
    if n == 0:
        print("No complete pairs -- nothing to decide.")
    else:
        ratio = verdict["ratio_candidate_over_control"]
        print(f"- pairs used            : {n}"
              + (f"  (dropped blocks {verdict['dropped_blocks']})"
                 if verdict.get("dropped_blocks") else ""))
        print(f"- ratio cand/ctrl       : {ratio:.4f}  ({verdict['pct_change']:+.2f}%)")
        print(f"- speedup ctrl/cand     : {verdict['speedup_control_over_candidate']:.4f}x")
        if "ratio_ci95_low" in verdict:
            print(f"- 95% CI on the ratio   : [{verdict['ratio_ci95_low']:.4f}, "
                  f"{verdict['ratio_ci95_high']:.4f}]  "
                  f"(half-width {verdict['ci95_half_width_pct']:.2f}%)")
        mr = verdict.get("min_ratio_candidate_over_control")
        if mr:
            print(f"- min/min cross-check   : {mr:.4f}  "
                  "(one-sided-contamination-robust; should agree in sign)")
        print(f"\n**{verdict['verdict']}**")
        if "ci95_half_width_pct" in verdict:
            hw = verdict["ci95_half_width_pct"]
            print(f"\nThis campaign can resolve effects larger than ~{hw:.2f}%. An effect "
                  f"smaller than that needs more blocks (half-width shrinks as 1/sqrt(n): "
                  f"~{math.ceil(n * (hw / 1.0) ** 2)} blocks for ~1%).")

    print("\n## recall@10 movement (reported, NOT part of the timing verdict)\n")
    print("| efSearch | control | candidate | delta |")
    print("|---------:|--------:|----------:|------:|")
    for ef in args.ef_list:
        c, k = ctl["recall_at_10"].get(ef), cnd["recall_at_10"].get(ef)
        d = recall_delta.get(ef)
        print(f"| {ef} | {('%.4f' % c) if c is not None else 'n/a'} | "
              f"{('%.4f' % k) if k is not None else 'n/a'} | "
              f"{('%+.4f' % d) if d is not None else 'n/a'} |")

    print("\n## Graph identity\n")
    for name, a in (("control", ctl), ("candidate", cnd)):
        h = a["graph_sha256_distinct"]
        if not h:
            shown = "not emitted"
        elif len(h) == 1:
            shown = h[0][:16] + "..."
        else:
            shown = f"{len(h)} DISTINCT hashes (expected at participants>1)"
        print(f"- {name:9s} edges={a['graph_directed_edges']}  graph_sha256: {shown}")
    if args.participants > 1:
        print("\n  Note: at participants>1 the graph is order-nondeterministic, so distinct "
              "hashes are expected and prove nothing. Use participants=1 for identity.")

    if not all_ok:
        print("\n## Notes\n- one or more runs FAILED -- see results.json")
    print(f"\nArtifacts: {run_dir}/  (raw-*.txt, results.json)")


if __name__ == "__main__":
    sys.exit(main())
