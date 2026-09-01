#!/usr/bin/env python3
"""
run_baseline.py -- Infinity-vs-FAISS HNSW index-build baseline driver (Apple Silicon).

This is the durable, minimal mechanism for comparing Infinity's HNSW index build
against FAISS's on the exact same workload. It is meant to be read in one sitting.

FAIRNESS CONTRACT (this is the whole contract; keep it this short):
  * Same dataset file, same vector count, same dimensions for both engines.
  * Same M and efConstruction; same thread count (participants).
  * recall@10 is compared at matched efSearch. The build-time ratio is only
    reportable when every compared efSearch has a recall deficit within
    --recall-deficit-threshold (default 0.005). Otherwise the run is flagged
    RECALL-UNMATCHED and the ratio is printed but marked not-comparable.
  * N paired runs with ALTERNATING order (infinity-first, faiss-first, ...) to
    cancel thermal/order drift. Each engine runs in a fresh process.
  * We report median and spread (min, max, relative MAD), never a single run.
  * We refuse to run on battery when --require-ac is set; otherwise we warn loudly.
  * We record host state: AC/battery, thermal pressure, CPU idle %, competing procs.

The two engine binaries take identical argv:
  DATASET N D M EFC EFSEARCH CHUNK QUERYCOUNT PARTICIPANTS BUILDGRAIN AUDIT_SIDECAR
and print key=value lines on stdout, keys prefixed infinity_ / faiss_. We parse
those. Build time comes from <engine>_cold_build_ns when present; if it is absent
we fall back to wall-clock of the subprocess and say so loudly (see LIMITATIONS in
README.md).
"""

import argparse
import json
import os
import random
import re
import shutil
import statistics
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone

import campaign  # 17-arg campaign argv + SIGCONT supervisor (see campaign.py)

# The engine's recall audit always emits these fixed efSearch points regardless of
# the efSearch argv value (the argv value only sets the echoed config line). We can
# therefore report recall for any subset of these from a single build+search run.
EMITTED_EF_POINTS = (32, 64, 128, 256, 512)


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def ensure_dataset(path, n, d, seed=0):
    """Ensure the raw little-endian float32 dataset exists (n*d*4 bytes, no header).

    Regenerates deterministically with random.Random(seed).random() float32 values,
    which reproduces the canonical smoke dataset sha256 for n=12288 d=128 seed=0.
    """
    want_bytes = n * d * 4
    if os.path.exists(path) and os.path.getsize(path) == want_bytes:
        return path
    if os.path.exists(path):
        raise SystemExit(
            f"dataset {path} exists but is {os.path.getsize(path)} bytes, "
            f"expected {want_bytes} (n={n} d={d}); refusing to overwrite")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rng = random.Random(seed)
    total = n * d
    print(f"[dataset] regenerating {path} ({n}x{d}, seed={seed}) ...", file=sys.stderr)
    with open(path, "wb") as fh:
        chunk = 1 << 16  # floats per write
        remaining = total
        while remaining > 0:
            k = min(chunk, remaining)
            fh.write(struct.pack("<%df" % k, *[rng.random() for _ in range(k)]))
            remaining -= k
    return path


# --------------------------------------------------------------------------- #
# host preflight
# --------------------------------------------------------------------------- #
def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:  # noqa: BLE001 - preflight must never crash the run
        return f"<{' '.join(cmd)} failed: {exc}>"


def preflight():
    """Snapshot host state. Best-effort: never raises."""
    batt = _run(["pmset", "-g", "batt"])
    on_ac = "AC Power" in batt
    m = re.search(r"(\d+)%", batt)
    battery_pct = int(m.group(1)) if m else None

    therm = _run(["pmset", "-g", "therm"])
    # No sysctl thermal oids on M-series; pmset -g therm is the portable signal.
    thermal_nominal = "No thermal warning level has been recorded" in therm

    # CPU idle % is accurate from a single top sample; per-process CPU is not
    # (a single top sample reports ~0 for every process), so we read process CPU
    # from `ps -r` (sorted by CPU) instead.
    top = _run(["top", "-l", "1", "-n", "0"])
    idle = None
    mi = re.search(r"CPU usage:.*?([\d.]+)%\s*idle", top)
    if mi:
        idle = float(mi.group(1))
    top_procs = []
    ps_out = _run(["ps", "-Ax", "-o", "pcpu=,comm=", "-r"])
    for line in ps_out.splitlines():
        mp = re.match(r"\s*([\d.]+)\s+(.+)$", line)
        if not mp:
            continue
        cpu, name = float(mp.group(1)), mp.group(2).strip()
        if cpu >= 5.0 and os.path.basename(name) not in ("ps", "top", "run_baseline.py"):
            top_procs.append({"name": os.path.basename(name), "cpu_pct": cpu})
        if len(top_procs) >= 8:
            break
    competing = bool(top_procs) or (idle is not None and idle < 85.0)

    load = os.getloadavg()
    return {
        "on_ac_power": on_ac,
        "battery_pct": battery_pct,
        "pmset_batt_raw": batt.strip(),
        "thermal_nominal": thermal_nominal,
        "pmset_therm_raw": therm.strip(),
        "cpu_idle_pct": idle,
        "loadavg_1_5_15": load,
        "competing_load_suspected": competing,
        "top_cpu_procs": top_procs,
    }


# --------------------------------------------------------------------------- #
# engine invocation + parsing
# --------------------------------------------------------------------------- #
def parse_kv(text):
    kv = {}
    for line in text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            kv[k.strip()] = v.strip()
    return kv


def run_engine(binary, engine, args, sidecar_path):
    """Run one engine binary once in a fresh process. Returns a result dict.

    Current binaries accept a plain 11-positional-argument invocation and do not
    suspend themselves, so they run under a bare subprocess. Older prebuilt
    binaries require the 17-arg campaign protocol and SIGSTOP at attestation
    barriers; we fall back to campaign.run_campaign (which supervises the SIGCONT
    resumes) when the plain form is rejected with the usage exit code.
    """
    plain_argv = campaign.build_plain_argv(
        binary, args.dataset, args.n, args.d, args.m, args.efc,
        args.ef_list[0],               # argv efSearch: only sets echoed config
        args.chunk_size, args.query_count, args.participants, args.build_grain,
        sidecar_path)
    res = campaign.run_plain(plain_argv, timeout=args.timeout)
    argv = plain_argv
    if res["returncode"] == campaign.USAGE_EXIT_CODE and not res["timed_out"]:
        # Legacy binary: retry with the campaign binding and a supervisor.
        argv = campaign.build_argv(
            binary, engine, args.dataset, args.n, args.d, args.m, args.efc,
            args.ef_list[0], args.chunk_size, args.query_count,
            args.participants, args.build_grain, sidecar_path)
        res = campaign.run_campaign(argv, timeout=args.timeout)
    stdout, stderr = res["stdout"], res["stderr"]
    returncode, wall_ns = res["returncode"], res["wall_ns"]
    if res["timed_out"]:
        return {"engine": engine, "ok": False, "reason": f"timeout after {args.timeout}s",
                "argv": argv, "stdout": stdout, "stderr": stderr,
                "wall_ns": wall_ns, "timed_out": True, "returncode": returncode}

    kv = parse_kv(stdout)
    prefix = engine + "_"

    def g(key, cast=str, default=None):
        raw = kv.get(prefix + key, kv.get(key))
        if raw is None:
            return default
        try:
            return cast(raw)
        except (TypeError, ValueError):
            return default

    cold_ns = g("cold_build_ns", int)
    build_ns = cold_ns if cold_ns is not None else wall_ns
    build_source = "engine_stdout:cold_build_ns" if cold_ns is not None else "wall_clock_fallback"

    recall = {}
    for ef in args.ef_list:
        key = f"{prefix}recall_at_10_ef_{ef}"
        if key in kv:
            recall[ef] = float(kv[key])

    # Query side. Throughput is measured with kHnswD0QueryThroughputConcurrency
    # threads issuing queries; latency percentiles are measured one query at a time.
    # Both engines run the same harness constants (ef=256, k=10), so they compare.
    throughput_ops = g("query_throughput_operations", int)
    throughput_wall_ns = g("query_throughput_wall_ns", int)
    qps = (throughput_ops / (throughput_wall_ns / 1e9)
           if throughput_ops and throughput_wall_ns else None)

    ok = (returncode == 0 and kv.get("status") == "PASS"
          and g("valid", int) == 1 and build_ns is not None)
    return {
        "engine": engine,
        "ok": ok,
        "argv": argv,
        "returncode": returncode,
        "status": kv.get("status"),
        "valid": g("valid", int),
        "threads": g("threads", int),
        "build_ns": build_ns,
        "build_source": build_source,
        "cold_build_ns_present": cold_ns is not None,
        "wall_ns": wall_ns,
        "recall_at_10": recall,
        "graph_directed_edges": g("graph_directed_edges", int),
        "graph_level0_directed_edges": g("graph_level0_directed_edges", int),
        "graph_level0_capacity": g("graph_level0_capacity", int),
        "graph_upper_capacity": g("graph_upper_capacity", int),
        "graph_level_histogram": kv.get(prefix + "graph_level_histogram"),
        "qps": qps,
        "query_throughput_operations": throughput_ops,
        "query_throughput_wall_ns": throughput_wall_ns,
        "query_latency_p50_ns": g("query_latency_p50_ns", int),
        "query_latency_p95_ns": g("query_latency_p95_ns", int),
        "query_latency_p99_ns": g("query_latency_p99_ns", int),
        "query_throughput_concurrency": g("query_throughput_concurrency", int)
                                        or (int(kv["query_throughput_concurrency"])
                                            if "query_throughput_concurrency" in kv else None),
        "build_buckets_per_worker": g("build_buckets_per_worker", int),
        "submitted_tasks": g("submitted_tasks", int),
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": False,
    }


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def _med(values):
    """Median of the non-None values, or None when there are none."""
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def spread(samples):
    """median, min, max, relative MAD (median abs deviation / median)."""
    if not samples:
        return {"median": None, "min": None, "max": None, "rel_mad": None, "n": 0}
    med = statistics.median(samples)
    mad = statistics.median([abs(x - med) for x in samples])
    return {
        "median": med, "min": min(samples), "max": max(samples),
        "rel_mad": (mad / med) if med else None, "n": len(samples),
    }


def fmt_ns(ns):
    """Human-readable duration. Query latencies are sub-millisecond, so keep
    microsecond resolution below 1 ms instead of rounding everything to 0.2 ms."""
    if ns is None:
        return "n/a"
    if ns < 1e6:
        return f"{ns/1e3:.0f} us"
    if ns < 1e9:
        return f"{ns/1e6:.1f} ms"
    return f"{ns/1e9:.3f} s"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Infinity-vs-FAISS HNSW build baseline")
    p.add_argument("--dataset", default="/tmp/d0test/d0-f32le-n12288-d128-seed0.bin")
    p.add_argument("--n", type=int, default=12288, help="vector count")
    p.add_argument("--d", type=int, default=128, help="dimensions")
    p.add_argument("--seed", type=int, default=0, help="dataset regen seed")
    p.add_argument("--m", type=int, default=32, help="HNSW M")
    p.add_argument("--efc", type=int, default=200, help="efConstruction")
    p.add_argument("--ef", default="32,64,128",
                   help=f"efSearch list to report; subset of {EMITTED_EF_POINTS}")
    p.add_argument("--participants", type=int, default=12, help="build thread count")
    p.add_argument("--pairs", type=int, default=3, help="number of alternating-order pairs")
    p.add_argument("--chunk-size", type=int, default=8192)
    p.add_argument("--query-count", type=int, default=256)
    p.add_argument("--build-grain", type=int, default=1)
    p.add_argument("--timeout", type=int, default=600, help="seconds per engine invocation")
    p.add_argument("--recall-deficit-threshold", type=float, default=0.005)
    p.add_argument("--require-ac", action="store_true",
                   help="refuse to run on battery power (default: warn loudly and continue)")
    p.add_argument("--results-root", default=os.path.join(here, "results"))
    p.add_argument("--infinity-bin", required=True)
    p.add_argument("--faiss-bin", required=True)
    args = p.parse_args()

    args.ef_list = [int(x) for x in args.ef.split(",") if x.strip()]
    unknown = [ef for ef in args.ef_list if ef not in EMITTED_EF_POINTS]
    if unknown:
        print(f"WARNING: efSearch {unknown} are not in the engine's emitted set "
              f"{EMITTED_EF_POINTS}; recall for those will be missing.", file=sys.stderr)

    for name, path in (("infinity-bin", args.infinity_bin), ("faiss-bin", args.faiss_bin)):
        if not (os.path.exists(path) and os.access(path, os.X_OK)):
            raise SystemExit(f"{name} not found or not executable: {path}")

    ensure_dataset(args.dataset, args.n, args.d, args.seed)

    host = preflight()
    print("=" * 72)
    print("HOST PREFLIGHT")
    print(f"  power           : {'AC' if host['on_ac_power'] else 'BATTERY'} "
          f"(battery {host['battery_pct']}%)")
    print(f"  thermal nominal : {host['thermal_nominal']}")
    print(f"  cpu idle        : {host['cpu_idle_pct']}%")
    print(f"  loadavg         : {host['loadavg_1_5_15']}")
    if host["competing_load_suspected"]:
        print("  !! COMPETING LOAD suspected (cpu idle < 85%). Top CPU procs: "
              + ", ".join(f"{x['name']}:{x['cpu_pct']}%" for x in host["top_cpu_procs"]))
    if not host["on_ac_power"]:
        banner = "!!! RUNNING ON BATTERY POWER -- results may be throttled/unstable !!!"
        print("\n" + "*" * len(banner) + f"\n{banner}\n" + "*" * len(banner) + "\n")
        if args.require_ac:
            raise SystemExit("refusing to run on battery (--require-ac set)")
    print("=" * 72)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.results_root, ts)
    os.makedirs(run_dir, exist_ok=True)

    engines = {"infinity": args.infinity_bin, "faiss": args.faiss_bin}
    runs = []
    for pair in range(args.pairs):
        # alternate order: even pairs infinity-first, odd pairs faiss-first
        order = ["infinity", "faiss"] if pair % 2 == 0 else ["faiss", "infinity"]
        print(f"\n--- pair {pair} (order: {' -> '.join(order)}) ---")
        for eng in order:
            sidecar = os.path.join(run_dir, f"sidecar-{eng}-p{pair}.json")
            r = run_engine(engines[eng], eng, args, sidecar)
            r["pair"] = pair
            r["order"] = order[0] + "-first"
            raw_path = os.path.join(run_dir, f"raw-{eng}-p{pair}-{r['order']}.txt")
            with open(raw_path, "w") as fh:
                fh.write("### ARGV\n" + " ".join(r.get("argv", [])) + "\n### STDOUT\n"
                         + r.get("stdout", "") + "\n### STDERR\n" + r.get("stderr", ""))
            r["raw_path"] = raw_path
            status = "OK" if r["ok"] else f"FAIL({r.get('reason', r.get('status'))})"
            print(f"  {eng:9s} {status:22s} build={fmt_ns(r.get('build_ns'))} "
                  f"src={r.get('build_source')}")
            # keep json small: drop bulky stdout/stderr from the in-memory record
            r.pop("stdout", None)
            r.pop("stderr", None)
            runs.append(r)
            # discard the large audit sidecar the engine wrote; stdout has what we need
            if os.path.exists(sidecar):
                os.remove(sidecar)

    # aggregate
    agg = {}
    for eng in ("infinity", "faiss"):
        er = [r for r in runs if r["engine"] == eng and r["ok"]]
        build_samples = [r["build_ns"] for r in er]
        recall_by_ef = {}
        for ef in args.ef_list:
            vals = [r["recall_at_10"][ef] for r in er if ef in r["recall_at_10"]]
            recall_by_ef[ef] = statistics.median(vals) if vals else None
        sp = spread(build_samples)
        vps = (args.n / (sp["median"] / 1e9)) if sp["median"] else None
        qps_spread = spread([r["qps"] for r in er if r.get("qps")])
        agg[eng] = {
            "ok_runs": len(er),
            "build_ns": sp,
            "vectors_per_sec": vps,
            "recall_at_10": recall_by_ef,
            "build_source": er[0]["build_source"] if er else None,
            "graph_directed_edges": er[0]["graph_directed_edges"] if er else None,
            "graph_level0_capacity": er[0]["graph_level0_capacity"] if er else None,
            "graph_upper_capacity": er[0]["graph_upper_capacity"] if er else None,
            "qps": qps_spread,
            "query_latency_p50_ns": _med([r.get("query_latency_p50_ns") for r in er]),
            "query_latency_p95_ns": _med([r.get("query_latency_p95_ns") for r in er]),
            "query_latency_p99_ns": _med([r.get("query_latency_p99_ns") for r in er]),
            "query_throughput_concurrency": er[0].get("query_throughput_concurrency") if er else None,
            "build_buckets_per_worker": er[0].get("build_buckets_per_worker") if er else None,
            "submitted_tasks": er[0].get("submitted_tasks") if er else None,
        }

    # recall-parity gate
    gate = {"threshold": args.recall_deficit_threshold, "per_ef": {}, "pass": True}
    for ef in args.ef_list:
        fi = agg["faiss"]["recall_at_10"].get(ef)
        ii = agg["infinity"]["recall_at_10"].get(ef)
        if fi is None or ii is None:
            gate["per_ef"][ef] = {"deficit": None, "pass": None}
            continue
        deficit = fi - ii  # positive => infinity worse
        ok = abs(deficit) <= args.recall_deficit_threshold
        gate["per_ef"][ef] = {"faiss": fi, "infinity": ii, "deficit": deficit, "pass": ok}
        if not ok:
            gate["pass"] = False

    ratio = None
    if agg["infinity"]["build_ns"]["median"] and agg["faiss"]["build_ns"]["median"]:
        ratio = agg["infinity"]["build_ns"]["median"] / agg["faiss"]["build_ns"]["median"]

    all_ok = all(r["ok"] for r in runs)
    wall_fallback = any(not r.get("cold_build_ns_present", True) for r in runs if r["ok"])

    result = {
        "timestamp": ts,
        "config": {k: getattr(args, k) for k in
                   ("dataset", "n", "d", "seed", "m", "efc", "ef_list", "participants",
                    "pairs", "chunk_size", "query_count", "build_grain", "timeout",
                    "recall_deficit_threshold")},
        "binaries": {"infinity": args.infinity_bin, "faiss": args.faiss_bin},
        "host": host,
        "runs": runs,
        "aggregate": agg,
        "recall_gate": gate,
        "build_time_ratio_infinity_over_faiss": ratio,
        "qps_ratio_infinity_over_faiss": (
            agg["infinity"]["qps"]["median"] / agg["faiss"]["qps"]["median"]
            if agg["infinity"]["qps"]["median"] and agg["faiss"]["qps"]["median"] else None),
        "all_runs_ok": all_ok,
        "build_timing_used_wall_clock_fallback": wall_fallback,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    print_report(args, agg, gate, ratio, all_ok, wall_fallback, run_dir)
    return 0 if all_ok else 1


def print_report(args, agg, gate, ratio, all_ok, wall_fallback, run_dir):
    inf, fai = agg["infinity"], agg["faiss"]
    print("\n" + "#" * 72)
    bpw = inf.get("build_buckets_per_worker")
    tasks = inf.get("submitted_tasks")
    sched = f" buckets/worker={bpw} tasks={tasks}" if bpw is not None else ""
    print(f"# BASELINE RESULT  (n={args.n} d={args.d} M={args.m} efC={args.efc} "
          f"threads={args.participants} pairs={args.pairs}{sched})")
    print("#" * 72 + "\n")

    print("## Build timing (median of paired runs)\n")
    print("| engine | ok runs | median build | min | max | rel MAD | vectors/sec |")
    print("|--------|--------:|-------------:|----:|----:|--------:|------------:|")
    for name, a in (("Infinity", inf), ("FAISS", fai)):
        b = a["build_ns"]
        print(f"| {name} | {a['ok_runs']} | {fmt_ns(b['median'])} | {fmt_ns(b['min'])} | "
              f"{fmt_ns(b['max'])} | {('%.1f%%' % (b['rel_mad']*100)) if b['rel_mad'] is not None else 'n/a'} | "
              f"{('%.0f' % a['vectors_per_sec']) if a['vectors_per_sec'] else 'n/a'} |")

    if ratio is not None:
        verdict = "reportable" if gate["pass"] else "NOT COMPARABLE (RECALL-UNMATCHED)"
        print(f"\n**Build-time ratio Infinity/FAISS = {ratio:.3f}x**  "
              f"(<1 means Infinity builds faster) -- {verdict}")

    print("\n## recall@10 parity gate (threshold "
          f"|deficit| <= {args.recall_deficit_threshold})\n")
    print("| efSearch | FAISS recall@10 | Infinity recall@10 | deficit (F-I) | gate |")
    print("|---------:|----------------:|-------------------:|--------------:|------|")
    for ef in args.ef_list:
        g = gate["per_ef"].get(ef, {})
        fi, ii, dfc = g.get("faiss"), g.get("infinity"), g.get("deficit")
        gp = g.get("pass")
        gate_s = "n/a" if gp is None else ("PASS" if gp else "FAIL")
        print(f"| {ef} | {('%.4f' % fi) if fi is not None else 'n/a'} | "
              f"{('%.4f' % ii) if ii is not None else 'n/a'} | "
              f"{('%+.4f' % dfc) if dfc is not None else 'n/a'} | {gate_s} |")
    print(f"\n**Recall-parity gate: {'PASS' if gate['pass'] else 'FAIL -> RECALL-UNMATCHED'}**")

    qps_ratio = None
    if inf["qps"]["median"] and fai["qps"]["median"]:
        qps_ratio = inf["qps"]["median"] / fai["qps"]["median"]
    conc = inf.get("query_throughput_concurrency") or fai.get("query_throughput_concurrency")
    print(f"\n## Query performance (k=10, efSearch=256, throughput at {conc} concurrent threads,\n"
          "   latency measured one query at a time; identical harness constants for both engines)\n")
    print("| engine | QPS (median) | QPS min | QPS max | p50 latency | p95 | p99 |")
    print("|--------|-------------:|--------:|--------:|------------:|----:|----:|")
    for name, a in (("Infinity", inf), ("FAISS", fai)):
        q = a["qps"]
        print(f"| {name} | {('%.0f' % q['median']) if q['median'] else 'n/a'} | "
              f"{('%.0f' % q['min']) if q['min'] else 'n/a'} | "
              f"{('%.0f' % q['max']) if q['max'] else 'n/a'} | "
              f"{fmt_ns(a['query_latency_p50_ns'])} | {fmt_ns(a['query_latency_p95_ns'])} | "
              f"{fmt_ns(a['query_latency_p99_ns'])} |")
    if qps_ratio is not None:
        print(f"\n**QPS ratio Infinity/FAISS = {qps_ratio:.3f}x**  (>1 means Infinity serves "
              "more queries/sec)")

    print("\n## Graph audit (single representative run)\n")
    print("| engine | directed edges | level0 capacity | upper capacity |")
    print("|--------|---------------:|----------------:|---------------:|")
    for name, a in (("Infinity", inf), ("FAISS", fai)):
        print(f"| {name} | {a['graph_directed_edges']} | {a['graph_level0_capacity']} | "
              f"{a['graph_upper_capacity']} |")

    notes = []
    if wall_fallback:
        notes.append("build timing used WALL-CLOCK FALLBACK for >=1 run "
                     "(engine did not emit *_cold_build_ns)")
    else:
        notes.append("build timing taken from engine stdout (*_cold_build_ns)")
    if not all_ok:
        notes.append("one or more runs FAILED -- see results.json")
    print("\n## Notes\n" + "\n".join(f"- {n}" for n in notes))
    print(f"\nArtifacts: {run_dir}/  (raw-*.txt, results.json)")


if __name__ == "__main__":
    sys.exit(main())
