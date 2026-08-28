#!/usr/bin/env python3
"""campaign.py -- run one HNSW-d0 harness binary and collect its stdout.

The prebuilt harness binaries (build/bench/{infinity,faiss}_hnsw_d0, built from
tools/apple_silicon/native_hnsw_smoke) do NOT use the old 11-arg contract. They
require a 17-arg "campaign" invocation and, at four points during the run, call
raise(SIGSTOP) on themselves at an "attestation barrier" -- they expect an
external supervisor to observe the stop and send SIGCONT to resume them. Run
standalone under plain subprocess.run they hang suspended forever.

This module is the whole, minimal mechanism needed to drive them:
  1. Build the 17-arg argv the binary demands. Two of those args are SHA-256
     hex digests the binary re-computes and verifies at runtime:
       argv[15] EXPECTED_EXECUTABLE_SHA256 == sha256(the binary file)
       argv[16] EXPECTED_DATASET_SHA256    == sha256(the dataset file)
     plus a role id (infinity->1, faiss->4; verified by RoleMatchesEngine), a
     64-hex campaign nonce (echoed only, not verified) and a schedule sequence.
  2. Supervise: send SIGCONT to the child whenever it suspends itself, until it
     exits. SIGCONT to a not-stopped process with no handler is ignored, so a
     periodic nudge is safe and needs no WUNTRACED bookkeeping.

The engine's own <engine>_cold_build_ns is measured with steady_clock strictly
INSIDE the timed region and OUTSIDE every SIGSTOP barrier (verified by reading
faiss_hnsw_bridge.cpp / infinity bridge), so the suspend/resume pauses do not
contaminate the reported build time. Wall-clock here is NOT a valid build time
and is returned only for diagnostics.

This is not a verification framework: it is the minimum required to satisfy an
existing binary's CLI and un-pause a process that pauses itself.
"""

import hashlib
import os
import signal
import subprocess
import tempfile
import time

# infinity accepts role ids {1,2,3}; faiss accepts {4} (RoleMatchesEngine in
# tools/apple_silicon/native_hnsw_smoke/hnsw_d0_runner.cpp).
ROLE_BY_ENGINE = {"infinity": 1, "faiss": 4}
# The nonce is parsed as 64 hex chars and echoed back; its value is not checked.
CAMPAIGN_NONCE_HEX = "00" * 32
SCHEDULE_SEQUENCE = "1"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_argv(binary, engine, dataset, n, d, m, efc, ef_search, chunk,
               query_count, participants, build_grain, sidecar):
    """Assemble the 17-arg campaign argv the harness binaries require."""
    return [
        binary,
        dataset,
        str(n), str(d),
        str(m), str(efc),
        str(ef_search),
        str(chunk),
        str(query_count),
        str(participants),
        str(build_grain),
        sidecar,
        CAMPAIGN_NONCE_HEX,
        SCHEDULE_SEQUENCE,
        str(ROLE_BY_ENGINE[engine]),
        sha256_file(binary),
        sha256_file(dataset),
    ]


def run_campaign(argv, timeout, nudge_interval=0.05):
    """Run argv, resuming the child past every self-imposed SIGSTOP barrier.

    Returns dict: stdout, stderr, returncode, wall_ns, timed_out.
    stdout/stderr are captured to temp files (never PIPE) so a suspended child
    can never deadlock us by filling a pipe buffer.
    """
    out_fh = tempfile.TemporaryFile(mode="w+")
    err_fh = tempfile.TemporaryFile(mode="w+")
    t0 = time.perf_counter()
    proc = subprocess.Popen(argv, stdout=out_fh, stderr=err_fh, text=True)
    timed_out = False
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        # The child suspends itself (SIGSTOP) at attestation barriers; resume it.
        try:
            os.kill(proc.pid, signal.SIGCONT)
        except ProcessLookupError:
            break
        if time.perf_counter() - t0 > timeout:
            timed_out = True
            proc.kill()
            proc.wait()
            break
        time.sleep(nudge_interval)
    wall_ns = int((time.perf_counter() - t0) * 1e9)
    out_fh.seek(0)
    err_fh.seek(0)
    stdout = out_fh.read()
    stderr = err_fh.read()
    out_fh.close()
    err_fh.close()
    return {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": proc.returncode,
        "wall_ns": wall_ns,
        "timed_out": timed_out,
    }
