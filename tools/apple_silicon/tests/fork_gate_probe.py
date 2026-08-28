from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.apple_silicon import native_hnsw_d0 as d0


def main() -> None:
    python_executable = Path(
        "/Applications/Xcode.app/Contents/Developer/usr/bin/python3"
    )
    python_process_executable = Path(
        "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/"
        "Python3.framework/Versions/3.9/Resources/Python.app/Contents/"
        "MacOS/Python"
    )
    child_script = (
        "import os, signal\n"
        f"for phase in {d0.ATTESTATION_BARRIER_PHASES!r}:\n"
        "    print('attestation_barrier=' + phase, flush=True)\n"
        "    os.kill(os.getpid(), signal.SIGSTOP)\n"
        "print('child-complete', flush=True)\n"
    )
    thread_count = d0.process_thread_count(d0.os.getpid())
    result = d0.launch_and_supervise_attested_process(
        ["/usr/bin/time", "-lp", str(python_executable), "-c", child_script],
        launch_cwd=REPO,
        environment=d0.benchmark_environment(1),
        cwd=REPO,
        timeout_seconds=10.0,
        expected_wrapper_executable=d0.process_executable_identity(
            Path("/usr/bin/time")
        ),
        expected_benchmark_executable=d0.process_executable_identity(
            python_process_executable
        ),
    )
    evidence = result["benchmark_process_evidence"]
    print(
        json.dumps(
            {
                "thread_count": thread_count,
                "pid": result["pid"],
                "returncode": result["returncode"],
                "timed_out": result["timed_out"],
                "errors": result["errors"],
                "phases": [
                    barrier["phase"] for barrier in result["barriers"]
                ],
                "child_complete": "child-complete\n" in result["stdout"],
                "member_pids": evidence["pre_reap_quiescence"][
                    "member_pids"
                ],
                "terminal_pid": evidence["terminal_observation"]["pid"],
                "reap_pid": evidence["reap"]["pid"],
                "status_validated": evidence["reap"]["status_validated"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
