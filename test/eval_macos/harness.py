"""Shared harness for the macOS arm64 end-to-end evaluation suites.

Every suite under ``test/eval_macos/`` needs the same four things, and getting any
of them subtly different produces failures that look like engine defects:

1. **Its own server instance.** These suites run concurrently. Two servers cannot
   share a port, and one suite's ``DROP DATABASE`` must not delete another's data,
   so each suite owns an instance name and a port offset (see ``INSTANCES`` below)
   and drives ``scripts/apple_silicon/run_server.sh`` with them.

2. **A writable data root.** The committed suites hardcode ``/var/infinity`` (both
   the ``.slt`` corpus and ``test_pysdk/common/common_values.py``). ``/var`` is
   root-owned on macOS. ``data_root()`` hands back a per-instance directory under
   the build tree instead, and ``stage_csv``/``stage_jsonl`` write import fixtures
   into it, so ``COPY ... FROM`` gets a path the server can actually read.

3. **A connection that fails loudly.** ``infinity.connect`` takes a
   ``NetworkAddress``, not a string: passing ``"127.0.0.1:23817"`` raises
   ``INVALID_SERVER_ADDRESS`` rather than connecting, which reads like the server
   is down. ``connect()`` builds the address from the instance's own port.

4. **Teardown that runs even when the test fails.** A suite that leaves its server
   running holds a port and ~1 GB of buffer pool for the rest of the evaluation.

The server binary is expected at ``build/macos-arm64-release/src/infinity``; build
it with ``scripts/apple_silicon/build_server.sh macos-arm64-release infinity``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SERVER = REPO_ROOT / "scripts" / "apple_silicon" / "run_server.sh"
DEFAULT_BINARY = REPO_ROOT / "build" / "macos-arm64-release" / "src" / "infinity"
EVAL_ROOT = REPO_ROOT / "build" / "eval"

# Port-offset registry. Offsets are assigned here rather than derived from a hash of
# the instance name so that a collision is a visible edit to this table instead of a
# silent bind failure at 3am. Offset 0 is deliberately absent: it is the shared `dev`
# instance a human may have running, and an evaluation suite must never adopt it and
# then --fresh it away.
#
# Each offset N moves all four listeners: pg 5432+N, http 23820+N, thrift 23817+N,
# peer 23850+N. Keep offsets >= 100 apart so no two instances overlap.
INSTANCES: dict[str, int] = {
    "eval-ddl": 100,
    "eval-insert": 200,
    "eval-mutate": 300,
    "eval-dense": 400,
    "eval-sparse": 500,
    "eval-fulltext": 600,
    "eval-hybrid": 700,
    "eval-secondary": 800,
    "eval-types": 900,
    "eval-snapshot": 1000,
    "eval-http": 1100,
    "eval-meta": 1200,
    "eval-negative": 1300,
    "eval-concurrency": 1400,
    "eval-recovery": 1500,
    "eval-load": 1600,
    "eval-soak": 1700,
    "eval-leak": 1800,
    "eval-spare-a": 1900,
    "eval-spare-b": 2000,
}


class HarnessError(RuntimeError):
    """Raised for setup failures, so they are never mistaken for engine defects."""


@dataclass
class Instance:
    """A single server instance: its ports, its directories, its lifecycle."""

    name: str
    offset: int
    wal_flush: str = "full_fsync"
    binary: Path = DEFAULT_BINARY
    _started: bool = field(default=False, init=False, repr=False)

    @property
    def pg_port(self) -> int:
        return 5432 + self.offset

    @property
    def http_port(self) -> int:
        return 23820 + self.offset

    @property
    def thrift_port(self) -> int:
        return 23817 + self.offset

    @property
    def peer_port(self) -> int:
        return 23850 + self.offset

    @property
    def root(self) -> Path:
        return REPO_ROOT / "build" / "instances" / self.name

    @property
    def log_file(self) -> Path:
        return self.root / "log" / "infinity.log"

    @property
    def http_base(self) -> str:
        return f"http://127.0.0.1:{self.http_port}"

    # ---- lifecycle ----------------------------------------------------------

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        cmd = [str(RUN_SERVER), *args, "--instance", self.name,
               "--port-offset", str(self.offset), "--wal-flush", self.wal_flush,
               "--binary", str(self.binary.relative_to(REPO_ROOT))]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if check and proc.returncode != 0:
            raise HarnessError(
                f"{' '.join(cmd)} exited {proc.returncode}\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )
        return proc

    def start(self, fresh: bool = True) -> "Instance":
        """Start this instance, replacing any previous one of the same name.

        ``fresh=True`` by default: a surviving data directory makes suites
        non-idempotent (a fixed-name ``CREATE SNAPSHOT`` fails the second time,
        row counts double), and a suite that silently inherits state produces
        failures that do not reproduce.
        """
        if not self.binary.exists():
            raise HarnessError(
                f"no server binary at {self.binary} — build it with:\n"
                f"  scripts/apple_silicon/build_server.sh macos-arm64-release infinity"
            )
        self._run("stop", check=False)
        args = ["start"]
        if fresh:
            args.append("--fresh")
        self._run(*args)
        self._started = True
        self.await_ready()
        return self

    def stop(self) -> None:
        self._run("stop", check=False)
        self._started = False

    def restart(self, *, graceful: bool = True) -> None:
        """Restart in place, keeping the data directory — for recovery tests.

        ``graceful=False`` sends SIGKILL instead of SIGTERM, so the server has no
        chance to checkpoint and startup must recover from the WAL alone.
        """
        if graceful:
            self.stop()
        else:
            pid = self.pid()
            if pid is None:
                raise HarnessError(f"instance {self.name} is not running; nothing to kill")
            os.kill(pid, 9)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if self.pid() is None:
                    break
                time.sleep(0.1)
            else:
                raise HarnessError(f"pid {pid} still alive 30s after SIGKILL")
            (self.root / "infinity.pid").unlink(missing_ok=True)
        self.start(fresh=False)

    def pid(self) -> int | None:
        pidfile = self.root / "infinity.pid"
        if not pidfile.exists():
            return None
        try:
            pid = int(pidfile.read_text().strip())
        except ValueError:
            return None
        try:
            os.kill(pid, 0)
        except OSError:
            return None
        return pid

    def await_ready(self, timeout: float = 120.0) -> None:
        """Block until the thrift port accepts AND a query round-trips.

        An accepted connection is not readiness: the listener binds before the
        catalog finishes replaying the WAL, so a query issued in that window fails
        with an error that looks like a defect. Requiring a successful query is the
        only check that covers both.
        """
        deadline = time.monotonic() + timeout
        last: Exception | None = None
        while time.monotonic() < deadline:
            if self._port_accepts(self.thrift_port):
                try:
                    conn = self.connect()
                    try:
                        conn.list_databases()
                    finally:
                        conn.disconnect()
                    return
                except Exception as exc:  # noqa: BLE001 - reported verbatim below
                    last = exc
            time.sleep(0.25)
        tail = ""
        if self.log_file.exists():
            tail = "\n".join(self.log_file.read_text(errors="replace").splitlines()[-40:])
        raise HarnessError(
            f"instance {self.name} not ready after {timeout}s "
            f"(thrift {self.thrift_port}); last error: {last!r}\n--- log tail ---\n{tail}"
        )

    @staticmethod
    def _port_accepts(port: int) -> bool:
        with socket.socket() as sock:
            sock.settimeout(1.0)
            try:
                sock.connect(("127.0.0.1", port))
            except OSError:
                return False
        return True

    # ---- clients ------------------------------------------------------------

    def connect(self):
        """Open a thrift SDK connection to this instance.

        ``infinity.connect`` requires a ``NetworkAddress``; a ``"host:port"``
        string raises ``INVALID_SERVER_ADDRESS`` instead of connecting.
        """
        import infinity
        from infinity.common import NetworkAddress

        return infinity.connect(NetworkAddress("127.0.0.1", self.thrift_port))

    def db(self, name: str = "default_db"):
        conn = self.connect()
        return conn, conn.get_database(name)

    # ---- data staging -------------------------------------------------------

    def data_root(self) -> Path:
        """A server-readable directory for import/export fixtures.

        The committed suites use ``/var/infinity/test_data``, which does not exist
        and is not creatable without root on macOS. This lives under the instance
        so ``--fresh`` cleans it up with everything else.
        """
        path = self.root / "test_data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stage_file(self, name: str, content: str) -> Path:
        path = self.data_root() / name
        path.write_text(content)
        return path

    def stage_csv(self, name: str, rows: Iterable[Sequence[Any]],
                  header: Sequence[str] | None = None, delimiter: str = ",") -> Path:
        lines = []
        if header is not None:
            lines.append(delimiter.join(str(h) for h in header))
        lines.extend(delimiter.join(_csv_cell(c) for c in row) for row in rows)
        return self.stage_file(name, "\n".join(lines) + "\n")

    def stage_jsonl(self, name: str, records: Iterable[dict]) -> Path:
        body = "\n".join(json.dumps(r) for r in records)
        return self.stage_file(name, body + "\n")

    # ---- observation --------------------------------------------------------

    def rss_bytes(self) -> int:
        """Resident set size of the server process, for leak detection."""
        pid = self.pid()
        if pid is None:
            raise HarnessError(f"instance {self.name} is not running")
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True, check=True)
        return int(out.stdout.strip()) * 1024

    def fd_count(self) -> int:
        """Open file descriptors held by the server, for descriptor-leak detection."""
        pid = self.pid()
        if pid is None:
            raise HarnessError(f"instance {self.name} is not running")
        out = subprocess.run(["lsof", "-p", str(pid)], capture_output=True, text=True)
        return max(0, len(out.stdout.splitlines()) - 1)

    def thread_count(self) -> int:
        pid = self.pid()
        if pid is None:
            raise HarnessError(f"instance {self.name} is not running")
        out = subprocess.run(["ps", "-M", "-p", str(pid)], capture_output=True, text=True)
        return max(0, len(out.stdout.splitlines()) - 1)

    # The server's spdlog pattern is "[HH:MM:SS.mmm] [tid] [level] message" — square
    # brackets, not the pipe-delimited format this originally matched. An earlier version
    # looked for "| error |" and therefore returned [] unconditionally, which silently
    # turned every suite's server-side error check into a no-op. Verified against a real
    # log: levels present are [info], [warning], [error], [critical].
    _LEVEL_RE = re.compile(r"\[(error|critical|fatal)\]", re.IGNORECASE)

    def log_errors(self, since_offset: int = 0) -> list[str]:
        """Lines the server itself logged at error level or above.

        A query that returns a clean result while the server logs an error is a defect the
        SDK cannot see, so suites should check this as well as the client-side result.

        Note this reports the *logged* level. A crash can bypass logging entirely — the
        index-scan SIGSEGV writes nothing here — so an empty list is not proof of health.
        Pair it with ``pid()``.
        """
        if not self.log_file.exists():
            return []
        text = self.log_file.read_text(errors="replace")[since_offset:]
        return [ln for ln in text.splitlines() if self._LEVEL_RE.search(ln)]

    def log_warnings(self, since_offset: int = 0) -> list[str]:
        """Lines logged at warning level.

        Separate from ``log_errors`` because the engine reports some genuinely destructive
        actions at warning level — deleting a corrupt WAL is a ``[warning]`` — so a suite
        checking only errors misses them.
        """
        if not self.log_file.exists():
            return []
        text = self.log_file.read_text(errors="replace")[since_offset:]
        return [ln for ln in text.splitlines() if "[warning]" in ln.lower()]

    def log_offset(self) -> int:
        return self.log_file.stat().st_size if self.log_file.exists() else 0

    # ---- context manager ----------------------------------------------------

    def __enter__(self) -> "Instance":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def instance(name: str, **kwargs: Any) -> Instance:
    """Look up a registered instance by name.

    Registered rather than free-form so that two suites cannot pick the same port
    offset; add new names to ``INSTANCES``.
    """
    if name not in INSTANCES:
        raise HarnessError(
            f"unknown instance {name!r}; register it in harness.INSTANCES "
            f"(known: {', '.join(sorted(INSTANCES))})"
        )
    return Instance(name=name, offset=INSTANCES[name], **kwargs)


def _csv_cell(value: Any) -> str:
    text = str(value)
    if any(ch in text for ch in ',"\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


# ---- result recording -------------------------------------------------------

def record(suite: str, payload: dict) -> Path:
    """Write a suite's machine-readable result under build/eval/results/."""
    out = EVAL_ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{suite}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def free_disk_bytes() -> int:
    return shutil.disk_usage(REPO_ROOT).free
