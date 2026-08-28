#!/usr/bin/env python3
"""Fail-closed, read-only post-restart Apple execution-policy gate."""

from __future__ import annotations

import copy
import ctypes
import datetime
import errno
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
GATE_ID = "native-hnsw-post-restart-execution-policy-v1"

BOOT_CUTOFF_TEXT = "2026-08-22T02:52:10-04:00"
BOOT_CUTOFF_SECONDS = 1_787_381_530
BOOT_CUTOFF_MICROSECONDS = 0

COMMAND_TIMEOUT_SECONDS = 5.0
LOG_TIMEOUT_SECONDS = 10.0
LOG_WINDOW = "10m"
CPU_SAMPLE_COUNT = 5
CPU_SAMPLE_INTERVAL_SECONDS = 1.0
CPU_SAMPLE_COMMAND_TIMEOUT_SECONDS = 1.0
CPU_SAMPLE_TOTAL_LIMIT_SECONDS = 10.0
CPU_REJECTION_THRESHOLD = Decimal("90")
CANARY_TIMEOUT_SECONDS = 5.0
POST_KILL_WAIT_SECONDS = 1.0
GROUP_POLL_SECONDS = 0.01
MAX_CANARY_BYTES = 16 * 1024 * 1024
P_PID = 1
WNOHANG = 0x00000001
WEXITED = 0x00000004
WNOWAIT = 0x00000020
CLD_EXITED = 1
CLD_KILLED = 2
CLD_DUMPED = 3
PROCESS_GROUP_PID_CAPACITY = 4096

SYSCTL_COMMAND = ("/usr/sbin/sysctl", "-n", "kern.boottime")
GATEKEEPER_COMMAND = ("/usr/sbin/spctl", "--status")
SIP_COMMAND = ("/usr/bin/csrutil", "status")
XCODE_SELECT_COMMAND = ("/usr/bin/xcode-select", "-p")
XCODE_FIRST_LAUNCH_COMMAND = (
    "/usr/bin/xcodebuild",
    "-checkFirstLaunchStatus",
)
XCODE_LICENSE_COMMAND = ("/usr/bin/xcodebuild", "-license", "check")
SYSPOLICY_DISCOVERY_COMMAND = ("/bin/ps", "-axo", "pid=,comm=")
LOG_PREDICATE = (
    'process == "syspolicyd" AND '
    '(eventMessage CONTAINS "failed to call driver" OR '
    'eventMessage CONTAINS "Unable to initialize qtn_proc")'
)
SYSPOLICY_LOG_COMMAND = (
    "/usr/bin/log",
    "show",
    "--last",
    LOG_WINDOW,
    "--style",
    "compact",
    "--predicate",
    LOG_PREDICATE,
    "--no-pager",
)
TRUE_PATH = Path("/usr/bin/true")
PRIVATE_TEMP_ROOT = Path("/private/tmp")
CANARY_FILENAME = "true-canary"

FORBIDDEN_LOG_MESSAGES = (
    "failed to call driver",
    "Unable to initialize qtn_proc",
)

FAILURE_EXIT_CODES = {
    "platform": 10,
    "boot_time": 11,
    "gatekeeper": 12,
    "sip": 13,
    "xcode_select": 14,
    "xcode_first_launch": 15,
    "xcode_license": 16,
    "syspolicyd_discovery": 17,
    "syspolicyd_cpu": 18,
    "syspolicyd_logs": 19,
    "canary_copy": 20,
    "canary_execution": 21,
    "canary_process_group": 22,
    "canary_cleanup": 23,
    "usage": 64,
    "internal": 70,
    "interrupted": 71,
}

READ_ONLY_PROHIBITIONS = (
    "no Gatekeeper or SIP changes",
    "no developer-mode changes",
    "no service restart or kickstart",
    "no provenance or quarantine removal",
    "no endpoint-extension changes",
)

_BOOT_TIME_PATTERN = re.compile(
    r"^\{\s*sec\s*=\s*([0-9]+)\s*,\s*usec\s*=\s*([0-9]+)\s*\}"
    r"(?:\s+.*)?$"
)
_DISCOVERY_LINE_PATTERN = re.compile(r"^\s*([0-9]+)\s+(.+?)\s*$")
_CPU_LINE_PATTERN = re.compile(
    r"^\s*([0-9]+)\s+([0-9]+(?:\.[0-9]+)?)\s+(\S+)\s*$"
)


class GateFailure(RuntimeError):
    """A named fail-closed gate outcome with a stable nonzero exit code."""

    def __init__(
        self,
        check: str,
        message: str,
        checks: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if check not in FAILURE_EXIT_CODES:
            raise ValueError("Unknown execution-policy failure check: %s" % check)
        super().__init__(message)
        self.check = check
        self.exit_code = FAILURE_EXIT_CODES[check]
        self.checks = copy.deepcopy(dict(checks or {}))


@dataclass(frozen=True)
class CommandResult:
    argv: Tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class CanaryMaterial:
    directory: Path
    executable: Path
    directory_device: int
    directory_inode: int
    evidence: Dict[str, Any]


class _Sigval(ctypes.Union):
    _fields_ = [
        ("sival_int", ctypes.c_int),
        ("sival_ptr", ctypes.c_void_p),
    ]


class _DarwinSiginfo(ctypes.Structure):
    _fields_ = [
        ("si_signo", ctypes.c_int),
        ("si_errno", ctypes.c_int),
        ("si_code", ctypes.c_int),
        ("si_pid", ctypes.c_int),
        ("si_uid", ctypes.c_uint),
        ("si_status", ctypes.c_int),
        ("si_addr", ctypes.c_void_p),
        ("si_value", _Sigval),
        ("si_band", ctypes.c_long),
        ("_reserved", ctypes.c_ulong * 7),
    ]


@dataclass(frozen=True)
class _TerminalObservation:
    pid: int
    code: int
    status: int


@dataclass
class _BoundedChild:
    process: Any
    group_verified: bool = False
    terminal: Optional[_TerminalObservation] = None
    pre_reap_member_pids: Optional[List[int]] = None
    reap_started: bool = False
    reaped: bool = False
    returncode: Optional[int] = None


_WAITID_LIBRARY: Optional[Any] = None
_LIBPROC_LIBRARY: Optional[Any] = None


def canonical_json_bytes(value: Any) -> bytes:
    """Return one ASCII, sorted-key, newline-terminated JSON record."""

    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_uid == right.st_uid
        and left.st_gid == right.st_gid
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _nofollow_flag() -> int:
    return int(getattr(os, "O_NOFOLLOW", 0))


def _cloexec_flag() -> int:
    return int(getattr(os, "O_CLOEXEC", 0))


def _read_stable_regular_file(path: Path, *, maximum_bytes: int) -> Tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | _cloexec_flag() | _nofollow_flag()
    descriptor = os.open(str(path), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("Path is not a regular file: %s" % path)
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise RuntimeError(
                "File size is outside the accepted range: %s" % before.st_size
            )
        chunks: List[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("File ended before its recorded size: %s" % path)
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1) != b"":
            raise RuntimeError("File grew while it was read: %s" % path)
        after = os.fstat(descriptor)
        path_status = os.lstat(str(path))
        if not _same_file_identity(before, after):
            raise RuntimeError("File changed while it was read: %s" % path)
        if not _same_file_identity(after, path_status):
            raise RuntimeError("File path changed while it was read: %s" % path)
        return b"".join(chunks), after
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise RuntimeError("Canary copy made no write progress")
        offset += written


def create_verified_canary(
    *,
    source: Path = TRUE_PATH,
    temporary_root: Path = PRIVATE_TEMP_ROOT,
) -> CanaryMaterial:
    """Create a private executable byte copy without changing source metadata."""

    source_bytes, source_status = _read_stable_regular_file(
        source,
        maximum_bytes=MAX_CANARY_BYTES,
    )
    if source_status.st_mode & 0o111 == 0:
        raise RuntimeError("Canary source is not executable: %s" % source)

    directory: Optional[Path] = None
    try:
        directory = Path(
            tempfile.mkdtemp(
                prefix="infinity-execution-policy-",
                dir=str(temporary_root),
            )
        )
        os.chmod(str(directory), 0o700)
        directory_status = os.lstat(str(directory))
        if not stat.S_ISDIR(directory_status.st_mode):
            raise RuntimeError("Private canary path is not a directory")
        if directory_status.st_uid != os.getuid():
            raise RuntimeError("Private canary directory has the wrong owner")
        if stat.S_IMODE(directory_status.st_mode) != 0o700:
            raise RuntimeError("Private canary directory mode is not 0700")

        executable = directory / CANARY_FILENAME
        descriptor = os.open(
            str(executable),
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | _cloexec_flag()
            | _nofollow_flag(),
            0o700,
        )
        try:
            _write_all(descriptor, source_bytes)
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
            destination_status = os.fstat(descriptor)
            if (
                not stat.S_ISREG(destination_status.st_mode)
                or destination_status.st_nlink != 1
                or destination_status.st_uid != os.getuid()
                or stat.S_IMODE(destination_status.st_mode) != 0o700
            ):
                raise RuntimeError("Canary destination metadata is invalid")
        finally:
            os.close(descriptor)

        copied_bytes, copied_status = _read_stable_regular_file(
            executable,
            maximum_bytes=MAX_CANARY_BYTES,
        )
        if copied_bytes != source_bytes:
            raise RuntimeError("Canary destination bytes differ from /usr/bin/true")
        if copied_status.st_dev != directory_status.st_dev:
            raise RuntimeError("Canary destination escaped its private filesystem")

        digest = _sha256(source_bytes)
        return CanaryMaterial(
            directory=directory,
            executable=executable,
            directory_device=int(directory_status.st_dev),
            directory_inode=int(directory_status.st_ino),
            evidence={
                "bytes": len(source_bytes),
                "copy_mode": "0700",
                "copy_name": CANARY_FILENAME,
                "copy_sha256": _sha256(copied_bytes),
                "private_directory_mode": "0700",
                "source_mode": "%04o" % stat.S_IMODE(source_status.st_mode),
                "source_path": str(source),
                "source_sha256": digest,
                "verified_byte_equal": _sha256(copied_bytes) == digest,
            },
        )
    except BaseException as primary_error:
        if directory is not None:
            try:
                shutil.rmtree(str(directory))
                if os.path.lexists(str(directory)):
                    raise RuntimeError(
                        "Private canary directory survived failed-copy cleanup"
                    )
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "Canary creation failed and private cleanup also failed: "
                    "%s: %s"
                    % (type(cleanup_error).__name__, cleanup_error)
                ) from primary_error
        raise


def cleanup_verified_canary(material: CanaryMaterial) -> Dict[str, Any]:
    status = os.lstat(str(material.directory))
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_dev != material.directory_device
        or status.st_ino != material.directory_inode
        or status.st_uid != os.getuid()
    ):
        raise RuntimeError("Private canary directory identity changed before cleanup")
    shutil.rmtree(str(material.directory))
    if os.path.lexists(str(material.directory)):
        raise RuntimeError("Private canary directory survived cleanup")
    return {"private_directory_removed": True}


def _waitid_library() -> Any:
    global _WAITID_LIBRARY
    if _WAITID_LIBRARY is None:
        if (
            sys.platform != "darwin"
            or ctypes.sizeof(_DarwinSiginfo) != 104
            or _DarwinSiginfo.si_pid.offset != 12
            or _DarwinSiginfo.si_status.offset != 20
        ):
            raise RuntimeError("Canary waitid requires the Darwin arm64 ABI")
        library = ctypes.CDLL(None, use_errno=True)
        library.waitid.argtypes = [
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.POINTER(_DarwinSiginfo),
            ctypes.c_int,
        ]
        library.waitid.restype = ctypes.c_int
        _WAITID_LIBRARY = library
    return _WAITID_LIBRARY


def _observe_terminal_nonconsuming(
    pid: int,
) -> Optional[_TerminalObservation]:
    if pid <= 0:
        raise RuntimeError("Canary child PID must be positive")
    information = _DarwinSiginfo()
    ctypes.set_errno(0)
    result = _waitid_library().waitid(
        P_PID,
        ctypes.c_uint(pid),
        ctypes.byref(information),
        WNOHANG | WNOWAIT | WEXITED,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.ECHILD:
            raise ChildProcessError(error_number, os.strerror(error_number))
        if error_number == errno.EINTR:
            raise InterruptedError(error_number, os.strerror(error_number))
        raise OSError(error_number, os.strerror(error_number))
    if information.si_pid == 0:
        return None
    if (
        information.si_signo != signal.SIGCHLD
        or information.si_pid != pid
        or information.si_code not in {CLD_EXITED, CLD_KILLED, CLD_DUMPED}
    ):
        raise RuntimeError("waitid returned malformed canary child information")
    return _TerminalObservation(
        int(information.si_pid),
        int(information.si_code),
        int(information.si_status),
    )


def _libproc_library() -> Any:
    global _LIBPROC_LIBRARY
    if _LIBPROC_LIBRARY is None:
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        library.proc_listpgrppids.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        library.proc_listpgrppids.restype = ctypes.c_int
        _LIBPROC_LIBRARY = library
    return _LIBPROC_LIBRARY


def _group_member_pids(process_group: int) -> List[int]:
    if process_group <= 0:
        raise RuntimeError("Canary process group must be positive")
    function = _libproc_library().proc_listpgrppids
    ctypes.set_errno(0)
    capacity_hint = function(process_group, None, 0)
    if capacity_hint < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    capacity = max(64, capacity_hint + 32)
    if capacity > PROCESS_GROUP_PID_CAPACITY:
        raise RuntimeError("Canary process group exceeds the PID capacity")
    values = (ctypes.c_int * capacity)()
    ctypes.set_errno(0)
    count = function(
        process_group,
        values,
        ctypes.sizeof(values),
    )
    if count < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if count > capacity:
        raise RuntimeError("Canary process group exceeds the PID capacity")
    pids = [int(values[index]) for index in range(count)]
    if (
        any(pid <= 0 for pid in pids)
        or len(pids) != len(set(pids))
    ):
        raise RuntimeError("Canary process group returned malformed PIDs")
    return sorted(pids)


def _kill_group(process_group: int) -> None:
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return


def _wait_for_terminal(
    child: _BoundedChild,
    *,
    deadline: float,
) -> bool:
    while child.terminal is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        observation = _observe_terminal_nonconsuming(child.process.pid)
        if observation is not None:
            child.terminal = observation
            return True
        time.sleep(min(GROUP_POLL_SECONDS, remaining))
    return True


def _wait_for_quiescent_group(
    child: _BoundedChild,
    *,
    deadline: float,
) -> bool:
    if not child.group_verified or child.terminal is None or child.reap_started:
        raise RuntimeError("Canary group cannot be inspected safely")
    while True:
        members = _group_member_pids(child.process.pid)
        if members == [child.process.pid]:
            child.pre_reap_member_pids = members
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(GROUP_POLL_SECONDS, remaining))


def _reap_terminal_child(child: _BoundedChild) -> None:
    if child.terminal is None or child.pre_reap_member_pids is None:
        raise RuntimeError("Canary child cannot be reaped before group quiescence")
    child.reap_started = True
    try:
        observed_pid, wait_status = os.waitpid(child.process.pid, 0)
    except ChildProcessError as error:
        child.reaped = True
        raise RuntimeError(
            "Canary child was already reaped; wait status is unavailable"
        ) from error
    if observed_pid != child.process.pid:
        raise RuntimeError("waitpid returned a different canary child")
    observation = child.terminal
    if observation.code == CLD_EXITED:
        if (
            not os.WIFEXITED(wait_status)
            or os.WEXITSTATUS(wait_status) != observation.status
        ):
            raise RuntimeError("Canary waitpid exit status differs from waitid")
        returncode = observation.status
    else:
        if (
            observation.code not in {CLD_KILLED, CLD_DUMPED}
            or not os.WIFSIGNALED(wait_status)
            or os.WTERMSIG(wait_status) != observation.status
        ):
            raise RuntimeError("Canary waitpid signal status differs from waitid")
        returncode = -observation.status
    child.returncode = returncode
    child.process.returncode = returncode
    child.reaped = True


def _force_settle_child(child: _BoundedChild) -> List[str]:
    errors: List[str] = []
    if child.reaped:
        return errors
    deadline = time.monotonic() + POST_KILL_WAIT_SECONDS
    if child.reap_started:
        try:
            _reap_terminal_child(child)
        except BaseException as error:
            errors.append("retry reap: %s: %s" % (type(error).__name__, error))
        return errors
    try:
        if child.group_verified:
            _kill_group(child.process.pid)
        else:
            child.process.kill()
    except BaseException as error:
        errors.append("kill child: %s: %s" % (type(error).__name__, error))
    try:
        if not _wait_for_terminal(child, deadline=deadline):
            errors.append("terminal observation timed out")
            return errors
    except BaseException as error:
        errors.append(
            "observe terminal child: %s: %s" % (type(error).__name__, error)
        )
        return errors
    if child.group_verified:
        try:
            if not _wait_for_quiescent_group(child, deadline=deadline):
                errors.append("process group did not become quiescent")
                return errors
        except BaseException as error:
            errors.append(
                "inspect process group: %s: %s" % (type(error).__name__, error)
            )
            return errors
    else:
        child.pre_reap_member_pids = []
    try:
        _reap_terminal_child(child)
    except BaseException as error:
        errors.append("reap child: %s: %s" % (type(error).__name__, error))
    return errors


def _bounded_popen(
    argv: Sequence[str],
    *,
    cwd: str,
    timeout_seconds: float,
) -> Tuple[int, bytes, bytes, bool, bool, List[int]]:
    with (
        tempfile.TemporaryFile() as stdout_file,
        tempfile.TemporaryFile() as stderr_file,
    ):
        process = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env={
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            },
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            close_fds=True,
            start_new_session=True,
        )
        child = _BoundedChild(process=process)
        timed_out = False
        forced_cleanup = False
        try:
            if os.getpgid(process.pid) != process.pid:
                raise RuntimeError("Bounded child is not its process-group leader")
            child.group_verified = True
            if not _wait_for_terminal(
                child,
                deadline=time.monotonic() + timeout_seconds,
            ):
                timed_out = True
                forced_cleanup = True
                _kill_group(process.pid)
                if not _wait_for_terminal(
                    child,
                    deadline=time.monotonic() + POST_KILL_WAIT_SECONDS,
                ):
                    raise RuntimeError("Timed-out child did not terminate")
            if not _wait_for_quiescent_group(
                child,
                deadline=time.monotonic() + POST_KILL_WAIT_SECONDS,
            ):
                forced_cleanup = True
                _kill_group(process.pid)
                if not _wait_for_quiescent_group(
                    child,
                    deadline=time.monotonic() + POST_KILL_WAIT_SECONDS,
                ):
                    raise RuntimeError(
                        "Bounded child process group did not become quiescent"
                    )
            _reap_terminal_child(child)
        except BaseException as primary_error:
            cleanup_errors = _force_settle_child(child)
            if cleanup_errors:
                raise RuntimeError(
                    "Bounded child failed and cleanup also failed: %s"
                    % "; ".join(cleanup_errors)
                ) from primary_error
            raise
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
        if type(stdout) is not bytes or type(stderr) is not bytes:
            raise RuntimeError("Bounded child output is not binary")
        if child.returncode is None or child.pre_reap_member_pids is None:
            raise RuntimeError("Bounded child completion evidence is incomplete")
        return (
            child.returncode,
            stdout,
            stderr,
            timed_out,
            forced_cleanup,
            child.pre_reap_member_pids,
        )


def execute_verified_canary(
    material: CanaryMaterial,
    *,
    timeout_seconds: float = CANARY_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    (
        returncode,
        stdout,
        stderr,
        timed_out,
        forced_cleanup,
        pre_reap_member_pids,
    ) = _bounded_popen(
        (str(material.executable),),
        cwd=str(material.directory),
        timeout_seconds=timeout_seconds,
    )
    return {
        "new_process_group": True,
        "new_session": True,
        "process_group_leader_pid": pre_reap_member_pids[0],
        "process_group_member_pids_before_reap": pre_reap_member_pids,
        "process_group_quiescent_before_reap": True,
        "process_group_required_forced_cleanup": forced_cleanup,
        "reaped": True,
        "returncode": returncode,
        "stderr_bytes": len(stderr),
        "stderr_sha256": _sha256(stderr),
        "stdout_bytes": len(stdout),
        "stdout_sha256": _sha256(stdout),
        "timed_out": timed_out,
        "timeout_milliseconds": int(timeout_seconds * 1000),
    }


class RealSystem:
    """The fixed read-only host interface used by the command-line gate."""

    def system_name(self) -> str:
        return platform.system()

    def machine(self) -> str:
        return platform.machine()

    def run_command(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
    ) -> CommandResult:
        (
            returncode,
            stdout,
            stderr,
            timed_out,
            forced_cleanup,
            _,
        ) = _bounded_popen(argv, cwd="/", timeout_seconds=timeout_seconds)
        if timed_out:
            raise subprocess.TimeoutExpired(list(argv), timeout_seconds)
        if forced_cleanup:
            raise RuntimeError("Host command left a surviving process-group member")
        return CommandResult(tuple(argv), returncode, stdout, stderr)

    def resolve_directory(self, path: str) -> str:
        candidate = Path(path)
        resolved = candidate.resolve(strict=True)
        if not resolved.is_dir():
            raise RuntimeError("Path is not a directory: %s" % path)
        return str(resolved)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def create_verified_canary(self) -> CanaryMaterial:
        return create_verified_canary()

    def execute_verified_canary(
        self,
        material: CanaryMaterial,
        *,
        timeout_seconds: float,
    ) -> Dict[str, Any]:
        return execute_verified_canary(
            material,
            timeout_seconds=timeout_seconds,
        )

    def cleanup_verified_canary(
        self,
        material: CanaryMaterial,
    ) -> Dict[str, Any]:
        return cleanup_verified_canary(material)


def _fail(check: str, message: str, checks: Mapping[str, Any]) -> None:
    raise GateFailure(check, message, checks)


def _decode_text(
    data: bytes,
    *,
    check: str,
    label: str,
    checks: Mapping[str, Any],
) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        _fail(check, "%s is not UTF-8: %s" % (label, error), checks)
    if "\x00" in text or "\r" in text:
        _fail(check, "%s contains forbidden control bytes" % label, checks)
    return text


def _command_evidence(result: CommandResult) -> Dict[str, Any]:
    return {
        "argv": list(result.argv),
        "returncode": result.returncode,
        "stderr_bytes": len(result.stderr),
        "stderr_sha256": _sha256(result.stderr),
        "stdout_bytes": len(result.stdout),
        "stdout_sha256": _sha256(result.stdout),
    }


def _required_command(
    operations: Any,
    argv: Sequence[str],
    *,
    timeout_seconds: float,
    check: str,
    checks: Mapping[str, Any],
) -> CommandResult:
    try:
        result = operations.run_command(
            argv,
            timeout_seconds=timeout_seconds,
        )
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            raise
        _fail(
            check,
            "Command failed before a valid result: %s: %s"
            % (type(error).__name__, error),
            checks,
        )
    if (
        not isinstance(result, CommandResult)
        or result.argv != tuple(argv)
        or type(result.returncode) is not int
        or type(result.stdout) is not bytes
        or type(result.stderr) is not bytes
    ):
        _fail(check, "Command backend returned a malformed result", checks)
    if result.returncode != 0:
        _fail(
            check,
            "Command exited nonzero: %s" % result.returncode,
            checks,
        )
    return result


def _one_output_line(
    result: CommandResult,
    *,
    check: str,
    label: str,
    checks: Mapping[str, Any],
) -> str:
    text = _decode_text(
        result.stdout,
        check=check,
        label=label,
        checks=checks,
    )
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0]:
        _fail(check, "%s is not exactly one nonempty line" % label, checks)
    return lines[0]


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _canonical_decimal(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return "0" if text in {"-0", ""} else text


def _discover_syspolicyd(
    operations: Any,
    checks: Mapping[str, Any],
) -> Tuple[int, CommandResult]:
    result = _required_command(
        operations,
        SYSPOLICY_DISCOVERY_COMMAND,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
        check="syspolicyd_discovery",
        checks=checks,
    )
    text = _decode_text(
        result.stdout,
        check="syspolicyd_discovery",
        label="syspolicyd process listing",
        checks=checks,
    )
    matches: List[int] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parsed = _DISCOVERY_LINE_PATTERN.fullmatch(line)
        if parsed is None:
            _fail(
                "syspolicyd_discovery",
                "Process listing contains a malformed line",
                checks,
            )
        pid = int(parsed.group(1))
        command = parsed.group(2)
        if _basename(command) == "syspolicyd":
            matches.append(pid)
    if len(matches) != 1 or matches[0] <= 1:
        _fail(
            "syspolicyd_discovery",
            "Expected exactly one syspolicyd process, found %d" % len(matches),
            checks,
        )
    return matches[0], result


def _sample_syspolicyd_cpu(
    operations: Any,
    pid: int,
    checks: Mapping[str, Any],
) -> Dict[str, Any]:
    samples: List[Decimal] = []
    started = operations.monotonic()
    for sample_index in range(CPU_SAMPLE_COUNT):
        command = (
            "/bin/ps",
            "-p",
            str(pid),
            "-o",
            "pid=,%cpu=,comm=",
        )
        result = _required_command(
            operations,
            command,
            timeout_seconds=CPU_SAMPLE_COMMAND_TIMEOUT_SECONDS,
            check="syspolicyd_cpu",
            checks=checks,
        )
        line = _one_output_line(
            result,
            check="syspolicyd_cpu",
            label="syspolicyd CPU sample",
            checks=checks,
        )
        parsed = _CPU_LINE_PATTERN.fullmatch(line)
        if parsed is None:
            _fail("syspolicyd_cpu", "Malformed syspolicyd CPU sample", checks)
        observed_pid = int(parsed.group(1))
        command_name = parsed.group(3)
        if observed_pid != pid or _basename(command_name) != "syspolicyd":
            _fail("syspolicyd_cpu", "syspolicyd identity changed", checks)
        try:
            cpu = Decimal(parsed.group(2))
        except InvalidOperation:
            _fail("syspolicyd_cpu", "Invalid syspolicyd CPU value", checks)
        if not cpu.is_finite() or cpu < 0 or cpu > Decimal("1200"):
            _fail("syspolicyd_cpu", "Out-of-range syspolicyd CPU value", checks)
        samples.append(cpu)
        if sample_index + 1 != CPU_SAMPLE_COUNT:
            operations.sleep(CPU_SAMPLE_INTERVAL_SECONDS)
    ended = operations.monotonic()
    if ended < started or ended - started > CPU_SAMPLE_TOTAL_LIMIT_SECONDS:
        _fail(
            "syspolicyd_cpu",
            "syspolicyd CPU sampling exceeded its bounded interval",
            checks,
        )
    persistent = all(value >= CPU_REJECTION_THRESHOLD for value in samples)
    record = {
        "pid": pid,
        "persistent_at_or_above_threshold": persistent,
        "sample_count": CPU_SAMPLE_COUNT,
        "sample_interval_milliseconds": int(
            CPU_SAMPLE_INTERVAL_SECONDS * 1000
        ),
        "samples_percent": [_canonical_decimal(value) for value in samples],
        "threshold_percent": _canonical_decimal(CPU_REJECTION_THRESHOLD),
    }
    if persistent:
        _fail(
            "syspolicyd_cpu",
            "syspolicyd CPU was at least 90% in every bounded sample",
            dict(checks, syspolicyd_cpu=record),
        )
    return record


def run_gate(operations: Optional[Any] = None) -> Dict[str, Any]:
    """Run all read-only checks and return canonicalizable evidence."""

    operations = operations or RealSystem()
    checks: Dict[str, Any] = {}

    system_name = operations.system_name()
    machine = operations.machine()
    checks["platform"] = {
        "machine": machine,
        "system": system_name,
    }
    if system_name != "Darwin" or machine != "arm64":
        _fail("platform", "Gate requires native macOS arm64", checks)

    boot_result = _required_command(
        operations,
        SYSCTL_COMMAND,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
        check="boot_time",
        checks=checks,
    )
    boot_line = _one_output_line(
        boot_result,
        check="boot_time",
        label="kern.boottime",
        checks=checks,
    )
    parsed_boot = _BOOT_TIME_PATTERN.fullmatch(boot_line)
    if parsed_boot is None:
        _fail("boot_time", "kern.boottime has an unrecognized format", checks)
    boot_seconds = int(parsed_boot.group(1))
    boot_microseconds = int(parsed_boot.group(2))
    if boot_microseconds >= 1_000_000:
        _fail("boot_time", "kern.boottime microseconds are invalid", checks)
    strictly_after = (boot_seconds, boot_microseconds) > (
        BOOT_CUTOFF_SECONDS,
        BOOT_CUTOFF_MICROSECONDS,
    )
    boot_utc = datetime.datetime.fromtimestamp(
        boot_seconds + boot_microseconds / 1_000_000,
        tz=datetime.timezone.utc,
    ).isoformat(timespec="microseconds")
    checks["boot_time"] = {
        "command": _command_evidence(boot_result),
        "cutoff": BOOT_CUTOFF_TEXT,
        "cutoff_epoch_microseconds": (
            BOOT_CUTOFF_SECONDS * 1_000_000 + BOOT_CUTOFF_MICROSECONDS
        ),
        "observed_epoch_microseconds": (
            boot_seconds * 1_000_000 + boot_microseconds
        ),
        "observed_utc": boot_utc,
        "strictly_after_cutoff": strictly_after,
    }
    if not strictly_after:
        _fail(
            "boot_time",
            "Boot time is not strictly after the required cutoff",
            checks,
        )

    gatekeeper_result = _required_command(
        operations,
        GATEKEEPER_COMMAND,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
        check="gatekeeper",
        checks=checks,
    )
    gatekeeper_line = _one_output_line(
        gatekeeper_result,
        check="gatekeeper",
        label="Gatekeeper status",
        checks=checks,
    )
    checks["gatekeeper"] = {
        "command": _command_evidence(gatekeeper_result),
        "enabled": gatekeeper_line == "assessments enabled",
        "status": gatekeeper_line,
    }
    if gatekeeper_line != "assessments enabled":
        _fail("gatekeeper", "Gatekeeper is not enabled", checks)

    sip_result = _required_command(
        operations,
        SIP_COMMAND,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
        check="sip",
        checks=checks,
    )
    sip_line = _one_output_line(
        sip_result,
        check="sip",
        label="SIP status",
        checks=checks,
    )
    expected_sip = "System Integrity Protection status: enabled."
    checks["sip"] = {
        "command": _command_evidence(sip_result),
        "enabled": sip_line == expected_sip,
        "status": sip_line,
    }
    if sip_line != expected_sip:
        _fail("sip", "System Integrity Protection is not fully enabled", checks)

    select_result = _required_command(
        operations,
        XCODE_SELECT_COMMAND,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
        check="xcode_select",
        checks=checks,
    )
    selected_path = _one_output_line(
        select_result,
        check="xcode_select",
        label="xcode-select path",
        checks=checks,
    )
    if not selected_path.startswith("/") or selected_path.endswith("/"):
        _fail("xcode_select", "xcode-select returned a noncanonical path", checks)
    try:
        resolved_path = operations.resolve_directory(selected_path)
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            raise
        _fail(
            "xcode_select",
            "Selected Xcode path does not resolve to a directory: %s" % error,
            checks,
        )
    checks["xcode_select"] = {
        "command": _command_evidence(select_result),
        "exists": True,
        "path": selected_path,
        "resolved_path": resolved_path,
    }

    first_launch = _required_command(
        operations,
        XCODE_FIRST_LAUNCH_COMMAND,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
        check="xcode_first_launch",
        checks=checks,
    )
    checks["xcode_first_launch"] = {
        "command": _command_evidence(first_launch),
        "passed": True,
    }

    license_check = _required_command(
        operations,
        XCODE_LICENSE_COMMAND,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
        check="xcode_license",
        checks=checks,
    )
    checks["xcode_license"] = {
        "command": _command_evidence(license_check),
        "passed": True,
    }

    syspolicyd_pid, discovery_result = _discover_syspolicyd(operations, checks)
    checks["syspolicyd_discovery"] = {
        "command": _command_evidence(discovery_result),
        "pid": syspolicyd_pid,
        "unique": True,
    }
    checks["syspolicyd_cpu"] = _sample_syspolicyd_cpu(
        operations,
        syspolicyd_pid,
        checks,
    )

    log_result = _required_command(
        operations,
        SYSPOLICY_LOG_COMMAND,
        timeout_seconds=LOG_TIMEOUT_SECONDS,
        check="syspolicyd_logs",
        checks=checks,
    )
    combined_logs = log_result.stdout + b"\n" + log_result.stderr
    try:
        combined_text = combined_logs.decode("utf-8")
    except UnicodeDecodeError as error:
        _fail("syspolicyd_logs", "Policy log output is not UTF-8: %s" % error, checks)
    match_counts = {
        message: combined_text.count(message)
        for message in FORBIDDEN_LOG_MESSAGES
    }
    checks["syspolicyd_logs"] = {
        "command": _command_evidence(log_result),
        "forbidden_match_counts": match_counts,
        "fresh_forbidden_message_found": any(match_counts.values()),
        "window": LOG_WINDOW,
    }
    if any(match_counts.values()):
        _fail(
            "syspolicyd_logs",
            "Fresh syspolicyd execution-policy errors were found",
            checks,
        )

    material: Optional[CanaryMaterial] = None
    primary_failure: Optional[GateFailure] = None
    try:
        try:
            material = operations.create_verified_canary()
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt):
                raise
            _fail(
                "canary_copy",
                "Could not create a byte-verified private canary: %s: %s"
                % (type(error).__name__, error),
                checks,
            )
        if (
            not isinstance(material, CanaryMaterial)
            or material.evidence.get("source_path") != str(TRUE_PATH)
            or material.evidence.get("verified_byte_equal") is not True
            or material.evidence.get("source_sha256")
            != material.evidence.get("copy_sha256")
        ):
            _fail("canary_copy", "Canary copy evidence is malformed", checks)
        checks["canary_copy"] = copy.deepcopy(material.evidence)

        try:
            execution = operations.execute_verified_canary(
                material,
                timeout_seconds=CANARY_TIMEOUT_SECONDS,
            )
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt):
                raise
            _fail(
                "canary_execution",
                "Canary execution failed before valid evidence: %s: %s"
                % (type(error).__name__, error),
                checks,
            )
        required_execution = {
            "new_process_group",
            "new_session",
            "process_group_leader_pid",
            "process_group_member_pids_before_reap",
            "process_group_quiescent_before_reap",
            "process_group_required_forced_cleanup",
            "reaped",
            "returncode",
            "stderr_bytes",
            "stderr_sha256",
            "stdout_bytes",
            "stdout_sha256",
            "timed_out",
            "timeout_milliseconds",
        }
        if type(execution) is not dict or set(execution) != required_execution:
            _fail("canary_execution", "Canary execution evidence is malformed", checks)
        checks["canary_execution"] = copy.deepcopy(execution)
        if (
            execution["timeout_milliseconds"] != 5000
            or execution["timed_out"] is not False
            or execution["reaped"] is not True
            or execution["new_session"] is not True
            or execution["new_process_group"] is not True
            or execution["returncode"] != 0
            or execution["stdout_bytes"] != 0
            or execution["stderr_bytes"] != 0
        ):
            _fail(
                "canary_execution",
                "Copied /usr/bin/true did not complete cleanly within five seconds",
                checks,
            )
        if (
            type(execution["process_group_leader_pid"]) is not int
            or execution["process_group_leader_pid"] <= 0
            or execution["process_group_member_pids_before_reap"]
            != [execution["process_group_leader_pid"]]
            or execution["process_group_quiescent_before_reap"] is not True
            or execution["process_group_required_forced_cleanup"] is not False
        ):
            _fail(
                "canary_process_group",
                "Canary process group was not leader-only before reap",
                checks,
            )
    except GateFailure as error:
        primary_failure = error
    finally:
        if material is not None:
            try:
                cleanup = operations.cleanup_verified_canary(material)
                if cleanup != {"private_directory_removed": True}:
                    raise RuntimeError("Canary cleanup evidence is malformed")
                checks["canary_cleanup"] = cleanup
            except BaseException as error:
                if isinstance(error, KeyboardInterrupt):
                    raise
                message = "Private canary cleanup failed: %s: %s" % (
                    type(error).__name__,
                    error,
                )
                if primary_failure is not None:
                    message += "; earlier failure: %s" % primary_failure
                raise GateFailure("canary_cleanup", message, checks) from error
    if primary_failure is not None:
        primary_failure.checks = copy.deepcopy(checks)
        raise primary_failure

    return {
        "checks": checks,
        "gate_id": GATE_ID,
        "read_only": True,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "write_scope": {
            "policy_or_service_changes": False,
            "prohibitions": list(READ_ONLY_PROHIBITIONS),
            "transient_private_canary_only": True,
        },
    }


def _failure_record(error: GateFailure) -> Dict[str, Any]:
    return {
        "checks": error.checks,
        "failure": {
            "check": error.check,
            "exit_code": error.exit_code,
            "message": str(error),
        },
        "gate_id": GATE_ID,
        "read_only": True,
        "schema_version": SCHEMA_VERSION,
        "status": "FAIL",
    }


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    operations: Optional[Any] = None,
    stdout: Optional[BinaryIO] = None,
    stderr: Optional[BinaryIO] = None,
) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    output = sys.stdout.buffer if stdout is None else stdout
    error_output = sys.stderr.buffer if stderr is None else stderr
    if arguments:
        failure = GateFailure(
            "usage",
            "This gate accepts no command-line arguments",
        )
        error_output.write(canonical_json_bytes(_failure_record(failure)))
        return failure.exit_code
    try:
        evidence = run_gate(operations)
    except KeyboardInterrupt:
        failure = GateFailure("interrupted", "Gate interrupted before completion")
        error_output.write(canonical_json_bytes(_failure_record(failure)))
        return failure.exit_code
    except GateFailure as error:
        error_output.write(canonical_json_bytes(_failure_record(error)))
        return error.exit_code
    except BaseException as error:
        failure = GateFailure(
            "internal",
            "Unexpected gate failure: %s: %s" % (type(error).__name__, error),
        )
        error_output.write(canonical_json_bytes(_failure_record(failure)))
        return failure.exit_code
    output.write(canonical_json_bytes(evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
