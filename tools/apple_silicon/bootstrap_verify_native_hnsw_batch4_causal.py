#!/usr/bin/env python3
"""Authenticate and execute a captured Batch4 verifier pair.

Invoke this trusted file with ``python3 -I -S -B``.
"""

from __future__ import annotations

import sys

if __name__ == "__main__" and not (
    sys.flags.isolated
    and sys.flags.no_site
    and sys.flags.dont_write_bytecode
):
    sys.stderr.write("bootstrap verification failed: requires python3 -I -S -B\n")
    raise SystemExit(2)

import argparse
import hashlib
import os
import re
import stat
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


VERIFIER_FILENAME = "verifier-native_hnsw_batch4_causal.py"
HELPER_FILENAME = "module-verify_native_hnsw_d0.py"
EXACTNESS_HELPER_FILENAME = "module-verify_native_hnsw_exactness.py"
HELPER_GLOBAL = "_AUTHENTICATED_NATIVE_HNSW_D0_HELPER"
EXACTNESS_HELPER_GLOBAL = "_AUTHENTICATED_NATIVE_HNSW_EXACTNESS_HELPER"
CONTEXT_GLOBAL = "_AUTHENTICATED_NATIVE_HNSW_BATCH4_CONTEXT"
HELPER_MODULE_NAME = "_authenticated_verify_native_hnsw_d0"
EXACTNESS_HELPER_MODULE_NAME = "_authenticated_verify_native_hnsw_exactness"
MAX_SOURCE_BYTES = 8 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}")
SINGLE_USE_OPTIONS = (
    "--verifier-sha256",
    "--verifier-bytes",
    "--helper-sha256",
    "--helper-bytes",
    "--exactness-helper-sha256",
    "--exactness-helper-bytes",
    "--preflight-only",
)


class BootstrapError(RuntimeError):
    """Raised when captured verifier authentication cannot be trusted."""


@dataclass(frozen=True)
class TrustedSource:
    filename: str
    sha256: str
    size: int


def _bounded_size(value: str) -> int:
    try:
        size = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("size must be a decimal integer") from error
    if not 1 <= size <= MAX_SOURCE_BYTES:
        raise argparse.ArgumentTypeError(
            f"size must be between 1 and {MAX_SOURCE_BYTES}"
        )
    return size


def _sha256(value: str) -> str:
    normalized = value.lower()
    if SHA256.fullmatch(normalized) is None:
        raise argparse.ArgumentTypeError("SHA-256 must be 64 hexadecimal characters")
    return normalized


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_stable(
    expected: os.stat_result,
    actual: os.stat_result,
    *,
    label: str,
) -> None:
    if _identity(expected) != _identity(actual):
        raise BootstrapError(f"{label} changed during authenticated read")


def _open_evidence_directory(path: Path) -> tuple[int, os.stat_result]:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise BootstrapError(f"Cannot lstat evidence directory: {error}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise BootstrapError("Evidence directory must be a non-symlink directory")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BootstrapError(f"Cannot open evidence directory: {error}") from error
    try:
        _require_stable(
            before,
            os.fstat(descriptor),
            label="Evidence directory",
        )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, before


def _read_source(directory_descriptor: int, trusted: TrustedSource) -> bytes:
    try:
        before = os.lstat(trusted.filename, dir_fd=directory_descriptor)
    except OSError as error:
        raise BootstrapError(f"Cannot lstat {trusted.filename}: {error}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise BootstrapError(f"{trusted.filename} must be a non-symlink regular file")
    if before.st_nlink != 1:
        raise BootstrapError(f"{trusted.filename} must not be hard-linked")
    if before.st_size != trusted.size:
        raise BootstrapError(f"{trusted.filename} size mismatch")
    if before.st_size > MAX_SOURCE_BYTES:
        raise BootstrapError(f"{trusted.filename} exceeds the source size limit")

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(
            trusted.filename,
            flags,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise BootstrapError(f"Cannot open {trusted.filename}: {error}") from error

    try:
        opened = os.fstat(descriptor)
        _require_stable(before, opened, label=trusted.filename)
        chunks: list[bytes] = []
        remaining = trusted.size + 1
        while remaining:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        _require_stable(opened, os.fstat(descriptor), label=trusted.filename)
    except OSError as error:
        raise BootstrapError(f"Cannot read {trusted.filename}: {error}") from error
    finally:
        os.close(descriptor)

    if len(data) != trusted.size:
        raise BootstrapError(f"{trusted.filename} size changed during read")
    if hashlib.sha256(data).hexdigest() != trusted.sha256:
        raise BootstrapError(f"{trusted.filename} SHA-256 mismatch")
    return data


def _authenticate_pair(
    evidence_directory: Path,
    verifier: TrustedSource,
    helper: TrustedSource,
    exactness_helper: TrustedSource,
) -> tuple[int, os.stat_result, bytes, bytes, bytes]:
    directory_descriptor, directory_before = _open_evidence_directory(
        evidence_directory
    )
    try:
        verifier_bytes = _read_source(directory_descriptor, verifier)
        helper_bytes = _read_source(directory_descriptor, helper)
        exactness_helper_bytes = _read_source(
            directory_descriptor,
            exactness_helper,
        )
        _require_stable(
            directory_before,
            os.fstat(directory_descriptor),
            label="Evidence directory",
        )
    except BaseException as error:
        os.close(directory_descriptor)
        if isinstance(error, OSError):
            raise BootstrapError(
                f"Cannot verify evidence directory: {error}"
            ) from error
        raise
    return (
        directory_descriptor,
        directory_before,
        verifier_bytes,
        helper_bytes,
        exactness_helper_bytes,
    )


def _load_helper(
    source: bytes,
    filename: str,
    module_name: str = HELPER_MODULE_NAME,
) -> types.ModuleType:
    module = types.ModuleType(module_name)
    module.__file__ = filename
    module.__package__ = None
    module.__loader__ = None
    module.__spec__ = None
    sys.modules[module_name] = module
    try:
        code = compile(source, filename, "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _execute_verifier(
    source: bytes,
    filename: str,
    helper: types.ModuleType,
    exactness_helper: types.ModuleType,
    context: dict[str, Any],
) -> None:
    globals_value = {
        "__name__": "__main__",
        "__file__": filename,
        "__package__": None,
        "__loader__": None,
        "__spec__": None,
        "__cached__": None,
        HELPER_GLOBAL: helper,
        EXACTNESS_HELPER_GLOBAL: exactness_helper,
        CONTEXT_GLOBAL: types.MappingProxyType(context),
    }
    sys.argv = [filename, context["evidence_directory"]]
    if context["preflight_only"]:
        sys.argv.append("--preflight-only")
    code = compile(source, filename, "exec", dont_inherit=True)
    exec(code, globals_value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("evidence_directory", type=Path)
    parser.add_argument("--verifier-sha256", required=True, type=_sha256)
    parser.add_argument("--verifier-bytes", required=True, type=_bounded_size)
    parser.add_argument("--helper-sha256", required=True, type=_sha256)
    parser.add_argument("--helper-bytes", required=True, type=_bounded_size)
    parser.add_argument(
        "--exactness-helper-sha256",
        required=True,
        type=_sha256,
    )
    parser.add_argument(
        "--exactness-helper-bytes",
        required=True,
        type=_bounded_size,
    )
    parser.add_argument("--preflight-only", action="store_true")
    arguments = list(sys.argv[1:] if argv is None else argv)
    for option in SINGLE_USE_OPTIONS:
        occurrences = sum(
            argument == option or argument.startswith(f"{option}=")
            for argument in arguments
        )
        if occurrences > 1:
            parser.error(f"{option} must not be repeated")
    return parser.parse_args(arguments)


def _require_isolated_runtime() -> None:
    if not (
        sys.flags.isolated
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
    ):
        raise BootstrapError("Bootstrap requires python3 -I -S -B")


def main(argv: Sequence[str] | None = None) -> int:
    directory_descriptor: int | None = None
    try:
        _require_isolated_runtime()
        args = parse_args(argv)
        verifier = TrustedSource(
            VERIFIER_FILENAME,
            args.verifier_sha256,
            args.verifier_bytes,
        )
        helper = TrustedSource(
            HELPER_FILENAME,
            args.helper_sha256,
            args.helper_bytes,
        )
        exactness_helper = TrustedSource(
            EXACTNESS_HELPER_FILENAME,
            args.exactness_helper_sha256,
            args.exactness_helper_bytes,
        )
        (
            directory_descriptor,
            directory_identity,
            verifier_bytes,
            helper_bytes,
            exactness_helper_bytes,
        ) = _authenticate_pair(
            args.evidence_directory,
            verifier,
            helper,
            exactness_helper,
        )
        evidence_directory = args.evidence_directory.resolve(strict=True)
        _require_stable(
            directory_identity,
            os.lstat(evidence_directory),
            label="Evidence directory path",
        )
        helper_module = _load_helper(
            helper_bytes,
            str(evidence_directory / HELPER_FILENAME),
        )
        exactness_helper_module = _load_helper(
            exactness_helper_bytes,
            str(evidence_directory / EXACTNESS_HELPER_FILENAME),
            EXACTNESS_HELPER_MODULE_NAME,
        )
        context = {
            "evidence_directory": str(evidence_directory),
            "directory_descriptor": directory_descriptor,
            "directory_identity": _identity(directory_identity),
            "preflight_only": args.preflight_only,
            "verifier": {
                "filename": verifier.filename,
                "sha256": verifier.sha256,
                "bytes": verifier.size,
            },
            "helper": {
                "filename": helper.filename,
                "sha256": helper.sha256,
                "bytes": helper.size,
            },
            "exactness_helper": {
                "filename": exactness_helper.filename,
                "sha256": exactness_helper.sha256,
                "bytes": exactness_helper.size,
            },
        }
        try:
            _execute_verifier(
                verifier_bytes,
                str(evidence_directory / VERIFIER_FILENAME),
                helper_module,
                exactness_helper_module,
                context,
            )
        finally:
            _require_stable(
                directory_identity,
                os.fstat(directory_descriptor),
                label="Evidence directory descriptor",
            )
            _require_stable(
                directory_identity,
                os.lstat(evidence_directory),
                label="Evidence directory path",
            )
    except (BootstrapError, OSError) as error:
        print(f"bootstrap verification failed: {error}", file=sys.stderr)
        return 2
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
