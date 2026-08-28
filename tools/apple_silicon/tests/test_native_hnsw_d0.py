from __future__ import annotations

import contextlib
import copy
import ctypes
import hashlib
import json
import os
import random
import re
import shlex
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from tools.apple_silicon import native_hnsw_d0 as d0


class DatasetTests(unittest.TestCase):
    def test_fnv1a64_known_vectors(self) -> None:
        self.assertEqual(d0.fnv1a64(b""), 0xCBF29CE484222325)
        self.assertEqual(d0.fnv1a64(b"hello"), 0xA430D84680AABD0B)

    def test_dataset_is_deterministic_little_endian_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.bin"
            record = d0.create_dataset(path, vectors=2, dimensions=3)

            generator = random.Random(0)
            expected = b"".join(struct.pack("<f", generator.random()) for _ in range(6))
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(record["bytes"], len(expected))
            self.assertEqual(record["sha256"], hashlib.sha256(expected).hexdigest())
            self.assertEqual(record["fnv1a64"], d0.fnv1a64(expected))
            self.assertEqual(stat.S_IMODE(path.stat().st_mode) & 0o222, 0)
            self.assertEqual(
                d0.verify_dataset(path, record)["sha256"],
                record["sha256"],
            )

    def test_dataset_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.bin"
            record = d0.create_dataset(path, vectors=1, dimensions=1)
            path.chmod(0o644)
            path.write_bytes(b"\0\0\0\0")
            path.chmod(0o444)
            with self.assertRaisesRegex(d0.D0Failure, "Dataset sha256 changed"):
                d0.verify_dataset(path, record)


class AuditSidecarReadTests(unittest.TestCase):
    def test_regular_sidecar_is_read_once_by_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.bin"
            path.write_bytes(b"sidecar")

            self.assertEqual(d0.read_audit_sidecar(path), b"sidecar")

    def test_symlink_and_open_race_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.bin"
            target.write_bytes(b"sidecar")
            symlink = root / "symlink.bin"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(d0.D0Failure, "must not be a symlink"):
                d0.read_audit_sidecar(symlink)

            path = root / "race.bin"
            replacement = root / "replacement.bin"
            path.write_bytes(b"original")
            replacement.write_bytes(b"replacement")
            real_open = d0.os.open

            def replace_then_open(
                requested: str | bytes | d0.os.PathLike[str],
                flags: int,
            ) -> int:
                path.unlink()
                replacement.rename(path)
                return real_open(requested, flags)

            with (
                patch.object(d0.os, "open", side_effect=replace_then_open),
                self.assertRaisesRegex(d0.D0Failure, "path changed before"),
            ):
                d0.read_audit_sidecar(path)


class SourceProvenanceTests(unittest.TestCase):
    def test_schema_9_faiss_build_policy_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            definition = "-DFAISS_DISABLE_GLOBAL_HNSW_STATS=1"
            entry = {
                "directory": str(root),
                "file": str(root / "IndexHNSW.cpp"),
                "command": f"/fixture/clang++ {definition} -c IndexHNSW.cpp -o x.o",
            }
            second_entry = {
                "directory": str(root),
                "file": str(root / "IndexBinaryHNSW.cpp"),
                "command": (
                    f"/fixture/clang++ {definition} "
                    "-c IndexBinaryHNSW.cpp -o y.o"
                ),
            }
            entries = [entry, second_entry]
            cache = {"FAISS_ENABLE_GLOBAL_HNSW_STATS": "OFF"}

            d0.validate_faiss_stats_disabled_build(
                cache,
                entries,
                label="fixture",
                response_references=set(),
            )

            for invalid_cache in ({}, {"FAISS_ENABLE_GLOBAL_HNSW_STATS": "ON"}):
                with self.subTest(cache=invalid_cache):
                    with self.assertRaisesRegex(
                        d0.D0Failure,
                        "FAISS_ENABLE_GLOBAL_HNSW_STATS=OFF",
                    ):
                        d0.validate_faiss_stats_disabled_build(
                            invalid_cache,
                            entries,
                            label="fixture",
                            response_references=set(),
                        )

            for replacement in ("", f"{definition} -UFAISS_DISABLE_GLOBAL_HNSW_STATS"):
                with self.subTest(replacement=replacement):
                    invalid_entry = dict(entry)
                    invalid_entry["command"] = entry["command"].replace(
                        definition,
                        replacement,
                        1,
                    )
                    with self.assertRaisesRegex(
                        d0.D0Failure,
                        "must define exactly FAISS_DISABLE_GLOBAL_HNSW_STATS=1",
                    ):
                        d0.validate_faiss_stats_disabled_build(
                            cache,
                            [invalid_entry, second_entry],
                            label="fixture",
                            response_references=set(),
                        )

    def test_valid_single_parent_faiss_derivative_is_captured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            faiss = root / "faiss"
            benchmark = root / "benchmark"
            evidence = root / "evidence"
            faiss.mkdir()
            benchmark.mkdir()
            evidence.mkdir()

            def git(*arguments: str, text: bool = True) -> subprocess.CompletedProcess:
                return subprocess.run(
                    ["git", *arguments],
                    cwd=faiss,
                    capture_output=True,
                    text=text,
                    check=True,
                )

            git("init", "-q")
            git("config", "user.name", "Fixture")
            git("config", "user.email", "fixture@example.com")
            for path in d0.PINNED_FAISS_STATS_PATCH_PATHS:
                target = faiss / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f"base {path}\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-q", "-m", "base")
            base_head = git("rev-parse", "HEAD").stdout.strip()
            base_tree = git("rev-parse", "HEAD^{tree}").stdout.strip()
            git("tag", "v1.15.0", base_head)

            for path in d0.PINNED_FAISS_STATS_PATCH_PATHS:
                (faiss / path).write_text(
                    f"derivative {path}\n",
                    encoding="utf-8",
                )
            git("add", ".")
            git("commit", "-q", "-m", "disable global stats")
            derivative_tree = git("rev-parse", "HEAD^{tree}").stdout.strip()
            patch_bytes = git(
                "diff",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                f"{base_head}..HEAD",
                text=False,
            ).stdout
            patch_source = benchmark / d0.FAISS_STATS_PATCH_SOURCE
            patch_source.parent.mkdir(parents=True)
            patch_source.write_bytes(patch_bytes)

            repository = d0.capture_git_repository(
                faiss,
                output_dir=evidence,
                label="source-repository-00",
            )
            source_records = d0.compile_source_records(
                [
                    {
                        "directory": str(faiss),
                        "file": str(faiss / "faiss/IndexHNSW.cpp"),
                    }
                ],
                repo=benchmark,
                output_dir=evidence,
            )
            with patch.multiple(
                d0,
                PINNED_FAISS_COMMIT=base_head,
                PINNED_FAISS_BASE_TREE=base_tree,
                PINNED_FAISS_STATS_DISABLED_TREE=derivative_tree,
                PINNED_FAISS_STATS_PATCH_SHA256=hashlib.sha256(
                    patch_bytes
                ).hexdigest(),
            ):
                reference = d0.capture_faiss_audited_reference(
                    benchmark_repo=benchmark,
                    faiss_source_root=faiss,
                    faiss_repository=repository,
                    faiss_source_records=source_records,
                    output_dir=evidence,
                )

            self.assertEqual(reference["base"]["commit"], base_head)
            self.assertEqual(reference["derivative"]["tree"], derivative_tree)
            self.assertEqual(
                reference["derivative"]["changed_paths"],
                list(d0.PINNED_FAISS_STATS_PATCH_PATHS),
            )
            self.assertEqual(
                (evidence / d0.FAISS_STATS_PATCH_FILENAME).read_bytes(),
                patch_bytes,
            )

    def test_content_blob_capture_does_not_copy_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "evidence"
            source.write_bytes(b"captured bytes\n")

            with patch.object(
                d0.shutil,
                "copy2",
                side_effect=PermissionError("protected file flags"),
            ):
                record = d0.capture_content_blob(source, output_dir=output)

            captured = output / record["captured_path"]
            self.assertEqual(captured.read_bytes(), source.read_bytes())
            self.assertEqual(
                record["sha256"],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )

    def test_target_compile_entries_match_target_object_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            compiler = "/toolchain/clang++"

            def entry(source: str, output: str) -> dict[str, str]:
                return {
                    "directory": str(root),
                    "file": str(root / source),
                    "command": (
                        f"{compiler} -O3 -DNDEBUG -c {root / source} " f"-o {output}"
                    ),
                }

            entries = [
                entry("selected.cpp", "objects/selected.o"),
                entry("unrelated.cpp", "objects/unrelated.o"),
            ]
            commands = (
                f"{compiler} -O3 -DNDEBUG -c {root / 'selected.cpp'} "
                "-o objects/selected.o\n"
            )

            selected = d0.target_compile_entries(
                entries,
                commands,
                compiler=compiler,
                build_directory=root,
                label="fixture",
                response_references=set(),
            )

            self.assertEqual(selected, entries[:1])

    def test_target_compile_entries_cover_configured_c_cxx_and_asm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cache = {
                "CMAKE_C_COMPILER": "/toolchain/clang",
                "CMAKE_CXX_COMPILER": "/toolchain/clang++",
                "CMAKE_ASM_COMPILER": "/toolchain/clang-asm",
                "CMAKE_C_COMPILER_CLANG_SCAN_DEPS": "/toolchain/clang-scan-deps",
                "CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS": "/toolchain/clang-scan-deps",
                "CMAKE_ASM_COMPILER_CLANG_SCAN_DEPS": "/toolchain/clang-scan-deps",
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_OSX_ARCHITECTURES": "arm64",
                "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0",
                "CMAKE_OSX_SYSROOT": "/fixture/MacOSX.sdk",
                "CMAKE_LINKER": "/toolchain/ld",
            }
            common = [
                "-O3",
                "-DNDEBUG",
                "-arch",
                "arm64",
                "-isysroot",
                cache["CMAKE_OSX_SYSROOT"],
                "-mmacosx-version-min=14.0",
            ]

            def entry(
                compiler: str,
                source_name: str,
                output_name: str,
            ) -> dict[str, object]:
                source = root / source_name
                output = root / output_name
                return {
                    "directory": str(root),
                    "file": str(source),
                    "arguments": [
                        compiler,
                        *common,
                        "-c",
                        str(source),
                        "-o",
                        str(output),
                    ],
                    "output": str(output),
                }

            entries = [
                entry(cache["CMAKE_C_COMPILER"], "unit.c", "unit-c.o"),
                entry(cache["CMAKE_CXX_COMPILER"], "unit.cpp", "unit-cxx.o"),
                entry(cache["CMAKE_ASM_COMPILER"], "unit.S", "unit-asm.o"),
            ]
            commands = "\n".join(
                [
                    *(
                        shlex.join(
                            [
                                cache["CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS"],
                                "-format=p1689",
                                "--",
                                *item["arguments"],
                            ]
                        )
                        for item in entries
                    ),
                    *(shlex.join(item["arguments"]) for item in entries),
                    shlex.join(entries[1]["arguments"]),
                ]
            )

            selected = d0.target_compile_entries(
                [*entries, copy.deepcopy(entries[1])],
                commands,
                cache=cache,
                build_directory=root,
                label="fixture",
                response_references=set(),
            )

            self.assertEqual(len(selected), 3)
            self.assertEqual(
                {item["arguments"][0] for item in selected},
                {
                    cache["CMAKE_C_COMPILER"],
                    cache["CMAKE_CXX_COMPILER"],
                    cache["CMAKE_ASM_COMPILER"],
                },
            )
            d0.validate_release_build(
                cache,
                selected,
                label="fixture",
                response_references=set(),
            )
            shared_driver_cache = {
                **cache,
                "CMAKE_ASM_COMPILER": cache["CMAKE_C_COMPILER"],
            }
            self.assertEqual(
                d0.configured_compiler_paths(
                    shared_driver_cache,
                    label="fixture",
                ),
                (
                    cache["CMAKE_C_COMPILER"],
                    cache["CMAKE_CXX_COMPILER"],
                ),
            )
            self.assertEqual(
                d0.configured_scan_deps_paths(
                    shared_driver_cache,
                    label="fixture",
                ),
                (cache["CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS"],),
            )

    def test_target_compile_entries_reject_malformed_dependency_scan(
        self,
    ) -> None:
        build = Path("/fixture/build")
        compiler = "/toolchain/clang++"
        scanner = "/toolchain/clang-scan-deps"
        arguments = [
            compiler,
            "-c",
            "/fixture/source.cpp",
            "-o",
            str(build / "source.o"),
        ]
        entry = {
            "directory": str(build),
            "file": "/fixture/source.cpp",
            "arguments": arguments,
        }
        cache = {
            "CMAKE_CXX_COMPILER": compiler,
            "CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS": scanner,
        }
        with self.assertRaisesRegex(
            d0.D0Failure,
            "malformed configured dependency scan",
        ):
            d0.target_compile_entries(
                [entry],
                "\n".join(
                    (
                        shlex.join(
                            [
                                scanner,
                                "-format=p1689",
                                compiler,
                                *arguments[1:],
                            ]
                        ),
                        shlex.join(arguments),
                    )
                ),
                cache=cache,
                build_directory=build,
                label="fixture",
                response_references=set(),
            )

        malformed_payloads = (
            [*arguments, "-o", str(build / "second.o")],
            [*arguments, "-o"],
            [*arguments, "-c"],
        )
        for malformed in malformed_payloads:
            with (
                self.subTest(payload=malformed),
                self.assertRaisesRegex(
                    d0.D0Failure,
                    "malformed configured dependency scan",
                ),
            ):
                d0.target_compile_entries(
                    [entry],
                    "\n".join(
                        (
                            shlex.join(
                                [
                                    scanner,
                                    "-format=p1689",
                                    "--",
                                    *malformed,
                                ]
                            ),
                            shlex.join(arguments),
                        )
                    ),
                    cache=cache,
                    build_directory=build,
                    label="fixture",
                    response_references=set(),
                )

        with self.assertRaisesRegex(
            d0.D0Failure,
            "configured compiler has an unsupported wrapper",
        ):
            d0.target_compile_entries(
                [entry],
                "\n".join(
                    (
                        shlex.join(
                            [
                                "/toolchain/unconfigured-scan-deps",
                                "-format=p1689",
                                "--",
                                *arguments,
                            ]
                        ),
                        shlex.join(arguments),
                    )
                ),
                cache=cache,
                build_directory=build,
                label="fixture",
                response_references=set(),
            )

    def test_target_compile_entries_reject_conflicting_duplicate_actions(
        self,
    ) -> None:
        root = Path("/fixture/build")
        compiler = "/toolchain/clang++"
        base_arguments = [
            compiler,
            "-O3",
            "-DNDEBUG",
            "-c",
            "/fixture/source.cpp",
            "-o",
            str(root / "source.o"),
        ]
        entry = {
            "directory": str(root),
            "file": "/fixture/source.cpp",
            "arguments": base_arguments,
        }
        with self.assertRaisesRegex(
            d0.D0Failure,
            "conflicting compiler invocations",
        ):
            d0.target_compile_entries(
                [entry],
                "\n".join(
                    [
                        shlex.join(base_arguments),
                        shlex.join([*base_arguments, "-DFORGED=1"]),
                    ]
                ),
                compiler=compiler,
                build_directory=root,
                label="fixture",
                response_references=set(),
            )

        conflicting_entry = copy.deepcopy(entry)
        conflicting_entry["arguments"] = [*base_arguments, "-DFORGED=1"]
        with self.assertRaisesRegex(
            d0.D0Failure,
            "compile database has conflicting records",
        ):
            d0.target_compile_entries(
                [entry, conflicting_entry],
                shlex.join(base_arguments),
                compiler=compiler,
                build_directory=root,
                label="fixture",
                response_references=set(),
            )

    def test_target_compile_entries_reject_unconfigured_compile_action(
        self,
    ) -> None:
        build = Path("/fixture/build")
        compiler = "/toolchain/clang++"
        known_arguments = [
            compiler,
            "-c",
            "/fixture/known.cpp",
            "-o",
            str(build / "known.o"),
        ]
        unknown_arguments = [
            "/toolchain/unrecorded-cc",
            "-c",
            "/fixture/unknown.c",
            "-o",
            str(build / "unknown.o"),
        ]
        with self.assertRaisesRegex(
            d0.D0Failure,
            "does not use exactly one configured compiler",
        ):
            d0.target_compile_entries(
                [
                    {
                        "directory": str(build),
                        "file": "/fixture/known.cpp",
                        "arguments": known_arguments,
                    }
                ],
                "\n".join(
                    (shlex.join(known_arguments), shlex.join(unknown_arguments))
                ),
                compiler=compiler,
                build_directory=build,
                label="fixture",
                response_references=set(),
            )

    def test_compiled_sources_use_lexical_absolute_path_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "neon" / "aba.h"
            sibling = root / "neon.h"
            nested.parent.mkdir()
            nested.write_text("nested\n", encoding="utf-8")
            sibling.write_text("sibling\n", encoding="utf-8")
            entries = [
                {"directory": str(root), "file": str(nested)},
                {"directory": str(root), "file": str(sibling)},
            ]

            records = d0.compile_source_records(
                entries,
                repo=root,
                output_dir=root,
            )

            self.assertEqual(
                [record["absolute_path"] for record in records],
                sorted(
                    (str(nested), str(sibling)),
                    key=lambda path: path.encode("utf-8"),
                ),
            )


class ProductionBuildIdentityTests(unittest.TestCase):
    def test_vcpkg_headers_and_production_switches_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            include = repo / "installed" / "arm64-osx" / "include"
            (include / "simde").mkdir(parents=True)
            (include / "ctpl_stl.h").write_text("ctpl\n", encoding="utf-8")
            (include / "simde" / "simde-common.h").write_text(
                "simde\n",
                encoding="utf-8",
            )
            binary = repo / "infinity_hnsw_d0_production"
            binary.write_bytes(b"binary")
            cache = {
                "CMAKE_HOME_DIRECTORY": str(repo),
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
                "ENABLE_JEMALLOC": "OFF",
                "VCPKG_MANIFEST_INSTALL": "OFF",
                "VCPKG_INSTALLED_DIR": str(repo / "installed"),
                "_VCPKG_INSTALLED_DIR": str(repo / "installed"),
                "VCPKG_TARGET_TRIPLET": "arm64-osx",
                "INFINITY_BUILD_APPLE_HNSW_D0_PRODUCTION": "ON",
                "INFINITY_BUILD_TIME_OVERRIDE": "2026-08-20 00:00.00",
                "INFINITY_DISABLE_APPLE_HNSW_BATCH4_TRAVERSAL": "OFF",
                "INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING": "ON",
            }

            ctpl, simde = d0.resolve_vcpkg_benchmark_headers(cache)
            self.assertEqual(ctpl, (include / "ctpl_stl.h").resolve())
            self.assertEqual(simde, (include / "simde").resolve())
            d0.validate_infinity_production_build(
                cache,
                binary=binary,
                repo=repo,
            )

            cache["INFINITY_DISABLE_APPLE_HNSW_BATCH4_TRAVERSAL"] = "ON"
            with self.assertRaisesRegex(d0.D0Failure, "TRAVERSAL=OFF"):
                d0.validate_infinity_production_build(
                    cache,
                    binary=binary,
                    repo=repo,
                )


class ResponseFileTests(unittest.TestCase):
    def expand(
        self,
        root: Path,
        arguments: list[str],
    ) -> tuple[list[str], set[tuple[Path, Path]]]:
        references: set[tuple[Path, Path]] = set()
        expanded = d0.expand_response_arguments(
            arguments,
            working_directory=root,
            label="fixture",
            response_references=references,
        )
        return expanded, references

    def test_nested_relative_and_repeated_response_files_expand(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "inner.rsp").write_text("-DVALUE=1\n", encoding="utf-8")
            (root / "outer.rsp").write_text(
                "@inner.rsp -O3\n",
                encoding="utf-8",
            )

            expanded, references = self.expand(
                root,
                ["clang++", "@outer.rsp", "@inner.rsp"],
            )

            self.assertEqual(
                expanded,
                ["clang++", "-DVALUE=1", "-O3", "-DVALUE=1"],
            )
            self.assertEqual(
                references,
                {
                    (root, (root / "inner.rsp").resolve()),
                    (root, (root / "outer.rsp").resolve()),
                },
            )

    def test_macho_dynamic_paths_are_not_response_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            arguments = [
                "clang++",
                "@rpath/libfaiss.dylib",
                "@loader_path/libomp.dylib",
                "@executable_path/libsupport.dylib",
            ]

            expanded, references = self.expand(root, arguments)

            self.assertEqual(expanded, arguments)
            self.assertEqual(references, set())

    def test_hidden_release_policy_overrides_are_rejected(self) -> None:
        valid = [
            "clang++",
            "-O3",
            "-DNDEBUG",
            "-arch",
            "arm64",
            "-isysroot",
            "/fixture/MacOSX.sdk",
            "-mmacosx-version-min=14.0",
        ]
        cache = {
            "CMAKE_OSX_SYSROOT": "/fixture/MacOSX.sdk",
            "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for hidden, message in (
                ("-O0", "effective optimization"),
                ("-UNDEBUG", "NDEBUG"),
            ):
                with self.subTest(hidden=hidden):
                    (root / "hidden.rsp").write_text(hidden, encoding="utf-8")
                    expanded, _ = self.expand(root, [*valid, "@hidden.rsp"])
                    with self.assertRaisesRegex(d0.D0Failure, message):
                        d0.validate_release_arguments(
                            expanded,
                            cache,
                            label="fixture",
                        )

    def test_missing_cycle_malformed_and_non_utf8_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "cycle-a.rsp").write_text("@cycle-b.rsp", encoding="utf-8")
            (root / "cycle-b.rsp").write_text("@cycle-a.rsp", encoding="utf-8")
            (root / "malformed.rsp").write_text("'unterminated", encoding="utf-8")
            (root / "binary.rsp").write_bytes(b"\xff")

            for argument, message in (
                ("@missing.rsp", "missing"),
                ("@cycle-a.rsp", "cycle"),
                ("@malformed.rsp", "cannot be tokenized"),
                ("@binary.rsp", "not UTF-8"),
            ):
                with self.subTest(argument=argument):
                    with self.assertRaisesRegex(d0.D0Failure, message):
                        self.expand(root, [argument])

    def test_response_file_resource_limits_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "large.rsp").write_text("12345", encoding="utf-8")
            (root / "tokens.rsp").write_text("one two", encoding="utf-8")
            (root / "depth-a.rsp").write_text("@depth-b.rsp", encoding="utf-8")
            (root / "depth-b.rsp").write_text("-DEND", encoding="utf-8")

            with patch.object(d0, "MAX_RESPONSE_FILE_BYTES", 4):
                with self.assertRaisesRegex(d0.D0Failure, "byte limit"):
                    self.expand(root, ["@large.rsp"])
            with patch.object(d0, "MAX_RESPONSE_TOKENS", 1):
                with self.assertRaisesRegex(d0.D0Failure, "token limit"):
                    self.expand(root, ["@tokens.rsp"])
            with patch.object(d0, "MAX_RESPONSE_DEPTH", 1):
                with self.assertRaisesRegex(d0.D0Failure, "nesting limit"):
                    self.expand(root, ["@depth-a.rsp"])


class BuildInputClosureParsingTests(unittest.TestCase):
    def test_ninja_deps_parser_preserves_graph_and_rejects_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            text = (
                "obj/main.o: #deps 2, deps mtime 123 (VALID)\n"
                "    ../src/main.cpp\n"
                "    generated/module.pcm\n\n"
            )

            records = d0.parse_ninja_deps(
                text,
                build_directory=root,
                label="fixture",
            )

            output = str(root / "obj/main.o")
            self.assertEqual(records[output]["dependency_count"], 2)
            self.assertEqual(
                records[output]["dependencies"],
                [
                    str((root / "../src/main.cpp").resolve()),
                    str(root / "generated/module.pcm"),
                ],
            )
            with self.assertRaisesRegex(d0.D0Failure, "truncates dependencies"):
                d0.parse_ninja_deps(
                    "obj/main.o: #deps 1, deps mtime 123 (VALID)\n",
                    build_directory=root,
                    label="fixture",
                )

    def test_compile_direct_inputs_cover_modules_and_forced_includes(self) -> None:
        root = Path("/fixture/build")
        entry = {
            "directory": str(root),
            "file": "/fixture/src/main.cpp",
        }
        arguments = [
            "clang++",
            "-include",
            "/fixture/include/forced.h",
            "-imacros=/fixture/include/macros.h",
            "-fmodule-file=core=modules/core.pcm",
            "-fmodule-map-file=modules/core.modulemap",
            "-fmodule-output=modules/main.pcm",
            "-c",
            "/fixture/src/main.cpp",
            "-o",
            "main.o",
        ]

        self.assertEqual(
            d0.module_output_paths(
                arguments,
                working_directory=root,
                label="fixture",
            ),
            ["/fixture/build/modules/main.pcm"],
        )
        self.assertEqual(
            d0.compile_direct_inputs(
                entry,
                arguments,
                working_directory=root,
                label="fixture",
            ),
            [
                {"role": "compile-source", "path": "/fixture/src/main.cpp"},
                {
                    "role": "forced-include",
                    "path": "/fixture/include/forced.h",
                },
                {
                    "role": "forced-macro-include",
                    "path": "/fixture/include/macros.h",
                },
                {
                    "role": "module-input",
                    "path": "/fixture/build/modules/core.pcm",
                },
                {
                    "role": "module-map",
                    "path": "/fixture/build/modules/core.modulemap",
                },
            ],
        )

    def test_link_inputs_cover_archives_dylibs_objects_and_frameworks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sysroot = root / "MacOSX.sdk"
            framework = (
                sysroot
                / "System/Library/Frameworks/Accelerate.framework/Accelerate.tbd"
            )
            framework.parent.mkdir(parents=True)
            framework.write_text("stub\n", encoding="utf-8")
            arguments = [
                "clang++",
                "obj/main.o",
                "libsupport.a",
                "/fixture/lib/libomp.dylib",
                "-framework",
                "Accelerate",
                "-o",
                "app",
            ]

            self.assertEqual(
                d0.derive_link_inputs(
                    arguments,
                    working_directory=root,
                    sysroot=sysroot,
                    label="fixture",
                ),
                [
                    {"role": "link-object", "path": str(root / "obj/main.o")},
                    {"role": "link-archive", "path": str(root / "libsupport.a")},
                    {
                        "role": "link-dynamic-library",
                        "path": "/fixture/lib/libomp.dylib",
                    },
                    {
                        "role": "framework-input",
                        "path": str(framework),
                    },
                ],
            )

    def test_link_inputs_resolve_library_shorthand_in_linker_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sysroot = root / "MacOSX.sdk"
            explicit = root / "explicit"
            explicit.mkdir()
            system_libraries = sysroot / "usr/lib"
            system_libraries.mkdir(parents=True)
            explicit_archive = explicit / "libsample.a"
            explicit_archive.write_bytes(b"archive\n")
            (system_libraries / "libsample.tbd").write_text(
                "system sample\n",
                encoding="utf-8",
            )
            system_math = system_libraries / "libm.tbd"
            system_math.write_text("system math\n", encoding="utf-8")

            arguments = [
                "clang++",
                "obj/main.o",
                "-L",
                str(explicit),
                "-Wl,-search_paths_first",
                "-lsample",
                "-l",
                "m",
                "-o",
                "app",
            ]
            self.assertEqual(
                d0.derive_link_inputs(
                    arguments,
                    working_directory=root,
                    sysroot=sysroot,
                    label="fixture",
                ),
                [
                    {"role": "link-object", "path": str(root / "obj/main.o")},
                    {
                        "role": "link-library-archive",
                        "path": str(explicit_archive),
                    },
                    {
                        "role": "link-library-text-based-stub",
                        "path": str(system_math),
                    },
                ],
            )

            arguments.remove("-Wl,-search_paths_first")
            resolved = d0.derive_link_inputs(
                arguments,
                working_directory=root,
                sysroot=sysroot,
                label="fixture",
            )
            self.assertEqual(
                resolved[1],
                {
                    "role": "link-library-text-based-stub",
                    "path": str(system_libraries / "libsample.tbd"),
                },
            )

    def test_link_inputs_reject_missing_library_shorthand(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaisesRegex(
                d0.D0Failure,
                "cannot resolve library 'missing'",
            ):
                d0.derive_link_inputs(
                    ["clang++", "obj/main.o", "-lmissing", "-o", "app"],
                    working_directory=root,
                    sysroot=root / "MacOSX.sdk",
                    label="fixture",
                )

    def test_wl_inputs_capture_bare_objects_and_section_payloads(self) -> None:
        root = Path("/fixture/build")
        arguments = [
            "clang++",
            "base.o",
            "-Wl,/fixture/build/injected.o",
            "-Wl,-sectcreate,__TEXT,__audit,/fixture/payload.bin",
            "-o",
            "app",
        ]
        self.assertEqual(
            d0.derive_link_inputs(
                arguments,
                working_directory=root,
                sysroot=Path("/fixture/MacOSX.sdk"),
                label="fixture",
            ),
            [
                {"role": "link-object", "path": "/fixture/build/base.o"},
                {"role": "link-object", "path": "/fixture/build/injected.o"},
                {
                    "role": "section-content-input",
                    "path": "/fixture/payload.bin",
                },
            ],
        )

    def test_wl_inputs_reject_unsupported_forms(self) -> None:
        with self.assertRaisesRegex(d0.D0Failure, "unsupported -Wl option"):
            d0.derive_link_inputs(
                ["clang++", "base.o", "-Wl,-unknown_linker_switch", "-o", "app"],
                working_directory=Path("/fixture/build"),
                sysroot=Path("/fixture/MacOSX.sdk"),
                label="fixture",
            )

    def test_ninja_link_selection_rejects_same_basename_different_path(self) -> None:
        compiler = "/toolchain/clang++"
        build = Path("/fixture/build")
        common = [
            compiler,
            "-O3",
            "-DNDEBUG",
            "-arch",
            "arm64",
            "-isysroot",
            "/fixture/MacOSX.sdk",
            "-mmacosx-version-min=14.0",
        ]
        compile_arguments = [
            *common,
            "-c",
            "/fixture/source.cpp",
            "-o",
            str(build / "main.o"),
        ]
        link_arguments = [
            *common,
            str(build / "main.o"),
            "-o",
            str(build / "alternate/runner"),
        ]
        entry = {
            "directory": str(build),
            "file": "/fixture/source.cpp",
            "arguments": compile_arguments,
            "output": str(build / "main.o"),
        }
        cache = {
            "CMAKE_CXX_COMPILER": compiler,
            "CMAKE_OSX_SYSROOT": "/fixture/MacOSX.sdk",
            "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0",
        }
        with self.assertRaisesRegex(
            d0.D0Failure,
            "exactly one expected link command",
        ):
            d0.validate_ninja_commands(
                "\n".join(
                    (" ".join(compile_arguments), " ".join(link_arguments))
                ),
                cache=cache,
                compile_entries=[entry],
                build_directory=build,
                expected_output=build / "real/runner",
                label="fixture",
                response_references=set(),
            )

    def test_graph_proves_only_reachable_material_archive(self) -> None:
        selected = "/fixture/build/app"
        archive = "/fixture/build/libderived.a"
        object_path = "/fixture/build/derived.o"
        nodes = [
            {
                "output": selected,
                "rule": "LINK",
                "inputs": [{"edge": "implicit", "path": archive}],
                "outputs": [],
            },
            {
                "output": archive,
                "rule": "ARCHIVE",
                "inputs": [{"edge": "explicit", "path": object_path}],
                "outputs": [],
            },
        ]
        link_inputs = [{"role": "link-archive", "path": archive}]
        self.assertEqual(
            d0.graph_proven_link_outputs(
                nodes=nodes,
                selected_output=selected,
                link_inputs=link_inputs,
                known_derived_outputs={object_path},
                build_directory=Path("/fixture/build"),
                label="fixture",
            ),
            [archive],
        )
        for mutation in ("unreachable", "phony", "opaque-input"):
            changed = [
                {
                    **node,
                    "inputs": [dict(item) for item in node["inputs"]],
                }
                for node in nodes
            ]
            if mutation == "unreachable":
                changed[0]["inputs"] = []
            elif mutation == "phony":
                changed[1]["rule"] = "phony"
            else:
                changed[1]["inputs"][0]["path"] = "/fixture/build/opaque.o"
            with self.subTest(mutation=mutation):
                self.assertEqual(
                    d0.graph_proven_link_outputs(
                        nodes=changed,
                        selected_output=selected,
                        link_inputs=link_inputs,
                        known_derived_outputs={object_path},
                        build_directory=Path("/fixture/build"),
                        label="fixture",
                    ),
                    [],
                )


def fixture_query_checksums() -> list[int]:
    return [index + 1 for index in range(d0.HELDOUT_QUERY_COUNT)]


def valid_output(
    engine: str,
    *,
    participants: int = d0.MEASURED_PARTICIPANTS,
    recall: str = "1.0",
    campaign_nonce: str = "c" * 64,
    schedule_sequence: int = 0,
    binary_sha256: str = "b" * 64,
) -> str:
    level0_histogram = ["0"] * (2 * d0.M + 1)
    level0_histogram[2] = str(d0.VECTORS)
    per_query_checksums = fixture_query_checksums()
    result_checksum = d0.query_benchmark_checksum(per_query_checksums)
    common = {
        "status": "PASS",
        "scope": d0.D0_SCOPE,
        "order": f"{engine}-only",
        "data_fnv1a64": "123",
        "data_sha256": "a" * 64,
        "data_bytes": "456",
        "vectors": str(d0.VECTORS),
        "dimensions": str(d0.DIMENSIONS),
        "M": str(d0.M),
        "ef_construction": str(d0.EF_CONSTRUCTION),
        "ef_search": str(d0.EF_SEARCH),
        "participants": str(participants),
        "build_grain": str(d0.BUILD_GRAIN),
        "query_benchmark_schema": str(d0.QUERY_BENCHMARK_SCHEMA_VERSION),
        "query_unique_queries": str(d0.HELDOUT_QUERY_COUNT),
        "query_k": str(d0.QUERY_K),
        "query_ef_search": str(d0.QUERY_EF_SEARCH),
        "query_recall_floor": str(d0.QUERY_RECALL_FLOOR),
        "query_maximum_absolute_recall_gap": str(
            d0.QUERY_MAXIMUM_ABSOLUTE_RECALL_GAP
        ),
        "query_warmup_passes": str(d0.QUERY_WARMUP_PASSES),
        "query_measured_passes": str(d0.QUERY_MEASURED_PASSES),
        "query_latency_concurrency": str(d0.QUERY_LATENCY_CONCURRENCY),
        "query_throughput_concurrency": str(d0.QUERY_THROUGHPUT_CONCURRENCY),
        "query_percentile_method": d0.QUERY_PERCENTILE_METHOD,
        "query_transaction_definition": d0.QUERY_TRANSACTION_DEFINITION,
        "query_timed_corpus_sha256": "d" * 64,
        f"{engine}_valid": "1",
        f"{engine}_threads": str(participants),
        f"{engine}_index_size": str(d0.VECTORS),
        f"{engine}_cold_build_ns": "200",
        f"{engine}_insert_call_ns": "100",
        f"{engine}_self_recall_at_1": recall,
        f"{engine}_distance_checksum": "12.5",
        f"{engine}_query_latency_sample_count": str(
            d0.QUERY_LATENCY_SAMPLE_COUNT
        ),
        f"{engine}_query_latency_validated_operations": str(
            d0.QUERY_LATENCY_SAMPLE_COUNT
        ),
        f"{engine}_query_latency_validated_result_checksum": str(
            result_checksum
        ),
        f"{engine}_query_throughput_operations": str(
            d0.QUERY_THROUGHPUT_OPERATIONS
        ),
        f"{engine}_query_throughput_validated_operations": str(
            d0.QUERY_THROUGHPUT_OPERATIONS
        ),
        f"{engine}_query_throughput_wall_ns": "1000000",
        f"{engine}_query_throughput_validated_result_checksum": str(
            result_checksum
        ),
        f"{engine}_query_result_checksum": str(result_checksum),
        f"{engine}_query_per_query_checksum_count": str(
            d0.HELDOUT_QUERY_COUNT
        ),
        f"{engine}_query_per_query_checksums": ",".join(
            str(checksum) for checksum in per_query_checksums
        ),
        "audit_sidecar_schema": str(d0.AUDIT_SIDECAR_SCHEMA_VERSION),
        "audit_sidecar_bytes": "1000",
        "audit_sidecar_sha256": "c" * 64,
        "campaign_nonce": campaign_nonce,
        "schedule_sequence": str(schedule_sequence),
        "role_id": str(d0.STANDALONE_ROLE_IDS[engine]),
        "binary_sha256": binary_sha256,
        "dataset_sha256": "a" * 64,
        "heldout_query_count": str(d0.HELDOUT_QUERY_COUNT),
        "heldout_query_seed": str(d0.HELDOUT_QUERY_SEED),
        "heldout_queries_sha256": "d" * 64,
        "heldout_truth_sha256": "e" * 64,
        "heldout_truth_tie_counts_at_10": ",".join(
            ["10"] * d0.HELDOUT_QUERY_COUNT
        ),
        "heldout_truth_tie_counts_at_100": ",".join(
            ["100"] * d0.HELDOUT_QUERY_COUNT
        ),
        f"{engine}_graph_valid": "1",
        f"{engine}_graph_vertex_count": str(d0.VECTORS),
        f"{engine}_graph_reachable_count": str(d0.VECTORS),
        f"{engine}_graph_directed_edges": str(2 * d0.VECTORS),
        f"{engine}_graph_level0_directed_edges": str(2 * d0.VECTORS),
        f"{engine}_graph_max_level": "0",
        f"{engine}_graph_entry_point": "0",
        f"{engine}_graph_level0_capacity": str(2 * d0.M),
        f"{engine}_graph_upper_capacity": str(d0.M),
        f"{engine}_graph_levels_sha256": "f" * 64,
        f"{engine}_graph_sha256": "1" * 64,
        f"{engine}_graph_level_histogram": f"0:{d0.VECTORS}",
        f"{engine}_graph_degree_histograms": "L0:"
        + ",".join(level0_histogram),
    }
    common.update(
        {f"{engine}_{name}": "0" for name in d0.EXECUTION_WITNESS_FIELDS}
    )
    for k, ef in d0.RECALL_POINTS:
        returned = ",".join(
            str(label)
            for _ in range(d0.HELDOUT_QUERY_COUNT)
            for label in range(k)
        )
        common[f"{engine}_recall_at_{k}_ef_{ef}"] = "1"
        common[f"{engine}_returned_ids_ef_{ef}_k_{k}"] = returned
        common[f"{engine}_returned_distances_sha256_ef_{ef}_k_{k}"] = "2" * 64
    if engine == "infinity":
        common.update(
            {
                "infinity_submitted_tasks": str(
                    d0.expected_infinity_tasks(participants)
                ),
                "infinity_build_start": "0",
                "infinity_build_end": str(d0.VECTORS),
            }
        )
    return "".join(f"{key}={value}\n" for key, value in common.items())


def valid_attestation_output(driver_output: str) -> str:
    lines = [f"attestation_schema={d0.ATTESTATION_SCHEMA_VERSION}"]
    for phase in ("before", "after"):
        prefix = f"attestation_{phase}_"
        lines.extend(
            (
                prefix + "dynamic_cdhash=" + "a" * 40,
                prefix + "held_executable_dev=1",
                prefix + "held_executable_ino=2",
                prefix + "held_executable_size=3",
                prefix + "mapped_executable_dev=1",
                prefix + "mapped_executable_ino=2",
                prefix + "mapped_executable_size=3",
                prefix + "mapped_vnode_matches_held_fd=1",
                prefix + "image_count=1",
                prefix + "image_0_path_hex=2f66697874757265",
                prefix + "image_0_uuid=" + "b" * 32,
                prefix + "image_0_shared_cache=0",
            )
        )
        if phase == "before":
            lines.extend(
                f"attestation_barrier={barrier}"
                for barrier in d0.ATTESTATION_BARRIER_PHASES[:3]
            )
    lines.extend(
        (
            "attestation_initial_image_adds=1",
            "attestation_later_image_adds=0",
            "attestation_image_removes=0",
            "attestation_image_set_unchanged=1",
            "attestation_barrier=after-work",
            "attestation_status=PASS",
        )
    )
    return "\n".join(lines) + "\n" + driver_output


class DriverParsingTests(unittest.TestCase):
    dataset = {"fnv1a64": 123, "sha256": "a" * 64, "bytes": 456}

    def test_accepts_exact_infinity_and_faiss_schemas(self) -> None:
        for engine in ("infinity", "faiss"):
            with self.subTest(engine=engine):
                fields = d0.parse_driver_result(
                    valid_output(engine),
                    engine=engine,
                    participants=d0.MEASURED_PARTICIPANTS,
                    dataset=self.dataset,
                )
                self.assertEqual(fields["order"], f"{engine}-only")

    def test_accepts_and_separates_strict_attestation_transcript(self) -> None:
        driver_output = valid_output("infinity")
        separated, attestation = d0.split_attested_driver_output(
            valid_attestation_output(driver_output)
        )
        self.assertEqual(separated, driver_output)
        self.assertEqual(
            attestation["barriers"],
            list(d0.ATTESTATION_BARRIER_PHASES),
        )
        self.assertEqual(attestation["before"], attestation["after"])
        self.assertEqual(attestation["initial_image_adds"], 1)
        self.assertEqual(attestation["status"], "PASS")

    def test_rejects_mutated_or_reordered_attestation_transcript(self) -> None:
        base = valid_attestation_output(valid_output("faiss"))
        mutations = (
            (
                "snapshots differ",
                base.replace(
                    "attestation_after_dynamic_cdhash=" + "a" * 40,
                    "attestation_after_dynamic_cdhash=" + "c" * 40,
                ),
            ),
            (
                "out of order",
                base.replace(
                    "attestation_barrier=before-work",
                    "attestation_barrier=after-work",
                    1,
                ),
            ),
            (
                "status is not PASS",
                base.replace("attestation_status=PASS", "attestation_status=FAIL"),
            ),
            (
                "after driver result",
                valid_output("faiss")
                + "\n".join(base.splitlines()[:2])
                + "\n",
            ),
        )
        for message, transcript in mutations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(d0.D0Failure, message):
                    d0.split_attested_driver_output(transcript)

    def test_infinity_task_count_tracks_participant_count(self) -> None:
        self.assertEqual(d0.expected_infinity_tasks(1), 1)
        self.assertEqual(
            d0.expected_infinity_tasks(d0.MEASURED_PARTICIPANTS),
            12,
        )

    def test_rejects_duplicate_unknown_missing_and_malformed_fields(self) -> None:
        base = valid_output("faiss")
        cases = (
            ("duplicate", base + "status=PASS\n"),
            ("unknown", base + "seed=0\n"),
            ("missing", base.replace("data_bytes=456\n", "")),
            ("not one key=value", base + "not-a-field\n"),
        )
        for expected, text in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(d0.D0Failure, expected):
                    d0.parse_driver_output(text, engine="faiss")

    def test_rejects_nonfinite_and_out_of_range_recall(self) -> None:
        for recall in ("nan", "inf", "-0.1", "1.1"):
            with self.subTest(recall=recall):
                with self.assertRaises(d0.D0Failure):
                    d0.parse_driver_result(
                        valid_output("faiss", recall=recall),
                        engine="faiss",
                        participants=d0.MEASURED_PARTICIPANTS,
                        dataset=self.dataset,
                    )

    def test_rejects_noncanonical_integer(self) -> None:
        text = valid_output("faiss").replace(
            f"vectors={d0.VECTORS}\n",
            f"vectors=0{d0.VECTORS}\n",
        )
        with self.assertRaisesRegex(d0.D0Failure, "canonical unsigned decimal"):
            d0.parse_driver_result(
                text,
                engine="faiss",
                participants=d0.MEASURED_PARTICIPANTS,
                dataset=self.dataset,
            )


class IndexTimingBindingTests(unittest.TestCase):
    @staticmethod
    def barriers(*, outer_ns: int) -> list[dict[str, object]]:
        observations: list[dict[str, object]] = []
        before_index_resume = 2_000_000_001
        for index, phase in enumerate(d0.ATTESTATION_BARRIER_PHASES):
            observed = 1_000_000_000 + index * 100
            resumed = observed + 1
            if phase == "before-index":
                observed = before_index_resume - 1
                resumed = before_index_resume
            elif phase == "after-index":
                observed = before_index_resume + outer_ns
                resumed = observed + 1
            observations.append(
                {
                    "phase": phase,
                    "observed_at_monotonic_ns": observed,
                    "resumed_at_monotonic_ns": resumed,
                }
            )
        return observations

    def test_binds_reported_duration_inside_parent_window(self) -> None:
        binding = d0.index_timing_binding(
            self.barriers(outer_ns=101_000_000),
            {"infinity_cold_build_ns": "100000000"},
            engine="infinity",
        )
        self.assertEqual(binding["outer_index_window_ns"], 101_000_000)
        self.assertEqual(binding["boundary_overhead_ns"], 1_000_000)

    def test_rejects_duration_outside_parent_window_or_excess_overhead(self) -> None:
        with self.assertRaisesRegex(
            d0.D0Failure,
            "exceeds the parent-observed index window",
        ):
            d0.index_timing_binding(
                self.barriers(outer_ns=99_999_999),
                {"infinity_cold_build_ns": "100000000"},
                engine="infinity",
            )
        with self.assertRaisesRegex(
            d0.D0Failure,
            "overhead exceeds the protocol maximum",
        ):
            d0.index_timing_binding(
                self.barriers(
                    outer_ns=(
                        100_000_000
                        + d0.INDEX_TIMING_MAXIMUM_BOUNDARY_OVERHEAD_NS
                        + 1
                    )
                ),
                {"infinity_cold_build_ns": "100000000"},
                engine="infinity",
            )


class ProcessMappedExecutableIdentityTests(unittest.TestCase):
    class FakeLibproc:
        def __init__(
            self,
            *,
            path: str = "/tmp/benchmark",
            device: int = 30,
            inode: int = 31,
            size: int = 32,
        ) -> None:
            self.path = path
            self.device = device
            self.inode = inode
            self.size = size
            self.calls: list[tuple[int, int, int, int]] = []

        def proc_pidinfo(
            self,
            pid: int,
            flavor: int,
            address: int,
            buffer: object,
            buffer_size: int,
        ) -> int:
            self.calls.append((pid, flavor, address, buffer_size))
            record = ctypes.cast(
                buffer,
                ctypes.POINTER(d0._ProcRegionWithPathInfo),
            ).contents
            record.prp_prinfo.pri_protection = (
                d0.VM_PROT_READ | d0.VM_PROT_EXECUTE
            )
            record.prp_prinfo.pri_address = 0x100000000
            record.prp_prinfo.pri_size = 0x4000
            record.prp_vip.vip_vi.vi_stat.vst_dev = self.device
            record.prp_vip.vip_vi.vi_stat.vst_mode = stat.S_IFREG | 0o555
            record.prp_vip.vip_vi.vi_stat.vst_ino = self.inode
            record.prp_vip.vip_vi.vi_stat.vst_size = self.size
            record.prp_vip.vip_path = os.fsencode(self.path)
            return buffer_size

    def test_region_structures_match_the_macos_abi(self) -> None:
        self.assertEqual(ctypes.sizeof(d0._ProcRegionInfo), 96)
        self.assertEqual(ctypes.sizeof(d0._VinfoStat), 136)
        self.assertEqual(ctypes.sizeof(d0._VnodeInfo), 152)
        self.assertEqual(ctypes.sizeof(d0._VnodeInfoPath), 1176)
        self.assertEqual(ctypes.sizeof(d0._ProcRegionWithPathInfo), 1272)

    def test_uses_mapped_region_vnode_instead_of_path_stat(self) -> None:
        library = self.FakeLibproc()
        with patch.object(d0, "_libproc", return_value=library):
            identity = d0._mapped_executable_identity(
                123,
                Path("/tmp/benchmark"),
            )

        self.assertEqual(
            identity,
            {
                "path": "/tmp/benchmark",
                "device": 30,
                "inode": 31,
                "size": 32,
            },
        )
        self.assertEqual(
            library.calls,
            [
                (
                    123,
                    d0.PROC_PIDREGIONPATHINFO,
                    0,
                    ctypes.sizeof(d0._ProcRegionWithPathInfo),
                )
            ],
        )

    def test_rejects_region_path_that_differs_from_proc_pidpath(self) -> None:
        library = self.FakeLibproc(path="/tmp/replacement")
        with (
            patch.object(d0, "_libproc", return_value=library),
            self.assertRaisesRegex(
                d0.D0Failure,
                "mapped region differs from proc_pidpath",
            ),
        ):
            d0._mapped_executable_identity(123, Path("/tmp/benchmark"))

    @unittest.skipUnless(
        sys.platform == "darwin",
        "mapped executable vnode evidence requires macOS",
    )
    def test_replace_launch_restore_does_not_accept_restored_path_vnode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launch_path = root / "sleep"
            held_original = root / "sleep.original"
            replacement = root / "sleep.replacement"
            source = root / "sleep.c"
            source.write_text(
                "#include <unistd.h>\n"
                "int main(void) { sleep(30); return 0; }\n",
                encoding="ascii",
            )
            subprocess.run(
                [
                    "/usr/bin/clang",
                    "-x",
                    "c",
                    str(source),
                    "-o",
                    str(launch_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            shutil.copyfile(launch_path, replacement)
            replacement.chmod(0o755)
            expected = d0.process_executable_identity(launch_path)
            launch_path.rename(held_original)
            replacement.rename(launch_path)
            launched = d0.process_executable_identity(launch_path)
            self.assertNotEqual(
                (expected["device"], expected["inode"]),
                (launched["device"], launched["inode"]),
            )

            process = subprocess.Popen(
                [str(launch_path), "30"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                observed_before_restore = d0.capture_process_identity(
                    process.pid
                )
                mapped_before_restore = d0._process_executable_from_snapshot(
                    observed_before_restore
                )
                self.assertEqual(
                    (
                        mapped_before_restore["device"],
                        mapped_before_restore["inode"],
                        mapped_before_restore["size"],
                    ),
                    (
                        launched["device"],
                        launched["inode"],
                        launched["size"],
                    ),
                )
                launch_path.unlink()
                held_original.rename(launch_path)
                self.assertEqual(
                    d0.process_executable_identity(launch_path),
                    expected,
                )

                observed = d0.capture_process_identity(process.pid)
                mapped = d0._process_executable_from_snapshot(observed)
                self.assertEqual(mapped, mapped_before_restore)
                self.assertEqual(
                    (
                        mapped["device"],
                        mapped["inode"],
                        mapped["size"],
                    ),
                    (
                        launched["device"],
                        launched["inode"],
                        launched["size"],
                    ),
                )
                self.assertNotEqual(mapped, expected)
            finally:
                process.terminate()
                process.wait(timeout=5.0)


class ProcessTopologyTests(unittest.TestCase):
    WRAPPER = {
        "path": "/usr/bin/time",
        "device": 10,
        "inode": 11,
        "size": 12,
    }
    BENCHMARK = {
        "path": "/tmp/benchmark",
        "device": 20,
        "inode": 21,
        "size": 22,
    }

    @classmethod
    def processes(cls) -> list[dict[str, object]]:
        return [
            {
                "pid": 100,
                "parent_pid": 50,
                "process_group": 100,
                "status_code": 3,
                "status": "sleeping",
                "start_time_seconds": 1,
                "start_time_microseconds": 2,
                "name": "time",
                "executable_path": cls.WRAPPER["path"],
                "executable_device": cls.WRAPPER["device"],
                "executable_inode": cls.WRAPPER["inode"],
                "executable_size": cls.WRAPPER["size"],
                "child_pids": [101],
            },
            {
                "pid": 101,
                "parent_pid": 100,
                "process_group": 100,
                "status_code": 4,
                "status": "stopped",
                "start_time_seconds": 3,
                "start_time_microseconds": 4,
                "name": "benchmark",
                "executable_path": cls.BENCHMARK["path"],
                "executable_device": cls.BENCHMARK["device"],
                "executable_inode": cls.BENCHMARK["inode"],
                "executable_size": cls.BENCHMARK["size"],
                "child_pids": [],
            },
        ]

    def validate(self, processes: list[dict[str, object]]) -> int:
        return d0.validate_live_benchmark_topology(
            processes,
            process_group=100,
            supervisor_pid=50,
            expected_wrapper_executable=self.WRAPPER,
            expected_benchmark_executable=self.BENCHMARK,
            phase="before-index",
        )

    def test_accepts_exact_wrapper_and_stopped_child(self) -> None:
        self.assertEqual(self.validate(self.processes()), 101)

    def test_rejects_topology_executable_and_descendant_mutations(self) -> None:
        mutations = (
            ("contains 3 processes", lambda value: value.append(copy.deepcopy(value[1]))),
            (
                "unexpected parent",
                lambda value: value[0].update({"parent_pid": 51}),
            ),
            (
                "not a direct child",
                lambda value: value[1].update({"parent_pid": 99}),
            ),
            (
                "not stopped",
                lambda value: value[1].update(
                    {"status_code": 3, "status": "sleeping"}
                ),
            ),
            (
                "spawned descendants",
                lambda value: value[1].update({"child_pids": [102]}),
            ),
            (
                "executable identity differs",
                lambda value: value[1].update({"executable_inode": 99}),
            ),
        )
        for message, mutate in mutations:
            with self.subTest(message=message):
                processes = self.processes()
                mutate(processes)
                with self.assertRaisesRegex(d0.D0Failure, message):
                    self.validate(processes)


class CampaignBindingTests(unittest.TestCase):
    def test_campaign_binding_freezes_engine_binary_dataset_and_schedule(self) -> None:
        binaries = {
            "infinity": {"sha256": "b" * 64},
            "faiss": {"sha256": "c" * 64},
        }
        with patch.object(d0.secrets, "token_hex", return_value="d" * 64):
            campaign = d0.create_campaign_binding(
                binaries=binaries,
                dataset={"sha256": "a" * 64},
                schedule_sha256="e" * 64,
            )
        self.assertEqual(
            campaign,
            {
                "schema_version": d0.CAMPAIGN_BINDING_SCHEMA_VERSION,
                "campaign_nonce": "d" * 64,
                "dataset_sha256": "a" * 64,
                "schedule_sha256": "e" * 64,
                "roles": {
                    "infinity": {
                        "role_id": 3,
                        "engine": "infinity",
                        "binary_sha256": "b" * 64,
                    },
                    "faiss": {
                        "role_id": 4,
                        "engine": "faiss",
                        "binary_sha256": "c" * 64,
                    },
                },
            },
        )
        self.assertEqual(
            d0.member_campaign_binding(
                {"sequence": 17, "engine": "faiss"},
                campaign,
            ),
            {
                "campaign_nonce": "d" * 64,
                "schedule_sequence": 17,
                "role_id": 4,
                "binary_sha256": "c" * 64,
                "dataset_sha256": "a" * 64,
            },
        )

    def test_driver_binding_cannot_be_self_declared(self) -> None:
        fields = {
            "campaign_nonce": "f" * 64,
            "schedule_sequence": "0",
            "role_id": "3",
            "binary_sha256": "b" * 64,
            "dataset_sha256": "a" * 64,
        }
        with self.assertRaisesRegex(d0.D0Failure, "campaign_nonce differs"):
            d0.validate_driver_campaign_binding(
                fields,
                {
                    "campaign_nonce": "c" * 64,
                    "schedule_sequence": 0,
                    "role_id": 3,
                    "binary_sha256": "b" * 64,
                    "dataset_sha256": "a" * 64,
                },
                context="fixture",
            )


SCHEMA2_FIXTURE_VECTORS = 4
SCHEMA2_FIXTURE_DIMENSIONS = 2
SCHEMA2_FIXTURE_M = 2
SCHEMA2_FIXTURE_EF_CONSTRUCTION = 4
SCHEMA2_FIXTURE_QUERY_COUNT = 2
SCHEMA2_FIXTURE_QUERY_K = 2
SCHEMA2_FIXTURE_QUERY_EF = 2
SCHEMA2_FIXTURE_RECALL_POINTS = ((2, 2), (3, 3))
SCHEMA2_FIXTURE_LATENCY_SAMPLES = (101, 102, 103, 104)
SCHEMA2_FIXTURE_THROUGHPUT_OPERATIONS = 4
SCHEMA2_FIXTURE_THROUGHPUT_WALL_NS = 100_000
SCHEMA2_FIXTURE_CAMPAIGN_NONCE = "11" * 32
SCHEMA2_FIXTURE_BINARY_SHA256 = "22" * 32
SCHEMA2_FIXTURE_DATASET_SHA256 = "33" * 32
SCHEMA2_FIXTURE_QUERIES_SHA256 = "44" * 32
SCHEMA2_FIXTURE_TRUTH_SHA256 = "55" * 32
SCHEMA2_FIXTURE_QUERY_OFFSET = 192
SCHEMA2_FIXTURE_CORPUS_OFFSET = 248
SCHEMA2_FIXTURE_MEASUREMENTS_OFFSET = 280
SCHEMA2_FIXTURE_PER_QUERY_OFFSET = 352


@contextlib.contextmanager
def schema2_small_protocol() -> Iterator[None]:
    values = {
        "VECTORS": SCHEMA2_FIXTURE_VECTORS,
        "DIMENSIONS": SCHEMA2_FIXTURE_DIMENSIONS,
        "M": SCHEMA2_FIXTURE_M,
        "EF_CONSTRUCTION": SCHEMA2_FIXTURE_EF_CONSTRUCTION,
        "HELDOUT_QUERY_COUNT": SCHEMA2_FIXTURE_QUERY_COUNT,
        "RECALL_POINTS": SCHEMA2_FIXTURE_RECALL_POINTS,
        "QUERY_K": SCHEMA2_FIXTURE_QUERY_K,
        "QUERY_EF_SEARCH": SCHEMA2_FIXTURE_QUERY_EF,
        "QUERY_WARMUP_PASSES": 1,
        "QUERY_MEASURED_PASSES": 2,
        "QUERY_LATENCY_SAMPLE_COUNT": len(SCHEMA2_FIXTURE_LATENCY_SAMPLES),
        "QUERY_THROUGHPUT_OPERATIONS": (
            SCHEMA2_FIXTURE_THROUGHPUT_OPERATIONS
        ),
    }
    with contextlib.ExitStack() as stack:
        for name, value in values.items():
            stack.enter_context(patch.object(d0, name, value))
        yield


def schema2_fixture_truth() -> dict[str, object]:
    distances = (
        (0.0, 1.0, 4.0, 9.0),
        (9.0, 4.0, 1.0, 0.0),
    )
    return {
        "queries_sha256": SCHEMA2_FIXTURE_QUERIES_SHA256,
        "truth_sha256": SCHEMA2_FIXTURE_TRUTH_SHA256,
        "_distances": distances,
        "_cutoffs_at_10": (100.0, 100.0),
        "_cutoffs_at_100": (100.0, 100.0),
    }


def schema2_fixture_labels(query_index: int, k: int) -> tuple[int, ...]:
    return ((0, 1, 2, 3), (3, 2, 1, 0))[query_index][:k]


def schema2_fixture_per_query_checksums() -> list[int]:
    distances = schema2_fixture_truth()["_distances"]
    assert isinstance(distances, tuple)
    checksums: list[int] = []
    for query_index in range(SCHEMA2_FIXTURE_QUERY_COUNT):
        labels = schema2_fixture_labels(query_index, SCHEMA2_FIXTURE_QUERY_K)
        distance_bits = [
            struct.unpack("<I", struct.pack("<f", distances[query_index][label]))[
                0
            ]
            for label in labels
        ]
        checksums.append(
            d0.query_result_checksum(query_index, list(labels), distance_bits)
        )
    return checksums


def schema2_sidecar_fixture() -> tuple[bytes, dict[str, str]]:
    engine = "infinity"
    role_id = d0.STANDALONE_ROLE_IDS[engine]
    per_query_checksums = schema2_fixture_per_query_checksums()
    result_checksum = d0.query_benchmark_checksum(per_query_checksums)
    prefix = struct.pack(
        "<8sII32sQII32s32s",
        d0.AUDIT_SIDECAR_MAGIC,
        d0.AUDIT_SIDECAR_SCHEMA_VERSION,
        d0.AUDIT_ENGINE_IDENTIFIERS[engine],
        bytes.fromhex(SCHEMA2_FIXTURE_CAMPAIGN_NONCE),
        7,
        role_id,
        0,
        bytes.fromhex(SCHEMA2_FIXTURE_BINARY_SHA256),
        bytes.fromhex(SCHEMA2_FIXTURE_DATASET_SHA256),
    )
    body = bytearray(
        struct.pack(
            "<QQIIiiIIQQII",
            SCHEMA2_FIXTURE_VECTORS,
            SCHEMA2_FIXTURE_DIMENSIONS,
            SCHEMA2_FIXTURE_M,
            SCHEMA2_FIXTURE_EF_CONSTRUCTION,
            0,
            0,
            2 * SCHEMA2_FIXTURE_M,
            SCHEMA2_FIXTURE_M,
            SCHEMA2_FIXTURE_QUERY_COUNT,
            d0.HELDOUT_QUERY_SEED,
            len(SCHEMA2_FIXTURE_RECALL_POINTS),
            0,
        )
    )
    body.extend(
        struct.pack(
            "<IIIIIIIIIIQQ",
            d0.QUERY_BENCHMARK_SCHEMA_VERSION,
            SCHEMA2_FIXTURE_QUERY_COUNT,
            SCHEMA2_FIXTURE_QUERY_K,
            SCHEMA2_FIXTURE_QUERY_EF,
            1,
            2,
            d0.QUERY_LATENCY_CONCURRENCY,
            d0.QUERY_THROUGHPUT_CONCURRENCY,
            d0.QUERY_PERCENTILE_METHOD_ID,
            d0.QUERY_TRANSACTION_DEFINITION_ID,
            struct.unpack("<Q", struct.pack("<d", d0.QUERY_RECALL_FLOOR))[0],
            struct.unpack(
                "<Q",
                struct.pack("<d", d0.QUERY_MAXIMUM_ABSOLUTE_RECALL_GAP),
            )[0],
        )
    )
    body.extend(bytes.fromhex(SCHEMA2_FIXTURE_QUERIES_SHA256))
    body.extend(
        struct.pack(
            "<9Q",
            len(SCHEMA2_FIXTURE_LATENCY_SAMPLES),
            len(SCHEMA2_FIXTURE_LATENCY_SAMPLES),
            result_checksum,
            SCHEMA2_FIXTURE_THROUGHPUT_OPERATIONS,
            SCHEMA2_FIXTURE_THROUGHPUT_OPERATIONS,
            SCHEMA2_FIXTURE_THROUGHPUT_WALL_NS,
            result_checksum,
            result_checksum,
            SCHEMA2_FIXTURE_QUERY_COUNT,
        )
    )
    body.extend(struct.pack("<2Q", *per_query_checksums))
    body.extend(
        struct.pack(
            "<4Q",
            *SCHEMA2_FIXTURE_LATENCY_SAMPLES,
        )
    )

    levels_digest = hashlib.sha256()
    levels_digest.update(b"infinity-hnsw-levels-v1")
    levels_digest.update(struct.pack("<Q", SCHEMA2_FIXTURE_VECTORS))
    graph_digest = hashlib.sha256()
    graph_digest.update(b"infinity-hnsw-graph-v1")
    graph_digest.update(
        struct.pack(
            "<QIIQQ",
            SCHEMA2_FIXTURE_VECTORS,
            0,
            0,
            2 * SCHEMA2_FIXTURE_M,
            SCHEMA2_FIXTURE_M,
        )
    )
    for vertex in range(SCHEMA2_FIXTURE_VECTORS):
        neighbor = (vertex + 1) % SCHEMA2_FIXTURE_VECTORS
        body.extend(
            struct.pack(
                "<iqIIi",
                0,
                vertex,
                2 * SCHEMA2_FIXTURE_M,
                1,
                neighbor,
            )
        )
        levels_digest.update(struct.pack("<II", vertex, 0))
        graph_digest.update(struct.pack("<IIQ", vertex, 0, vertex))
        graph_digest.update(struct.pack("<IQI", 0, 1, neighbor))

    fields = {
        "audit_sidecar_schema": str(d0.AUDIT_SIDECAR_SCHEMA_VERSION),
        "campaign_nonce": SCHEMA2_FIXTURE_CAMPAIGN_NONCE,
        "schedule_sequence": "7",
        "role_id": str(role_id),
        "binary_sha256": SCHEMA2_FIXTURE_BINARY_SHA256,
        "dataset_sha256": SCHEMA2_FIXTURE_DATASET_SHA256,
        "query_benchmark_schema": str(d0.QUERY_BENCHMARK_SCHEMA_VERSION),
        "query_unique_queries": str(SCHEMA2_FIXTURE_QUERY_COUNT),
        "query_k": str(SCHEMA2_FIXTURE_QUERY_K),
        "query_ef_search": str(SCHEMA2_FIXTURE_QUERY_EF),
        "query_warmup_passes": "1",
        "query_measured_passes": "2",
        "query_latency_concurrency": str(d0.QUERY_LATENCY_CONCURRENCY),
        "query_throughput_concurrency": str(d0.QUERY_THROUGHPUT_CONCURRENCY),
        "query_timed_corpus_sha256": SCHEMA2_FIXTURE_QUERIES_SHA256,
        "infinity_query_latency_sample_count": str(
            len(SCHEMA2_FIXTURE_LATENCY_SAMPLES)
        ),
        "infinity_query_latency_validated_operations": str(
            len(SCHEMA2_FIXTURE_LATENCY_SAMPLES)
        ),
        "infinity_query_latency_validated_result_checksum": str(
            result_checksum
        ),
        "infinity_query_throughput_operations": str(
            SCHEMA2_FIXTURE_THROUGHPUT_OPERATIONS
        ),
        "infinity_query_throughput_validated_operations": str(
            SCHEMA2_FIXTURE_THROUGHPUT_OPERATIONS
        ),
        "infinity_query_throughput_wall_ns": str(
            SCHEMA2_FIXTURE_THROUGHPUT_WALL_NS
        ),
        "infinity_query_throughput_validated_result_checksum": str(
            result_checksum
        ),
        "infinity_query_result_checksum": str(result_checksum),
        "infinity_query_per_query_checksum_count": str(
            SCHEMA2_FIXTURE_QUERY_COUNT
        ),
        "infinity_query_per_query_checksums": ",".join(
            str(value) for value in per_query_checksums
        ),
        "heldout_queries_sha256": SCHEMA2_FIXTURE_QUERIES_SHA256,
        "heldout_truth_sha256": SCHEMA2_FIXTURE_TRUTH_SHA256,
        "infinity_graph_vertex_count": str(SCHEMA2_FIXTURE_VECTORS),
        "infinity_graph_reachable_count": str(SCHEMA2_FIXTURE_VECTORS),
        "infinity_graph_directed_edges": str(SCHEMA2_FIXTURE_VECTORS),
        "infinity_graph_level0_directed_edges": str(SCHEMA2_FIXTURE_VECTORS),
        "infinity_graph_max_level": "0",
        "infinity_graph_entry_point": "0",
        "infinity_graph_level0_capacity": str(2 * SCHEMA2_FIXTURE_M),
        "infinity_graph_upper_capacity": str(SCHEMA2_FIXTURE_M),
        "infinity_graph_levels_sha256": levels_digest.hexdigest(),
        "infinity_graph_sha256": graph_digest.hexdigest(),
        "infinity_graph_level_histogram": (
            f"0:{SCHEMA2_FIXTURE_VECTORS}"
        ),
        "infinity_graph_degree_histograms": "L0:0,4,0,0,0",
    }
    fields.update(
        {f"infinity_{name}": "0" for name in d0.EXECUTION_WITNESS_FIELDS}
    )

    distances = schema2_fixture_truth()["_distances"]
    assert isinstance(distances, tuple)
    for k, ef in SCHEMA2_FIXTURE_RECALL_POINTS:
        body.extend(
            struct.pack(
                "<IIQ",
                k,
                ef,
                SCHEMA2_FIXTURE_QUERY_COUNT * k,
            )
        )
        returned_ids: list[str] = []
        distance_digest = hashlib.sha256()
        for query_index in range(SCHEMA2_FIXTURE_QUERY_COUNT):
            for label in schema2_fixture_labels(query_index, k):
                distance_bytes = struct.pack(
                    "<f", distances[query_index][label]
                )
                body.extend(
                    struct.pack(
                        "<qI",
                        label,
                        struct.unpack("<I", distance_bytes)[0],
                    )
                )
                returned_ids.append(str(label))
                distance_digest.update(distance_bytes)
        fields[f"infinity_recall_at_{k}_ef_{ef}"] = "1.0"
        fields[f"infinity_returned_ids_ef_{ef}_k_{k}"] = ",".join(
            returned_ids
        )
        fields[
            f"infinity_returned_distances_sha256_ef_{ef}_k_{k}"
        ] = distance_digest.hexdigest()

    sidecar = prefix + bytes(body)
    fields["audit_sidecar_bytes"] = str(len(sidecar))
    fields["audit_sidecar_sha256"] = hashlib.sha256(sidecar).hexdigest()
    return sidecar, fields


def parse_schema2_fixture(
    sidecar: bytes,
    fields: dict[str, str],
) -> dict[str, object]:
    return d0.parse_audit_sidecar(
        Path("fixture.audit-v4.bin"),
        engine="infinity",
        fields=fields,
        dataset={"sha256": SCHEMA2_FIXTURE_DATASET_SHA256},
        truth=schema2_fixture_truth(),
        expected_campaign_nonce=SCHEMA2_FIXTURE_CAMPAIGN_NONCE,
        expected_schedule_sequence=7,
        expected_role_id=d0.STANDALONE_ROLE_IDS["infinity"],
        expected_binary_sha256=SCHEMA2_FIXTURE_BINARY_SHA256,
        expected_dataset_sha256=SCHEMA2_FIXTURE_DATASET_SHA256,
        data=sidecar,
    )


def refresh_schema2_sidecar_fields(
    sidecar: bytes,
    fields: dict[str, str],
) -> dict[str, str]:
    updated = dict(fields)
    updated["audit_sidecar_bytes"] = str(len(sidecar))
    updated["audit_sidecar_sha256"] = hashlib.sha256(sidecar).hexdigest()
    return updated


class ProducerSchema2SidecarTests(unittest.TestCase):
    def test_valid_schema2_layout_and_returned_evidence(self) -> None:
        with schema2_small_protocol():
            sidecar, fields = schema2_sidecar_fixture()
            self.assertEqual(
                struct.unpack_from("<I", sidecar, SCHEMA2_FIXTURE_QUERY_OFFSET)[0],
                2,
            )
            self.assertEqual(
                sidecar[
                    SCHEMA2_FIXTURE_CORPUS_OFFSET:
                    SCHEMA2_FIXTURE_MEASUREMENTS_OFFSET
                ],
                bytes.fromhex(SCHEMA2_FIXTURE_QUERIES_SHA256),
            )
            expected_measurements = (
                4,
                4,
                int(fields["infinity_query_result_checksum"]),
                4,
                4,
                SCHEMA2_FIXTURE_THROUGHPUT_WALL_NS,
                int(fields["infinity_query_result_checksum"]),
                int(fields["infinity_query_result_checksum"]),
                2,
            )
            self.assertEqual(
                struct.unpack_from(
                    "<9Q", sidecar, SCHEMA2_FIXTURE_MEASUREMENTS_OFFSET
                ),
                expected_measurements,
            )

            audit = parse_schema2_fixture(sidecar, fields)
            query = audit["query"]
            assert isinstance(query, dict)
            self.assertEqual(query["schema_version"], 2)
            self.assertEqual(
                query["timed_query_corpus_sha256"],
                SCHEMA2_FIXTURE_QUERIES_SHA256,
            )
            self.assertEqual(query["latency_validated_operations"], 4)
            self.assertEqual(
                query["latency_validated_result_checksum"],
                expected_measurements[2],
            )
            self.assertEqual(query["throughput_operations"], 4)
            self.assertEqual(query["throughput_validated_operations"], 4)
            self.assertEqual(
                query["throughput_wall_ns"],
                SCHEMA2_FIXTURE_THROUGHPUT_WALL_NS,
            )
            self.assertEqual(
                query["throughput_validated_result_checksum"],
                expected_measurements[2],
            )
            self.assertEqual(query["result_checksum"], expected_measurements[2])
            self.assertEqual(query["per_query_checksum_count"], 2)
            self.assertEqual(
                query["per_query_checksums"],
                schema2_fixture_per_query_checksums(),
            )

    def test_binary_scalar_mutations_are_rejected(self) -> None:
        with schema2_small_protocol():
            sidecar, fields = schema2_sidecar_fixture()
            checksum = int(fields["infinity_query_result_checksum"])
            mutations = (
                (SCHEMA2_FIXTURE_QUERY_OFFSET, "<I", 1),
                (SCHEMA2_FIXTURE_MEASUREMENTS_OFFSET, "<Q", 3),
                (SCHEMA2_FIXTURE_MEASUREMENTS_OFFSET + 8, "<Q", 3),
                (SCHEMA2_FIXTURE_MEASUREMENTS_OFFSET + 16, "<Q", checksum ^ 1),
                (SCHEMA2_FIXTURE_MEASUREMENTS_OFFSET + 24, "<Q", 3),
                (SCHEMA2_FIXTURE_MEASUREMENTS_OFFSET + 32, "<Q", 3),
                (SCHEMA2_FIXTURE_MEASUREMENTS_OFFSET + 40, "<Q", 0),
                (SCHEMA2_FIXTURE_MEASUREMENTS_OFFSET + 48, "<Q", checksum ^ 1),
                (SCHEMA2_FIXTURE_MEASUREMENTS_OFFSET + 56, "<Q", checksum ^ 1),
                (SCHEMA2_FIXTURE_MEASUREMENTS_OFFSET + 64, "<Q", 1),
                (SCHEMA2_FIXTURE_PER_QUERY_OFFSET, "<Q", 0),
            )
            for offset, encoding, value in mutations:
                with self.subTest(offset=offset):
                    mutated = bytearray(sidecar)
                    struct.pack_into(encoding, mutated, offset, value)
                    mutated_bytes = bytes(mutated)
                    mutated_fields = refresh_schema2_sidecar_fields(
                        mutated_bytes, fields
                    )
                    with self.assertRaises(d0.D0Failure):
                        parse_schema2_fixture(mutated_bytes, mutated_fields)

    def test_synchronized_checksum_forgery_fails_elementwise_replay(self) -> None:
        with schema2_small_protocol():
            sidecar, fields = schema2_sidecar_fixture()
            forged_per_query = list(reversed(schema2_fixture_per_query_checksums()))
            forged_aggregate = d0.query_benchmark_checksum(forged_per_query)
            mutated = bytearray(sidecar)
            struct.pack_into(
                "<2Q",
                mutated,
                SCHEMA2_FIXTURE_PER_QUERY_OFFSET,
                *forged_per_query,
            )
            for offset in (
                SCHEMA2_FIXTURE_MEASUREMENTS_OFFSET + 16,
                SCHEMA2_FIXTURE_MEASUREMENTS_OFFSET + 48,
                SCHEMA2_FIXTURE_MEASUREMENTS_OFFSET + 56,
            ):
                struct.pack_into("<Q", mutated, offset, forged_aggregate)
            mutated_bytes = bytes(mutated)
            mutated_fields = refresh_schema2_sidecar_fields(
                mutated_bytes, fields
            )
            for name in (
                "infinity_query_latency_validated_result_checksum",
                "infinity_query_throughput_validated_result_checksum",
                "infinity_query_result_checksum",
            ):
                mutated_fields[name] = str(forged_aggregate)
            mutated_fields["infinity_query_per_query_checksums"] = ",".join(
                str(value) for value in forged_per_query
            )

            with self.assertRaisesRegex(
                d0.D0Failure,
                "per-query checksums differ from raw recall results",
            ):
                parse_schema2_fixture(mutated_bytes, mutated_fields)

    def test_synchronized_corpus_forgery_fails_independent_identity(self) -> None:
        with schema2_small_protocol():
            sidecar, fields = schema2_sidecar_fixture()
            forged_sha256 = "66" * 32
            mutated = bytearray(sidecar)
            mutated[
                SCHEMA2_FIXTURE_CORPUS_OFFSET:
                SCHEMA2_FIXTURE_MEASUREMENTS_OFFSET
            ] = bytes.fromhex(forged_sha256)
            mutated_bytes = bytes(mutated)
            mutated_fields = refresh_schema2_sidecar_fields(
                mutated_bytes, fields
            )
            mutated_fields["query_timed_corpus_sha256"] = forged_sha256
            mutated_fields["heldout_queries_sha256"] = forged_sha256

            with self.assertRaisesRegex(
                d0.D0Failure,
                "timed query corpus differs from independent reconstruction",
            ):
                parse_schema2_fixture(mutated_bytes, mutated_fields)

    def test_stdout_replay_mutations_are_rejected(self) -> None:
        with schema2_small_protocol():
            sidecar, fields = schema2_sidecar_fixture()
            mutations = {
                "query_timed_corpus_sha256": "66" * 32,
                "infinity_query_latency_validated_operations": "3",
                "infinity_query_throughput_validated_operations": "3",
                "infinity_query_per_query_checksum_count": "1",
                "infinity_query_per_query_checksums": ",".join(
                    str(value)
                    for value in reversed(schema2_fixture_per_query_checksums())
                ),
            }
            for name, value in mutations.items():
                with self.subTest(name=name):
                    mutated_fields = dict(fields)
                    mutated_fields[name] = value
                    with self.assertRaises(d0.D0Failure):
                        parse_schema2_fixture(sidecar, mutated_fields)

    def test_truncation_and_trailing_bytes_are_rejected(self) -> None:
        with schema2_small_protocol():
            sidecar, fields = schema2_sidecar_fixture()
            for mutated in (sidecar[:-1], sidecar + b"\0"):
                with self.subTest(size=len(mutated)):
                    mutated_fields = refresh_schema2_sidecar_fields(
                        mutated, fields
                    )
                    with self.assertRaises(d0.D0Failure):
                        parse_schema2_fixture(mutated, mutated_fields)


class QueryProtocolTests(unittest.TestCase):
    def test_nearest_rank_uses_ceil_rank_without_interpolation(self) -> None:
        samples = list(range(1, 101))
        self.assertEqual(d0.nearest_rank(samples, 0.50), 50)
        self.assertEqual(d0.nearest_rank(samples, 0.95), 95)
        self.assertEqual(d0.nearest_rank(samples, 0.99), 99)

    def test_query_record_keeps_raw_samples_and_derives_qps_tps(self) -> None:
        samples = list(range(1, d0.QUERY_LATENCY_SAMPLE_COUNT + 1))
        per_query_checksums = fixture_query_checksums()
        result_checksum = d0.query_benchmark_checksum(per_query_checksums)
        record = d0.query_performance_record(
            timed_query_corpus_sha256="d" * 64,
            latency_samples_ns=samples,
            latency_validated_operations=d0.QUERY_LATENCY_SAMPLE_COUNT,
            latency_validated_result_checksum=result_checksum,
            throughput_operations=d0.QUERY_THROUGHPUT_OPERATIONS,
            throughput_validated_operations=d0.QUERY_THROUGHPUT_OPERATIONS,
            throughput_wall_ns=2_000_000,
            throughput_validated_result_checksum=result_checksum,
            result_checksum=result_checksum,
            per_query_checksums=per_query_checksums,
        )

        self.assertIs(record["latency_samples_ns"], samples)
        self.assertEqual(record["timed_query_corpus_sha256"], "d" * 64)
        self.assertEqual(
            record["latency_validated_operations"],
            d0.QUERY_LATENCY_SAMPLE_COUNT,
        )
        self.assertEqual(
            record["throughput_validated_operations"],
            d0.QUERY_THROUGHPUT_OPERATIONS,
        )
        self.assertEqual(record["per_query_checksum_count"], 64)
        self.assertIs(record["per_query_checksums"], per_query_checksums)
        self.assertEqual(record["qps"], 5_120_000.0)
        self.assertEqual(record["tps"], record["qps"])
        self.assertLessEqual(record["p50_ns"], record["p95_ns"])
        self.assertLessEqual(record["p95_ns"], record["p99_ns"])

    def test_driver_rejects_query_count_and_protocol_tampering(self) -> None:
        dataset = {"fnv1a64": 123, "sha256": "a" * 64, "bytes": 456}
        result_checksum = d0.query_benchmark_checksum(fixture_query_checksums())
        cases = (
            (
                f"faiss_query_latency_sample_count={d0.QUERY_LATENCY_SAMPLE_COUNT}\n",
                "faiss_query_latency_sample_count=1\n",
            ),
            (
                f"query_ef_search={d0.QUERY_EF_SEARCH}\n",
                "query_ef_search=32\n",
            ),
            (
                f"query_percentile_method={d0.QUERY_PERCENTILE_METHOD}\n",
                "query_percentile_method=interpolated\n",
            ),
            (
                "query_timed_corpus_sha256=" + "d" * 64 + "\n",
                "query_timed_corpus_sha256=" + "f" * 64 + "\n",
            ),
            (
                "faiss_query_latency_validated_operations="
                f"{d0.QUERY_LATENCY_SAMPLE_COUNT}\n",
                "faiss_query_latency_validated_operations=1\n",
            ),
            (
                "faiss_query_latency_validated_result_checksum="
                f"{result_checksum}\n",
                "faiss_query_latency_validated_result_checksum="
                f"{result_checksum ^ 1}\n",
            ),
            (
                "faiss_query_per_query_checksum_count="
                f"{d0.HELDOUT_QUERY_COUNT}\n",
                "faiss_query_per_query_checksum_count=63\n",
            ),
            (
                "faiss_query_per_query_checksums="
                + ",".join(str(value) for value in fixture_query_checksums())
                + "\n",
                "faiss_query_per_query_checksums="
                + ",".join(
                    str(value)
                    for value in reversed(fixture_query_checksums())
                )
                + "\n",
            ),
        )
        for original, replacement in cases:
            with self.subTest(replacement=replacement):
                with self.assertRaises(d0.D0Failure):
                    d0.parse_driver_result(
                        valid_output("faiss").replace(original, replacement),
                        engine="faiss",
                        participants=d0.MEASURED_PARTICIPANTS,
                        dataset=dataset,
                    )

    def test_driver_rejects_noncanonical_or_overflowing_query_checksums(
        self,
    ) -> None:
        dataset = {"fnv1a64": 123, "sha256": "a" * 64, "bytes": 456}
        valid_csv = ",".join(str(value) for value in fixture_query_checksums())
        cases = (
            "01," + valid_csv.split(",", 1)[1],
            str(1 << 64) + "," + valid_csv.split(",", 1)[1],
            valid_csv + ",1",
        )
        for replacement in cases:
            with self.subTest(replacement=replacement[:32]):
                mutated = valid_output("faiss").replace(
                    f"faiss_query_per_query_checksums={valid_csv}\n",
                    f"faiss_query_per_query_checksums={replacement}\n",
                )
                with self.assertRaises(d0.D0Failure):
                    d0.parse_driver_result(
                        mutated,
                        engine="faiss",
                        participants=d0.MEASURED_PARTICIPANTS,
                        dataset=dataset,
                    )

    def test_bridge_phase_order_keeps_query_work_after_ingestion_timer(self) -> None:
        source_directory = (
            Path(d0.__file__).resolve().parent / "native_hnsw_smoke"
        )
        for filename in ("infinity_hnsw_bridge.cpp", "faiss_hnsw_bridge.cpp"):
            with self.subTest(filename=filename):
                source = (source_directory / filename).read_text(encoding="utf-8")
                positions = [
                    source.index("const auto insert_end"),
                    source.index("result->cold_build_ns"),
                    source.index("GenerateHnswD0HeldOutQueries"),
                    source.index("BenchmarkHnswD0Queries"),
                    source.index("result->self_recall_at_1"),
                    source.index("AuditHnswD0Recall"),
                    source.index("AuditHnswD0Graph"),
                ]
                self.assertEqual(positions, sorted(positions))
                self.assertIn(
                    "insert_end - cold_begin",
                    source,
                )


class ProductionSourceInvariantTests(unittest.TestCase):
    def test_mutable_hnsw_distance_has_no_production_callers(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        algorithm_path = (
            repository
            / "src/storage/knn_index/knn_hnsw/hnsw_alg.cppm"
        )
        algorithm_source = algorithm_path.read_text(encoding="utf-8")
        self.assertEqual(
            len(
                re.findall(
                    r"\bDistance\s*&\s*distance\s*\(\s*\)",
                    algorithm_source,
                )
            ),
            1,
            "The mutable KnnHnsw distance accessor changed; update this guard",
        )

        member_call = re.compile(r"(?:->|\.)\s*distance\s*\(")
        member_pointer = re.compile(
            r"&[^\n;{}]*\bKnnHnsw\b[^\n;{}]*::\s*distance\b"
        )
        source_suffixes = {".c", ".cc", ".cpp", ".cppm", ".cxx", ".h", ".hpp"}
        offenders: list[str] = []
        for path in sorted((repository / "src").rglob("*")):
            if (
                not path.is_file()
                or path.suffix not in source_suffixes
                or "unit_test" in path.parts
            ):
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if member_call.search(line) or member_pointer.search(line):
                    offenders.append(
                        f"{path.relative_to(repository)}:{line_number}:"
                        f"{line.strip()}"
                    )

        self.assertEqual(
            offenders,
            [],
            "Production code must not obtain mutable distance state:\n"
            + "\n".join(offenders),
        )


class HostParsingTests(unittest.TestCase):
    def test_battery_percentage_is_strictly_parsed(self) -> None:
        self.assertEqual(
            d0.parse_battery_percent(
                " -InternalBattery-0 (id=1)\t88%; discharging; present: true"
            ),
            88,
        )
        for invalid in ("no battery", " -InternalBattery-0\t101%; charging"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(d0.D0Failure):
                    d0.parse_battery_percent(invalid)

    def test_power_source_and_active_profile_are_selected(self) -> None:
        self.assertEqual(
            d0.parse_power_source("Now drawing from 'Battery Power'"),
            "battery",
        )
        profiles = (
            "Battery Power:\n"
            " lowpowermode 0\n"
            " sleep 1\n"
            "AC Power:\n"
            " lowpowermode 1\n"
            " sleep 0\n"
        )
        self.assertIn("lowpowermode 0", d0.power_profile(profiles, "battery"))
        self.assertIn("lowpowermode 1", d0.power_profile(profiles, "ac"))
        self.assertEqual(d0.POWER_POLICY, "source-record-only")
        self.assertEqual(d0.BATTERY_MINIMUM_PERCENT, 20)

    def test_process_swap_counter_is_parsed(self) -> None:
        self.assertEqual(d0.parse_process_swaps("          0  swaps\n"), 0)
        self.assertEqual(d0.parse_process_swaps("          2  swaps\n"), 2)
        self.assertIsNone(d0.parse_process_swaps("no time counters"))


def passing_idle(
    *,
    include_failed_attempt: bool = False,
    minimum_idle_percent: float = 95.0,
    observed_idle_percent: float = 96.0,
) -> dict:
    attempts = []
    if include_failed_attempt:
        attempts.append(
            {
                "attempt": 1,
                "status": "fail",
                "minimum_idle_percent": 90.0,
                "covered_seconds": 15,
            }
        )
    attempts.append(
        {
            "attempt": len(attempts) + 1,
            "status": "pass",
            "minimum_idle_percent": observed_idle_percent,
            "covered_seconds": 15,
        }
    )
    return {
        "status": "pass",
        "accepted_at_monotonic_ns": d0.time.monotonic_ns(),
        "accepted_at_unix_ns": d0.time.time_ns(),
        "required_minimum_idle_percent": minimum_idle_percent,
        "required_window_seconds": 15,
        "accepted_attempt": attempts[-1]["attempt"],
        "attempts": attempts,
    }


def summary_records(dataset: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    per_query_checksums = fixture_query_checksums()
    result_checksum = d0.query_benchmark_checksum(per_query_checksums)
    for entry in d0.schedule():
        engine = str(entry["engine"])
        participants = int(entry["participants"])
        fields = d0.parse_driver_result(
            valid_output(engine, participants=participants),
            engine=engine,
            participants=participants,
            dataset=dataset,
        )
        if entry["treatment"] == "measured":
            fields[f"{engine}_cold_build_ns"] = "100" if engine == "infinity" else "200"
            fields[f"{engine}_insert_call_ns"] = "100"
            fields[f"{engine}_query_throughput_wall_ns"] = (
                "1000000" if engine == "infinity" else "2000000"
            )
        query_wall_ns = int(fields[f"{engine}_query_throughput_wall_ns"])
        latency_ns = 100 if engine == "infinity" else 200
        fingerprint = {
            "sha256": dataset["sha256"],
            "fnv1a64": dataset["fnv1a64"],
        }
        records.append(
            {
                **entry,
                "status": "pass",
                "started_at_unix_ns": entry["sequence"] * 2 + 1,
                "ended_at_unix_ns": entry["sequence"] * 2 + 2,
                "fields": fields,
                "idle": passing_idle(include_failed_attempt=True),
                "idle_to_launch_ns": 1,
                "process_swaps": 0,
                "host_before": {
                    "power_source": "battery",
                    "power_policy": "source-record-only",
                    "battery_percent": 80,
                    "battery_minimum_percent": 20,
                    "low_power_mode": False,
                    "thermal_nominal": True,
                    "swap": {"used_bytes": 0},
                },
                "host_after": {
                    "power_source": "battery",
                    "battery_percent": 80,
                    "battery_minimum_percent": 20,
                    "low_power_mode": False,
                    "thermal_nominal": True,
                    "swap": {"used_bytes": 0},
                },
                "global_swap_growth_bytes": 0,
                "dataset_before": fingerprint,
                "dataset_after": fingerprint,
                "audit": {
                    "schema_version": d0.AUDIT_SIDECAR_SCHEMA_VERSION,
                    "query": d0.query_performance_record(
                        timed_query_corpus_sha256="d" * 64,
                        latency_samples_ns=(
                            [latency_ns] * d0.QUERY_LATENCY_SAMPLE_COUNT
                        ),
                        latency_validated_operations=(
                            d0.QUERY_LATENCY_SAMPLE_COUNT
                        ),
                        latency_validated_result_checksum=result_checksum,
                        throughput_operations=d0.QUERY_THROUGHPUT_OPERATIONS,
                        throughput_validated_operations=(
                            d0.QUERY_THROUGHPUT_OPERATIONS
                        ),
                        throughput_wall_ns=query_wall_ns,
                        throughput_validated_result_checksum=result_checksum,
                        result_checksum=result_checksum,
                        per_query_checksums=list(per_query_checksums),
                    ),
                    "graph": {
                        "valid": True,
                        "vertex_count": d0.VECTORS,
                        "reachable_count": d0.VECTORS,
                        "directed_edges": 2 * d0.VECTORS,
                        "level0_directed_edges": 2 * d0.VECTORS,
                        "max_level": 0,
                        "entry_point": 0,
                        "level0_capacity": 2 * d0.M,
                        "upper_capacity": d0.M,
                        "levels_sha256": "f" * 64,
                        "graph_sha256": "1" * 64,
                        "level_histogram": f"0:{d0.VECTORS}",
                        "degree_histograms": fields[
                            f"{engine}_graph_degree_histograms"
                        ],
                    },
                    "recall": {
                        "query_count": d0.HELDOUT_QUERY_COUNT,
                        "query_seed": d0.HELDOUT_QUERY_SEED,
                        "queries_sha256": "d" * 64,
                        "truth_sha256": "e" * 64,
                        "points": [
                            {
                                "k": k,
                                "ef": ef,
                                "eligible_hits": d0.HELDOUT_QUERY_COUNT * k,
                                "possible_hits": d0.HELDOUT_QUERY_COUNT * k,
                                "recall": 1.0,
                            }
                            for k, ef in d0.RECALL_POINTS
                        ],
                    },
                },
            }
        )
    return records


def set_recall_hits(point: dict[str, object], eligible_hits: int) -> None:
    possible_hits = int(point["possible_hits"])
    point["eligible_hits"] = eligible_hits
    point["recall"] = eligible_hits / float(possible_hits)


class ScheduleAndSummaryTests(unittest.TestCase):
    dataset = {
        "sha256": "a" * 64,
        "fnv1a64": 123,
        "bytes": 456,
    }

    def test_schedule_is_exact_and_balanced(self) -> None:
        entries = d0.schedule()
        self.assertEqual(len(entries), 18)
        self.assertEqual([entry["sequence"] for entry in entries], list(range(18)))
        measured_orders = [
            entry["pair_order"]
            for entry in entries
            if entry["treatment"] == "measured" and entry["engine"] == "infinity"
        ]
        self.assertEqual(measured_orders.count("infinity/faiss"), 3)
        self.assertEqual(measured_orders.count("faiss/infinity"), 3)

    def test_schedule_rejects_reordering(self) -> None:
        entries = d0.schedule()
        entries[0], entries[1] = entries[1], entries[0]
        with self.assertRaisesRegex(d0.D0Failure, "frozen D0 member order"):
            d0.validate_schedule(entries)

    def test_summary_has_known_ratio_and_accepts_final_idle_attempt(self) -> None:
        records = summary_records(self.dataset)
        summary = d0.summarize(
            records,
            dataset=self.dataset,
            schedule_sha256="b" * 64,
        )
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(
            summary["ingestion"]["stratified_geometric_mean_ratio"],
            2.0,
        )
        self.assertEqual(
            summary["query"]["stratified_geometric_mean_qps_ratio"],
            2.0,
        )
        self.assertTrue(summary["acceptance"]["idle_protocol_met"])

    def test_summary_accepts_exact_recall_deficit_boundary(self) -> None:
        records = summary_records(self.dataset)
        for record in records:
            if (
                record["treatment"] == "measured"
                and record["pair"] == "P1"
                and record["engine"] == "infinity"
            ):
                point = next(
                    point
                    for point in record["audit"]["recall"]["points"]
                    if point["k"] == 100 and point["ef"] == 128
                )
                set_recall_hits(point, int(point["possible_hits"]) - 32)

        summary = d0.summarize(
            records,
            dataset=self.dataset,
            schedule_sha256="b" * 64,
        )

        self.assertEqual(summary["status"], "pass")
        self.assertTrue(
            summary["acceptance"]["paired_recall_deficit_at_most_0_005"]
        )
        point_summary = next(
            point
            for point in summary["quality"]["recall"]["points"]
            if point["k"] == 100 and point["ef"] == 128
        )
        self.assertEqual(
            point_summary["maximum_infinity_deficit_exact"],
            {"numerator": 1, "denominator": 200},
        )

    def test_summary_rejects_recall_deficit_beyond_boundary(self) -> None:
        records = summary_records(self.dataset)
        for record in records:
            if (
                record["treatment"] == "measured"
                and record["pair"] == "P1"
                and record["engine"] == "infinity"
            ):
                point = next(
                    point
                    for point in record["audit"]["recall"]["points"]
                    if point["k"] == 100 and point["ef"] == 128
                )
                set_recall_hits(point, int(point["possible_hits"]) - 33)

        summary = d0.summarize(
            records,
            dataset=self.dataset,
            schedule_sha256="b" * 64,
        )

        self.assertEqual(summary["status"], "fail")
        self.assertFalse(
            summary["acceptance"]["paired_recall_deficit_at_most_0_005"]
        )

    def test_recall_hit_count_records_are_strict(self) -> None:
        mutations = {
            "boolean hits": lambda point: point.update({"eligible_hits": True}),
            "negative hits": lambda point: point.update({"eligible_hits": -1}),
            "too many hits": lambda point: point.update(
                {"eligible_hits": int(point["possible_hits"]) + 1}
            ),
            "wrong denominator": lambda point: point.update(
                {"possible_hits": int(point["possible_hits"]) + 1}
            ),
            "float mismatch": lambda point: point.update({"recall": 0.5}),
            "extra key": lambda point: point.update({"unexpected": 1}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                record = summary_records(self.dataset)[0]
                point = record["audit"]["recall"]["points"][0]
                mutate(point)
                with self.assertRaises(d0.D0Failure):
                    d0.recall_point_fraction(
                        record,
                        int(point["k"]),
                        int(point["ef"]),
                    )

    def test_summary_rejects_query_recall_below_floor(self) -> None:
        records = summary_records(self.dataset)
        for record in records:
            if record["treatment"] == "measured" and record["pair"] == "P1":
                point = next(
                    point
                    for point in record["audit"]["recall"]["points"]
                    if (point["k"], point["ef"])
                    == (d0.QUERY_K, d0.QUERY_EF_SEARCH)
                )
                set_recall_hits(point, int(point["possible_hits"]) - 7)

        summary = d0.summarize(
            records,
            dataset=self.dataset,
            schedule_sha256="b" * 64,
        )

        self.assertEqual(summary["status"], "fail")
        self.assertFalse(summary["acceptance"]["query_recall_floor_met"])

    def test_summary_uses_absolute_query_recall_gap(self) -> None:
        records = summary_records(self.dataset)
        target = next(
            record
            for record in records
            if record["treatment"] == "measured"
            and record["pair"] == "P1"
            and record["engine"] == "faiss"
        )
        point = next(
            point
            for point in target["audit"]["recall"]["points"]
            if (point["k"], point["ef"])
            == (d0.QUERY_K, d0.QUERY_EF_SEARCH)
        )
        set_recall_hits(point, int(point["possible_hits"]) - 4)

        summary = d0.summarize(
            records,
            dataset=self.dataset,
            schedule_sha256="b" * 64,
        )

        self.assertEqual(summary["status"], "fail")
        self.assertTrue(summary["acceptance"]["query_recall_floor_met"])
        self.assertFalse(
            summary["acceptance"]["query_absolute_recall_gap_at_most_0_005"]
        )
        self.assertEqual(
            summary["query"]["pairs"][0]["absolute_recall_gap_exact"],
            {"numerator": 1, "denominator": 160},
        )

    def test_power_source_transition_is_recorded_not_rejected(self) -> None:
        records = summary_records(self.dataset)
        records[0]["host_after"]["power_source"] = "ac"
        summary = d0.summarize(
            records,
            dataset=self.dataset,
            schedule_sha256="b" * 64,
        )
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["power"]["sources"], ["ac", "battery"])

    def test_global_swap_growth_is_recorded_not_rejected(self) -> None:
        records = summary_records(self.dataset)
        records[0]["host_after"]["swap"]["used_bytes"] = 4096
        records[0]["global_swap_growth_bytes"] = 4096
        summary = d0.summarize(
            records,
            dataset=self.dataset,
            schedule_sha256="b" * 64,
        )
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(
            summary["resources"]["maximum_observed_global_swap_growth_bytes"],
            4096,
        )

    def test_idle_acceptance_uses_only_final_attempt(self) -> None:
        idle = passing_idle(include_failed_attempt=True)
        self.assertTrue(d0.idle_record_passes(idle))
        idle["attempts"].append(
            {
                "attempt": 3,
                "status": "fail",
                "minimum_idle_percent": 94.0,
                "covered_seconds": 15,
            }
        )
        self.assertFalse(d0.idle_record_passes(idle))

    def test_development_idle_override_is_explicitly_validated(self) -> None:
        idle = passing_idle(
            minimum_idle_percent=25.0,
            observed_idle_percent=26.0,
        )
        self.assertTrue(
            d0.idle_record_passes(
                idle,
                expected_minimum_idle_percent=25.0,
            )
        )
        self.assertFalse(d0.idle_record_passes(idle))
        self.assertEqual(d0.idle_policy(25.0), "development-only-override")

    def test_development_idle_override_boundaries_are_strict(self) -> None:
        for accepted in (25.0, 60.0, 95.0):
            with self.subTest(accepted=accepted):
                self.assertEqual(
                    d0.validate_idle_minimum_percent(accepted),
                    accepted,
                )
        for rejected in (24.9, 95.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(rejected=rejected):
                with self.assertRaisesRegex(
                    d0.D0Failure,
                    "must be finite and between",
                ):
                    d0.validate_idle_minimum_percent(rejected)

    def test_cli_defaults_to_95_percent_idle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            infinity_binary = root / "infinity"
            faiss_binary = root / "faiss"
            infinity_binary.write_bytes(b"infinity")
            faiss_binary.write_bytes(b"faiss")
            args = d0.parse_args(
                [
                    "--repo",
                    str(root),
                    "--infinity-binary",
                    str(infinity_binary),
                    "--faiss-binary",
                    str(faiss_binary),
                    "--output-dir",
                    str(root / "evidence"),
                ]
            )
        self.assertEqual(args.idle_minimum_percent, 95.0)
        self.assertIs(args.preflight_only, False)

    def test_cli_accepts_preflight_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            infinity_binary = root / "infinity"
            faiss_binary = root / "faiss"
            infinity_binary.write_bytes(b"infinity")
            faiss_binary.write_bytes(b"faiss")

            args = d0.parse_args(
                [
                    "--repo",
                    str(root),
                    "--infinity-binary",
                    str(infinity_binary),
                    "--faiss-binary",
                    str(faiss_binary),
                    "--output-dir",
                    str(root / "evidence"),
                    "--preflight-only",
                ]
            )

        self.assertIs(args.preflight_only, True)

    def test_summary_rejects_missing_or_duplicate_member(self) -> None:
        records = summary_records(self.dataset)
        with self.assertRaisesRegex(d0.D0Failure, "full schedule"):
            d0.summarize(
                records[:-1],
                dataset=self.dataset,
                schedule_sha256="b" * 64,
            )
        records[1] = records[0]
        with self.assertRaisesRegex(d0.D0Failure, "scheduled member"):
            d0.summarize(
                records,
                dataset=self.dataset,
                schedule_sha256="b" * 64,
            )

    def test_summary_rejects_overlapping_members(self) -> None:
        records = summary_records(self.dataset)
        records[1]["started_at_unix_ns"] = records[0]["ended_at_unix_ns"] - 1
        with self.assertRaisesRegex(d0.D0Failure, "overlaps or predates"):
            d0.summarize(
                records,
                dataset=self.dataset,
                schedule_sha256="b" * 64,
            )

    def test_failed_gate_is_a_completed_campaign(self) -> None:
        d0.validate_completed_campaign({"status": "fail"})
        d0.validate_completed_campaign({"status": "pass"})
        with self.assertRaisesRegex(d0.D0Failure, "summary status"):
            d0.validate_completed_campaign({"status": "interrupted"})


class MemberOrderingTests(unittest.TestCase):
    def test_unsafe_parent_environment_policy_is_fail_closed(self) -> None:
        for name in sorted(d0.UNSAFE_PARENT_ENVIRONMENT_NAMES):
            with self.subTest(name=name):
                self.assertEqual(
                    d0.unsafe_parent_environment_names({name: "injected"}),
                    [name],
                )
        for prefix in d0.UNSAFE_PARENT_ENVIRONMENT_PREFIXES:
            name = prefix + "INJECTED"
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    d0.unsafe_parent_environment_names({name: "injected"}),
                    [name],
                )
        self.assertEqual(
            d0.unsafe_parent_environment_names(
                {
                    "GIT_PAGER": "cat",
                    "HOME": "/tmp/home",
                    "PATH": "/tmp/bin",
                }
            ),
            [],
        )
        with self.assertRaisesRegex(
            d0.D0Failure,
            "unsafe compiler, loader, toolchain, Git, or Python",
        ):
            d0.reject_unsafe_parent_environment({"CFLAGS": "-march=native"})

    def test_executable_resolution_uses_fixed_path_and_rejects_home_expansion(
        self,
    ) -> None:
        with patch.object(
            d0.shutil,
            "which",
            return_value="/usr/bin/true",
        ) as which:
            self.assertEqual(
                d0.resolve_executable("tool"),
                Path("/usr/bin/true"),
            )
        which.assert_called_once_with(
            "tool",
            path=d0.tool_environment()["PATH"],
        )
        with self.assertRaisesRegex(d0.D0Failure, "home expansion"):
            d0.resolve_executable("~/tool")

    def test_main_rejects_parent_environment_before_creating_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence"
            with (
                patch.object(
                    d0,
                    "parse_args",
                    return_value=SimpleNamespace(output_dir=output),
                ),
                patch.object(
                    d0,
                    "reject_unsafe_parent_environment",
                    side_effect=d0.D0Failure("unsafe parent"),
                ),
            ):
                with self.assertRaisesRegex(d0.D0Failure, "unsafe parent"):
                    d0.main()
            self.assertFalse(output.exists())

    def test_child_environment_does_not_inherit_performance_controls(self) -> None:
        poisoned = {
            "DYLD_INSERT_LIBRARIES": "/tmp/injected.dylib",
            "KMP_AFFINITY": "compact",
            "MallocStackLogging": "1",
            "OMP_WAIT_POLICY": "ACTIVE",
        }
        with patch.dict(d0.os.environ, poisoned):
            environment = d0.benchmark_environment(12)

        for name in poisoned:
            self.assertNotIn(name, environment)
        self.assertEqual(environment["OMP_NUM_THREADS"], "12")
        self.assertEqual(environment["OMP_PROC_BIND"], "FALSE")
        self.assertEqual(environment["LC_ALL"], "C")

    def test_expensive_checks_precede_final_idle_gate(self) -> None:
        operations: list[str] = []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "infinity_hnsw_d0"
            binary.write_bytes(b"binary")
            dataset_path = root / "dataset.bin"
            dataset_path.write_bytes(b"dataset")
            dataset = {
                "path": "dataset.bin",
                "sha256": "a" * 64,
                "fnv1a64": 123,
                "bytes": 456,
            }
            args = SimpleNamespace(
                repo=root,
                infinity_binary=binary,
                faiss_binary=root / "faiss_hnsw_d0",
                idle_minimum_percent=95.0,
                idle_timeout_seconds=15.0,
                member_timeout_seconds=30.0,
            )

            def verified(*args: object, **kwargs: object) -> dict[str, object]:
                del args, kwargs
                operations.append("dataset")
                return {
                    "sha256": dataset["sha256"],
                    "fnv1a64": dataset["fnv1a64"],
                }

            def hashed(*args: object, **kwargs: object) -> str:
                del args, kwargs
                operations.append("binary")
                return "b" * 64

            def host_checked(*args: object, **kwargs: object) -> dict[str, object]:
                del args, kwargs
                operations.append("host")
                return {
                    "power_source": "battery",
                    "battery_minimum_percent": 20,
                    "swap": {"used_bytes": 0},
                }

            def artifacts_checked(*args: object, **kwargs: object) -> dict[str, object]:
                del args, kwargs
                operations.append("artifacts")
                return {"artifacts": []}

            def idled(*args: object, **kwargs: object) -> dict[str, object]:
                del args, kwargs
                operations.append("idle")
                return passing_idle()

            def launched_and_supervised(
                *args: object,
                **kwargs: object,
            ) -> dict[str, object]:
                del args, kwargs
                operations.append("launch-supervise")
                return {
                    "pid": 1234,
                    "returncode": 0,
                    "stdout": valid_attestation_output(
                        valid_output("infinity", participants=1)
                    ),
                    "stderr": "123 maximum resident set size\n0 swaps\n",
                    "timed_out": False,
                    "barriers": IndexTimingBindingTests.barriers(outer_ns=201),
                    "benchmark_process_evidence": {
                        "schema_version": d0.PROCESS_EVIDENCE_SCHEMA_VERSION
                    },
                    "errors": [],
                }

            with (
                patch.object(d0, "verify_dataset", side_effect=verified),
                patch.object(d0, "sha256", side_effect=hashed),
                patch.object(
                    d0,
                    "capture_member_preflight",
                    side_effect=host_checked,
                ),
                patch.object(
                    d0,
                    "verify_runtime_artifacts",
                    side_effect=artifacts_checked,
                ),
                patch.object(d0, "wait_for_idle", side_effect=idled),
                patch.object(
                    d0,
                    "launch_and_supervise_attested_process",
                    side_effect=launched_and_supervised,
                    create=True,
                ) as launch_and_supervise,
                patch.object(d0.subprocess, "Popen") as popen,
                patch.object(d0, "supervise_attested_process") as supervise,
                patch.object(d0, "read_audit_sidecar", return_value=b"x" * 1000),
                patch.object(
                    d0,
                    "parse_audit_sidecar",
                    return_value={
                        "schema_version": d0.AUDIT_SIDECAR_SCHEMA_VERSION,
                        "graph": {},
                        "recall": {},
                    },
                ),
                patch.object(
                    d0,
                    "capture_member_postflight",
                    return_value={"swap": {"used_bytes": 0}},
                ),
            ):
                d0.run_member(
                    d0.schedule()[0],
                    args=args,
                    output_dir=root,
                    binary_hashes={"infinity": "b" * 64},
                    campaign_binding={
                        "schema_version": d0.CAMPAIGN_BINDING_SCHEMA_VERSION,
                        "campaign_nonce": "c" * 64,
                        "dataset_sha256": "a" * 64,
                        "schedule_sha256": "e" * 64,
                        "roles": {
                            "infinity": {
                                "role_id": 3,
                                "engine": "infinity",
                                "binary_sha256": "b" * 64,
                            }
                        },
                    },
                    runtime_artifact_hashes={
                        str(binary): "b" * 64,
                    },
                    dataset_path=dataset_path,
                    dataset=dataset,
                    truth={
                        "queries_sha256": "d" * 64,
                        "truth_sha256": "e" * 64,
                        "tie_counts_at_10": [10] * d0.HELDOUT_QUERY_COUNT,
                        "tie_counts_at_100": [100] * d0.HELDOUT_QUERY_COUNT,
                    },
                )

        self.assertEqual(
            operations[:6],
            [
                "dataset",
                "binary",
                "artifacts",
                "host",
                "idle",
                "launch-supervise",
            ],
        )
        launch_and_supervise.assert_called_once()
        launch_args, launch_kwargs = launch_and_supervise.call_args
        self.assertEqual(launch_args[0][0:3], ["/usr/bin/time", "-lp", str(binary)])
        self.assertEqual(launch_kwargs["launch_cwd"], root)
        self.assertEqual(
            launch_kwargs["environment"],
            d0.benchmark_environment(1),
        )
        self.assertEqual(launch_kwargs["cwd"], root)
        self.assertEqual(launch_kwargs["timeout_seconds"], 30.0)
        self.assertIn("expected_wrapper_executable", launch_kwargs)
        self.assertIn("expected_benchmark_executable", launch_kwargs)
        popen.assert_not_called()
        supervise.assert_not_called()


class AttestationSupervisorTests(unittest.TestCase):
    WRAPPER_EXECUTABLE = {
        "path": "/usr/bin/time",
        "device": 10,
        "inode": 11,
        "size": 12,
    }
    BENCHMARK_EXECUTABLE = {
        "path": "/usr/bin/python3",
        "device": 20,
        "inode": 21,
        "size": 22,
    }

    @staticmethod
    def mocked_supervision_objects() -> tuple[Mock, Mock, Mock]:
        process = Mock()
        process.pid = 12345
        process.returncode = None
        process.stdout = Mock()
        process.stderr = Mock()
        stdout_thread = Mock()
        stderr_thread = Mock()
        stdout_thread.is_alive.return_value = False
        stderr_thread.is_alive.return_value = False
        return process, stdout_thread, stderr_thread

    @staticmethod
    def publish_terminal(
        child: d0._AttestedChildState,
        *,
        code: int,
        status: int,
    ) -> None:
        d0._publish_attested_terminal(
            child,
            d0._TerminalObservation(
                child.process.pid,
                code,
                status,
            ),
        )

    @staticmethod
    def exception_notes(error: BaseException) -> tuple[str, ...]:
        return (
            *getattr(error, "__notes__", ()),
            *getattr(error, "_infinity_notes", ()),
        )

    def test_cleanup_reporting_lock_and_append_faults_are_noexcept(
        self,
    ) -> None:
        class AcquireThenRaiseRLock:
            def __init__(self) -> None:
                self.owned = False
                self.release_calls = 0

            def acquire(self) -> bool:
                self.owned = True
                raise KeyboardInterrupt("post-effect acquire failure")

            def release(self) -> None:
                self.release_calls += 1
                if not self.owned:
                    raise RuntimeError("lock is not owned")
                self.owned = False

        value = KeyboardInterrupt("cleanup fault")
        append_fault = Mock()
        append_fault.append.side_effect = MemoryError("append failed")
        release_fault = Mock()
        release_fault.acquire.return_value = True
        release_fault.release.side_effect = RuntimeError("release failed")

        d0._append_under_lock_noexcept(
            release_fault,
            append_fault,
            value,
        )

        append_fault.append.assert_called_once_with(value)
        release_fault.release.assert_called_once_with()

        acquire_fault = AcquireThenRaiseRLock()
        untouched = Mock()

        d0._append_under_lock_noexcept(acquire_fault, untouched, value)

        untouched.append.assert_not_called()
        self.assertFalse(acquire_fault.owned)
        self.assertEqual(acquire_fault.release_calls, 1)

    @staticmethod
    def launch(phases: tuple[str, ...], tail_sleep: float = 0.0) -> subprocess.Popen:
        script = (
            "import os, signal, sys, time\n"
            f"phases = {phases!r}\n"
            "for phase in phases:\n"
            "    print('attestation_barrier=' + phase, flush=True)\n"
            "    os.kill(os.getpid(), signal.SIGSTOP)\n"
            f"time.sleep({tail_sleep!r})\n"
            "print('child-complete', flush=True)\n"
        )
        return subprocess.Popen(
            ["/usr/bin/python3", "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

    @staticmethod
    def resume_after_stop(
        process_group: int,
        phase: str,
        *,
        cwd: Path,
        deadline: float,
        supervisor_pid: int,
        expected_wrapper_executable: dict[str, object],
        expected_benchmark_executable: dict[str, object],
    ) -> dict[str, object]:
        del cwd
        if time.monotonic() >= deadline:
            raise d0.D0Failure("test deadline expired")
        time.sleep(0.05)
        os.kill(process_group, signal.SIGCONT)
        now = time.monotonic_ns()
        benchmark_pid = process_group + 100_000
        return {
            "phase": phase,
            "process_group": process_group,
            "stopped_pid": benchmark_pid,
            "observed_at_monotonic_ns": now,
            "resumed_at_monotonic_ns": now,
            "processes": [
                {
                    "pid": process_group,
                    "parent_pid": supervisor_pid,
                    "process_group": process_group,
                    "status_code": 3,
                    "status": "sleeping",
                    "start_time_seconds": 100,
                    "start_time_microseconds": 1,
                    "name": "time",
                    "executable_path": expected_wrapper_executable["path"],
                    "executable_device": expected_wrapper_executable["device"],
                    "executable_inode": expected_wrapper_executable["inode"],
                    "executable_size": expected_wrapper_executable["size"],
                    "child_pids": [benchmark_pid],
                },
                {
                    "pid": benchmark_pid,
                    "parent_pid": process_group,
                    "process_group": process_group,
                    "status_code": 4,
                    "status": "stopped",
                    "start_time_seconds": 101,
                    "start_time_microseconds": 2,
                    "name": "benchmark",
                    "executable_path": expected_benchmark_executable["path"],
                    "executable_device": expected_benchmark_executable["device"],
                    "executable_inode": expected_benchmark_executable["inode"],
                    "executable_size": expected_benchmark_executable["size"],
                    "child_pids": [],
                },
            ],
        }

    def test_supervisor_resumes_exact_ordered_four_barrier_protocol(self) -> None:
        process = self.launch(d0.ATTESTATION_BARRIER_PHASES)
        owner_thread = threading.get_ident()
        resume_threads: list[int] = []

        def resume(*args: object, **kwargs: object) -> dict[str, object]:
            resume_threads.append(threading.get_ident())
            return self.resume_after_stop(*args, **kwargs)

        with patch.object(
            d0,
            "wait_for_attestation_stop",
            side_effect=resume,
        ):
            result = d0.supervise_attested_process(
                process,
                cwd=Path.cwd(),
                timeout_seconds=5.0,
                expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
            )
        self.assertEqual(process.returncode, 0, result)
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(
            [record["phase"] for record in result["barriers"]],
            list(d0.ATTESTATION_BARRIER_PHASES),
        )
        self.assertIn("child-complete\n", result["stdout"])
        self.assertEqual(
            resume_threads,
            [owner_thread] * len(d0.ATTESTATION_BARRIER_PHASES),
        )

    def test_supervisor_kills_out_of_order_or_incomplete_protocol(self) -> None:
        owner_thread = threading.get_ident()
        real_killpg = os.killpg
        for phases in (("after-work",), ("before-work",)):
            with self.subTest(phases=phases):
                process = self.launch(phases, tail_sleep=2.0)
                signal_threads: list[int] = []

                def killpg(process_group: int, signal_number: int) -> None:
                    signal_threads.append(threading.get_ident())
                    real_killpg(process_group, signal_number)

                with (
                    patch.object(
                        d0,
                        "wait_for_attestation_stop",
                        side_effect=self.resume_after_stop,
                    ),
                    patch.object(d0.os, "killpg", side_effect=killpg),
                ):
                    result = d0.supervise_attested_process(
                        process,
                        cwd=Path.cwd(),
                        timeout_seconds=0.5,
                        expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                        expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
                    )
                self.assertNotEqual(process.returncode, 0)
                self.assertTrue(result["errors"])
                self.assertEqual(signal_threads, [owner_thread])

    def test_launch_time_sigint_is_deferred_until_owned_child_is_reaped(
        self,
    ) -> None:
        process, stdout_thread, stderr_thread = self.mocked_supervision_objects()
        events: list[str] = []

        class DeferredKeyboardInterrupt(KeyboardInterrupt):
            instance: DeferredKeyboardInterrupt | None = None

            def __new__(
                cls,
                *args: object,
                **kwargs: object,
            ) -> DeferredKeyboardInterrupt:
                del args, kwargs
                if cls.instance is None:
                    cls.instance = super().__new__(cls)
                return cls.instance

        interrupt = DeferredKeyboardInterrupt("launch-time SIGINT")

        def prior_sigint_handler(
            signal_number: int,
            frame: object,
        ) -> None:
            del signal_number, frame
            raise interrupt

        installed_handlers: dict[int, object] = {
            signal_number: (
                prior_sigint_handler
                if signal_number == signal.SIGINT
                else signal.SIG_DFL
            )
            for signal_number in d0.ATTESTATION_DEFERRED_SIGNALS
        }

        def install_handler(signal_number: int, handler: object) -> object:
            self.assertIn(signal_number, d0.ATTESTATION_DEFERRED_SIGNALS)
            previous = installed_handlers[signal_number]
            installed_handlers[signal_number] = handler
            return previous

        def current_handler(signal_number: int) -> object:
            self.assertIn(signal_number, d0.ATTESTATION_DEFERRED_SIGNALS)
            return installed_handlers[signal_number]

        def forked(
            command: list[str],
            *,
            launch_cwd: Path,
            environment: dict[str, str],
            state: d0._GatedLaunchState,
            deadline: float,
            deferred: d0._DeferredSigintState,
        ) -> Mock:
            del command, launch_cwd, environment, deadline, deferred
            events.append("launch")
            handler = installed_handlers[signal.SIGINT]
            self.assertTrue(callable(handler))
            handler(signal.SIGINT, None)
            handler(signal.SIGINT, None)
            state.process = process
            state.phase = d0._AttestedLaunchPhase.GROUP_VERIFIED
            events.append("launch-returned")
            return process

        def wait_for_terminal(
            child: d0._AttestedChildState,
            *,
            deadline: float,
            cancellation_event: threading.Event | None = None,
            cancellation_action: object | None = None,
            **kwargs: object,
        ) -> bool:
            del deadline, kwargs
            if cancellation_event is not None and cancellation_event.is_set():
                self.assertTrue(callable(cancellation_action))
                cancellation_action()
            self.publish_terminal(
                child,
                code=d0.CLD_KILLED,
                status=signal.SIGKILL,
            )
            events.append("terminal")
            return True

        def killed(process_group: int, signal_number: int) -> None:
            self.assertEqual(process_group, process.pid)
            self.assertEqual(signal_number, signal.SIGKILL)
            events.append("killpg")

        def group_members(process_group: int) -> list[int]:
            self.assertEqual(process_group, process.pid)
            events.append("group")
            return [process.pid]

        def reaped(pid: int, options: int) -> tuple[int, int]:
            self.assertEqual((pid, options), (process.pid, 0))
            events.append("waitpid")
            return process.pid, signal.SIGKILL

        with (
            patch.object(
                d0,
                "KeyboardInterrupt",
                DeferredKeyboardInterrupt,
                create=True,
            ),
            patch.object(
                d0.signal,
                "default_int_handler",
                prior_sigint_handler,
            ),
            patch.object(d0.signal, "getsignal", side_effect=current_handler),
            patch.object(d0.signal, "signal", side_effect=install_handler),
            patch.object(
                d0,
                "_fork_gated_attested_process",
                side_effect=forked,
            ) as gated_fork,
            patch.object(
                d0,
                "_complete_gated_attested_exec",
            ) as complete_exec,
            patch.object(
                d0.threading,
                "Thread",
                side_effect=(stdout_thread, stderr_thread),
            ),
            patch.object(
                d0,
                "_wait_for_attested_terminal",
                side_effect=wait_for_terminal,
            ),
            patch.object(
                d0,
                "process_group_member_pids",
                side_effect=group_members,
            ),
            patch.object(d0.os, "killpg", side_effect=killed) as killpg,
            patch.object(d0.os, "waitpid", side_effect=reaped) as waitpid,
            self.assertRaises(DeferredKeyboardInterrupt) as raised,
        ):
            d0.launch_and_supervise_attested_process(
                ["/usr/bin/time", "-lp", "/usr/bin/true"],
                launch_cwd=Path.cwd(),
                environment={"LC_ALL": "C"},
                cwd=Path("/fixture/repo"),
                timeout_seconds=0.25,
                expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
            )

        self.assertIs(raised.exception, interrupt)
        self.assertIn("launch-returned", events)
        self.assertLess(events.index("launch-returned"), events.index("killpg"))
        self.assertLess(events.index("killpg"), events.index("terminal"))
        self.assertLess(events.index("terminal"), events.index("group"))
        self.assertLess(events.index("group"), events.index("waitpid"))
        self.assertIs(
            installed_handlers[signal.SIGINT],
            prior_sigint_handler,
        )
        for signal_number in d0.ATTESTATION_DEFERRED_SIGNALS:
            if signal_number != signal.SIGINT:
                self.assertEqual(
                    installed_handlers[signal_number],
                    signal.SIG_DFL,
                )
        gated_fork.assert_called_once()
        self.assertEqual(
            gated_fork.call_args.args[0],
            ["/usr/bin/time", "-lp", "/usr/bin/true"],
        )
        self.assertEqual(
            gated_fork.call_args.kwargs["launch_cwd"],
            Path.cwd(),
        )
        self.assertEqual(
            gated_fork.call_args.kwargs["environment"],
            {"LC_ALL": "C"},
        )
        complete_exec.assert_called_once()
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        waitpid.assert_called_once_with(process.pid, 0)
        process.stdout.close.assert_called_once_with()
        process.stderr.close.assert_called_once_with()

    def test_launch_time_sigterm_is_deferred_until_owned_child_is_reaped(
        self,
    ) -> None:
        process, _, _ = self.mocked_supervision_objects()
        installed_handlers: dict[int, object] = {
            signal.SIGINT: d0.signal.default_int_handler,
            signal.SIGTERM: signal.SIG_DFL,
            signal.SIGHUP: signal.SIG_DFL,
            signal.SIGQUIT: signal.SIG_DFL,
        }

        def current_handler(signal_number: int) -> object:
            return installed_handlers[signal_number]

        def install_handler(signal_number: int, handler: object) -> object:
            previous = installed_handlers[signal_number]
            installed_handlers[signal_number] = handler
            return previous

        def forked(
            command: list[str],
            *,
            launch_cwd: Path,
            environment: dict[str, str],
            state: d0._GatedLaunchState,
            deadline: float,
            deferred: d0._DeferredSigintState,
        ) -> Mock:
            del command, launch_cwd, environment, deadline, deferred
            handler = installed_handlers[signal.SIGTERM]
            self.assertTrue(callable(handler))
            handler(signal.SIGTERM, None)
            later_handler = installed_handlers[signal.SIGINT]
            self.assertTrue(callable(later_handler))
            later_handler(signal.SIGINT, None)
            state.process = process
            state.phase = d0._AttestedLaunchPhase.GROUP_VERIFIED
            return process

        def complete_exec(
            state: d0._GatedLaunchState,
            *,
            deadline: float,
            deferred: d0._DeferredSigintState,
            expected_wrapper_executable: dict[str, object],
        ) -> None:
            del state, deadline, expected_wrapper_executable
            self.assertTrue(deferred.requested)
            self.assertEqual(deferred.signal_number, signal.SIGTERM)
            raise d0._deferred_signal_exception(deferred)

        def wait_for_terminal(
            child: d0._AttestedChildState,
            *,
            deadline: float,
            **kwargs: object,
        ) -> bool:
            del deadline, kwargs
            self.publish_terminal(
                child,
                code=d0.CLD_KILLED,
                status=signal.SIGKILL,
            )
            return True

        with (
            patch.object(d0.signal, "getsignal", side_effect=current_handler),
            patch.object(d0.signal, "signal", side_effect=install_handler),
            patch.object(
                d0,
                "_fork_gated_attested_process",
                side_effect=forked,
            ),
            patch.object(
                d0,
                "_complete_gated_attested_exec",
                side_effect=complete_exec,
            ),
            patch.object(
                d0,
                "_wait_for_attested_terminal",
                side_effect=wait_for_terminal,
            ),
            patch.object(
                d0,
                "process_group_member_pids",
                return_value=[process.pid],
            ),
            patch.object(d0.os, "killpg") as killpg,
            patch.object(
                d0.os,
                "waitpid",
                return_value=(process.pid, signal.SIGKILL),
            ) as waitpid,
            self.assertRaises(SystemExit) as raised,
        ):
            d0.launch_and_supervise_attested_process(
                ["/usr/bin/time", "-lp", "/usr/bin/true"],
                launch_cwd=Path.cwd(),
                environment={"LC_ALL": "C"},
                cwd=Path("/fixture/repo"),
                timeout_seconds=0.25,
                expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
            )

        self.assertEqual(raised.exception.code, 128 + signal.SIGTERM)
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        waitpid.assert_called_once_with(process.pid, 0)
        process.stdout.close.assert_called_once_with()
        process.stderr.close.assert_called_once_with()
        self.assertEqual(
            installed_handlers,
            {
                signal.SIGINT: d0.signal.default_int_handler,
                signal.SIGTERM: signal.SIG_DFL,
                signal.SIGHUP: signal.SIG_DFL,
                signal.SIGQUIT: signal.SIG_DFL,
            },
        )

    def test_gated_exec_resets_every_inherited_caught_signal(self) -> None:
        custom_handler = Mock()
        blockable_signals = {
            signal.SIGINT,
            signal.SIGTERM,
            signal.SIGHUP,
            signal.SIGQUIT,
            signal.SIGUSR1,
            signal.SIGUSR2,
            signal.SIGPIPE,
        }
        if hasattr(signal, "SIGXFSZ"):
            blockable_signals.add(signal.SIGXFSZ)

        def current_handler(signal_number: int) -> object:
            if signal_number == signal.SIGUSR1:
                return custom_handler
            if signal_number in {signal.SIGUSR2, signal.SIGPIPE}:
                return signal.SIG_IGN
            return signal.SIG_DFL

        with patch.object(
            d0.signal,
            "getsignal",
            side_effect=current_handler,
        ):
            defaults = d0._gated_exec_default_signal_numbers(
                blockable_signals
            )

        self.assertIn(signal.SIGUSR1, defaults)
        self.assertNotIn(signal.SIGUSR2, defaults)
        self.assertIn(signal.SIGPIPE, defaults)
        for signal_number in d0.ATTESTATION_DEFERRED_SIGNALS:
            self.assertIn(signal_number, defaults)
        if hasattr(signal, "SIGXFSZ"):
            self.assertIn(signal.SIGXFSZ, defaults)

    def test_gated_child_restores_termination_defaults_before_unmask(
        self,
    ) -> None:
        class ChildExit(BaseException):
            pass

        events: list[tuple[object, ...]] = []
        default_signal_numbers = {
            int(signal_number)
            for signal_number in d0.ATTESTATION_DEFERRED_SIGNALS
        }
        default_signal_numbers.update({int(signal.SIGPIPE), int(signal.SIGUSR1)})
        if hasattr(signal, "SIGXFSZ"):
            default_signal_numbers.add(int(signal.SIGXFSZ))

        def write(descriptor: int, payload: bytes) -> int:
            events.append(("write", descriptor, payload))
            return len(payload)

        def install_handler(signal_number: int, handler: object) -> object:
            events.append(("signal", signal_number, handler))
            return signal.SIG_DFL

        def restore_mask(operation: int, mask: object) -> set[signal.Signals]:
            events.append(("mask", operation, mask))
            return set()

        def execve(
            path: str,
            arguments: tuple[str, ...],
            environment: dict[str, str],
        ) -> None:
            events.append(("execve", path, arguments, environment))
            raise OSError(2, "missing")

        with (
            patch.object(d0.os, "close"),
            patch.object(d0.os, "setsid"),
            patch.object(d0.os, "write", side_effect=write),
            patch.object(d0.os, "read", return_value=b"E"),
            patch.object(d0.os, "dup2"),
            patch.object(d0.os, "fchdir"),
            patch.object(d0.signal, "signal", side_effect=install_handler),
            patch.object(
                d0.signal,
                "pthread_sigmask",
                side_effect=restore_mask,
            ),
            patch.object(d0.os, "execve", side_effect=execve),
            patch.object(d0.os, "_exit", side_effect=ChildExit),
            self.assertRaises(ChildExit),
        ):
            d0._run_gated_attested_child(
                command=("/fixture/time", "-lp", "/fixture/benchmark"),
                environment={"LC_ALL": "C"},
                previous_signal_mask={signal.SIGUSR1},
                default_signal_numbers=tuple(sorted(default_signal_numbers)),
                command_descriptor=10,
                status_descriptor=11,
                stdout_descriptor=12,
                stderr_descriptor=13,
                devnull_descriptor=14,
                cwd_descriptor=15,
                parent_descriptors=(16, 17, 18, 19),
            )

        mask_index = next(
            index for index, event in enumerate(events) if event[0] == "mask"
        )
        exec_index = next(
            index for index, event in enumerate(events) if event[0] == "execve"
        )
        for signal_number in default_signal_numbers:
            reset_index = events.index(
                ("signal", signal_number, signal.SIG_DFL)
            )
            self.assertLess(reset_index, mask_index)
        self.assertLess(mask_index, exec_index)

    def test_pre_exec_settlement_never_resignals_after_ambiguous_reap(
        self,
    ) -> None:
        pid = 12345
        terminal = d0._TerminalObservation(
            pid=pid,
            code=d0.CLD_KILLED,
            status=signal.SIGKILL,
        )
        with (
            patch.object(
                d0,
                "_observe_attested_terminal_nonconsuming",
                side_effect=(
                    None,
                    terminal,
                    ChildProcessError("status was consumed"),
                ),
            ) as observe,
            patch.object(
                d0.os,
                "killpg",
            ) as killpg,
            patch.object(
                d0.os,
                "waitpid",
                side_effect=OSError("wait fault after consuming status"),
            ) as waitpid,
            patch.object(d0.time, "sleep") as sleep,
        ):
            d0._settle_pre_exec_gated_child(pid, group_verified=True)

        killpg.assert_called_once_with(pid, signal.SIGKILL)
        waitpid.assert_called_once_with(pid, 0)
        self.assertEqual(observe.call_count, 3)
        sleep.assert_called_once_with(d0.ATTESTATION_STOP_POLL_SECONDS)

    def test_pre_exec_settlement_retries_transient_observation_and_reap(
        self,
    ) -> None:
        pid = 12345
        terminal = d0._TerminalObservation(
            pid=pid,
            code=d0.CLD_KILLED,
            status=signal.SIGKILL,
        )
        with (
            patch.object(
                d0,
                "_observe_attested_terminal_nonconsuming",
                side_effect=(
                    KeyboardInterrupt("observe interrupted"),
                    None,
                    terminal,
                    terminal,
                ),
            ) as observe,
            patch.object(
                d0.os,
                "killpg",
                side_effect=KeyboardInterrupt("kill interrupted"),
            ) as killpg,
            patch.object(
                d0.os,
                "waitpid",
                side_effect=(
                    OSError("wait fault"),
                    (pid, signal.SIGKILL),
                ),
            ) as waitpid,
            patch.object(
                d0.time,
                "sleep",
                side_effect=(KeyboardInterrupt("sleep interrupted"), None),
            ) as sleep,
        ):
            d0._settle_pre_exec_gated_child(pid, group_verified=True)

        killpg.assert_called_once_with(pid, signal.SIGKILL)
        self.assertEqual(waitpid.call_args_list, [call(pid, 0)] * 2)
        self.assertEqual(observe.call_count, 4)
        self.assertEqual(
            sleep.call_args_list,
            [call(d0.ATTESTATION_STOP_POLL_SECONDS)] * 3,
        )

    def test_fork_gate_preserves_primary_when_mask_restore_fails(
        self,
    ) -> None:
        primary = RuntimeError("fork failed")
        restoration = KeyboardInterrupt("mask restore failed")

        with (
            patch.object(d0.threading, "active_count", return_value=1),
            patch.object(d0, "process_thread_count", return_value=1),
            patch.object(d0, "process_child_pids", return_value=[]),
            patch.object(d0.os, "open", side_effect=(10, 11)),
            patch.object(
                d0.os,
                "pipe",
                side_effect=((12, 13), (14, 15), (16, 17), (18, 19)),
            ),
            patch.object(d0.os, "get_inheritable", return_value=False),
            patch.object(
                d0.signal,
                "pthread_sigmask",
                side_effect=(set(), set(), restoration, set()),
            ) as pthread_sigmask,
            patch.object(d0.os, "fork", side_effect=primary),
            patch.object(
                d0.os,
                "waitpid",
                side_effect=ChildProcessError(),
            ) as waitpid,
            patch.object(d0.os, "close") as close,
            patch.object(d0.time, "sleep") as sleep,
            self.assertRaises(RuntimeError) as raised,
        ):
            d0._fork_gated_attested_process(
                ["/usr/bin/time", "-lp", "/usr/bin/true"],
                launch_cwd=Path.cwd(),
                environment={"LC_ALL": "C"},
                state=d0._GatedLaunchState(),
                deadline=time.monotonic() + 1.0,
                deferred=d0._DeferredSigintState(
                    interruption=KeyboardInterrupt()
                ),
            )

        self.assertIs(raised.exception, primary)
        notes = "\n".join(self.exception_notes(raised.exception))
        self.assertIn(type(restoration).__name__, notes)
        self.assertEqual(
            pthread_sigmask.call_args_list[-2:],
            [call(signal.SIG_SETMASK, set())] * 2,
        )
        waitpid.assert_called_once_with(-1, 0)
        sleep.assert_called_once_with(d0.ATTESTATION_STOP_POLL_SECONDS)
        self.assertEqual(
            {entry.args[0] for entry in close.call_args_list},
            set(range(10, 20)),
        )

    def test_fork_gate_restores_snapshot_if_block_call_fails(self) -> None:
        previous_mask = {signal.SIGUSR1}
        block_error = MemoryError("mask result allocation failed")

        with (
            patch.object(d0.threading, "active_count", return_value=1),
            patch.object(d0, "process_thread_count", return_value=1),
            patch.object(d0, "process_child_pids", return_value=[]),
            patch.object(d0.os, "open", side_effect=(10, 11)),
            patch.object(
                d0.os,
                "pipe",
                side_effect=((12, 13), (14, 15), (16, 17), (18, 19)),
            ),
            patch.object(d0.os, "get_inheritable", return_value=False),
            patch.object(
                d0.signal,
                "pthread_sigmask",
                side_effect=(previous_mask, block_error, set()),
            ) as pthread_sigmask,
            patch.object(d0.os, "fork") as fork,
            patch.object(d0.os, "close"),
            self.assertRaises(MemoryError) as raised,
        ):
            d0._fork_gated_attested_process(
                ["/usr/bin/time", "-lp", "/usr/bin/true"],
                launch_cwd=Path.cwd(),
                environment={"LC_ALL": "C"},
                state=d0._GatedLaunchState(),
                deadline=time.monotonic() + 1.0,
                deferred=d0._DeferredSigintState(
                    interruption=KeyboardInterrupt()
                ),
            )

        self.assertIs(raised.exception, block_error)
        fork.assert_not_called()
        self.assertEqual(
            pthread_sigmask.call_args_list[0],
            call(signal.SIG_BLOCK, set()),
        )
        self.assertEqual(
            pthread_sigmask.call_args_list[-1],
            call(signal.SIG_SETMASK, previous_mask),
        )

    def test_fork_gate_restore_failure_settles_owned_child(self) -> None:
        pid = 12345
        restoration = KeyboardInterrupt("mask restore failed")
        stdout = Mock()
        stderr = Mock()
        state = d0._GatedLaunchState()
        events: list[str] = []
        mask_calls = 0

        def update_mask(
            operation: int,
            mask: object,
        ) -> set[signal.Signals]:
            del mask
            nonlocal mask_calls
            mask_calls += 1
            if operation == signal.SIG_SETMASK and mask_calls == 3:
                events.append("restore-failure")
                raise restoration
            if operation == signal.SIG_SETMASK:
                events.append("restore-success")
            return set()

        def settle_child(
            child_pid: int,
            *,
            group_verified: bool,
        ) -> None:
            self.assertEqual(child_pid, pid)
            self.assertTrue(group_verified)
            events.append("settle")

        with (
            patch.object(d0.threading, "active_count", return_value=1),
            patch.object(d0, "process_thread_count", return_value=1),
            patch.object(d0, "process_child_pids", return_value=[]),
            patch.object(d0.os, "open", side_effect=(10, 11)),
            patch.object(
                d0.os,
                "pipe",
                side_effect=((12, 13), (14, 15), (16, 17), (18, 19)),
            ),
            patch.object(d0.os, "get_inheritable", return_value=False),
            patch.object(
                d0.signal,
                "pthread_sigmask",
                side_effect=update_mask,
            ) as pthread_sigmask,
            patch.object(d0.os, "fork", return_value=pid),
            patch.object(d0.os, "fdopen", side_effect=(stdout, stderr)),
            patch.object(d0.os, "getpgid", return_value=pid),
            patch.object(d0.os, "getsid", return_value=pid),
            patch.object(d0.os, "close"),
            patch.object(d0, "_read_gated_launch_byte", return_value=b"H"),
            patch.object(
                d0,
                "_settle_pre_exec_gated_child",
                side_effect=settle_child,
            ) as settle,
            patch.object(d0.time, "sleep") as sleep,
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            d0._fork_gated_attested_process(
                ["/usr/bin/time", "-lp", "/usr/bin/true"],
                launch_cwd=Path.cwd(),
                environment={"LC_ALL": "C"},
                state=state,
                deadline=time.monotonic() + 1.0,
                deferred=d0._DeferredSigintState(
                    interruption=KeyboardInterrupt()
                ),
            )

        self.assertIs(raised.exception, restoration)
        settle.assert_called_once_with(pid, group_verified=True)
        self.assertEqual(
            pthread_sigmask.call_args_list[-2:],
            [call(signal.SIG_SETMASK, set())] * 2,
        )
        self.assertEqual(
            events,
            ["restore-failure", "settle", "restore-success"],
        )
        sleep.assert_called_once_with(d0.ATTESTATION_STOP_POLL_SECONDS)
        stdout.close.assert_called_once_with()
        stderr.close.assert_called_once_with()
        self.assertEqual(state.command_descriptor, -1)
        self.assertEqual(state.status_descriptor, -1)

    def test_fork_gate_settles_ambiguous_post_effect_fork(self) -> None:
        primary = MemoryError("fork result allocation failed")
        child_pid = 54321
        events: list[tuple[str, int]] = []

        def close_descriptor(descriptor: int) -> None:
            events.append(("close", descriptor))

        wait_results: list[object] = [
            (child_pid + 1, 0),
            (child_pid, 125 << 8),
            ChildProcessError(),
        ]

        def reap(pid: int, options: int) -> tuple[int, int]:
            self.assertEqual((pid, options), (-1, 0))
            events.append(("waitpid", pid))
            result = wait_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result

        with (
            patch.object(d0, "_require_waitable_sigchld_policy"),
            patch.object(d0.threading, "active_count", return_value=1),
            patch.object(d0, "process_thread_count", return_value=1),
            patch.object(d0, "process_child_pids", return_value=[]),
            patch.object(d0.os, "open", side_effect=(10, 11)),
            patch.object(
                d0.os,
                "pipe",
                side_effect=((12, 13), (14, 15), (16, 17), (18, 19)),
            ),
            patch.object(d0.os, "get_inheritable", return_value=False),
            patch.object(
                d0.signal,
                "pthread_sigmask",
                side_effect=(set(), set(), set()),
            ),
            patch.object(d0.os, "fork", side_effect=primary),
            patch.object(d0.os, "close", side_effect=close_descriptor),
            patch.object(d0.os, "waitpid", side_effect=reap) as waitpid,
            self.assertRaises(MemoryError) as raised,
        ):
            d0._fork_gated_attested_process(
                ["/usr/bin/time", "-lp", "/usr/bin/true"],
                launch_cwd=Path.cwd(),
                environment={"LC_ALL": "C"},
                state=d0._GatedLaunchState(),
                deadline=time.monotonic() + 1.0,
                deferred=d0._DeferredSigintState(
                    interruption=KeyboardInterrupt()
                ),
            )

        self.assertIs(raised.exception, primary)
        self.assertEqual(waitpid.call_args_list, [call(-1, 0)] * 3)
        self.assertLess(
            events.index(("close", 13)),
            events.index(("waitpid", -1)),
        )
        self.assertEqual(
            {value for operation, value in events if operation == "close"},
            set(range(10, 20)),
        )

    def test_fork_gate_rejects_nonwaitable_sigchld_before_resources(
        self,
    ) -> None:
        with (
            patch.object(
                d0,
                "_darwin_sigchld_action",
                return_value=(int(signal.SIG_DFL), d0.SA_NOCLDWAIT),
            ),
            patch.object(
                d0.signal,
                "getsignal",
                return_value=signal.SIG_DFL,
            ),
            patch.object(d0.os, "open") as open_descriptor,
            patch.object(d0.os, "fork") as fork,
            self.assertRaisesRegex(
                d0.D0Failure,
                "waitable default SIGCHLD",
            ),
        ):
            d0._fork_gated_attested_process(
                ["/usr/bin/time", "-lp", "/usr/bin/true"],
                launch_cwd=Path.cwd(),
                environment={"LC_ALL": "C"},
                state=d0._GatedLaunchState(),
                deadline=time.monotonic() + 1.0,
                deferred=d0._DeferredSigintState(
                    interruption=KeyboardInterrupt()
                ),
            )

        open_descriptor.assert_not_called()
        fork.assert_not_called()

    def test_fork_gate_rejects_uncontrolled_signal_handler_before_resources(
        self,
    ) -> None:
        custom_handler = Mock()
        valid_signals = {
            signal.SIGKILL,
            signal.SIGSTOP,
            signal.SIGUSR1,
            *d0.ATTESTATION_DEFERRED_SIGNALS,
        }

        def current_handler(signal_number: int) -> object:
            if signal_number == signal.SIGUSR1:
                return custom_handler
            return signal.SIG_DFL

        with (
            patch.object(d0, "_require_waitable_sigchld_policy"),
            patch.object(
                d0.signal,
                "valid_signals",
                return_value=valid_signals,
            ),
            patch.object(
                d0.signal,
                "getsignal",
                side_effect=current_handler,
            ),
            patch.object(d0.os, "open") as open_descriptor,
            patch.object(d0.os, "fork") as fork,
            self.assertRaisesRegex(
                d0.D0Failure,
                "uncontrolled signal handlers",
            ),
        ):
            d0._fork_gated_attested_process(
                ["/usr/bin/time", "-lp", "/usr/bin/true"],
                launch_cwd=Path.cwd(),
                environment={"LC_ALL": "C"},
                state=d0._GatedLaunchState(),
                deadline=time.monotonic() + 1.0,
                deferred=d0._DeferredSigintState(
                    interruption=KeyboardInterrupt()
                ),
            )

        open_descriptor.assert_not_called()
        fork.assert_not_called()

    def test_partial_signal_handler_installation_is_restored(self) -> None:
        install_error = RuntimeError("SIGTERM handler install failed")
        original_handlers: dict[int, object] = {
            signal.SIGINT: d0.signal.default_int_handler,
            signal.SIGTERM: signal.SIG_DFL,
            signal.SIGHUP: signal.SIG_DFL,
            signal.SIGQUIT: signal.SIG_DFL,
        }
        installed_handlers = dict(original_handlers)
        installation_attempts = 0

        def current_handler(signal_number: int) -> object:
            return installed_handlers[signal_number]

        def install_handler(signal_number: int, handler: object) -> object:
            nonlocal installation_attempts
            installation_attempts += 1
            if installation_attempts == 2:
                raise install_error
            previous = installed_handlers[signal_number]
            installed_handlers[signal_number] = handler
            return previous

        with (
            patch.object(d0.signal, "getsignal", side_effect=current_handler),
            patch.object(d0.signal, "signal", side_effect=install_handler),
            patch.object(d0, "_fork_gated_attested_process") as gated_fork,
            self.assertRaises(RuntimeError) as raised,
        ):
            d0.launch_and_supervise_attested_process(
                ["/usr/bin/time", "-lp", "/usr/bin/true"],
                launch_cwd=Path.cwd(),
                environment={"LC_ALL": "C"},
                cwd=Path("/fixture/repo"),
                timeout_seconds=0.25,
                expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
            )

        self.assertIs(raised.exception, install_error)
        self.assertEqual(installed_handlers, original_handlers)
        gated_fork.assert_not_called()

    def test_fork_gate_rejects_native_threads_before_resources(self) -> None:
        with (
            patch.object(d0.threading, "active_count", return_value=1),
            patch.object(d0, "process_thread_count", return_value=2),
            patch.object(d0.os, "open") as open_descriptor,
            patch.object(d0.os, "fork") as fork,
            self.assertRaisesRegex(
                d0.D0Failure,
                "process-wide single-threaded",
            ),
        ):
            d0._fork_gated_attested_process(
                ["/usr/bin/time", "-lp", "/usr/bin/true"],
                launch_cwd=Path.cwd(),
                environment={"LC_ALL": "C"},
                state=d0._GatedLaunchState(),
                deadline=time.monotonic() + 1.0,
                deferred=d0._DeferredSigintState(
                    interruption=KeyboardInterrupt()
                ),
            )
        open_descriptor.assert_not_called()
        fork.assert_not_called()
        self.assertNotIn("__del__", d0._GatedTextProcess.__dict__)

    def test_prequeued_sigint_never_releases_exec_gate(self) -> None:
        process, _, _ = self.mocked_supervision_objects()
        state = d0._GatedLaunchState(
            phase=d0._AttestedLaunchPhase.GROUP_VERIFIED,
            process=process,
            command_descriptor=10,
            status_descriptor=11,
        )
        interrupt = KeyboardInterrupt("prequeued")
        deferred = d0._DeferredSigintState(
            interruption=interrupt,
            request=(signal.SIGINT, interrupt),
        )
        with (
            patch.object(d0.os, "write") as write,
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            d0._complete_gated_attested_exec(
                state,
                deadline=time.monotonic() + 1.0,
                deferred=deferred,
                expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
            )
        self.assertIs(raised.exception, interrupt)
        write.assert_not_called()
        self.assertEqual(
            state.phase,
            d0._AttestedLaunchPhase.GROUP_VERIFIED,
        )

    def test_reap_requires_exact_leader_only_quiescence(self) -> None:
        process, _, _ = self.mocked_supervision_objects()
        cases = (
            ("missing", None),
            ("empty", []),
            ("extra-member", [process.pid, process.pid + 1]),
        )
        for label, members in cases:
            with self.subTest(label=label):
                child = d0._AttestedChildState(process=process)
                self.publish_terminal(
                    child,
                    code=d0.CLD_EXITED,
                    status=0,
                )
                child.group_state = d0._AttestedGroupState.QUIESCENT
                child.pre_reap_member_pids = members
                child.pre_reap_observed_at_monotonic_ns = time.monotonic_ns()
                with (
                    patch.object(d0.os, "waitpid") as waitpid,
                    self.assertRaisesRegex(
                        d0.D0Failure,
                        "(?i)(leader|quiescen)",
                    ),
                ):
                    d0._reap_attested_child(child)
                waitpid.assert_not_called()
                self.assertEqual(
                    child.lifecycle,
                    d0._AttestedLifecycle.TERMINAL_OBSERVED,
                )

    def test_exec_confirmation_stall_is_settled_before_reraise(self) -> None:
        process, stdout_thread, stderr_thread = self.mocked_supervision_objects()
        launch_error = TimeoutError("exec confirmation stalled")

        def forked(
            command: list[str],
            *,
            launch_cwd: Path,
            environment: dict[str, str],
            state: d0._GatedLaunchState,
            deadline: float,
            deferred: d0._DeferredSigintState,
        ) -> Mock:
            del command, launch_cwd, environment, deadline, deferred
            state.process = process
            state.phase = d0._AttestedLaunchPhase.GROUP_VERIFIED
            return process

        def wait_for_terminal(
            child: d0._AttestedChildState,
            *,
            deadline: float,
            **kwargs: object,
        ) -> bool:
            del deadline, kwargs
            self.publish_terminal(
                child,
                code=d0.CLD_KILLED,
                status=signal.SIGKILL,
            )
            return True

        with (
            patch.object(
                d0,
                "_fork_gated_attested_process",
                side_effect=forked,
            ),
            patch.object(
                d0,
                "_complete_gated_attested_exec",
                side_effect=launch_error,
            ),
            patch.object(
                d0.threading,
                "Thread",
                side_effect=(stdout_thread, stderr_thread),
            ),
            patch.object(
                d0,
                "_wait_for_attested_terminal",
                side_effect=wait_for_terminal,
            ),
            patch.object(
                d0,
                "process_group_member_pids",
                return_value=[process.pid],
            ),
            patch.object(d0.os, "killpg") as killpg,
            patch.object(
                d0.os,
                "waitpid",
                return_value=(process.pid, signal.SIGKILL),
            ) as waitpid,
            self.assertRaises(TimeoutError) as raised,
        ):
            d0.launch_and_supervise_attested_process(
                ["/usr/bin/time", "-lp", "/usr/bin/true"],
                launch_cwd=Path.cwd(),
                environment={"LC_ALL": "C"},
                cwd=Path("/fixture/repo"),
                timeout_seconds=0.25,
                expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
            )

        self.assertIs(raised.exception, launch_error)
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        waitpid.assert_called_once_with(process.pid, 0)
        self.assertEqual(process.returncode, -signal.SIGKILL)

    def test_supervisor_setup_fault_retries_under_production_ownership(
        self,
    ) -> None:
        setup_error = MemoryError("supervisor setup fault")
        process = d0._GatedTextProcess(
            pid=12345,
            stdout=Mock(),
            stderr=Mock(),
        )
        original_supervise_once = d0._supervise_attested_process_once
        attempts = 0

        def fault_then_supervise(*args: object, **kwargs: object) -> object:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise setup_error
            return original_supervise_once(*args, **kwargs)

        def wait_for_terminal(
            child: d0._AttestedChildState,
            *,
            deadline: float,
            **kwargs: object,
        ) -> bool:
            del deadline, kwargs
            self.publish_terminal(
                child,
                code=d0.CLD_KILLED,
                status=signal.SIGKILL,
            )
            return True

        with (
            patch.object(
                d0,
                "_supervise_attested_process_once",
                side_effect=fault_then_supervise,
            ),
            patch.object(
                d0,
                "_wait_for_attested_terminal",
                side_effect=wait_for_terminal,
            ),
            patch.object(
                d0,
                "process_group_member_pids",
                return_value=[process.pid],
            ),
            patch.object(d0.os, "killpg") as killpg,
            patch.object(
                d0.os,
                "waitpid",
                return_value=(process.pid, signal.SIGKILL),
            ) as waitpid,
            patch.object(d0.time, "sleep") as sleep,
            self.assertRaises(MemoryError) as raised,
        ):
            d0.supervise_attested_process(
                process,
                cwd=Path.cwd(),
                timeout_seconds=0.25,
                expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
            )

        self.assertIs(raised.exception, setup_error)
        self.assertEqual(attempts, 2)
        sleep.assert_called_once_with(d0.ATTESTATION_STOP_POLL_SECONDS)
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        waitpid.assert_called_once_with(process.pid, 0)
        self.assertEqual(process.returncode, -signal.SIGKILL)

    def test_launcher_gate_close_fault_settles_owned_child_before_reraise(
        self,
    ) -> None:
        control_error = KeyboardInterrupt("launch control close failed")
        process = d0._GatedTextProcess(
            pid=12345,
            stdout=Mock(),
            stderr=Mock(),
        )

        def forked(
            command: list[str],
            *,
            launch_cwd: Path,
            environment: dict[str, str],
            state: d0._GatedLaunchState,
            deadline: float,
            deferred: d0._DeferredSigintState,
        ) -> d0._GatedTextProcess:
            del command, launch_cwd, environment, deadline, deferred
            state.process = process
            state.phase = d0._AttestedLaunchPhase.GROUP_VERIFIED
            return process

        def wait_for_terminal(
            child: d0._AttestedChildState,
            *,
            deadline: float,
            **kwargs: object,
        ) -> bool:
            del deadline, kwargs
            self.publish_terminal(
                child,
                code=d0.CLD_KILLED,
                status=signal.SIGKILL,
            )
            return True

        with (
            patch.object(
                d0,
                "_fork_gated_attested_process",
                side_effect=forked,
            ),
            patch.object(d0, "_complete_gated_attested_exec"),
            patch.object(
                d0,
                "_close_gated_launch_control",
                side_effect=(control_error, None),
            ) as close_control,
            patch.object(
                d0,
                "_wait_for_attested_terminal",
                side_effect=wait_for_terminal,
            ),
            patch.object(
                d0,
                "process_group_member_pids",
                return_value=[process.pid],
            ),
            patch.object(d0.os, "killpg") as killpg,
            patch.object(
                d0.os,
                "waitpid",
                return_value=(process.pid, signal.SIGKILL),
            ) as waitpid,
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            d0.launch_and_supervise_attested_process(
                ["/usr/bin/time", "-lp", "/usr/bin/true"],
                launch_cwd=Path.cwd(),
                environment={"LC_ALL": "C"},
                cwd=Path("/fixture/repo"),
                timeout_seconds=0.25,
                expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
            )

        self.assertIs(raised.exception, control_error)
        self.assertEqual(close_control.call_count, 2)
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        waitpid.assert_called_once_with(process.pid, 0)
        self.assertEqual(process.returncode, -signal.SIGKILL)

    def test_launcher_recovers_process_published_before_fork_helper_raises(
        self,
    ) -> None:
        handoff_error = MemoryError("fork helper handoff failed")
        process = d0._GatedTextProcess(
            pid=12345,
            stdout=Mock(),
            stderr=Mock(),
        )

        def publish_then_raise(
            command: list[str],
            *,
            launch_cwd: Path,
            environment: dict[str, str],
            state: d0._GatedLaunchState,
            deadline: float,
            deferred: d0._DeferredSigintState,
        ) -> d0._GatedTextProcess:
            del command, launch_cwd, environment, deadline, deferred
            state.process = process
            state.phase = d0._AttestedLaunchPhase.GROUP_VERIFIED
            raise handoff_error

        def wait_for_terminal(
            child: d0._AttestedChildState,
            *,
            deadline: float,
            **kwargs: object,
        ) -> bool:
            del deadline, kwargs
            self.publish_terminal(
                child,
                code=d0.CLD_KILLED,
                status=signal.SIGKILL,
            )
            return True

        with (
            patch.object(
                d0,
                "_fork_gated_attested_process",
                side_effect=publish_then_raise,
            ),
            patch.object(
                d0,
                "_wait_for_attested_terminal",
                side_effect=wait_for_terminal,
            ),
            patch.object(
                d0,
                "process_group_member_pids",
                return_value=[process.pid],
            ),
            patch.object(d0.os, "killpg") as killpg,
            patch.object(
                d0.os,
                "waitpid",
                return_value=(process.pid, signal.SIGKILL),
            ) as waitpid,
            self.assertRaises(MemoryError) as raised,
        ):
            d0.launch_and_supervise_attested_process(
                ["/usr/bin/time", "-lp", "/usr/bin/true"],
                launch_cwd=Path.cwd(),
                environment={"LC_ALL": "C"},
                cwd=Path("/fixture/repo"),
                timeout_seconds=0.25,
                expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
            )

        self.assertIs(raised.exception, handoff_error)
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        waitpid.assert_called_once_with(process.pid, 0)
        self.assertEqual(process.returncode, -signal.SIGKILL)

    def test_production_quarantine_retries_after_sleep_baseexception(
        self,
    ) -> None:
        interrupt = KeyboardInterrupt("quarantine sleep interrupted")
        process = d0._GatedTextProcess(
            pid=12345,
            stdout=Mock(),
            stderr=Mock(),
        )
        stdout_thread = Mock()
        stderr_thread = Mock()
        stdout_thread.is_alive.return_value = False
        stderr_thread.is_alive.return_value = False
        wait_calls = 0

        def wait_for_terminal(
            child: d0._AttestedChildState,
            *,
            deadline: float,
            **kwargs: object,
        ) -> bool:
            del deadline, kwargs
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls <= 4:
                return False
            self.publish_terminal(
                child,
                code=d0.CLD_KILLED,
                status=signal.SIGKILL,
            )
            return True

        with (
            patch.object(
                d0.threading,
                "Thread",
                side_effect=(stdout_thread, stderr_thread),
            ),
            patch.object(
                d0,
                "_wait_for_attested_terminal",
                side_effect=wait_for_terminal,
            ),
            patch.object(
                d0,
                "process_group_member_pids",
                return_value=[process.pid],
            ),
            patch.object(d0.os, "killpg") as killpg,
            patch.object(
                d0.os,
                "waitpid",
                return_value=(process.pid, signal.SIGKILL),
            ) as waitpid,
            patch.object(
                d0.time,
                "sleep",
                side_effect=interrupt,
            ) as sleep,
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            d0.supervise_attested_process(
                process,
                cwd=Path.cwd(),
                timeout_seconds=0.25,
                expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
            )

        self.assertIs(raised.exception, interrupt)
        self.assertEqual(wait_calls, 5)
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        waitpid.assert_called_once_with(process.pid, 0)
        sleep.assert_called_once_with(d0.ATTESTATION_STOP_POLL_SECONDS)
        self.assertEqual(process.returncode, -signal.SIGKILL)

    def test_production_quarantine_guards_ownership_policy_read(self) -> None:
        class UnprintableInterrupt(KeyboardInterrupt):
            def __str__(self) -> str:
                raise RuntimeError("exception formatting failed")

        interrupt = UnprintableInterrupt()

        class FaultingOwnedProcess:
            def __init__(self) -> None:
                self.pid = 12345
                self.stdout = Mock()
                self.stderr = Mock()
                self.returncode: int | None = None
                self.ownership_reads = 0

            @property
            def production_owned(self) -> bool:
                self.ownership_reads += 1
                if self.ownership_reads == 1:
                    raise interrupt
                return True

        process = FaultingOwnedProcess()
        stdout_thread = Mock()
        stderr_thread = Mock()
        stdout_thread.is_alive.return_value = False
        stderr_thread.is_alive.return_value = False
        wait_calls = 0

        def wait_for_terminal(
            child: d0._AttestedChildState,
            *,
            deadline: float,
            **kwargs: object,
        ) -> bool:
            del deadline, kwargs
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls <= 3:
                return False
            self.publish_terminal(
                child,
                code=d0.CLD_KILLED,
                status=signal.SIGKILL,
            )
            return True

        with (
            patch.object(
                d0.threading,
                "Thread",
                side_effect=(stdout_thread, stderr_thread),
            ),
            patch.object(
                d0,
                "_wait_for_attested_terminal",
                side_effect=wait_for_terminal,
            ),
            patch.object(
                d0,
                "process_group_member_pids",
                return_value=[process.pid],
            ),
            patch.object(d0.os, "killpg") as killpg,
            patch.object(
                d0.os,
                "waitpid",
                return_value=(process.pid, signal.SIGKILL),
            ) as waitpid,
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            d0.supervise_attested_process(
                process,
                cwd=Path.cwd(),
                timeout_seconds=0.25,
                expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
            )

        self.assertIs(raised.exception, interrupt)
        self.assertEqual(wait_calls, 4)
        self.assertEqual(process.ownership_reads, 2)
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        waitpid.assert_called_once_with(process.pid, 0)
        self.assertEqual(process.returncode, -signal.SIGKILL)

    def test_production_owner_quarantines_until_quiescence_is_proved(
        self,
    ) -> None:
        stdout = Mock()
        stderr = Mock()
        process = d0._GatedTextProcess(
            pid=12345,
            stdout=stdout,
            stderr=stderr,
        )
        stdout_thread = Mock()
        stderr_thread = Mock()
        stdout_thread.is_alive.return_value = False
        stderr_thread.is_alive.return_value = False
        wait_calls = 0
        quiescence_calls = 0

        def wait_for_terminal(
            child: d0._AttestedChildState,
            *,
            deadline: float,
            **kwargs: object,
        ) -> bool:
            del deadline, kwargs
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls == 1:
                return False
            self.publish_terminal(
                child,
                code=d0.CLD_KILLED,
                status=signal.SIGKILL,
            )
            return True

        def quiescence(
            child: d0._AttestedChildState,
            *,
            deadline: float,
        ) -> bool:
            del deadline
            nonlocal quiescence_calls
            quiescence_calls += 1
            if quiescence_calls < 3:
                return False
            child.pre_reap_member_pids = [child.process.pid]
            child.pre_reap_observed_at_monotonic_ns = time.monotonic_ns()
            child.group_state = d0._AttestedGroupState.QUIESCENT
            return True

        with (
            patch.object(
                d0.threading,
                "Thread",
                side_effect=(stdout_thread, stderr_thread),
            ),
            patch.object(
                d0,
                "_wait_for_attested_terminal",
                side_effect=wait_for_terminal,
            ),
            patch.object(
                d0,
                "_wait_for_attested_group_quiescence",
                side_effect=quiescence,
            ),
            patch.object(d0.os, "killpg") as killpg,
            patch.object(
                d0.os,
                "waitpid",
                return_value=(process.pid, signal.SIGKILL),
            ) as waitpid,
        ):
            result = d0.supervise_attested_process(
                process,
                cwd=Path.cwd(),
                timeout_seconds=0.25,
                expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
            )

        self.assertTrue(result["timed_out"])
        self.assertEqual(quiescence_calls, 3)
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        waitpid.assert_called_once_with(process.pid, 0)
        self.assertEqual(process.returncode, -signal.SIGKILL)

    def test_supervisor_cleans_up_before_reraising_keyboard_interrupt(self) -> None:
        interrupt = KeyboardInterrupt()
        process, stdout_thread, stderr_thread = self.mocked_supervision_objects()
        process.stdout.close.side_effect = OSError("secondary cleanup fault")
        wait_calls = 0

        def wait_for_terminal(
            child: d0._AttestedChildState,
            *,
            deadline: float,
            **kwargs: object,
        ) -> bool:
            del deadline, kwargs
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls == 1:
                raise interrupt
            self.publish_terminal(
                child,
                code=d0.CLD_KILLED,
                status=signal.SIGKILL,
            )
            return True

        with (
            patch.object(
                d0.threading,
                "Thread",
                side_effect=(stdout_thread, stderr_thread),
            ),
            patch.object(
                d0,
                "_wait_for_attested_terminal",
                side_effect=wait_for_terminal,
            ),
            patch.object(
                d0,
                "process_group_member_pids",
                return_value=[process.pid],
            ),
            patch.object(d0.os, "killpg") as killpg,
            patch.object(
                d0.os,
                "waitpid",
                return_value=(process.pid, signal.SIGKILL),
            ) as waitpid,
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            d0.supervise_attested_process(
                process,
                cwd=Path.cwd(),
                timeout_seconds=0.25,
                expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
            )

        self.assertIs(raised.exception, interrupt)
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        waitpid.assert_called_once_with(process.pid, 0)
        process.stdout.close.assert_called_once_with()
        process.stderr.close.assert_called_once_with()
        self.assertEqual(wait_calls, 2)
        stdout_thread.join.assert_called_once_with(
            timeout=d0.ATTESTATION_THREAD_JOIN_SECONDS
        )
        stderr_thread.join.assert_called_once_with(
            timeout=d0.ATTESTATION_THREAD_JOIN_SECONDS
        )

    def test_cleanup_quiescence_failure_never_reaps(self) -> None:
        for mode in ("timeout", "exception"):
            with self.subTest(mode=mode):
                process, stdout_thread, stderr_thread = (
                    self.mocked_supervision_objects()
                )
                wait_calls = 0

                def wait_for_terminal(
                    child: d0._AttestedChildState,
                    *,
                    deadline: float,
                    **kwargs: object,
                ) -> bool:
                    del deadline, kwargs
                    nonlocal wait_calls
                    wait_calls += 1
                    if wait_calls == 1:
                        return False
                    self.publish_terminal(
                        child,
                        code=d0.CLD_KILLED,
                        status=signal.SIGKILL,
                    )
                    return True

                quiescence_effect: object
                if mode == "timeout":
                    quiescence_effect = lambda *args, **kwargs: False
                else:
                    quiescence_effect = OSError("group snapshot fault")

                with (
                    patch.object(
                        d0.threading,
                        "Thread",
                        side_effect=(stdout_thread, stderr_thread),
                    ),
                    patch.object(
                        d0,
                        "_wait_for_attested_terminal",
                        side_effect=wait_for_terminal,
                    ),
                    patch.object(
                        d0,
                        "_wait_for_attested_group_quiescence",
                        side_effect=quiescence_effect,
                    ) as quiescence,
                    patch.object(d0.os, "killpg") as killpg,
                    patch.object(d0.os, "waitpid") as waitpid,
                ):
                    result = d0.supervise_attested_process(
                        process,
                        cwd=Path.cwd(),
                        timeout_seconds=0.25,
                        expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                        expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
                    )

                self.assertTrue(result["timed_out"])
                self.assertGreaterEqual(quiescence.call_count, 1)
                waitpid.assert_not_called()
                self.assertIsNone(process.returncode)
                self.assertTrue(
                    any(
                        "quiescen" in error.lower()
                        for error in result["errors"]
                    ),
                    result,
                )
                killpg.assert_called_once_with(process.pid, signal.SIGKILL)

    def test_second_baseexception_during_cleanup_is_recorded_and_retried(
        self,
    ) -> None:
        process, stdout_thread, stderr_thread = self.mocked_supervision_objects()
        primary = SystemExit("primary supervision failure")
        secondary = KeyboardInterrupt("secondary cleanup interrupt")
        wait_calls = 0
        monotonic_calls = 0
        real_monotonic = time.monotonic

        def monotonic() -> float:
            nonlocal monotonic_calls
            monotonic_calls += 1
            if monotonic_calls == 1:
                return real_monotonic()
            if monotonic_calls == 2:
                raise secondary
            return real_monotonic()

        def wait_for_terminal(
            child: d0._AttestedChildState,
            *,
            deadline: float,
            **kwargs: object,
        ) -> bool:
            del deadline, kwargs
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls == 1:
                raise primary
            self.publish_terminal(
                child,
                code=d0.CLD_KILLED,
                status=signal.SIGKILL,
            )
            return True

        with (
            patch.object(d0.time, "monotonic", side_effect=monotonic),
            patch.object(
                d0.threading,
                "Thread",
                side_effect=(stdout_thread, stderr_thread),
            ),
            patch.object(
                d0,
                "_wait_for_attested_terminal",
                side_effect=wait_for_terminal,
            ),
            patch.object(
                d0,
                "process_group_member_pids",
                return_value=[process.pid],
            ) as group_members,
            patch.object(d0.os, "killpg") as killpg,
            patch.object(
                d0.os,
                "waitpid",
                return_value=(process.pid, signal.SIGKILL),
            ) as waitpid,
            self.assertRaises(SystemExit) as raised,
        ):
            d0.supervise_attested_process(
                process,
                cwd=Path.cwd(),
                timeout_seconds=0.25,
                expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
            )

        self.assertIs(raised.exception, primary)
        notes = "\n".join(self.exception_notes(raised.exception))
        self.assertIn(type(secondary).__name__, notes)
        self.assertIn(str(secondary), notes)
        self.assertEqual(wait_calls, 2)
        self.assertGreaterEqual(monotonic_calls, 3)
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        group_members.assert_called_once_with(process.pid)
        waitpid.assert_called_once_with(process.pid, 0)
        process.stdout.close.assert_called_once_with()
        process.stderr.close.assert_called_once_with()

    def test_supervisor_records_bounded_post_kill_timeout(self) -> None:
        process, stdout_thread, stderr_thread = self.mocked_supervision_objects()

        with (
            patch.object(
                d0.threading,
                "Thread",
                side_effect=(stdout_thread, stderr_thread),
            ),
            patch.object(
                d0,
                "_wait_for_attested_terminal",
                side_effect=(False, False),
            ),
            patch.object(d0.os, "killpg") as killpg,
        ):
            result = d0.supervise_attested_process(
                process,
                cwd=Path.cwd(),
                timeout_seconds=0.25,
                expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
            )

        self.assertTrue(result["timed_out"])
        self.assertTrue(
            any(
                "cleanup failed during observe terminal process" in error
                and "did not terminate within" in error
                for error in result["errors"]
            )
        )
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        process.stdout.close.assert_called_once_with()
        process.stderr.close.assert_called_once_with()
        stdout_thread.join.assert_called_once_with(
            timeout=d0.ATTESTATION_THREAD_JOIN_SECONDS
        )
        stderr_thread.join.assert_called_once_with(
            timeout=d0.ATTESTATION_THREAD_JOIN_SECONDS
        )

    def test_supervisor_attempts_and_records_every_cleanup_after_faults(self) -> None:
        process, stdout_thread, stderr_thread = self.mocked_supervision_objects()
        process.stdout.close.side_effect = OSError("stdout close fault")
        process.stderr.close.side_effect = OSError("stderr close fault")
        stdout_thread.join.side_effect = RuntimeError("stdout join fault")
        stderr_thread.join.side_effect = RuntimeError("stderr join fault")
        wait_calls = 0

        def wait_for_terminal(
            child: d0._AttestedChildState,
            *,
            deadline: float,
            **kwargs: object,
        ) -> bool:
            del deadline, kwargs
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls == 1:
                return False
            self.publish_terminal(
                child,
                code=d0.CLD_KILLED,
                status=signal.SIGKILL,
            )
            return True

        with (
            patch.object(
                d0.threading,
                "Thread",
                side_effect=(stdout_thread, stderr_thread),
            ),
            patch.object(
                d0,
                "_wait_for_attested_terminal",
                side_effect=wait_for_terminal,
            ),
            patch.object(
                d0,
                "process_group_member_pids",
                return_value=[process.pid],
            ),
            patch.object(
                d0.os,
                "killpg",
                side_effect=PermissionError("kill fault"),
            ) as killpg,
            patch.object(
                d0.os,
                "waitpid",
                side_effect=OSError("reap fault"),
            ) as waitpid,
        ):
            result = d0.supervise_attested_process(
                process,
                cwd=Path.cwd(),
                timeout_seconds=0.25,
                expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
            )

        cleanup_errors = "\n".join(result["errors"])
        for operation, detail in (
            ("kill process group", "kill fault"),
            ("close stdout", "stdout close fault"),
            ("close stderr", "stderr close fault"),
            ("reap process", "reap fault"),
            ("join stdout thread", "stdout join fault"),
            ("join stderr thread", "stderr join fault"),
        ):
            with self.subTest(operation=operation):
                self.assertIn(
                    f"cleanup failed during {operation}",
                    cleanup_errors,
                )
                self.assertIn(detail, cleanup_errors)
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        process.stdout.close.assert_called_once_with()
        process.stderr.close.assert_called_once_with()
        self.assertEqual(wait_calls, 2)
        self.assertEqual(waitpid.call_count, 2)
        expected_join = call(timeout=d0.ATTESTATION_THREAD_JOIN_SECONDS)
        self.assertEqual(stdout_thread.join.call_args_list, [expected_join] * 2)
        self.assertEqual(stderr_thread.join.call_args_list, [expected_join] * 2)

    def test_interrupted_reap_never_performs_a_later_group_syscall(self) -> None:
        process, stdout_thread, stderr_thread = self.mocked_supervision_objects()

        def wait_for_terminal(
            child: d0._AttestedChildState,
            *,
            deadline: float,
            **kwargs: object,
        ) -> bool:
            del deadline, kwargs
            self.publish_terminal(
                child,
                code=d0.CLD_EXITED,
                status=0,
            )
            return True

        with (
            patch.object(
                d0.threading,
                "Thread",
                side_effect=(stdout_thread, stderr_thread),
            ),
            patch.object(
                d0,
                "_wait_for_attested_terminal",
                side_effect=wait_for_terminal,
            ),
            patch.object(
                d0,
                "process_group_member_pids",
                return_value=[process.pid],
            ) as group_members,
            patch.object(
                d0.os,
                "waitpid",
                side_effect=(KeyboardInterrupt(), ChildProcessError()),
            ) as waitpid,
            patch.object(d0.os, "killpg") as killpg,
            self.assertRaises(KeyboardInterrupt),
        ):
            d0.supervise_attested_process(
                process,
                cwd=Path.cwd(),
                timeout_seconds=0.25,
                expected_wrapper_executable=self.WRAPPER_EXECUTABLE,
                expected_benchmark_executable=self.BENCHMARK_EXECUTABLE,
            )

        group_members.assert_called_once_with(process.pid)
        self.assertEqual(waitpid.call_count, 2)
        killpg.assert_not_called()
        self.assertEqual(process.returncode, sys.maxsize)

    @unittest.skipUnless(
        sys.platform == "darwin",
        "live process-group attestation requires macOS",
    )
    def test_live_time_wrapper_process_group_reaches_all_four_stops(self) -> None:
        repo = Path(__file__).resolve().parents[3]
        probe = Path(__file__).with_name("fork_gate_probe.py")
        completed = subprocess.run(
            [sys.executable, str(probe)],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=20.0,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["thread_count"], 1)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["errors"], [])
        self.assertFalse(result["timed_out"])
        self.assertTrue(result["child_complete"])
        self.assertEqual(
            result["phases"],
            list(d0.ATTESTATION_BARRIER_PHASES),
        )
        self.assertEqual(result["member_pids"], [result["pid"]])
        self.assertEqual(result["terminal_pid"], result["pid"])
        self.assertEqual(result["reap_pid"], result["pid"])
        self.assertTrue(result["status_validated"])


if __name__ == "__main__":
    unittest.main()
