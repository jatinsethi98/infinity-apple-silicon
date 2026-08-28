from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import math
import os
import random
import shlex
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

from tools.apple_silicon import verify_native_hnsw_d0 as verifier

VECTORS = 12_288
DIMENSIONS = 128
DATASET_BYTES = VECTORS * DIMENSIONS * 4
DATASET_NAME = "d0-f32le-n12288-d128-seed0.bin"
SCOPE = "d0-development-only"
CAMPAIGN_NONCE = "a5" * 32
AUDIT_BODY_HEADER_FORMAT = "<8sII32sQII32s32sQQIIiiIIQQII"
QUERY_SCHEMA2_FIXED_FORMAT = "<IIIIIIIIIIQQ32sQQQQQQQQQ"
QUERY_SCHEMA2_FIXED_BYTES = struct.calcsize(QUERY_SCHEMA2_FIXED_FORMAT)
QUERY_SCHEMA2_PER_QUERY_BYTES = verifier.HELDOUT_QUERY_COUNT * struct.calcsize("<Q")

PAIR_PLAN = (
    ("C1", "correctness", 1, ("infinity", "faiss")),
    ("W1", "warmup", 12, ("infinity", "faiss")),
    ("W2", "warmup", 12, ("faiss", "infinity")),
    ("P1", "measured", 12, ("infinity", "faiss")),
    ("P2", "measured", 12, ("faiss", "infinity")),
    ("P3", "measured", 12, ("faiss", "infinity")),
    ("P4", "measured", 12, ("infinity", "faiss")),
    ("P5", "measured", 12, ("infinity", "faiss")),
    ("P6", "measured", 12, ("faiss", "infinity")),
)

INFINITY_DURATIONS = {
    "P1": 100_000_000,
    "P2": 102_000_000,
    "P3": 98_000_000,
    "P4": 101_000_000,
    "P5": 99_000_000,
    "P6": 100_000_000,
}
FAISS_DURATIONS = {pair: duration * 2 for pair, duration in INFINITY_DURATIONS.items()}


def set_recall_hits(point: dict[str, Any], eligible_hits: int) -> None:
    possible_hits = point["possible_hits"]
    point["eligible_hits"] = eligible_hits
    point["recall"] = eligible_hits / float(possible_hits)


def synthetic_macho(
    libraries: tuple[str, ...] = (),
    *,
    file_type: int = 2,
) -> bytes:
    commands = []
    for name in libraries:
        encoded = name.encode("utf-8") + b"\0"
        command_size = (24 + len(encoded) + 7) & ~7
        command = struct.pack(
            "<IIIIII",
            0xC,
            command_size,
            24,
            0,
            0,
            0,
        )
        commands.append(command + encoded + b"\0" * (command_size - 24 - len(encoded)))
    command_bytes = b"".join(commands)
    header = struct.pack(
        "<IiiIIIII",
        0xFEEDFACF,
        0x0100000C,
        0,
        file_type,
        len(commands),
        len(command_bytes),
        0,
        0,
    )
    return header + command_bytes


INFINITY_BINARY_BYTES = synthetic_macho()
FAISS_BINARY_BYTES = synthetic_macho(("@rpath/libfaiss.dylib", "@rpath/libomp.dylib"))
LIBFAISS_BYTES = synthetic_macho(file_type=6)
LIBOMP_BYTES = synthetic_macho(file_type=6)

BINARY_RECORDS = {
    "infinity": {
        "path": "/fixture/build/infinity_hnsw_d0_production",
        "description": (
            "/fixture/build/infinity_hnsw_d0_production: "
            "Mach-O 64-bit executable arm64"
        ),
        "sha256": hashlib.sha256(INFINITY_BINARY_BYTES).hexdigest(),
        "bytes": len(INFINITY_BINARY_BYTES),
    },
    "faiss": {
        "path": "/fixture/build/faiss_hnsw_d0",
        "description": (
            "/fixture/build/faiss_hnsw_d0: " "Mach-O 64-bit executable arm64"
        ),
        "sha256": hashlib.sha256(FAISS_BINARY_BYTES).hexdigest(),
        "bytes": len(FAISS_BINARY_BYTES),
    },
}
RUNNER_SOURCE_PATH = "/fixture/repo/tools/apple_silicon/native_hnsw_d0.py"
VERIFIER_SOURCE_PATH = "/fixture/repo/tools/apple_silicon/verify_native_hnsw_d0.py"
LIBFAISS_PATH = "/fixture/build/libfaiss.dylib"
LIBFAISS_SHA256 = hashlib.sha256(LIBFAISS_BYTES).hexdigest()
LIBOMP_PATH = "/fixture/build/libomp.dylib"
LIBOMP_SHA256 = hashlib.sha256(LIBOMP_BYTES).hexdigest()
FAISS_SOURCE_ROOT = "/fixture/faiss"
REPO_SOURCE_ROOT = "/fixture/repo"
VCPKG_INSTALLED_ROOT = f"{REPO_SOURCE_ROOT}/vcpkg_installed"
VCPKG_INCLUDE_ROOT = f"{VCPKG_INSTALLED_ROOT}/arm64-osx/include"
PYTHON_PATH = "/fixture/tools/python3"
PYTHON_BYTES = b"synthetic Python executable\n"
TOOL_CONTENTS = {
    "compiler": b"synthetic clang executable\n",
    "linker": b"synthetic linker executable\n",
    "build_tool": b"synthetic Ninja executable\n",
}
INFINITY_SOURCE_BYTES = b"// synthetic Infinity harness source\n"
FAISS_HARNESS_SOURCE_BYTES = b"// synthetic FAISS harness source\n"
FAISS_LIBRARY_SOURCE_BYTES = b"// synthetic linked FAISS source\n"
FAISS_PATCH_BYTES = b"synthetic pinned FAISS stats-disabled patch\n"
FAISS_BASE_PATH_BYTES = {
    "CMakeLists.txt": b"base top-level CMake\n",
    "faiss/CMakeLists.txt": b"base FAISS CMake\n",
    "faiss/IndexBinaryHNSW.cpp": b"base binary HNSW\n",
    "faiss/IndexHNSW.cpp": b"base float HNSW\n",
}
FAISS_DERIVATIVE_PATH_BYTES = {
    "CMakeLists.txt": b"derivative top-level CMake\n",
    "faiss/CMakeLists.txt": b"derivative FAISS CMake\n",
    "faiss/IndexBinaryHNSW.cpp": b"derivative binary HNSW\n",
    "faiss/IndexHNSW.cpp": b"derivative float HNSW\n",
}
TOOL_METADATA = {
    "compiler": {
        "configured_path": "/fixture/tools/clang++",
        "resolved_path": "/fixture/tools/clang++",
        "version_command": ["/fixture/tools/clang++", "--version"],
        "version_returncode": 0,
        "version": "synthetic clang 20",
    },
    "linker": {
        "configured_path": "/fixture/tools/ld",
        "resolved_path": "/fixture/tools/ld",
        "version_command": ["/fixture/tools/ld", "-v"],
        "version_returncode": 0,
        "version": "synthetic ld",
    },
    "build_tool": {
        "configured_path": "/fixture/tools/ninja",
        "resolved_path": "/fixture/tools/ninja",
        "version_command": ["/fixture/tools/ninja", "--version"],
        "version_returncode": 0,
        "version": "1.13.2",
    },
}


def write_content_blob(directory: Path, content: bytes) -> dict[str, Any]:
    digest = hashlib.sha256(content).hexdigest()
    relative = Path("provenance") / "blobs" / digest[:2] / digest
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
    return {
        "captured_path": relative.as_posix(),
        "bytes": len(content),
        "sha256": digest,
    }


def executable_records(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        name: {
            **TOOL_METADATA[name],
            **write_content_blob(directory, TOOL_CONTENTS[name]),
        }
        for name in ("compiler", "linker", "build_tool")
    }


def synthetic_commit(
    index: int,
    entries: dict[str, tuple[str, str, str]],
    *,
    parents: tuple[str, ...] = (),
) -> tuple[str, str, bytes]:
    tree = verifier.git_tree_id(entries, f"synthetic repository {index}")
    commit_bytes = (
        f"tree {tree}\n"
        + "".join(f"parent {parent}\n" for parent in parents)
        + "author Fixture <fixture@example.com> 0 +0000\n"
        + "committer Fixture <fixture@example.com> 0 +0000\n"
        + "\n"
        + f"Synthetic repository {index}\n"
    ).encode("utf-8")
    head = verifier.git_object_id("commit", commit_bytes)
    return head, tree, commit_bytes


def synthetic_tree_listing(
    entries: dict[str, tuple[str, str, str]],
) -> str:
    return "".join(
        f"{mode} {kind} {object_id}\t{path}\n"
        for path, (mode, kind, object_id) in sorted(entries.items())
    )


def synthetic_git_repository(
    directory: Path,
    *,
    index: int,
    root: str,
    entries: dict[str, tuple[str, str, str]],
    status: str,
    diff: str,
    parents: tuple[str, ...] = (),
) -> dict[str, Any]:
    head, tree, commit_bytes = synthetic_commit(
        index,
        entries,
        parents=parents,
    )
    tree_listing = synthetic_tree_listing(entries)
    prefix = f"source-repository-{index:02d}"
    status_name = f"{prefix}-git-status.txt"
    diff_name = f"{prefix}-source-diff.patch"
    commit_name = f"{prefix}-commit-object.txt"
    tree_name = f"{prefix}-tree-listing.txt"
    (directory / status_name).write_text(status, encoding="utf-8")
    (directory / diff_name).write_text(diff, encoding="utf-8")
    (directory / commit_name).write_bytes(commit_bytes)
    (directory / tree_name).write_text(tree_listing, encoding="utf-8")
    return {
        "root": root,
        "head": head,
        "tree": tree,
        "dirty": bool(status),
        "status_path": status_name,
        "status_sha256": sha256(directory / status_name),
        "diff_path": diff_name,
        "diff_sha256": sha256(directory / diff_name),
        "commit_object_path": commit_name,
        "commit_object_sha256": sha256(directory / commit_name),
        "tree_listing_path": tree_name,
        "tree_listing_sha256": sha256(directory / tree_name),
    }


def faiss_base_git_entries() -> dict[str, tuple[str, str, str]]:
    entries = {
        path: (
            "100644",
            "blob",
            verifier.git_object_id("blob", content),
        )
        for path, content in FAISS_BASE_PATH_BYTES.items()
    }
    entries["faiss/IndexFixture.cpp"] = (
        "100644",
        "blob",
        verifier.git_object_id("blob", FAISS_LIBRARY_SOURCE_BYTES),
    )
    return entries


def faiss_derivative_git_entries() -> dict[str, tuple[str, str, str]]:
    entries = faiss_base_git_entries()
    entries.update(
        {
            path: (
                "100644",
                "blob",
                verifier.git_object_id("blob", content),
            )
            for path, content in FAISS_DERIVATIVE_PATH_BYTES.items()
        }
    )
    return entries


def synthetic_faiss_history() -> dict[str, Any]:
    base_entries = faiss_base_git_entries()
    base_head, base_tree, base_commit = synthetic_commit(100, base_entries)
    derivative_entries = faiss_derivative_git_entries()
    derivative_head, derivative_tree, derivative_commit = synthetic_commit(
        0,
        derivative_entries,
        parents=(base_head,),
    )
    return {
        "base_entries": base_entries,
        "base_head": base_head,
        "base_tree": base_tree,
        "base_commit": base_commit,
        "derivative_entries": derivative_entries,
        "derivative_head": derivative_head,
        "derivative_tree": derivative_tree,
        "derivative_commit": derivative_commit,
    }


def source_record(directory: Path, path: str, content: bytes) -> dict[str, Any]:
    return {
        "path": path.removeprefix("/fixture/repo/"),
        "absolute_path": path,
        **write_content_blob(directory, content),
    }


def dependency_record(
    path: str,
    digest: str,
    *,
    referenced_by: str,
    size: int,
) -> dict[str, Any]:
    return {
        "install_names": [f"@rpath/{Path(path).name}"],
        "referenced_by": [referenced_by],
        "resolved_path": path,
        "sha256": digest,
        "bytes": size,
        "unavailable_reason": None,
    }


def cmake_cache_text(
    *,
    linked_faiss: bool = False,
    production_infinity: bool = False,
) -> str:
    lines = [
        "CMAKE_BUILD_TYPE:STRING=Release",
        "CMAKE_OSX_ARCHITECTURES:STRING=arm64",
        "CMAKE_OSX_DEPLOYMENT_TARGET:STRING=14.0",
        "CMAKE_OSX_SYSROOT:PATH=/fixture/MacOSX.sdk",
        "CMAKE_CXX_COMPILER:FILEPATH=/fixture/tools/clang++",
        "CMAKE_LINKER:FILEPATH=/fixture/tools/ld",
        "CMAKE_MAKE_PROGRAM:FILEPATH=/fixture/tools/ninja",
        "CMAKE_HOME_DIRECTORY:INTERNAL="
        + (
            REPO_SOURCE_ROOT
            if production_infinity
            else f"{REPO_SOURCE_ROOT}/tools/apple_silicon/native_hnsw_smoke"
        ),
    ]
    if production_infinity:
        lines.extend(
            [
                "CMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON",
                "ENABLE_JEMALLOC:BOOL=OFF",
                "VCPKG_MANIFEST_INSTALL:BOOL=OFF",
                f"VCPKG_INSTALLED_DIR:PATH={VCPKG_INSTALLED_ROOT}",
                "VCPKG_TARGET_TRIPLET:STRING=arm64-osx",
                "INFINITY_BUILD_APPLE_HNSW_D0_PRODUCTION:BOOL=ON",
                "INFINITY_BUILD_TIME_OVERRIDE:STRING=2026-08-20 00:00.00",
                "INFINITY_DISABLE_APPLE_HNSW_BATCH4_TRAVERSAL:BOOL=OFF",
                "INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING:BOOL=ON",
            ]
        )
    if linked_faiss:
        lines.extend(
            [
                "CMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG",
                "BUILD_SHARED_LIBS:BOOL=ON",
                "FAISS_ENABLE_GPU:BOOL=OFF",
                "FAISS_ENABLE_PYTHON:BOOL=OFF",
                "FAISS_ENABLE_GLOBAL_HNSW_STATS:BOOL=OFF",
                "OpenMP_CXX_FLAGS:STRING=-fopenmp=libomp",
                "OpenMP_CXX_INCLUDE_DIR:PATH=/fixture/build/include",
                f"OpenMP_libomp_LIBRARY:FILEPATH={LIBOMP_PATH}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_synthetic_build_input_closure(
    directory: Path,
    *,
    capture_prefix: str,
    requested_target: str,
    expected_output: str,
    build_directory: str,
    source: dict[str, Any],
    object_name: str,
    link_arguments: list[str],
    extra_dependencies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    extra_dependencies = extra_dependencies or []
    object_path = f"{build_directory}/{object_name}"
    deps_name = f"{capture_prefix}-ninja-deps.txt"
    deps_text = (
        f"{object_name}: #deps {1 + len(extra_dependencies)}, "
        "deps mtime 1 (VALID)\n"
        f"    {source['absolute_path']}\n"
        + "".join(
            f"    {dependency['absolute_path']}\n"
            for dependency in extra_dependencies
        )
        + "\n"
    )
    (directory / deps_name).write_text(deps_text, encoding="utf-8")
    object_blob = write_content_blob(
        directory,
        f"synthetic object {capture_prefix}\n".encode(),
    )
    files = [
        {
            "path": object_path,
            "resolved_path": object_path,
            "type": "object",
            "roles": ["compile-output", "link-object", "ninja-output"],
            **object_blob,
        },
        {
            "path": source["absolute_path"],
            "resolved_path": source["absolute_path"],
            "type": "file",
            "roles": ["compile-source", "compiler-dependency"],
            "captured_path": source["captured_path"],
            "sha256": source["sha256"],
            "bytes": source["bytes"],
        },
    ]
    files.extend(
        {
            "path": dependency["absolute_path"],
            "resolved_path": dependency["absolute_path"],
            "type": "file",
            "roles": ["compiler-dependency"],
            "captured_path": dependency["captured_path"],
            "sha256": dependency["sha256"],
            "bytes": dependency["bytes"],
        }
        for dependency in extra_dependencies
    )
    files.sort(key=lambda record: record["path"].encode("utf-8"))
    compile_units = [
        {
            "output": object_path,
            "module_outputs": [],
            "direct_inputs": [
                {
                    "role": "compile-source",
                    "path": source["absolute_path"],
                }
            ],
        }
    ]
    dependency_graph = [
        {
            "output": object_path,
            "dependency_count": 1 + len(extra_dependencies),
            "deps_mtime": 1,
            "status": "VALID",
            "dependencies": [
                source["absolute_path"],
                *[
                    dependency["absolute_path"]
                    for dependency in extra_dependencies
                ],
            ],
        }
    ]
    targets_name = f"{capture_prefix}-ninja-targets.txt"
    targets_text = (
        f"{object_name}: CXX_COMPILER__synthetic_Release\n"
        f"{requested_target}: CXX_LINKER__synthetic_Release\n"
    )
    (directory / targets_name).write_text(targets_text, encoding="utf-8")
    query_name = f"{capture_prefix}-ninja-query.txt"
    query_text = (
        f"{requested_target}:\n"
        "  input: CXX_LINKER__synthetic_Release\n"
        f"    {object_name}\n"
        "  outputs:\n"
        f"{object_name}:\n"
        "  input: CXX_COMPILER__synthetic_Release\n"
        f"    {source['absolute_path']}\n"
        "  outputs:\n"
    )
    (directory / query_name).write_text(query_text, encoding="utf-8")
    query_nodes = verifier.parse_ninja_query(
        query_text,
        build_directory=Path(build_directory),
        context="synthetic Ninja query",
    )
    clean_stdout_name = f"{capture_prefix}-ninja-clean-plan.stdout"
    clean_stderr_name = f"{capture_prefix}-ninja-clean-plan.stderr"
    clean_outputs = [object_path, expected_output]
    clean_text = (
        "Cleaning...\n"
        f"Target {requested_target}\n"
        f"Remove {object_name}\n"
        f"Remove {requested_target}\n"
        "2 files.\n"
    )
    (directory / clean_stdout_name).write_text(clean_text, encoding="utf-8")
    (directory / clean_stderr_name).write_text("", encoding="utf-8")
    closure = {
        "closure_schema_version": verifier.BUILD_INPUT_CLOSURE_SCHEMA_VERSION,
        "products": [
            {
                "name": "benchmark",
                "requested_target": requested_target,
                "expected_output": expected_output,
                "compile_outputs": [object_path],
                "link": {
                    "arguments": link_arguments,
                    "inputs": [
                        {
                            "role": "link-object",
                            "path": object_path,
                        }
                    ],
                },
            }
        ],
        "build_directory": build_directory,
        "ninja_deps": {
            "command": [
                "/fixture/tools/ninja",
                "-C",
                build_directory,
                "-t",
                "deps",
            ],
            "captured_path": deps_name,
            "sha256": sha256(directory / deps_name),
            "records": 1,
        },
        "compile_units": compile_units,
        "dependency_graph": dependency_graph,
        "selected_target_graph": {
            "targets": {
                "command": [
                    "/fixture/tools/ninja",
                    "-C",
                    build_directory,
                    "-t",
                    "targets",
                    "all",
                ],
                "captured_path": targets_name,
                "sha256": sha256(directory / targets_name),
                "records": 2,
            },
            "query": {
                "command": [
                    "/fixture/tools/ninja",
                    "-C",
                    build_directory,
                    "-t",
                    "query",
                    requested_target,
                    object_name,
                ],
                "captured_path": query_name,
                "sha256": sha256(directory / query_name),
                "nodes": query_nodes,
            },
            "clean_plan": {
                "command": [
                    "/fixture/tools/ninja",
                    "-v",
                    "-n",
                    "-C",
                    build_directory,
                    "-t",
                    "clean",
                    requested_target,
                ],
                "returncode": 0,
                "stdout": {
                    "captured_path": clean_stdout_name,
                    "sha256": sha256(directory / clean_stdout_name),
                },
                "stderr": {
                    "captured_path": clean_stderr_name,
                    "sha256": sha256(directory / clean_stderr_name),
                },
                "outputs": clean_outputs,
            },
            "selected_outputs": [expected_output],
            "material_outputs": sorted([expected_output, object_path]),
            "derived_link_outputs": [],
        },
        "files": files,
    }
    closure_name = f"{capture_prefix}-build-input-closure.json"
    write_json(directory / closure_name, closure)
    return {
        "closure_schema_version": verifier.BUILD_INPUT_CLOSURE_SCHEMA_VERSION,
        "captured_path": closure_name,
        "sha256": sha256(directory / closure_name),
        "products": 1,
        "compile_units": 1,
        "dependency_outputs": 1,
        "files": 2 + len(extra_dependencies),
    }


def write_build_capture(
    directory: Path,
    *,
    label: str,
    source: dict[str, Any],
    binary: dict[str, Any],
    dependencies: list[dict[str, Any]],
    tools: dict[str, dict[str, Any]],
    extra_build_dependencies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cache_name = f"{label}-CMakeCache.txt"
    compile_name = f"{label}-compile_commands.json"
    commands_name = f"{label}-ninja-commands.txt"
    noop_name = f"{label}-ninja-noop.txt"
    dependencies_name = f"{label}-dependencies.json"
    production_infinity = label == "infinity"
    (directory / cache_name).write_text(
        cmake_cache_text(production_infinity=production_infinity),
        encoding="utf-8",
    )
    release_flags = (
        "-O3 -DNDEBUG -arch arm64 -isysroot /fixture/MacOSX.sdk "
        "-mmacosx-version-min=14.0"
    )
    write_json(
        directory / compile_name,
        [
            {
                "directory": "/fixture/build",
                "file": source["absolute_path"],
                "command": (
                    f"/fixture/tools/clang++ {release_flags} "
                    f"-c {source['absolute_path']} -o {label}.o"
                ),
            }
        ],
    )
    (directory / commands_name).write_text(
        f"/fixture/tools/clang++ {release_flags} "
        f"-c {source['absolute_path']} -o {label}.o\n"
        f"/fixture/tools/clang++ {release_flags} "
        f"{label}.o -o {Path(binary['path']).name}\n",
        encoding="utf-8",
    )
    link_arguments = [
        "/fixture/tools/clang++",
        *release_flags.split(),
        f"{label}.o",
        "-o",
        Path(binary["path"]).name,
    ]
    build_input_closure = write_synthetic_build_input_closure(
        directory,
        capture_prefix=label,
        requested_target=Path(binary["path"]).name,
        expected_output=binary["path"],
        build_directory="/fixture/build",
        source=source,
        object_name=f"{label}.o",
        link_arguments=link_arguments,
        extra_dependencies=extra_build_dependencies,
    )
    (directory / noop_name).write_text(
        "ninja: no work to do.\n",
        encoding="utf-8",
    )
    images = {binary["path"]}
    images.update(
        dependency["resolved_path"]
        for dependency in dependencies
        if dependency["sha256"] is not None
    )
    otool_outputs = {}
    for image in sorted(images):
        install_names = sorted(
            {
                install_name
                for dependency in dependencies
                if image in dependency["referenced_by"]
                for install_name in dependency["install_names"]
            }
        )
        otool_outputs[image] = f"{image}:\n" + "".join(
            f"\t{name} (compatibility version 1.0.0, " "current version 1.0.0)\n"
            for name in install_names
        )
    write_json(
        directory / dependencies_name,
        {
            "binary": binary["path"],
            "binary_sha256": binary["sha256"],
            "dependencies": dependencies,
            "otool_outputs": otool_outputs,
        },
    )
    return {
        "build_directory": "/fixture/build",
        "cmake_home_directory": (
            REPO_SOURCE_ROOT
            if production_infinity
            else f"{REPO_SOURCE_ROOT}/tools/apple_silicon/native_hnsw_smoke"
        ),
        "cmake_cache": {
            "captured_path": cache_name,
            "sha256": sha256(directory / cache_name),
        },
        "compile_commands": {
            "captured_path": compile_name,
            "sha256": sha256(directory / compile_name),
            "entries": 1,
        },
        "settings": {
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_OSX_ARCHITECTURES": "arm64",
            "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0",
            "CMAKE_OSX_SYSROOT": "/fixture/MacOSX.sdk",
            **(
                {
                    "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
                    "ENABLE_JEMALLOC": "OFF",
                    "VCPKG_MANIFEST_INSTALL": "OFF",
                    "VCPKG_INSTALLED_DIR": VCPKG_INSTALLED_ROOT,
                    "VCPKG_TARGET_TRIPLET": "arm64-osx",
                    "INFINITY_BUILD_APPLE_HNSW_D0_PRODUCTION": "ON",
                    "INFINITY_BUILD_TIME_OVERRIDE": "2026-08-20 00:00.00",
                    "INFINITY_DISABLE_APPLE_HNSW_BATCH4_TRAVERSAL": "OFF",
                    "INFINITY_DISABLE_APPLE_HNSW_BATCH4_RECIPROCAL_PRUNING": "ON",
                }
                if production_infinity
                else {}
            ),
        },
        **tools,
        "ninja_commands": {
            "captured_path": commands_name,
            "sha256": sha256(directory / commands_name),
        },
        "ninja_noop": {
            "captured_path": noop_name,
            "sha256": sha256(directory / noop_name),
        },
        "response_files": [],
        "dependencies": {
            "captured_path": dependencies_name,
            "sha256": sha256(directory / dependencies_name),
            "count": len(dependencies),
        },
        "build_input_closure": build_input_closure,
        "compiled_sources": 1,
    }


def raw_host_capture(*, battery_percent: int = 100) -> str:
    return (
        "[battery]\n"
        "Now drawing from 'AC Power'\n"
        f" -InternalBattery-0\t{battery_percent}%; charging; present: true\n\n"
        "[custom]\n"
        "Battery Power:\n"
        " lowpowermode 0\n"
        "AC Power:\n"
        " lowpowermode 0\n\n"
        "[thermal]\n"
        "Note: No thermal warning level has been recorded\n"
        "Note: No performance warning level has been recorded\n\n"
        "[swap]\n"
        "vm.swapusage: total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)\n"
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fnv1a64(data: bytes) -> int:
    value = 0xCBF29CE484222325
    for byte in data:
        value ^= byte
        value = (value * 0x100000001B3) & ((1 << 64) - 1)
    return value


def canonical_dataset_bytes() -> bytes:
    generator = random.Random(0)
    pack_float = struct.Struct("<f").pack
    result = bytearray()
    for _ in range(VECTORS * DIMENSIONS):
        result.extend(pack_float(generator.random()))
    data = bytes(result)
    if hashlib.sha256(data).hexdigest() != verifier.DATASET_SHA256:
        raise AssertionError("Canonical fixture SHA-256 differs")
    if fnv1a64(data) != verifier.DATASET_FNV1A64:
        raise AssertionError("Canonical fixture FNV-1a differs")
    return data


def synthetic_heldout_manifest(truth: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": verifier.HELDOUT_MANIFEST_SCHEMA_VERSION,
        "query_count": verifier.HELDOUT_QUERY_COUNT,
        "dimension": DIMENSIONS,
        "seed": verifier.HELDOUT_QUERY_SEED,
        "generator": "splitmix64-high24-float32",
        "queries_sha256": truth["queries_sha256"],
        "truth_sha256": truth["truth_sha256"],
        "tie_counts_at_10": list(truth["tie_counts_at_10"]),
        "tie_counts_at_100": list(truth["tie_counts_at_100"]),
        "recall_points": verifier.expected_recall_points_json(),
    }


def synthetic_audit_sidecar(
    engine: str,
    truth: dict[str, Any],
    *,
    campaign_nonce: str,
    schedule_sequence: int,
    role_id: int,
    binary_sha256: str,
    dataset_sha256: str,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    engine_number = {"infinity": 1, "faiss": 2}[engine]
    latency_base = 100 if engine == "infinity" else 200
    latency_samples = [
        latency_base + (index % verifier.HELDOUT_QUERY_COUNT)
        for index in range(verifier.QUERY_LATENCY_SAMPLE_COUNT)
    ]
    throughput_wall_ns = 1_000_000 if engine == "infinity" else 2_000_000
    per_query_checksums: list[int] = []
    for query_index in range(verifier.HELDOUT_QUERY_COUNT):
        labels = truth["top_100"][query_index][: verifier.QUERY_K]
        distance_bits = [
            struct.unpack(
                "<I",
                struct.pack("<f", truth["distances"][query_index][label]),
            )[0]
            for label in labels
        ]
        per_query_checksums.append(
            verifier.query_result_checksum(
                query_index,
                labels,
                distance_bits,
            )
        )
    result_checksum = verifier.query_benchmark_checksum(per_query_checksums)
    data = bytearray(
        struct.pack(
            AUDIT_BODY_HEADER_FORMAT,
            verifier.AUDIT_SIDECAR_MAGIC,
            verifier.AUDIT_SIDECAR_SCHEMA_VERSION,
            engine_number,
            bytes.fromhex(campaign_nonce),
            schedule_sequence,
            role_id,
            0,
            bytes.fromhex(binary_sha256),
            bytes.fromhex(dataset_sha256),
            VECTORS,
            DIMENSIONS,
            32,
            200,
            0,
            0,
            64,
            32,
            verifier.HELDOUT_QUERY_COUNT,
            verifier.HELDOUT_QUERY_SEED,
            len(verifier.RECALL_POINTS),
            0,
        )
    )
    data.extend(
        struct.pack(
            "<IIIIIIIIIIQQ",
            verifier.QUERY_BENCHMARK_SCHEMA_VERSION,
            verifier.HELDOUT_QUERY_COUNT,
            verifier.QUERY_K,
            verifier.QUERY_EF_SEARCH,
            verifier.QUERY_WARMUP_PASSES,
            verifier.QUERY_MEASURED_PASSES,
            verifier.QUERY_LATENCY_CONCURRENCY,
            verifier.QUERY_THROUGHPUT_CONCURRENCY,
            verifier.QUERY_PERCENTILE_METHOD_ID,
            verifier.QUERY_TRANSACTION_DEFINITION_ID,
            struct.unpack("<Q", struct.pack("<d", verifier.QUERY_RECALL_FLOOR))[0],
            struct.unpack(
                "<Q",
                struct.pack(
                    "<d",
                    verifier.QUERY_MAXIMUM_ABSOLUTE_RECALL_GAP,
                ),
            )[0],
        )
    )
    data.extend(bytes.fromhex(truth["queries_sha256"]))
    data.extend(
        struct.pack(
            "<QQQQQQQQQ",
            verifier.QUERY_LATENCY_SAMPLE_COUNT,
            verifier.QUERY_LATENCY_SAMPLE_COUNT,
            result_checksum,
            verifier.QUERY_THROUGHPUT_OPERATIONS,
            verifier.QUERY_THROUGHPUT_OPERATIONS,
            throughput_wall_ns,
            result_checksum,
            result_checksum,
            verifier.HELDOUT_QUERY_COUNT,
        )
    )
    data.extend(
        struct.pack(
            "<" + "Q" * len(per_query_checksums),
            *per_query_checksums,
        )
    )
    data.extend(struct.pack("<" + "Q" * len(latency_samples), *latency_samples))

    levels_digest = hashlib.sha256()
    levels_digest.update(b"infinity-hnsw-levels-v1")
    levels_digest.update(struct.pack("<Q", VECTORS))
    graph_digest = hashlib.sha256()
    graph_digest.update(b"infinity-hnsw-graph-v1")
    graph_digest.update(struct.pack("<QIIQQ", VECTORS, 0, 0, 64, 32))
    for ordinal in range(VECTORS):
        neighbor = (ordinal + 1) % VECTORS
        data.extend(struct.pack("<iqIIi", 0, ordinal, 64, 1, neighbor))
        levels_digest.update(struct.pack("<II", ordinal, 0))
        graph_digest.update(struct.pack("<IIQ", ordinal, 0, ordinal))
        graph_digest.update(struct.pack("<IQI", 0, 1, neighbor))

    points: list[dict[str, Any]] = []
    point_details: list[dict[str, str]] = []
    for k, ef in verifier.RECALL_POINTS:
        data.extend(
            struct.pack(
                "<IIQ",
                k,
                ef,
                verifier.HELDOUT_QUERY_COUNT * k,
            )
        )
        returned_ids: list[str] = []
        distance_digest = hashlib.sha256()
        for query_index in range(verifier.HELDOUT_QUERY_COUNT):
            for label in truth["top_100"][query_index][:k]:
                distance_bytes = struct.pack(
                    "<f",
                    truth["distances"][query_index][label],
                )
                distance_bits = struct.unpack("<I", distance_bytes)[0]
                data.extend(struct.pack("<qI", label, distance_bits))
                returned_ids.append(str(label))
                distance_digest.update(distance_bytes)
        points.append(
            {
                "k": k,
                "ef": ef,
                "eligible_hits": verifier.HELDOUT_QUERY_COUNT * k,
                "possible_hits": verifier.HELDOUT_QUERY_COUNT * k,
                "recall": 1.0,
            }
        )
        point_details.append(
            {
                "returned_ids": ",".join(returned_ids),
                "returned_distances_sha256": distance_digest.hexdigest(),
            }
        )

    degree_counts = [0] * 65
    degree_counts[1] = VECTORS
    audit = {
        "schema_version": verifier.AUDIT_SIDECAR_SCHEMA_VERSION,
        "query": verifier.query_performance_record(
            truth["queries_sha256"],
            latency_samples,
            verifier.QUERY_LATENCY_SAMPLE_COUNT,
            result_checksum,
            verifier.QUERY_THROUGHPUT_OPERATIONS,
            verifier.QUERY_THROUGHPUT_OPERATIONS,
            throughput_wall_ns,
            result_checksum,
            result_checksum,
            per_query_checksums,
        ),
        "execution_witness": {
            name: 0 for name in verifier.EXECUTION_WITNESS_FIELDS
        },
        "graph": {
            "valid": True,
            "vertex_count": VECTORS,
            "reachable_count": VECTORS,
            "directed_edges": VECTORS,
            "level0_directed_edges": VECTORS,
            "max_level": 0,
            "entry_point": 0,
            "level0_capacity": 64,
            "upper_capacity": 32,
            "levels_sha256": levels_digest.hexdigest(),
            "graph_sha256": graph_digest.hexdigest(),
            "level_histogram": f"0:{VECTORS}",
            "degree_histograms": "L0:" + ",".join(map(str, degree_counts)),
        },
        "recall": {
            "query_count": verifier.HELDOUT_QUERY_COUNT,
            "query_seed": verifier.HELDOUT_QUERY_SEED,
            "queries_sha256": truth["queries_sha256"],
            "truth_sha256": truth["truth_sha256"],
            "points": points,
        },
    }
    details = {
        "truth_tie_counts_at_10": list(truth["tie_counts_at_10"]),
        "truth_tie_counts_at_100": list(truth["tie_counts_at_100"]),
        "points": point_details,
    }
    return bytes(data), audit, details


def write_manifest(directory: Path) -> None:
    lines = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            lines.append(f"{sha256(path)}  {path.relative_to(directory).as_posix()}")
    (directory / "MANIFEST.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def frozen_schedule() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for pair, treatment, participants, engines in PAIR_PLAN:
        for engine in engines:
            entries.append(
                {
                    "sequence": len(entries),
                    "pair": pair,
                    "pair_order": "/".join(engines),
                    "treatment": treatment,
                    "engine": engine,
                    "participants": participants,
                }
            )
    return entries


def benchmark_environment(participants: int) -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "OMP_NUM_THREADS": str(participants),
        "OMP_PROC_BIND": "FALSE",
        "OMP_THREAD_LIMIT": str(participants),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": "/private/tmp",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def attestation_fixture() -> tuple[str, dict[str, Any]]:
    snapshot = {
        "dynamic_cdhash": "a" * 40,
        "held_executable": {"device": 1, "inode": 2, "size": 3},
        "mapped_executable": {"device": 1, "inode": 2, "size": 3},
        "mapped_vnode_matches_held_fd": True,
        "image_count": 1,
        "images": [
            {
                "path_hex": "2f66697874757265",
                "uuid": "b" * 32,
                "shared_cache": False,
            }
        ],
    }
    lines = [f"attestation_schema={verifier.ATTESTATION_SCHEMA_VERSION}"]
    for phase in ("before", "after"):
        prefix = f"attestation_{phase}_"
        lines.extend(
            (
                prefix + "dynamic_cdhash=" + snapshot["dynamic_cdhash"],
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
                for barrier in verifier.ATTESTATION_BARRIER_PHASES[:3]
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
    return (
        "\n".join(lines) + "\n",
        {
            "schema_version": verifier.ATTESTATION_SCHEMA_VERSION,
            "barriers": list(verifier.ATTESTATION_BARRIER_PHASES),
            "before": snapshot,
            "after": snapshot,
            "initial_image_adds": 1,
            "later_image_adds": 0,
            "image_removes": 0,
            "image_set_unchanged": True,
            "status": "PASS",
        },
    )


def attestation_barrier_fixture(
    process_group: int,
    cold_build_ns: int,
    *,
    benchmark_path: str,
    benchmark_bytes: int,
) -> list[dict[str, Any]]:
    stopped_pid = process_group + 100_000
    result: list[dict[str, Any]] = []
    before_index_resume = 2_000_000_001 + process_group * 1_000_000_000
    after_index_observed = before_index_resume + cold_build_ns + 1_000_000
    for index, phase in enumerate(verifier.ATTESTATION_BARRIER_PHASES):
        observed = 1_000_000_000 + process_group * 1_000_000_000 + index * 100
        resumed = observed + 1
        if phase == "before-index":
            observed = before_index_resume - 1
            resumed = before_index_resume
        elif phase == "after-index":
            observed = after_index_observed
            resumed = observed + 1
        elif phase == "after-work":
            observed = after_index_observed + 100
            resumed = observed + 1
        result.append(
            {
                "phase": phase,
                "process_group": process_group,
                "stopped_pid": stopped_pid,
                "observed_at_monotonic_ns": observed,
                "resumed_at_monotonic_ns": resumed,
                "processes": [
                    {
                        "pid": process_group,
                        "parent_pid": 4242,
                        "process_group": process_group,
                        "status_code": 3,
                        "status": "sleeping",
                        "start_time_seconds": 100,
                        "start_time_microseconds": 1,
                        "name": "time",
                        "executable_path": "/usr/bin/time",
                        "executable_device": 10,
                        "executable_inode": 11,
                        "executable_size": 12,
                        "child_pids": [stopped_pid],
                    },
                    {
                        "pid": stopped_pid,
                        "parent_pid": process_group,
                        "process_group": process_group,
                        "status_code": 4,
                        "status": "stopped",
                        "start_time_seconds": 101,
                        "start_time_microseconds": 2,
                        "name": "benchmark",
                        "executable_path": benchmark_path,
                        "executable_device": 20,
                        "executable_inode": 21,
                        "executable_size": benchmark_bytes,
                        "child_pids": [],
                    },
                ],
            }
        )
    return result


def index_timing_binding_fixture(
    barriers: list[dict[str, Any]],
    cold_build_ns: int,
) -> dict[str, Any]:
    outer_start = barriers[1]["resumed_at_monotonic_ns"]
    outer_end = barriers[2]["observed_at_monotonic_ns"]
    outer_ns = outer_end - outer_start
    return {
        "schema_version": verifier.INDEX_TIMING_BINDING_SCHEMA_VERSION,
        "clock": "python-time-monotonic-ns",
        "outer_start_phase": "before-index.resumed",
        "outer_end_phase": "after-index.observed",
        "outer_start_monotonic_ns": outer_start,
        "outer_end_monotonic_ns": outer_end,
        "outer_index_window_ns": outer_ns,
        "reported_cold_build_ns": cold_build_ns,
        "boundary_overhead_ns": outer_ns - cold_build_ns,
        "maximum_boundary_overhead_ns": (
            verifier.INDEX_TIMING_MAXIMUM_BOUNDARY_OVERHEAD_NS
        ),
    }


def process_evidence_fixture(
    barriers: list[dict[str, Any]],
    *,
    process_group: int,
    benchmark_path: str,
    benchmark_bytes: int,
) -> dict[str, Any]:
    return {
        "schema_version": verifier.PROCESS_EVIDENCE_SCHEMA_VERSION,
        "supervisor_pid": 4242,
        "process_group": process_group,
        "wrapper_executable": {
            "path": "/usr/bin/time",
            "device": 10,
            "inode": 11,
            "size": 12,
        },
        "benchmark_executable": {
            "path": benchmark_path,
            "device": 20,
            "inode": 21,
            "size": benchmark_bytes,
        },
        "barrier_phases": list(verifier.ATTESTATION_BARRIER_PHASES),
        "benchmark_pid": process_group + 100_000,
        "terminal_observation": {
            "observed_at_monotonic_ns": (
                barriers[-1]["resumed_at_monotonic_ns"] + 1
            ),
            "pid": process_group,
            "code": verifier.CLD_EXITED,
            "status": 0,
        },
        "pre_reap_quiescence": {
            "observed_at_monotonic_ns": (
                barriers[-1]["resumed_at_monotonic_ns"] + 2
            ),
            "process_group": process_group,
            "leader_pid": process_group,
            "member_pids": [process_group],
            "no_members_except_leader": True,
        },
        "reap": {
            "completed_at_monotonic_ns": (
                barriers[-1]["resumed_at_monotonic_ns"] + 3
            ),
            "pid": process_group,
            "status_validated": True,
            "returncode": 0,
        },
    }


def driver_fields(
    engine: str,
    *,
    participants: int,
    dataset: dict[str, Any],
    cold_build_ns: int,
    audit: dict[str, Any],
    audit_details: dict[str, Any],
    sidecar: bytes,
    campaign_nonce: str,
    schedule_sequence: int,
    role_id: int,
    binary_sha256: str,
) -> dict[str, str]:
    fields = {
        "status": "PASS",
        "scope": SCOPE,
        "order": f"{engine}-only",
        "data_fnv1a64": str(dataset["fnv1a64"]),
        "data_sha256": str(dataset["sha256"]),
        "data_bytes": str(dataset["bytes"]),
        "vectors": str(VECTORS),
        "dimensions": str(DIMENSIONS),
        "M": "32",
        "ef_construction": "200",
        "ef_search": "32",
        "participants": str(participants),
        "build_grain": "0",
        "query_benchmark_schema": str(verifier.QUERY_BENCHMARK_SCHEMA_VERSION),
        "query_unique_queries": str(verifier.HELDOUT_QUERY_COUNT),
        "query_k": str(verifier.QUERY_K),
        "query_ef_search": str(verifier.QUERY_EF_SEARCH),
        "query_recall_floor": str(verifier.QUERY_RECALL_FLOOR),
        "query_maximum_absolute_recall_gap": str(
            verifier.QUERY_MAXIMUM_ABSOLUTE_RECALL_GAP
        ),
        "query_warmup_passes": str(verifier.QUERY_WARMUP_PASSES),
        "query_measured_passes": str(verifier.QUERY_MEASURED_PASSES),
        "query_latency_concurrency": str(verifier.QUERY_LATENCY_CONCURRENCY),
        "query_throughput_concurrency": str(
            verifier.QUERY_THROUGHPUT_CONCURRENCY
        ),
        "query_percentile_method": verifier.QUERY_PERCENTILE_METHOD,
        "query_transaction_definition": verifier.QUERY_TRANSACTION_DEFINITION,
        "query_timed_corpus_sha256": audit["query"][
            "timed_query_corpus_sha256"
        ],
        f"{engine}_valid": "1",
        f"{engine}_threads": str(participants),
        f"{engine}_index_size": str(VECTORS),
        f"{engine}_cold_build_ns": str(cold_build_ns),
        f"{engine}_insert_call_ns": str(cold_build_ns - 1_000),
        f"{engine}_self_recall_at_1": "1.0",
        f"{engine}_distance_checksum": "12.5",
        f"{engine}_query_latency_sample_count": str(
            verifier.QUERY_LATENCY_SAMPLE_COUNT
        ),
        f"{engine}_query_latency_validated_operations": str(
            audit["query"]["latency_validated_operations"]
        ),
        f"{engine}_query_latency_validated_result_checksum": str(
            audit["query"]["latency_validated_result_checksum"]
        ),
        f"{engine}_query_throughput_operations": str(
            audit["query"]["throughput_operations"]
        ),
        f"{engine}_query_throughput_validated_operations": str(
            audit["query"]["throughput_validated_operations"]
        ),
        f"{engine}_query_throughput_wall_ns": str(
            audit["query"]["throughput_wall_ns"]
        ),
        f"{engine}_query_throughput_validated_result_checksum": str(
            audit["query"]["throughput_validated_result_checksum"]
        ),
        f"{engine}_query_result_checksum": str(
            audit["query"]["result_checksum"]
        ),
        f"{engine}_query_per_query_checksum_count": str(
            len(audit["query"]["per_query_checksums"])
        ),
        f"{engine}_query_per_query_checksums": ",".join(
            str(value) for value in audit["query"]["per_query_checksums"]
        ),
        "audit_sidecar_schema": str(verifier.AUDIT_SIDECAR_SCHEMA_VERSION),
        "audit_sidecar_bytes": str(len(sidecar)),
        "audit_sidecar_sha256": hashlib.sha256(sidecar).hexdigest(),
        "campaign_nonce": campaign_nonce,
        "schedule_sequence": str(schedule_sequence),
        "role_id": str(role_id),
        "binary_sha256": binary_sha256,
        "dataset_sha256": str(dataset["sha256"]),
    }
    graph = audit["graph"]
    fields.update(
        {
            f"{engine}_{name}": str(value)
            for name, value in audit["execution_witness"].items()
        }
    )
    fields.update(
        {
            f"{engine}_graph_valid": "1",
            f"{engine}_graph_vertex_count": str(graph["vertex_count"]),
            f"{engine}_graph_reachable_count": str(graph["reachable_count"]),
            f"{engine}_graph_directed_edges": str(graph["directed_edges"]),
            f"{engine}_graph_level0_directed_edges": str(
                graph["level0_directed_edges"]
            ),
            f"{engine}_graph_max_level": str(graph["max_level"]),
            f"{engine}_graph_entry_point": str(graph["entry_point"]),
            f"{engine}_graph_level0_capacity": str(graph["level0_capacity"]),
            f"{engine}_graph_upper_capacity": str(graph["upper_capacity"]),
            f"{engine}_graph_levels_sha256": graph["levels_sha256"],
            f"{engine}_graph_sha256": graph["graph_sha256"],
            f"{engine}_graph_level_histogram": graph["level_histogram"],
            f"{engine}_graph_degree_histograms": graph["degree_histograms"],
        }
    )
    recall = audit["recall"]
    fields.update(
        {
            "heldout_query_count": str(recall["query_count"]),
            "heldout_query_seed": str(recall["query_seed"]),
            "heldout_queries_sha256": recall["queries_sha256"],
            "heldout_truth_sha256": recall["truth_sha256"],
            "heldout_truth_tie_counts_at_10": ",".join(
                str(value) for value in audit_details["truth_tie_counts_at_10"]
            ),
            "heldout_truth_tie_counts_at_100": ",".join(
                str(value) for value in audit_details["truth_tie_counts_at_100"]
            ),
        }
    )
    for point, detail in zip(recall["points"], audit_details["points"]):
        k = point["k"]
        ef = point["ef"]
        fields.update(
            {
                f"{engine}_recall_at_{k}_ef_{ef}": repr(point["recall"]),
                f"{engine}_returned_ids_ef_{ef}_k_{k}": detail["returned_ids"],
                (
                    f"{engine}_returned_distances_sha256_ef_{ef}_k_{k}"
                ): detail["returned_distances_sha256"],
            }
        )
    if engine == "infinity":
        average_bucket_size = (VECTORS - 1) // participants + 1
        bucket_size = max(1024, average_bucket_size)
        submitted_tasks = (VECTORS - 1) // bucket_size + 1
        fields.update(
            {
                "infinity_submitted_tasks": str(submitted_tasks),
                "infinity_build_start": "0",
                "infinity_build_end": str(VECTORS),
            }
        )
    return fields


def median(values: list[int | float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def variability(values: list[int]) -> dict[str, Any]:
    center = median(values)
    deviations = [abs(value - center) for value in values]
    return {
        "cold_build_ns": values,
        "median_cold_build_ns": center,
        "relative_mad": median(deviations) / center,
    }


def summary_for(
    records: list[dict[str, Any]],
    *,
    dataset: dict[str, Any],
    schedule_sha256: str,
) -> dict[str, Any]:
    metrics = verifier.recompute_metrics(records)
    quality = verifier.recompute_audit_quality(records)
    acceptance = {
        "all_runs_valid": True,
        "ingestion_relative_mad_at_most_0_10": all(
            details["relative_mad"] <= 0.10
            for details in metrics["ingestion"]["engines"].values()
        ),
        "ingestion_paired_ratio_relative_mad_at_most_0_10": (
            metrics["ingestion"]["paired_ratio_relative_mad"] <= 0.10
        ),
        "query_qps_relative_mad_at_most_0_10": all(
            details["qps_relative_mad"] <= 0.10
            for details in metrics["query"]["engines"].values()
        ),
        "query_paired_qps_ratio_relative_mad_at_most_0_10": (
            metrics["query"]["paired_qps_ratio_relative_mad"] <= 0.10
        ),
        "query_evidence_complete": True,
        "query_recall_floor_met": metrics["query"]["recall_floor_met"],
        "query_absolute_recall_gap_at_most_0_005": metrics["query"][
            "absolute_recall_gap_met"
        ],
        "idle_protocol_met": True,
        "idle_to_launch_at_most_1_second": True,
        "host_resource_protocol_met": True,
        "dataset_integrity_met": True,
        "raw_graph_audit_met": quality["graph"]["all_valid_and_reachable"],
        "heldout_recall_audit_met": True,
        "paired_recall_deficit_at_most_0_005": quality["recall"][
            "all_points_within_deficit"
        ],
        "exact_balanced_schedule_met": True,
    }
    return {
        "status": "pass" if all(acceptance.values()) else "fail",
        "scope": SCOPE,
        "claim_eligible": False,
        "power": {
            "sources": ["ac"],
            "policies": ["source-record-only"],
        },
        "idle": {
            "minimum_percent": 95.0,
            "window_seconds": 15,
            "policy": "default-95-percent",
        },
        "resources": {
            "global_swap_policy": "development-record-only",
            "maximum_observed_global_swap_growth_bytes": 0,
            "indexing_window_peak_rss_established": False,
        },
        "quality": quality,
        "workload": {
            "vectors": VECTORS,
            "dimensions": DIMENSIONS,
            "M": 32,
            "ef_construction": 200,
            "ef_search": 32,
            "participants": 12,
            "data_fnv1a64": dataset["fnv1a64"],
            "data_sha256": dataset["sha256"],
            "data_bytes": dataset["bytes"],
            "schedule_sha256": schedule_sha256,
        },
        "ingestion": metrics["ingestion"],
        "query": metrics["query"],
        "acceptance": acceptance,
        "limitations": [
            "Synthetic unit-test evidence.",
            verifier.DEVELOPMENT_SWAP_LIMITATION,
            verifier.INDEXING_WINDOW_RSS_LIMITATION,
            verifier.QUERY_SCOPE_LIMITATION,
        ],
    }


def build_valid_evidence(directory: Path) -> None:
    directory.mkdir()
    dataset_bytes = canonical_dataset_bytes()
    dataset_path = directory / DATASET_NAME
    dataset_path.write_bytes(dataset_bytes)
    dataset_path.chmod(0o444)
    dataset = {
        "path": DATASET_NAME,
        "seed": 0,
        "generator": "python-random.Random.random",
        "encoding": "raw-little-endian-float32",
        "vectors": VECTORS,
        "dimensions": DIMENSIONS,
        "values": VECTORS * DIMENSIONS,
        "bytes_per_value": 4,
        "mode": "0o444",
        "sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "fnv1a64": fnv1a64(dataset_bytes),
        "bytes": len(dataset_bytes),
    }
    write_json(directory / "dataset.json", dataset)
    truth = verifier.heldout_truth_for_validated_dataset(
        dataset_path,
        dataset["sha256"],
    )
    heldout = synthetic_heldout_manifest(truth)
    heldout_path = directory / verifier.HELDOUT_MANIFEST_FILENAME
    write_json(heldout_path, heldout)

    schedule = frozen_schedule()
    write_json(directory / "schedule.json", schedule)
    schedule_digest = sha256(directory / "schedule.json")
    campaign_binding = {
        "schema_version": verifier.CAMPAIGN_BINDING_SCHEMA_VERSION,
        "campaign_nonce": CAMPAIGN_NONCE,
        "dataset_sha256": dataset["sha256"],
        "schedule_sha256": schedule_digest,
        "roles": {
            engine: {
                "role_id": verifier.STANDALONE_ROLE_IDS[engine],
                "engine": engine,
                "binary_sha256": BINARY_RECORDS[engine]["sha256"],
            }
            for engine in ("infinity", "faiss")
        },
    }

    runner_capture = directory / verifier.RUNNER_CAPTURE_FILENAME
    runner_capture.write_text(
        "#!/usr/bin/env python3\n# synthetic archived campaign runner\n",
        encoding="utf-8",
    )
    runner_digest = sha256(runner_capture)
    verifier_capture = directory / verifier.VERIFIER_CAPTURE_FILENAME
    verifier_capture.write_text(
        "#!/usr/bin/env python3\n# synthetic standalone verifier\n",
        encoding="utf-8",
    )
    verifier_digest = sha256(verifier_capture)
    tools = executable_records(directory)
    python_record = {
        "configured_path": PYTHON_PATH,
        "resolved_path": PYTHON_PATH,
        **write_content_blob(directory, PYTHON_BYTES),
        "version_command": [PYTHON_PATH, "--version"],
        "version_returncode": 0,
        "version": "Python 3.13.3",
    }
    runtime_artifact_hashes = {
        details["path"]: details["sha256"] for details in BINARY_RECORDS.values()
    }
    runtime_artifact_hashes.update(
        {
            LIBFAISS_PATH: LIBFAISS_SHA256,
            LIBOMP_PATH: LIBOMP_SHA256,
            RUNNER_SOURCE_PATH: runner_digest,
            VERIFIER_SOURCE_PATH: verifier_digest,
            PYTHON_PATH: python_record["sha256"],
            **{record["resolved_path"]: record["sha256"] for record in tools.values()},
        }
    )
    runtime_artifact_sizes = {
        details["path"]: details["bytes"] for details in BINARY_RECORDS.values()
    }
    runtime_artifact_sizes.update(
        {
            LIBFAISS_PATH: len(LIBFAISS_BYTES),
            LIBOMP_PATH: len(LIBOMP_BYTES),
            RUNNER_SOURCE_PATH: runner_capture.stat().st_size,
            VERIFIER_SOURCE_PATH: verifier_capture.stat().st_size,
            PYTHON_PATH: len(PYTHON_BYTES),
            **{record["resolved_path"]: record["bytes"] for record in tools.values()},
        }
    )
    runtime_artifact_contents = {
        BINARY_RECORDS["infinity"]["path"]: INFINITY_BINARY_BYTES,
        BINARY_RECORDS["faiss"]["path"]: FAISS_BINARY_BYTES,
        LIBFAISS_PATH: LIBFAISS_BYTES,
        LIBOMP_PATH: LIBOMP_BYTES,
        RUNNER_SOURCE_PATH: runner_capture.read_bytes(),
        VERIFIER_SOURCE_PATH: verifier_capture.read_bytes(),
        PYTHON_PATH: PYTHON_BYTES,
        **{
            record["resolved_path"]: TOOL_CONTENTS[name]
            for name, record in tools.items()
        },
    }
    runtime_artifact_snapshots = [
        {
            "original_path": path,
            **write_content_blob(directory, runtime_artifact_contents[path]),
        }
        for path in sorted(runtime_artifact_hashes)
    ]

    infinity_source = source_record(
        directory,
        f"{REPO_SOURCE_ROOT}/tools/apple_silicon/native_hnsw_smoke/infinity_fixture.cpp",
        INFINITY_SOURCE_BYTES,
    )
    faiss_source = source_record(
        directory,
        f"{REPO_SOURCE_ROOT}/tools/apple_silicon/native_hnsw_smoke/faiss_fixture.cpp",
        FAISS_HARNESS_SOURCE_BYTES,
    )
    faiss_library_source = source_record(
        directory,
        f"{FAISS_SOURCE_ROOT}/faiss/IndexFixture.cpp",
        FAISS_LIBRARY_SOURCE_BYTES,
    )
    ctpl_source = source_record(
        directory,
        f"{VCPKG_INCLUDE_ROOT}/ctpl_stl.h",
        b"// synthetic ctpl header\n",
    )
    simde_source = source_record(
        directory,
        f"{VCPKG_INCLUDE_ROOT}/simde/simde-common.h",
        b"// synthetic SIMDe header\n",
    )
    faiss_dependencies = [
        dependency_record(
            LIBFAISS_PATH,
            LIBFAISS_SHA256,
            referenced_by=BINARY_RECORDS["faiss"]["path"],
            size=runtime_artifact_sizes[LIBFAISS_PATH],
        ),
        dependency_record(
            LIBOMP_PATH,
            LIBOMP_SHA256,
            referenced_by=BINARY_RECORDS["faiss"]["path"],
            size=runtime_artifact_sizes[LIBOMP_PATH],
        ),
    ]
    builds = {
        "infinity": write_build_capture(
            directory,
            label="infinity",
            source=infinity_source,
            binary=BINARY_RECORDS["infinity"],
            dependencies=[],
            tools=tools,
            extra_build_dependencies=[ctpl_source, simde_source],
        ),
        "faiss": write_build_capture(
            directory,
            label="faiss",
            source=faiss_source,
            binary=BINARY_RECORDS["faiss"],
            dependencies=faiss_dependencies,
            tools=tools,
        ),
    }

    faiss_cache_name = "faiss-library-CMakeCache.txt"
    faiss_compile_name = "faiss-library-compile_commands.json"
    faiss_commands_name = "faiss-library-ninja-commands.txt"
    faiss_noop_name = "faiss-library-ninja-noop.txt"
    (directory / faiss_cache_name).write_text(
        cmake_cache_text(linked_faiss=True),
        encoding="utf-8",
    )
    write_json(
        directory / faiss_compile_name,
        [
            {
                "directory": "/fixture/build",
                "file": faiss_library_source["absolute_path"],
                "command": (
                    "/fixture/tools/clang++ -O3 -DNDEBUG "
                    "-DFAISS_DISABLE_GLOBAL_HNSW_STATS=1 -arch arm64 "
                    "-isysroot /fixture/MacOSX.sdk "
                    "-mmacosx-version-min=14.0 "
                    f"-c {faiss_library_source['absolute_path']} -o faiss.o"
                ),
            }
        ],
    )
    (directory / faiss_commands_name).write_text(
        "/fixture/tools/clang++ -O3 -DNDEBUG "
        "-DFAISS_DISABLE_GLOBAL_HNSW_STATS=1 -arch arm64 "
        "-isysroot /fixture/MacOSX.sdk -mmacosx-version-min=14.0 "
        f"-c {faiss_library_source['absolute_path']} -o faiss.o\n"
        "/fixture/tools/clang++ -O3 -DNDEBUG -arch arm64 "
        "-isysroot /fixture/MacOSX.sdk -mmacosx-version-min=14.0 "
        f"faiss.o -o {LIBFAISS_PATH}\n",
        encoding="utf-8",
    )
    (directory / faiss_noop_name).write_text(
        "ninja: no work to do.\n",
        encoding="utf-8",
    )
    faiss_library_link_arguments = [
        "/fixture/tools/clang++",
        "-O3",
        "-DNDEBUG",
        "-arch",
        "arm64",
        "-isysroot",
        "/fixture/MacOSX.sdk",
        "-mmacosx-version-min=14.0",
        "faiss.o",
        "-o",
        LIBFAISS_PATH,
    ]
    faiss_library_build_input_closure = write_synthetic_build_input_closure(
        directory,
        capture_prefix="faiss-library",
        requested_target=Path(LIBFAISS_PATH).name,
        expected_output=LIBFAISS_PATH,
        build_directory="/fixture/build",
        source=faiss_library_source,
        object_name="faiss.o",
        link_arguments=faiss_library_link_arguments,
    )
    linked_faiss_build = {
        "library": {
            "path": LIBFAISS_PATH,
            "sha256": LIBFAISS_SHA256,
            "bytes": runtime_artifact_sizes[LIBFAISS_PATH],
        },
        "build_directory": "/fixture/build",
        "cmake_home_directory": FAISS_SOURCE_ROOT,
        "cmake_cache": {
            "captured_path": faiss_cache_name,
            "sha256": sha256(directory / faiss_cache_name),
        },
        "compile_commands": {
            "captured_path": faiss_compile_name,
            "sha256": sha256(directory / faiss_compile_name),
            "entries": 1,
        },
        "settings": {
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_OSX_ARCHITECTURES": "arm64",
            "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0",
            "CMAKE_OSX_SYSROOT": "/fixture/MacOSX.sdk",
            "CMAKE_CXX_FLAGS_RELEASE": "-O3 -DNDEBUG",
            "BUILD_SHARED_LIBS": "ON",
            "FAISS_ENABLE_GPU": "OFF",
            "FAISS_ENABLE_PYTHON": "OFF",
            "FAISS_ENABLE_GLOBAL_HNSW_STATS": "OFF",
            "OpenMP_CXX_FLAGS": "-fopenmp=libomp",
            "OpenMP_CXX_INCLUDE_DIR": "/fixture/build/include",
            "OpenMP_libomp_LIBRARY": LIBOMP_PATH,
        },
        **tools,
        "ninja_commands": {
            "captured_path": faiss_commands_name,
            "sha256": sha256(directory / faiss_commands_name),
        },
        "ninja_noop": {
            "captured_path": faiss_noop_name,
            "sha256": sha256(directory / faiss_noop_name),
        },
        "build_input_closure": faiss_library_build_input_closure,
        "response_files": [],
        "compiled_sources": 1,
        "openmp_runtime": dependency_record(
            LIBOMP_PATH,
            LIBOMP_SHA256,
            referenced_by=LIBFAISS_PATH,
            size=runtime_artifact_sizes[LIBOMP_PATH],
        ),
    }

    write_json(
        directory / "compiled-source-hashes.json",
        {
            "infinity": [infinity_source],
            "faiss": [faiss_source],
            "faiss_library": [faiss_library_source],
        },
    )
    libomp_recipe_blob = write_content_blob(
        directory,
        b"#!/bin/sh\n# synthetic libomp recipe\n",
    )
    omp_header_blob = write_content_blob(
        directory,
        b"/* synthetic omp.h */\n",
    )
    libomp_blob = write_content_blob(directory, LIBOMP_BYTES)
    self_check_libomp_digest = libomp_blob["sha256"]
    if self_check_libomp_digest != LIBOMP_SHA256:
        raise AssertionError("Synthetic libomp fixture digest differs")
    write_json(
        directory / "benchmark-input-source-hashes.json",
        {
            "native_harness": {
                "root": (f"{REPO_SOURCE_ROOT}/tools/apple_silicon/native_hnsw_smoke"),
                "files": [faiss_source, infinity_source],
            },
            "ctpl": {"root": VCPKG_INCLUDE_ROOT, "files": [ctpl_source]},
            "simde_headers": {
                "root": f"{VCPKG_INCLUDE_ROOT}/simde",
                "files": [simde_source],
            },
            "libomp_recipe": {
                "path": "tools/apple_silicon/prepare_d0_libomp.sh",
                **libomp_recipe_blob,
            },
            "omp_header": {
                "path": "/fixture/build/include/omp.h",
                **omp_header_blob,
            },
            "libomp_runtime": {
                "path": LIBOMP_PATH,
                **libomp_blob,
            },
        },
    )
    faiss_history = synthetic_faiss_history()
    repo_git_entries = {
        "README.md": (
            "100644",
            "blob",
            verifier.git_object_id("blob", b"synthetic repository\n"),
        )
    }
    repository_records = [
        synthetic_git_repository(
            directory,
            index=0,
            root=FAISS_SOURCE_ROOT,
            entries=faiss_history["derivative_entries"],
            status="",
            diff="",
            parents=(faiss_history["base_head"],),
        ),
        synthetic_git_repository(
            directory,
            index=1,
            root=REPO_SOURCE_ROOT,
            entries=repo_git_entries,
            status=" M fixture\n",
            diff="diff\n",
        ),
    ]
    if faiss_history["base_head"] != verifier.PINNED_FAISS_COMMIT:
        raise AssertionError("Synthetic FAISS base commit is not the patched pin")
    if faiss_history["base_tree"] != verifier.PINNED_FAISS_BASE_TREE:
        raise AssertionError("Synthetic FAISS base tree is not the patched pin")
    if (
        faiss_history["derivative_tree"]
        != verifier.PINNED_FAISS_STATS_DISABLED_TREE
    ):
        raise AssertionError("Synthetic FAISS derivative tree is not the patched pin")
    if (
        hashlib.sha256(FAISS_PATCH_BYTES).hexdigest()
        != verifier.PINNED_FAISS_STATS_PATCH_SHA256
    ):
        raise AssertionError("Synthetic FAISS patch is not the patched pin")
    (directory / verifier.FAISS_BASE_COMMIT_OBJECT_FILENAME).write_bytes(
        faiss_history["base_commit"]
    )
    (
        directory / verifier.FAISS_BASE_TREE_LISTING_FILENAME
    ).write_text(
        synthetic_tree_listing(faiss_history["base_entries"]),
        encoding="utf-8",
    )
    (directory / verifier.FAISS_DERIVATIVE_DIFF_FILENAME).write_bytes(
        FAISS_PATCH_BYTES
    )
    (directory / verifier.FAISS_CHANGED_PATHS_FILENAME).write_text(
        "".join(f"{path}\n" for path in verifier.PINNED_FAISS_STATS_PATCH_PATHS),
        encoding="utf-8",
    )
    (directory / verifier.FAISS_STATS_PATCH_FILENAME).write_bytes(
        FAISS_PATCH_BYTES
    )
    source_provenance = {
        "benchmark_input_sources_path": "benchmark-input-source-hashes.json",
        "benchmark_input_sources_sha256": sha256(
            directory / "benchmark-input-source-hashes.json"
        ),
        "compiled_source_hashes_path": "compiled-source-hashes.json",
        "compiled_source_hashes_sha256": sha256(
            directory / "compiled-source-hashes.json"
        ),
        "git_repositories": repository_records,
        "faiss_audited_reference": {
            "policy": "single-parent-pinned-tree-v1",
            "repository_root": FAISS_SOURCE_ROOT,
            "base": {
                "commit": verifier.PINNED_FAISS_COMMIT,
                "tree": verifier.PINNED_FAISS_BASE_TREE,
                "tag": "v1.15.0",
            },
            "derivative": {
                "head": faiss_history["derivative_head"],
                "tree": verifier.PINNED_FAISS_STATS_DISABLED_TREE,
                "parent": verifier.PINNED_FAISS_COMMIT,
                "dirty": False,
                "changed_paths": list(
                    verifier.PINNED_FAISS_STATS_PATCH_PATHS
                ),
            },
            "patch": {
                "source_path": (
                    f"{REPO_SOURCE_ROOT}/{verifier.FAISS_STATS_PATCH_SOURCE}"
                ),
                "captured_path": verifier.FAISS_STATS_PATCH_FILENAME,
                "sha256": verifier.PINNED_FAISS_STATS_PATCH_SHA256,
            },
            "feature": {
                "cmake_option": "FAISS_ENABLE_GLOBAL_HNSW_STATS",
                "required_value": "OFF",
                "compile_definition": "FAISS_DISABLE_GLOBAL_HNSW_STATS=1",
            },
        },
    }
    preflight = {
        "evidence_schema_version": verifier.EVIDENCE_SCHEMA_VERSION,
        "execution_mode": verifier.FULL_CAMPAIGN_MODE,
        "captured_at": "2026-08-20T00:00:00Z",
        "scope": SCOPE,
        "repo": REPO_SOURCE_ROOT,
        "runner": {
            "source_path": RUNNER_SOURCE_PATH,
            "captured_path": verifier.RUNNER_CAPTURE_FILENAME,
            "sha256": runner_digest,
            "bytes": runner_capture.stat().st_size,
            "argv": [
                RUNNER_SOURCE_PATH,
                "--repo",
                REPO_SOURCE_ROOT,
                "--infinity-binary",
                BINARY_RECORDS["infinity"]["path"],
                "--faiss-binary",
                BINARY_RECORDS["faiss"]["path"],
                "--output-dir",
                "/fixture/evidence",
            ],
            "invocation_working_directory": REPO_SOURCE_ROOT,
            "python_executable": python_record,
            "effective_arguments": {
                "repo": REPO_SOURCE_ROOT,
                "infinity_binary": BINARY_RECORDS["infinity"]["path"],
                "faiss_binary": BINARY_RECORDS["faiss"]["path"],
                "output_directory": "/fixture/evidence",
                "idle_minimum_percent": 95.0,
                "idle_timeout_seconds": 300.0,
                "member_timeout_seconds": 300.0,
                "preflight_only": False,
            },
        },
        "verifier": {
            "source_path": VERIFIER_SOURCE_PATH,
            "captured_path": verifier.VERIFIER_CAPTURE_FILENAME,
            "sha256": verifier_digest,
            "bytes": verifier_capture.stat().st_size,
        },
        "host": {
            "machine": "arm64",
            "macos": "26.2",
            "kernel": "25.2.0",
            "python": "3.9.6",
            "disk": {
                "total_bytes": 100_000_000_000,
                "used_bytes": 10_000_000_000,
                "free_bytes": 90_000_000_000,
                "minimum_reserve_bytes": 12_884_901_888,
                "declared_worst_case_new_bytes": 1_073_741_824,
                "required_free_bytes": 13_958_643_712,
            },
        },
        "environment_control": {
            "parent_performance_environment": {},
            "unsafe_parent_environment_policy": (
                verifier.expected_parent_environment_policy()
            ),
            "child_environment_template": benchmark_environment(12),
            "inherits_parent_environment": False,
        },
        "dataset": {
            **dataset,
            "record_path": "dataset.json",
            "record_sha256": sha256(directory / "dataset.json"),
        },
        "heldout": {
            **heldout,
            "record_path": verifier.HELDOUT_MANIFEST_FILENAME,
            "record_sha256": sha256(heldout_path),
        },
        "schedule": {
            "path": "schedule.json",
            "sha256": schedule_digest,
            "members": 18,
            "pair_plan": [
                {
                    "pair": pair,
                    "treatment": treatment,
                    "participants": participants,
                    "order": "/".join(engines),
                }
                for pair, treatment, participants, engines in PAIR_PLAN
            ],
        },
        "campaign_binding": campaign_binding,
        "binaries": BINARY_RECORDS,
        "runtime_artifact_hashes": runtime_artifact_hashes,
        "runtime_artifact_snapshots": runtime_artifact_snapshots,
        "builds": builds,
        "linked_faiss_build": linked_faiss_build,
        "source_provenance": source_provenance,
        "protocol": {
            "audit_sidecar_schema_version": verifier.AUDIT_SIDECAR_SCHEMA_VERSION,
            "attestation_schema_version": verifier.ATTESTATION_SCHEMA_VERSION,
            "attestation_barrier_phases": list(
                verifier.ATTESTATION_BARRIER_PHASES
            ),
            "index_timing_binding_schema_version": (
                verifier.INDEX_TIMING_BINDING_SCHEMA_VERSION
            ),
            "index_timing_maximum_boundary_overhead_ns": (
                verifier.INDEX_TIMING_MAXIMUM_BOUNDARY_OVERHEAD_NS
            ),
            "process_evidence_schema_version": (
                verifier.PROCESS_EVIDENCE_SCHEMA_VERSION
            ),
            "time_wrapper_path": "/usr/bin/time",
            "query_performance": verifier.expected_query_protocol(),
            "heldout_query_count": verifier.HELDOUT_QUERY_COUNT,
            "heldout_query_seed": verifier.HELDOUT_QUERY_SEED,
            "recall_points": verifier.expected_recall_points_json(),
            "idle_minimum_percent": 95.0,
            "idle_policy": "default-95-percent",
            "idle_window_seconds": 15,
            "idle_sample_count": 16,
            "idle_to_launch_max_ns": 1_000_000_000,
            "server_ports": [23871, 23872, 23873, 23874],
            "power_policy": "source-record-only",
            "battery_minimum_percent": 20,
            "global_swap_policy": "development-record-only",
            "member_timeout_seconds": 300.0,
            "idle_timeout_seconds": 300.0,
        },
    }
    write_json(directory / "preflight.json", preflight)

    records = []
    for entry in schedule:
        sequence = entry["sequence"]
        pair = entry["pair"]
        engine = entry["engine"]
        participants = entry["participants"]
        prefix = f"{sequence:02d}-{pair}-{engine}"
        if entry["treatment"] == "measured":
            durations = INFINITY_DURATIONS if engine == "infinity" else FAISS_DURATIONS
            cold_build_ns = durations[pair]
        else:
            cold_build_ns = 100_000_000 if engine == "infinity" else 200_000_000
        role = campaign_binding["roles"][engine]
        sidecar, audit, audit_details = synthetic_audit_sidecar(
            engine,
            truth,
            campaign_nonce=campaign_binding["campaign_nonce"],
            schedule_sequence=sequence,
            role_id=role["role_id"],
            binary_sha256=role["binary_sha256"],
            dataset_sha256=campaign_binding["dataset_sha256"],
        )
        sidecar_name = f"{prefix}{verifier.AUDIT_SIDECAR_SUFFIX}"
        (directory / sidecar_name).write_bytes(sidecar)
        fields = driver_fields(
            engine,
            participants=participants,
            dataset=dataset,
            cold_build_ns=cold_build_ns,
            audit=audit,
            audit_details=audit_details,
            sidecar=sidecar,
            campaign_nonce=campaign_binding["campaign_nonce"],
            schedule_sequence=sequence,
            role_id=role["role_id"],
            binary_sha256=role["binary_sha256"],
        )

        raw_idle_name = f"{prefix}.idle-attempt-01.top.txt"
        raw_idle = "".join(
            "CPU usage: 1.0% user, 3.0% sys, 96.0% idle\n" for _ in range(16)
        )
        (directory / raw_idle_name).write_text(
            "[stdout]\n" + raw_idle + "\n[stderr]\n",
            encoding="utf-8",
        )
        host_preflight_name = f"{prefix}.host-preflight.txt"
        process_name = f"{prefix}.processes.txt"
        host_postflight_name = f"{prefix}.host-postflight.txt"
        (directory / host_preflight_name).write_text(
            raw_host_capture(),
            encoding="utf-8",
        )
        (directory / process_name).write_text(
            "  1  0  0.0  0.1 S  00:01 /sbin/launchd\n",
            encoding="utf-8",
        )
        (directory / host_postflight_name).write_text(
            raw_host_capture(),
            encoding="utf-8",
        )
        accepted_unix_ns = 10_000_000_000 + sequence * 1_000_000_000
        dataset_fingerprint = {
            "verified_at": "2026-08-20T00:00:00Z",
            "path": DATASET_NAME,
            "mode": "0o444",
            "read_only": True,
            "sha256": dataset["sha256"],
            "fnv1a64": dataset["fnv1a64"],
            "bytes": dataset["bytes"],
        }
        dataset_argument = DATASET_NAME
        swap_record = {
            "total_bytes": 0,
            "used_bytes": 0,
            "free_bytes": 0,
            "raw": (
                "vm.swapusage: total = 0.00M  used = 0.00M  "
                "free = 0.00M  (encrypted)"
            ),
        }
        process_group = 1000 + sequence
        attestation_text, attestation = attestation_fixture()
        attestation_barriers = attestation_barrier_fixture(
            process_group,
            cold_build_ns,
            benchmark_path=BINARY_RECORDS[engine]["path"],
            benchmark_bytes=BINARY_RECORDS[engine]["bytes"],
        )
        record = {
            **entry,
            "status": "pass",
            "scope": SCOPE,
            "errors": [],
            "pid": process_group,
            "started_at": "2026-08-20T00:00:01Z",
            "started_at_unix_ns": accepted_unix_ns + 100_000_000,
            "ended_at_unix_ns": accepted_unix_ns + cold_build_ns + 100_000_000,
            "command": [
                "/usr/bin/time",
                "-lp",
                BINARY_RECORDS[engine]["path"],
                dataset_argument,
                str(VECTORS),
                str(DIMENSIONS),
                "32",
                "200",
                "32",
                "8192",
                "1000",
                str(participants),
                "0",
                sidecar_name,
                campaign_binding["campaign_nonce"],
                str(sequence),
                str(role["role_id"]),
                role["binary_sha256"],
                campaign_binding["dataset_sha256"],
            ],
            "working_directory": ".",
            "binary": {
                "path": BINARY_RECORDS[engine]["path"],
                "sha256": BINARY_RECORDS[engine]["sha256"],
            },
            "environment": benchmark_environment(participants),
            "idle": {
                "status": "pass",
                "accepted_at_monotonic_ns": 20_000_000_000 + sequence,
                "accepted_at_unix_ns": accepted_unix_ns,
                "required_minimum_idle_percent": 95.0,
                "required_window_seconds": 15,
                "accepted_attempt": 1,
                "attempts": [
                    {
                        "attempt": 1,
                        "captured_at": "2026-08-20T00:00:00Z",
                        "raw_path": raw_idle_name,
                        "idle_percentages": [96.0] * 16,
                        "minimum_idle_percent": 96.0,
                        "sample_count": 16,
                        "sample_interval_seconds": 1,
                        "covered_seconds": 15,
                        "command_elapsed_seconds": 15.1,
                        "status": "pass",
                    }
                ],
            },
            "idle_to_launch_ns": 100_000_000,
            "host_before": {
                "captured_at": "2026-08-20T00:00:00Z",
                "host_raw_path": host_preflight_name,
                "process_listing_path": process_name,
                "power_source": "ac",
                "power_policy": "source-record-only",
                "ac_power": True,
                "battery_percent": 100,
                "battery_minimum_percent": 20,
                "low_power_mode": False,
                "thermal_nominal": True,
                "swap": swap_record,
                "ports": [
                    {
                        "host": "127.0.0.1",
                        "port": port,
                        "connect_errno": 61,
                        "closed": True,
                    }
                    for port in verifier.SERVER_PORTS
                ],
            },
            "host_after": {
                "captured_at": "2026-08-20T00:00:01Z",
                "raw_path": host_postflight_name,
                "power_source": "ac",
                "battery_percent": 100,
                "battery_minimum_percent": 20,
                "low_power_mode": False,
                "thermal_nominal": True,
                "swap": swap_record,
            },
            "dataset_before": dataset_fingerprint,
            "dataset_after": dataset_fingerprint,
            "runtime_artifacts_before": {
                "verified_at": "2026-08-20T00:00:00Z",
                "artifacts": [
                    {
                        "path": path,
                        "sha256": digest,
                        "bytes": runtime_artifact_sizes[path],
                    }
                    for path, digest in sorted(runtime_artifact_hashes.items())
                ],
            },
            "runtime_artifacts_after": {
                "verified_at": "2026-08-20T00:00:01Z",
                "artifacts": [
                    {
                        "path": path,
                        "sha256": digest,
                        "bytes": runtime_artifact_sizes[path],
                    }
                    for path, digest in sorted(runtime_artifact_hashes.items())
                ],
            },
            "stdout_path": f"{prefix}.stdout",
            "stderr_path": f"{prefix}.stderr",
            "fields": fields,
            "attestation": attestation,
            "attestation_barriers": attestation_barriers,
            "benchmark_process_evidence": process_evidence_fixture(
                attestation_barriers,
                process_group=process_group,
                benchmark_path=BINARY_RECORDS[engine]["path"],
                benchmark_bytes=BINARY_RECORDS[engine]["bytes"],
            ),
            "index_timing_binding": index_timing_binding_fixture(
                attestation_barriers,
                cold_build_ns,
            ),
            "audit_sidecar": {
                "path": sidecar_name,
                "sha256": hashlib.sha256(sidecar).hexdigest(),
                "bytes": len(sidecar),
            },
            "audit": audit,
            "process_lifetime_maximum_resident_set_size": 123_456,
            "process_swaps": 0,
            "global_swap_growth_bytes": 0,
        }
        (directory / record["stdout_path"]).write_text(
            attestation_text
            + "".join(f"{key}={value}\n" for key, value in fields.items()),
            encoding="utf-8",
        )
        (directory / record["stderr_path"]).write_text(
            "123456 maximum resident set size\n0 swaps\n",
            encoding="utf-8",
        )
        write_json(directory / f"{prefix}.json", record)
        records.append(record)

    write_json(directory / "runs.json", records)
    write_json(
        directory / "summary.json",
        summary_for(
            records,
            dataset=dataset,
            schedule_sha256=schedule_digest,
        ),
    )
    (directory / "README.md").write_text(
        "Synthetic completed D0 evidence.\n",
        encoding="utf-8",
    )
    write_manifest(directory)


class LinkClosureParserTests(unittest.TestCase):
    def test_executable_record_binds_configured_path_to_resolved_binary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            target = evidence / "target"
            target.write_bytes(b"target executable\n")
            configured = evidence / "configured"
            configured.symlink_to(target.name)
            unrelated = evidence / "unrelated"
            unrelated.write_bytes(b"unrelated executable\n")
            resolved_target = target.resolve()
            resolved_unrelated = unrelated.resolve()
            blob = write_content_blob(evidence, target.read_bytes())
            digest = blob["sha256"]
            record = {
                "configured_path": str(configured),
                "resolved_path": str(resolved_target),
                **blob,
                "version_command": [str(resolved_target), "--version"],
                "version_returncode": 0,
                "version": "fixture tool",
            }

            verifier.validate_executable_record(
                evidence,
                record,
                "fixture",
                runtime_artifacts={str(resolved_target): digest},
                require_configured_path_resolution=True,
            )

            forged = copy.deepcopy(record)
            forged["resolved_path"] = str(resolved_unrelated)
            forged["version_command"][0] = str(resolved_unrelated)
            with self.assertRaisesRegex(
                verifier.VerificationError,
                "resolves to a different executable",
            ):
                verifier.validate_executable_record(
                    evidence,
                    forged,
                    "fixture",
                    runtime_artifacts={str(resolved_unrelated): digest},
                    require_configured_path_resolution=True,
                )

    def test_wl_inputs_capture_bare_objects_and_section_payloads(self) -> None:
        arguments = [
            "clang++",
            "base.o",
            "-Wl,/fixture/build/injected.o",
            "-Wl,-sectcreate,__TEXT,__audit,/fixture/payload.bin",
            "-o",
            "app",
        ]
        self.assertEqual(
            verifier.derive_link_inputs(
                arguments,
                working_directory=Path("/fixture/build"),
                sysroot=Path("/fixture/MacOSX.sdk"),
                available_paths={
                    "/fixture/build/base.o",
                    "/fixture/build/injected.o",
                    "/fixture/payload.bin",
                },
                context="fixture",
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
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "unsupported -Wl option",
        ):
            verifier.derive_link_inputs(
                ["clang++", "base.o", "-Wl,-unknown_linker_switch", "-o", "app"],
                working_directory=Path("/fixture/build"),
                sysroot=Path("/fixture/MacOSX.sdk"),
                available_paths={"/fixture/build/base.o"},
                context="fixture",
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
            verifier.VerificationError,
            "exactly one expected link command",
        ):
            verifier.validate_ninja_commands(
                "\n".join(
                    (" ".join(compile_arguments), " ".join(link_arguments))
                ),
                cache=cache,
                compile_entries=[entry],
                build_directory=build,
                expected_output=build / "real/runner",
                context="fixture",
                response_files={},
                used_response_files=set(),
            )

    def test_ninja_commands_verify_configured_c_cxx_and_asm_actions(self) -> None:
        build = Path("/fixture/build")
        cache = {
            "CMAKE_C_COMPILER": "/toolchain/clang",
            "CMAKE_CXX_COMPILER": "/toolchain/clang++",
            "CMAKE_ASM_COMPILER": "/toolchain/clang-asm",
            "CMAKE_C_COMPILER_CLANG_SCAN_DEPS": "/toolchain/clang-scan-deps",
            "CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS": "/toolchain/clang-scan-deps",
            "CMAKE_ASM_COMPILER_CLANG_SCAN_DEPS": "/toolchain/clang-scan-deps",
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_OSX_ARCHITECTURES": "arm64",
            "CMAKE_OSX_SYSROOT": "/fixture/MacOSX.sdk",
            "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0",
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
        entries = []
        for compiler_key, source_name, output_name in (
            ("CMAKE_C_COMPILER", "unit.c", "unit-c.o"),
            ("CMAKE_CXX_COMPILER", "unit.cpp", "unit-cxx.o"),
            ("CMAKE_ASM_COMPILER", "unit.S", "unit-asm.o"),
        ):
            arguments = [
                cache[compiler_key],
                *common,
                "-c",
                f"/fixture/{source_name}",
                "-o",
                str(build / output_name),
            ]
            entries.append(
                {
                    "directory": str(build),
                    "file": f"/fixture/{source_name}",
                    "arguments": arguments,
                    "output": str(build / output_name),
                }
            )
        link_arguments = [
            cache["CMAKE_CXX_COMPILER"],
            *common,
            *(str(build / name) for name in ("unit-c.o", "unit-cxx.o", "unit-asm.o")),
            "-o",
            str(build / "app"),
        ]
        commands = "\n".join(
            [
                *(
                    shlex.join(
                        [
                            cache["CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS"],
                            "-format=p1689",
                            "--",
                            *entry["arguments"],
                        ]
                    )
                    for entry in entries
                ),
                *(shlex.join(entry["arguments"]) for entry in entries),
                shlex.join(entries[1]["arguments"]),
                shlex.join(link_arguments),
            ]
        )

        verifier.validate_release_compile_inputs(
            cache,
            entries,
            context="fixture",
            response_files={},
            used_response_files=set(),
        )
        self.assertEqual(
            verifier.validate_ninja_commands(
                commands,
                cache=cache,
                compile_entries=entries,
                build_directory=build,
                expected_output=build / "app",
                context="fixture",
                response_files={},
                used_response_files=set(),
            ),
            link_arguments,
        )

    def test_ninja_commands_reject_malformed_dependency_scan(self) -> None:
        build = Path("/fixture/build")
        compiler = "/toolchain/clang++"
        scanner = "/toolchain/clang-scan-deps"
        compile_arguments = [
            compiler,
            "-O3",
            "-DNDEBUG",
            "-arch",
            "arm64",
            "-isysroot",
            "/fixture/MacOSX.sdk",
            "-mmacosx-version-min=14.0",
            "-c",
            "/fixture/source.cpp",
            "-o",
            str(build / "source.o"),
        ]
        link_arguments = [
            compiler,
            "-O3",
            "-DNDEBUG",
            "-arch",
            "arm64",
            "-isysroot",
            "/fixture/MacOSX.sdk",
            "-mmacosx-version-min=14.0",
            str(build / "source.o"),
            "-o",
            str(build / "app"),
        ]
        entry = {
            "directory": str(build),
            "file": "/fixture/source.cpp",
            "arguments": compile_arguments,
            "output": str(build / "source.o"),
        }
        cache = {
            "CMAKE_CXX_COMPILER": compiler,
            "CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS": scanner,
            "CMAKE_OSX_SYSROOT": "/fixture/MacOSX.sdk",
            "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0",
        }
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "malformed configured dependency scan",
        ):
            verifier.validate_ninja_commands(
                "\n".join(
                    (
                        shlex.join(
                            [
                                scanner,
                                "-format=p1689",
                                compiler,
                                *compile_arguments[1:],
                            ]
                        ),
                        shlex.join(compile_arguments),
                        shlex.join(link_arguments),
                    )
                ),
                cache=cache,
                compile_entries=[entry],
                build_directory=build,
                expected_output=build / "app",
                context="fixture",
                response_files={},
                used_response_files=set(),
            )

        malformed_payloads = (
            [*compile_arguments, "-o", str(build / "second.o")],
            [*compile_arguments, "-o"],
            [*compile_arguments, "-c"],
        )
        for malformed in malformed_payloads:
            with (
                self.subTest(payload=malformed),
                self.assertRaisesRegex(
                    verifier.VerificationError,
                    "malformed configured dependency scan",
                ),
            ):
                verifier.validate_ninja_commands(
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
                            shlex.join(compile_arguments),
                            shlex.join(link_arguments),
                        )
                    ),
                    cache=cache,
                    compile_entries=[entry],
                    build_directory=build,
                    expected_output=build / "app",
                    context="fixture",
                    response_files={},
                    used_response_files=set(),
                )

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "configured compiler has an unsupported wrapper",
        ):
            verifier.validate_ninja_commands(
                "\n".join(
                    (
                        shlex.join(
                            [
                                "/toolchain/unconfigured-scan-deps",
                                "-format=p1689",
                                "--",
                                *compile_arguments,
                            ]
                        ),
                        shlex.join(compile_arguments),
                        shlex.join(link_arguments),
                    )
                ),
                cache=cache,
                compile_entries=[entry],
                build_directory=build,
                expected_output=build / "app",
                context="fixture",
                response_files={},
                used_response_files=set(),
            )

    def test_ninja_commands_reject_conflicting_duplicate_action(self) -> None:
        build = Path("/fixture/build")
        compiler = "/toolchain/clang++"
        common = [
            "-O3",
            "-DNDEBUG",
            "-arch",
            "arm64",
            "-isysroot",
            "/fixture/MacOSX.sdk",
            "-mmacosx-version-min=14.0",
        ]
        compile_arguments = [
            compiler,
            *common,
            "-c",
            "/fixture/source.cpp",
            "-o",
            str(build / "source.o"),
        ]
        conflicting_arguments = [
            compiler,
            *common,
            "-DFORGED=1",
            "-c",
            "/fixture/source.cpp",
            "-o",
            str(build / "source.o"),
        ]
        link_arguments = [
            compiler,
            *common,
            str(build / "source.o"),
            "-o",
            str(build / "app"),
        ]
        entry = {
            "directory": str(build),
            "file": "/fixture/source.cpp",
            "arguments": compile_arguments,
            "output": str(build / "source.o"),
        }
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "conflicting compiler invocations",
        ):
            verifier.validate_ninja_commands(
                "\n".join(
                    shlex.join(arguments)
                    for arguments in (
                        compile_arguments,
                        conflicting_arguments,
                        link_arguments,
                    )
                ),
                cache={
                    "CMAKE_CXX_COMPILER": compiler,
                    "CMAKE_OSX_SYSROOT": "/fixture/MacOSX.sdk",
                    "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0",
                },
                compile_entries=[entry],
                build_directory=build,
                expected_output=build / "app",
                context="fixture",
                response_files={},
                used_response_files=set(),
            )

    def test_ninja_commands_reject_unconfigured_compile_action(self) -> None:
        build = Path("/fixture/build")
        compiler = "/toolchain/clang++"
        common = [
            "-O3",
            "-DNDEBUG",
            "-arch",
            "arm64",
            "-isysroot",
            "/fixture/MacOSX.sdk",
            "-mmacosx-version-min=14.0",
        ]
        compile_arguments = [
            compiler,
            *common,
            "-c",
            "/fixture/source.cpp",
            "-o",
            str(build / "source.o"),
        ]
        unknown_arguments = [
            "/toolchain/unrecorded-cc",
            *common,
            "-c",
            "/fixture/unknown.c",
            "-o",
            str(build / "unknown.o"),
        ]
        link_arguments = [
            compiler,
            *common,
            str(build / "source.o"),
            "-o",
            str(build / "app"),
        ]
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "does not use exactly one configured compiler",
        ):
            verifier.validate_ninja_commands(
                "\n".join(
                    shlex.join(arguments)
                    for arguments in (
                        compile_arguments,
                        unknown_arguments,
                        link_arguments,
                    )
                ),
                cache={
                    "CMAKE_CXX_COMPILER": compiler,
                    "CMAKE_OSX_SYSROOT": "/fixture/MacOSX.sdk",
                    "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0",
                },
                compile_entries=[
                    {
                        "directory": str(build),
                        "file": "/fixture/source.cpp",
                        "arguments": compile_arguments,
                    }
                ],
                build_directory=build,
                expected_output=build / "app",
                context="fixture",
                response_files={},
                used_response_files=set(),
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
        self.assertEqual(
            verifier.graph_proven_link_outputs(
                nodes=nodes,
                selected_output=selected,
                link_inputs=[{"role": "link-archive", "path": archive}],
                known_derived_outputs={object_path},
                build_directory=Path("/fixture/build"),
                context="fixture",
            ),
            [archive],
        )
        nodes[1]["inputs"][0]["path"] = "/fixture/build/opaque.o"
        self.assertEqual(
            verifier.graph_proven_link_outputs(
                nodes=nodes,
                selected_output=selected,
                link_inputs=[{"role": "link-archive", "path": archive}],
                known_derived_outputs={object_path},
                build_directory=Path("/fixture/build"),
                context="fixture",
            ),
            [],
        )


class BindingContractTests(unittest.TestCase):
    def test_expected_command_appends_required_binding_values(self) -> None:
        command = verifier.expected_command(
            "/fixture/infinity",
            DATASET_NAME,
            12,
            "00-C1-infinity.audit-v4.bin",
            campaign_nonce="a" * 64,
            schedule_sequence=17,
            role_id=3,
            binary_sha256="b" * 64,
            dataset_sha256="c" * 64,
        )
        self.assertEqual(len(command), 19)
        self.assertEqual(
            command[13:],
            [
                "00-C1-infinity.audit-v4.bin",
                "a" * 64,
                "17",
                "3",
                "b" * 64,
                "c" * 64,
            ],
        )
        with self.assertRaises(TypeError):
            verifier.expected_command(
                "/fixture/infinity",
                DATASET_NAME,
                12,
                "00-C1-infinity.audit-v4.bin",
                "a" * 64,
                17,
                3,
                "b" * 64,
                "c" * 64,
            )

    def test_campaign_binding_is_checked_against_frozen_inputs(self) -> None:
        binding = {
            "schema_version": verifier.CAMPAIGN_BINDING_SCHEMA_VERSION,
            "campaign_nonce": "a" * 64,
            "dataset_sha256": "b" * 64,
            "schedule_sha256": "c" * 64,
            "roles": {
                engine: {
                    "role_id": verifier.STANDALONE_ROLE_IDS[engine],
                    "engine": engine,
                    "binary_sha256": BINARY_RECORDS[engine]["sha256"],
                }
                for engine in ("infinity", "faiss")
            },
        }
        self.assertEqual(
            verifier.validate_campaign_binding(
                binding,
                dataset_sha256="b" * 64,
                schedule_sha256="c" * 64,
                binaries=BINARY_RECORDS,
            ),
            binding,
        )
        binding["roles"]["infinity"]["binary_sha256"] = "d" * 64
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "infinity role differs",
        ):
            verifier.validate_campaign_binding(
                binding,
                dataset_sha256="b" * 64,
                schedule_sha256="c" * 64,
                binaries=BINARY_RECORDS,
            )
        binding["roles"]["infinity"]["binary_sha256"] = BINARY_RECORDS["infinity"][
            "sha256"
        ]
        binding["schema_version"] = True
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "schema.*must be an integer",
        ):
            verifier.validate_campaign_binding(
                binding,
                dataset_sha256="b" * 64,
                schedule_sha256="c" * 64,
                binaries=BINARY_RECORDS,
            )
        binding["schema_version"] = verifier.CAMPAIGN_BINDING_SCHEMA_VERSION
        binding["roles"]["infinity"]["role_id"] = True
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "role ID must be an integer",
        ):
            verifier.validate_campaign_binding(
                binding,
                dataset_sha256="b" * 64,
                schedule_sha256="c" * 64,
                binaries=BINARY_RECORDS,
            )


class VerifiedEvidenceRaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "evidence"
        self.root.mkdir()
        (self.root / "nested").mkdir()
        self.target = self.root / "nested" / "record.json"
        self.target.write_text('{"value":1}\n', encoding="ascii")
        write_manifest(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def open_reader(self) -> verifier.VerifiedEvidence:
        reader = verifier.VerifiedEvidence(self.root)
        reader.open()
        return reader

    def test_descriptor_bound_file_and_directory_metadata(self) -> None:
        reader = self.open_reader()
        try:
            file_value = verifier.verified_file_stat(
                self.target,
                "record metadata",
            )
            directory_value, names = verifier.verified_directory_snapshot(
                self.target.parent,
                "nested metadata",
            )
            self.assertEqual(file_value.st_ino, self.target.stat().st_ino)
            self.assertEqual(directory_value.st_ino, self.target.parent.stat().st_ino)
            self.assertEqual(names, ["record.json"])
        finally:
            reader.close()

    def test_atomic_replacement_with_different_bytes_is_rejected(self) -> None:
        reader = self.open_reader()
        replacement = self.root / "replacement"
        replacement.write_text('{"value":2}\n', encoding="ascii")
        replacement.replace(self.target)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "changed|differs|hash mismatch",
        ):
            reader.read_bytes(self.target, context="record")
        with self.assertRaises(verifier.VerificationError):
            reader.close()

    def test_atomic_replacement_with_identical_bytes_is_rejected(self) -> None:
        reader = self.open_reader()
        replacement = self.root / "replacement"
        replacement.write_bytes(b'{"value":1}\n')
        replacement.replace(self.target)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "changed|differs",
        ):
            reader.read_bytes(self.target, context="record")
        with self.assertRaises(verifier.VerificationError):
            reader.close()

    def test_in_place_mutation_after_manifest_validation_is_rejected(self) -> None:
        reader = self.open_reader()
        self.target.write_text('{"value":2}\n', encoding="ascii")
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "changed|differs|hash mismatch",
        ):
            reader.read_bytes(self.target, context="record")
        with self.assertRaises(verifier.VerificationError):
            reader.close()

    def test_root_replacement_after_authentication_is_rejected(self) -> None:
        reader = self.open_reader()
        original = self.root.with_name("evidence-original")
        self.root.rename(original)
        self.root.mkdir()
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "Evidence directory .*changed",
        ):
            reader.read_bytes(self.target, context="record")
        with self.assertRaises(verifier.VerificationError):
            reader.close()

    def test_nested_directory_replacement_with_identical_tree_is_rejected(
        self,
    ) -> None:
        reader = self.open_reader()
        nested = self.root / "nested"
        original = self.root / "nested-original"
        nested.rename(original)
        nested.mkdir()
        (nested / "record.json").write_bytes(b'{"value":1}\n')
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "changed|differs",
        ):
            reader.read_bytes(self.target, context="record")
        with self.assertRaises(verifier.VerificationError):
            reader.close()

    def test_hard_link_added_after_manifest_validation_is_rejected(self) -> None:
        reader = self.open_reader()
        os.link(self.target, self.root / "record-alias.json")
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "hard-linked|changed",
        ):
            reader.read_bytes(self.target, context="record")
        with self.assertRaises(verifier.VerificationError):
            reader.close()

    def test_manifest_mutation_before_acceptance_is_rejected(self) -> None:
        reader = self.open_reader()
        manifest = self.root / "MANIFEST.sha256"
        text = manifest.read_text(encoding="ascii")
        replacement = ("0" if text[0] != "0" else "1") + text[1:]
        manifest.write_text(replacement, encoding="ascii")
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "MANIFEST.sha256 identity differs|MANIFEST.sha256 changed|hash mismatch",
        ):
            reader.close()

    def test_symlinked_and_hard_linked_files_are_rejected(self) -> None:
        for link_type in ("symlink", "hardlink"):
            with self.subTest(link_type=link_type):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = root / "source"
                    source.write_text("payload\n", encoding="ascii")
                    link = root / "link"
                    if link_type == "symlink":
                        link.symlink_to(source.name)
                    else:
                        os.link(source, link)
                    write_manifest(root)
                    with self.assertRaisesRegex(
                        verifier.VerificationError,
                        "symbolic link|hard-linked",
                    ):
                        verifier.VerifiedEvidence(root).open()


class VerifyNativeHnswD0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_pinned_libomp_sha256 = verifier.PINNED_LIBOMP_SHA256
        cls.original_faiss_pins = {
            "commit": verifier.PINNED_FAISS_COMMIT,
            "base_tree": verifier.PINNED_FAISS_BASE_TREE,
            "derivative_tree": verifier.PINNED_FAISS_STATS_DISABLED_TREE,
            "patch_sha256": verifier.PINNED_FAISS_STATS_PATCH_SHA256,
        }
        verifier.PINNED_LIBOMP_SHA256 = LIBOMP_SHA256
        faiss_history = synthetic_faiss_history()
        verifier.PINNED_FAISS_COMMIT = faiss_history["base_head"]
        verifier.PINNED_FAISS_BASE_TREE = faiss_history["base_tree"]
        verifier.PINNED_FAISS_STATS_DISABLED_TREE = faiss_history[
            "derivative_tree"
        ]
        verifier.PINNED_FAISS_STATS_PATCH_SHA256 = hashlib.sha256(
            FAISS_PATCH_BYTES
        ).hexdigest()
        cls.base_temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.base_temporary.name) / "base-evidence"
        build_valid_evidence(cls.base)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.base_temporary.cleanup()
        verifier.PINNED_LIBOMP_SHA256 = cls.original_pinned_libomp_sha256
        verifier.PINNED_FAISS_COMMIT = cls.original_faiss_pins["commit"]
        verifier.PINNED_FAISS_BASE_TREE = cls.original_faiss_pins["base_tree"]
        verifier.PINNED_FAISS_STATS_DISABLED_TREE = cls.original_faiss_pins[
            "derivative_tree"
        ]
        verifier.PINNED_FAISS_STATS_PATCH_SHA256 = cls.original_faiss_pins[
            "patch_sha256"
        ]

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.evidence = Path(self.temporary.name) / "evidence"
        shutil.copytree(self.base, self.evidence)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def mutate_json(
        self,
        relative: str,
        mutation: Callable[[Any], None],
    ) -> Any:
        path = self.evidence / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        mutation(value)
        write_json(path, value)
        return value

    def reset_evidence(self) -> None:
        shutil.rmtree(self.evidence)
        shutil.copytree(self.base, self.evidence)

    def load_preflight(self) -> dict[str, Any]:
        return json.loads(
            (self.evidence / "preflight.json").read_text(encoding="utf-8")
        )

    def store_preflight(self, preflight: dict[str, Any]) -> None:
        write_json(self.evidence / "preflight.json", preflight)

    def store_build_input_closure(
        self,
        label: str,
        closure: dict[str, Any],
    ) -> None:
        closure_path = self.evidence / f"{label}-build-input-closure.json"
        write_json(closure_path, closure)
        preflight = self.load_preflight()
        preflight["builds"][label]["build_input_closure"]["sha256"] = sha256(
            closure_path
        )
        self.store_preflight(preflight)

    @staticmethod
    def faiss_repository(preflight: dict[str, Any]) -> dict[str, Any]:
        return next(
            repository
            for repository in preflight["source_provenance"]["git_repositories"]
            if repository["root"] == FAISS_SOURCE_ROOT
        )

    def write_records(self, records: list[dict[str, Any]]) -> None:
        write_json(self.evidence / "runs.json", records)
        for record in records:
            prefix = f"{record['sequence']:02d}-{record['pair']}-{record['engine']}"
            write_json(self.evidence / f"{prefix}.json", record)

    def records(self) -> list[dict[str, Any]]:
        return json.loads(
            (self.evidence / "runs.json").read_text(encoding="utf-8")
        )

    def rewrite_stdout(self, record: dict[str, Any]) -> None:
        path = self.evidence / record["stdout_path"]
        attestation_lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("attestation_")
        ]
        path.write_text(
            "\n".join(attestation_lines)
            + "\n"
            + "".join(
                f"{key}={value}\n" for key, value in record["fields"].items()
            ),
            encoding="utf-8",
        )

    def synchronize_sidecar_metadata(self, index: int = 0) -> None:
        records = self.records()
        record = records[index]
        sidecar_path = self.evidence / record["audit_sidecar"]["path"]
        sidecar = sidecar_path.read_bytes()
        digest = hashlib.sha256(sidecar).hexdigest()
        record["audit_sidecar"].update(
            {"sha256": digest, "bytes": len(sidecar)}
        )
        record["fields"].update(
            {
                "audit_sidecar_sha256": digest,
                "audit_sidecar_bytes": str(len(sidecar)),
            }
        )
        self.rewrite_stdout(record)
        self.write_records(records)

    def mutate_first_sidecar(
        self,
        mutation: Callable[[bytearray], None],
        *,
        synchronize_metadata: bool,
    ) -> None:
        records = self.records()
        path = self.evidence / records[0]["audit_sidecar"]["path"]
        data = bytearray(path.read_bytes())
        mutation(data)
        path.write_bytes(data)
        if synchronize_metadata:
            self.synchronize_sidecar_metadata()
        write_manifest(self.evidence)

    def convert_to_preflight_only(self) -> None:
        schedule = json.loads(
            (self.evidence / "schedule.json").read_text(encoding="utf-8")
        )
        member_prefixes = tuple(
            f"{member['sequence']:02d}-{member['pair']}-{member['engine']}."
            for member in schedule
        )
        for path in list(self.evidence.rglob("*")):
            if not path.is_file() or path.name == "MANIFEST.sha256":
                continue
            relative = path.relative_to(self.evidence).as_posix()
            if relative in {
                "runs.json",
                "summary.json",
                "README.md",
            } or relative.startswith(member_prefixes):
                path.unlink()

        preflight = json.loads(
            (self.evidence / "preflight.json").read_text(encoding="utf-8")
        )
        preflight["execution_mode"] = verifier.PREFLIGHT_ONLY_MODE
        preflight["runner"]["argv"].append("--preflight-only")
        preflight["runner"]["effective_arguments"]["preflight_only"] = True
        write_json(self.evidence / "preflight.json", preflight)
        write_json(
            self.evidence / "preflight-result.json",
            {
                "evidence_schema_version": verifier.EVIDENCE_SCHEMA_VERSION,
                "scope": SCOPE,
                "execution_mode": verifier.PREFLIGHT_ONLY_MODE,
                "status": "pass",
                "completed_at": "2026-08-20T00:01:00Z",
                "benchmark_members_executed": 0,
                "campaign_complete": False,
                "artifact_sha256": {
                    "dataset.json": sha256(self.evidence / "dataset.json"),
                    verifier.HELDOUT_MANIFEST_FILENAME: sha256(
                        self.evidence / verifier.HELDOUT_MANIFEST_FILENAME
                    ),
                    "preflight.json": sha256(self.evidence / "preflight.json"),
                    "schedule.json": sha256(self.evidence / "schedule.json"),
                },
            },
        )
        write_manifest(self.evidence)

    def test_valid_synthetic_evidence_emits_strict_pass_json(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            return_code = verifier.main([str(self.evidence)])

        self.assertEqual(return_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["schedule_members"], 18)
        self.assertEqual(result["measured_pairs"], 6)
        self.assertEqual(
            self.records()[0]["audit"]["query"]["per_query_checksum_count"],
            verifier.HELDOUT_QUERY_COUNT,
        )
        self.assertEqual(
            result["idle"],
            {
                "minimum_percent": 95.0,
                "window_seconds": 15,
                "policy": "default-95-percent",
            },
        )
        self.assertEqual(
            result["metrics"]["ingestion"]["stratified_geometric_mean_ratio"],
            2.0,
        )
        self.assertEqual(
            result["metrics"]["ingestion"]["ratio_range"],
            [2.0, 2.0],
        )
        self.assertEqual(
            result["metrics"]["ingestion"]["engines"]["infinity"][
                "relative_mad"
            ],
            0.01,
        )
        self.assertEqual(
            result["metrics"]["query"]["stratified_geometric_mean_qps_ratio"],
            2.0,
        )
        self.assertLessEqual(
            result["metrics"]["query"]["engines"]["infinity"]["median_p50_ns"],
            result["metrics"]["query"]["engines"]["infinity"]["median_p99_ns"],
        )

    def test_base_commit_used_directly_as_derivative_is_rejected(self) -> None:
        history = synthetic_faiss_history()
        preflight = self.load_preflight()
        repository = self.faiss_repository(preflight)
        commit_path = self.evidence / repository["commit_object_path"]
        listing_path = self.evidence / repository["tree_listing_path"]
        commit_path.write_bytes(history["base_commit"])
        listing_path.write_text(
            synthetic_tree_listing(history["base_entries"]),
            encoding="utf-8",
        )
        repository.update(
            {
                "head": history["base_head"],
                "tree": history["base_tree"],
                "commit_object_sha256": sha256(commit_path),
                "tree_listing_sha256": sha256(listing_path),
            }
        )
        derivative = preflight["source_provenance"]["faiss_audited_reference"][
            "derivative"
        ]
        derivative["head"] = history["base_head"]
        derivative["tree"] = history["base_tree"]
        self.store_preflight(preflight)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "cannot use the pinned base commit directly",
        ):
            verifier.verify_evidence(self.evidence)

    def test_wrong_or_multiple_derivative_parents_are_rejected(self) -> None:
        history = synthetic_faiss_history()
        parent_cases = {
            "wrong": ("f" * 40,),
            "multiple": (history["base_head"], "f" * 40),
        }
        for label, parents in parent_cases.items():
            with self.subTest(label=label):
                self.reset_evidence()
                head, tree, commit = synthetic_commit(
                    0,
                    history["derivative_entries"],
                    parents=parents,
                )
                self.assertEqual(tree, history["derivative_tree"])
                preflight = self.load_preflight()
                repository = self.faiss_repository(preflight)
                commit_path = self.evidence / repository["commit_object_path"]
                commit_path.write_bytes(commit)
                repository["head"] = head
                repository["commit_object_sha256"] = sha256(commit_path)
                preflight["source_provenance"]["faiss_audited_reference"][
                    "derivative"
                ]["head"] = head
                self.store_preflight(preflight)
                write_manifest(self.evidence)

                with self.assertRaisesRegex(
                    verifier.VerificationError,
                    "exactly the pinned base commit as its parent",
                ):
                    verifier.verify_evidence(self.evidence)

    def test_wrong_base_or_derivative_tree_is_rejected_from_captured_bytes(
        self,
    ) -> None:
        cases = (
            (
                verifier.FAISS_BASE_TREE_LISTING_FILENAME,
                None,
                "FAISS base recursive tree differs",
            ),
            (
                "source-repository-00-tree-listing.txt",
                "tree_listing_sha256",
                "recursive tree listing does not match",
            ),
        )
        for filename, repository_hash_key, message in cases:
            with self.subTest(filename=filename):
                self.reset_evidence()
                path = self.evidence / filename
                text = path.read_text(encoding="utf-8")
                path.write_text(
                    text.replace(
                        verifier.git_object_id(
                            "blob",
                            (
                                FAISS_BASE_PATH_BYTES
                                if repository_hash_key is None
                                else FAISS_DERIVATIVE_PATH_BYTES
                            )["CMakeLists.txt"],
                        ),
                        "0" * 40,
                        1,
                    ),
                    encoding="utf-8",
                )
                if repository_hash_key is not None:
                    preflight = self.load_preflight()
                    self.faiss_repository(preflight)[repository_hash_key] = sha256(
                        path
                    )
                    self.store_preflight(preflight)
                write_manifest(self.evidence)

                with self.assertRaisesRegex(verifier.VerificationError, message):
                    verifier.verify_evidence(self.evidence)

    def test_dirty_derivative_is_rejected(self) -> None:
        preflight = self.load_preflight()
        repository = self.faiss_repository(preflight)
        status_path = self.evidence / repository["status_path"]
        status_path.write_text(" M faiss/IndexHNSW.cpp\n", encoding="utf-8")
        repository["status_sha256"] = sha256(status_path)
        repository["dirty"] = True
        preflight["source_provenance"]["faiss_audited_reference"]["derivative"][
            "dirty"
        ] = True
        self.store_preflight(preflight)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "FAISS derivative is dirty",
        ):
            verifier.verify_evidence(self.evidence)

    def test_changed_path_capture_rejects_missing_extra_and_reordered_paths(
        self,
    ) -> None:
        expected = list(verifier.PINNED_FAISS_STATS_PATCH_PATHS)
        cases = {
            "missing": expected[:-1],
            "extra": [*expected, "faiss/Unexpected.cpp"],
            "reordered": list(reversed(expected)),
        }
        for label, paths in cases.items():
            with self.subTest(label=label):
                self.reset_evidence()
                (self.evidence / verifier.FAISS_CHANGED_PATHS_FILENAME).write_text(
                    "".join(f"{path}\n" for path in paths),
                    encoding="utf-8",
                )
                write_manifest(self.evidence)
                with self.assertRaisesRegex(
                    verifier.VerificationError,
                    "changed-path capture differs",
                ):
                    verifier.verify_evidence(self.evidence)

    def test_patch_byte_and_recorded_hash_mismatches_are_rejected(self) -> None:
        for update_record in (False, True):
            with self.subTest(update_record=update_record):
                self.reset_evidence()
                patch_path = self.evidence / verifier.FAISS_STATS_PATCH_FILENAME
                patch_path.write_bytes(b"substituted patch bytes\n")
                if update_record:
                    preflight = self.load_preflight()
                    preflight["source_provenance"]["faiss_audited_reference"][
                        "patch"
                    ]["sha256"] = sha256(patch_path)
                    self.store_preflight(preflight)
                write_manifest(self.evidence)
                with self.assertRaisesRegex(
                    verifier.VerificationError,
                    "patch recorded hash differs|captured-file hash differs",
                ):
                    verifier.verify_evidence(self.evidence)

        self.reset_evidence()
        (
            self.evidence / verifier.FAISS_DERIVATIVE_DIFF_FILENAME
        ).write_bytes(b"substituted derivative diff\n")
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "derivative diff hash differs",
        ):
            verifier.verify_evidence(self.evidence)

    def test_faiss_stats_cache_option_missing_or_on_is_rejected(self) -> None:
        for value in (None, "ON"):
            with self.subTest(value=value):
                self.reset_evidence()
                cache_path = self.evidence / "faiss-library-CMakeCache.txt"
                line = "FAISS_ENABLE_GLOBAL_HNSW_STATS:BOOL=OFF"
                cache_text = cache_path.read_text(encoding="utf-8")
                replacement = (
                    "" if value is None else f"FAISS_ENABLE_GLOBAL_HNSW_STATS:BOOL={value}"
                )
                cache_path.write_text(
                    cache_text.replace(line, replacement, 1),
                    encoding="utf-8",
                )
                preflight = self.load_preflight()
                build = preflight["linked_faiss_build"]
                build["settings"]["FAISS_ENABLE_GLOBAL_HNSW_STATS"] = value
                build["cmake_cache"]["sha256"] = sha256(cache_path)
                self.store_preflight(preflight)
                write_manifest(self.evidence)
                with self.assertRaisesRegex(
                    verifier.VerificationError,
                    "FAISS_ENABLE_GLOBAL_HNSW_STATS=OFF",
                ):
                    verifier.verify_evidence(self.evidence)

    def test_faiss_compile_definition_missing_from_selected_unit_is_rejected(
        self,
    ) -> None:
        definition = " -DFAISS_DISABLE_GLOBAL_HNSW_STATS=1"
        compile_path = self.evidence / "faiss-library-compile_commands.json"
        compile_entries = json.loads(compile_path.read_text(encoding="utf-8"))
        compile_entries[0]["command"] = compile_entries[0]["command"].replace(
            definition,
            "",
            1,
        )
        write_json(compile_path, compile_entries)
        commands_path = self.evidence / "faiss-library-ninja-commands.txt"
        commands_path.write_text(
            commands_path.read_text(encoding="utf-8").replace(
                definition,
                "",
                1,
            ),
            encoding="utf-8",
        )
        preflight = self.load_preflight()
        build = preflight["linked_faiss_build"]
        build["compile_commands"]["sha256"] = sha256(compile_path)
        build["ninja_commands"]["sha256"] = sha256(commands_path)
        self.store_preflight(preflight)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "must define exactly FAISS_DISABLE_GLOBAL_HNSW_STATS=1",
        ):
            verifier.verify_evidence(self.evidence)

    def test_wrapper_resolving_different_faiss_dylib_is_rejected(self) -> None:
        alternate = "/fixture/alternate/libfaiss.dylib"
        dependencies_path = self.evidence / "faiss-dependencies.json"
        dependency_manifest = json.loads(
            dependencies_path.read_text(encoding="utf-8")
        )
        dependency = next(
            item
            for item in dependency_manifest["dependencies"]
            if item["resolved_path"] == LIBFAISS_PATH
        )
        dependency["resolved_path"] = alternate
        old_output = dependency_manifest["otool_outputs"].pop(LIBFAISS_PATH)
        dependency_manifest["otool_outputs"][alternate] = old_output.replace(
            LIBFAISS_PATH,
            alternate,
            1,
        )
        write_json(dependencies_path, dependency_manifest)

        preflight = self.load_preflight()
        preflight["builds"]["faiss"]["dependencies"]["sha256"] = sha256(
            dependencies_path
        )
        preflight["runtime_artifact_hashes"][alternate] = LIBFAISS_SHA256
        preflight["runtime_artifact_snapshots"].append(
            {
                "original_path": alternate,
                **write_content_blob(self.evidence, LIBFAISS_BYTES),
            }
        )
        preflight["runtime_artifact_snapshots"].sort(
            key=lambda record: record["original_path"]
        )
        self.store_preflight(preflight)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "wrapper resolves a different libfaiss.dylib",
        ):
            verifier.verify_evidence(self.evidence)

    def test_attestation_transcript_mutation_is_rejected(self) -> None:
        records = self.records()
        stdout_path = self.evidence / records[0]["stdout_path"]
        stdout_path.write_text(
            stdout_path.read_text(encoding="utf-8").replace(
                "attestation_after_dynamic_cdhash=" + "a" * 40,
                "attestation_after_dynamic_cdhash=" + "c" * 40,
                1,
            ),
            encoding="utf-8",
        )
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "snapshots differ",
        ):
            verifier.verify_evidence(self.evidence)

    def test_attestation_record_or_stop_evidence_mutation_is_rejected(self) -> None:
        records = self.records()
        records[0]["attestation"]["status"] = "FAIL"
        self.write_records(records)
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "attestation status",
        ):
            verifier.verify_evidence(self.evidence)

        shutil.rmtree(self.evidence)
        shutil.copytree(self.base, self.evidence)
        records = self.records()
        records[0]["attestation_barriers"][0]["processes"][1][
            "status_code"
        ] = 3
        records[0]["attestation_barriers"][0]["processes"][1][
            "status"
        ] = "sleeping"
        self.write_records(records)
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "stopped-process evidence differs",
        ):
            verifier.verify_evidence(self.evidence)

    def test_process_evidence_rejects_extra_helper_and_identity_drift(self) -> None:
        records = self.records()
        helper = copy.deepcopy(
            records[0]["attestation_barriers"][0]["processes"][1]
        )
        helper["pid"] += 1
        helper["status_code"] = 3
        helper["status"] = "sleeping"
        records[0]["attestation_barriers"][0]["processes"].append(helper)
        self.write_records(records)
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "does not contain exactly two processes",
        ):
            verifier.verify_evidence(self.evidence)

        shutil.rmtree(self.evidence)
        shutil.copytree(self.base, self.evidence)
        records = self.records()
        records[0]["attestation_barriers"][2]["processes"][1][
            "start_time_microseconds"
        ] += 1
        self.write_records(records)
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "benchmark identity changed across barriers",
        ):
            verifier.verify_evidence(self.evidence)

    def test_process_evidence_rejects_descendants_and_pre_reap_members(
        self,
    ) -> None:
        records = self.records()
        records[0]["attestation_barriers"][0]["processes"][1][
            "child_pids"
        ] = [999_999]
        self.write_records(records)
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "descendant topology differs",
        ):
            verifier.verify_evidence(self.evidence)

        shutil.rmtree(self.evidence)
        shutil.copytree(self.base, self.evidence)
        records = self.records()
        records[0]["benchmark_process_evidence"]["pre_reap_quiescence"][
            "member_pids"
        ] = [
            records[0]["benchmark_process_evidence"]["process_group"],
            records[0]["benchmark_process_evidence"]["benchmark_pid"],
        ]
        self.write_records(records)
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "was not quiescent before reap",
        ):
            verifier.verify_evidence(self.evidence)

    def test_process_evidence_rejects_missing_pre_reap_leader(self) -> None:
        records = self.records()
        records[0]["benchmark_process_evidence"]["pre_reap_quiescence"][
            "member_pids"
        ] = []
        self.write_records(records)
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "was not quiescent before reap",
        ):
            verifier.verify_evidence(self.evidence)

    def test_process_evidence_rejects_out_of_order_lifecycle_times(self) -> None:
        records = self.records()
        evidence = records[0]["benchmark_process_evidence"]
        evidence["terminal_observation"]["observed_at_monotonic_ns"] = (
            evidence["pre_reap_quiescence"]["observed_at_monotonic_ns"] + 1
        )
        self.write_records(records)
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "pre-reap observation time must be at least",
        ):
            verifier.verify_evidence(self.evidence)

        shutil.rmtree(self.evidence)
        shutil.copytree(self.base, self.evidence)
        records = self.records()
        evidence = records[0]["benchmark_process_evidence"]
        evidence["reap"]["completed_at_monotonic_ns"] = (
            evidence["pre_reap_quiescence"]["observed_at_monotonic_ns"] - 1
        )
        self.write_records(records)
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "reap completion time must be at least",
        ):
            verifier.verify_evidence(self.evidence)

    def test_process_evidence_requires_normal_zero_exit(self) -> None:
        for code, status, returncode in (
            (verifier.CLD_EXITED, 7, 7),
            (verifier.CLD_KILLED, 9, -9),
            (verifier.CLD_KILLED, 0, 0),
        ):
            with self.subTest(code=code, status=status, returncode=returncode):
                self.reset_evidence()
                records = self.records()
                evidence = records[0]["benchmark_process_evidence"]
                evidence["terminal_observation"].update(
                    {"code": code, "status": status}
                )
                evidence["reap"]["returncode"] = returncode
                self.write_records(records)
                write_manifest(self.evidence)

                with self.assertRaisesRegex(
                    verifier.VerificationError,
                    "did not exit successfully",
                ):
                    verifier.verify_evidence(self.evidence)

    def test_unsafe_parent_environment_policy_mutation_is_rejected(self) -> None:
        preflight = self.load_preflight()
        preflight["environment_control"]["unsafe_parent_environment_policy"][
            "observed_unsafe_names"
        ] = ["CFLAGS"]
        self.store_preflight(preflight)
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "differs from the frozen rejection policy",
        ):
            verifier.verify_evidence(self.evidence)

    def test_index_timing_binding_mutation_is_rejected(self) -> None:
        records = self.records()
        records[0]["index_timing_binding"]["outer_index_window_ns"] += 1
        self.write_records(records)
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "differs from independent reconstruction",
        ):
            verifier.verify_evidence(self.evidence)

    def test_index_timing_outer_window_must_bound_inner_timer(self) -> None:
        records = self.records()
        record = records[0]
        engine = record["engine"]
        cold_build_ns = int(record["fields"][f"{engine}_cold_build_ns"])
        barriers = record["attestation_barriers"]
        outer_start = barriers[1]["resumed_at_monotonic_ns"]
        barriers[2]["observed_at_monotonic_ns"] = outer_start + cold_build_ns - 1
        barriers[2]["resumed_at_monotonic_ns"] = (
            barriers[2]["observed_at_monotonic_ns"] + 1
        )
        barriers[3]["observed_at_monotonic_ns"] = (
            barriers[2]["resumed_at_monotonic_ns"] + 1
        )
        barriers[3]["resumed_at_monotonic_ns"] = (
            barriers[3]["observed_at_monotonic_ns"] + 1
        )
        record["index_timing_binding"] = index_timing_binding_fixture(
            barriers,
            cold_build_ns,
        )
        self.write_records(records)
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "cold-build duration exceeds the parent-observed index window",
        ):
            verifier.verify_evidence(self.evidence)

    def test_completed_negative_campaign_is_verified(self) -> None:
        records = self.records()
        for record in records:
            if (
                record["treatment"] == "measured"
                and record["pair"] in {"P1", "P4", "P5"}
                and record["engine"] == "infinity"
            ):
                cold_build_ns = 400_000_000
                record["fields"]["infinity_cold_build_ns"] = str(cold_build_ns)
                record["fields"]["infinity_insert_call_ns"] = "399999000"
                barriers = attestation_barrier_fixture(
                    record["pid"],
                    cold_build_ns,
                    benchmark_path=record["binary"]["path"],
                    benchmark_bytes=record["benchmark_process_evidence"][
                        "benchmark_executable"
                    ]["size"],
                )
                record["attestation_barriers"] = barriers
                record["benchmark_process_evidence"] = process_evidence_fixture(
                    barriers,
                    process_group=record["pid"],
                    benchmark_path=record["binary"]["path"],
                    benchmark_bytes=record["benchmark_process_evidence"][
                        "benchmark_executable"
                    ]["size"],
                )
                record["index_timing_binding"] = index_timing_binding_fixture(
                    barriers,
                    cold_build_ns,
                )
                self.rewrite_stdout(record)
        self.write_records(records)
        dataset = json.loads(
            (self.evidence / "dataset.json").read_text(encoding="utf-8")
        )
        summary = summary_for(
            records,
            dataset=dataset,
            schedule_sha256=sha256(self.evidence / "schedule.json"),
        )
        self.assertEqual(summary["status"], "fail")
        write_json(self.evidence / "summary.json", summary)
        write_manifest(self.evidence)

        result = verifier.verify_evidence(self.evidence)

        self.assertEqual(result["status"], "PASS")
        self.assertGreater(
            result["metrics"]["ingestion"]["paired_ratio_relative_mad"],
            0.10,
        )

    def test_run_chronology_rejects_overlap(self) -> None:
        records = [
            {"started_at_unix_ns": 10, "ended_at_unix_ns": 20},
            {"started_at_unix_ns": 20, "ended_at_unix_ns": 30},
        ]
        verifier.validate_run_chronology(records)
        records[1]["started_at_unix_ns"] = 19
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "overlaps or predates",
        ):
            verifier.validate_run_chronology(records)

    def test_production_batch4_switch_is_independently_verified(self) -> None:
        cache_path = self.evidence / "infinity-CMakeCache.txt"
        cache_path.write_text(
            cache_path.read_text(encoding="utf-8").replace(
                "INFINITY_DISABLE_APPLE_HNSW_BATCH4_TRAVERSAL:BOOL=OFF",
                "INFINITY_DISABLE_APPLE_HNSW_BATCH4_TRAVERSAL:BOOL=ON",
            ),
            encoding="utf-8",
        )
        preflight = json.loads(
            (self.evidence / "preflight.json").read_text(encoding="utf-8")
        )
        preflight["builds"]["infinity"]["settings"][
            "INFINITY_DISABLE_APPLE_HNSW_BATCH4_TRAVERSAL"
        ] = "ON"
        preflight["builds"]["infinity"]["cmake_cache"]["sha256"] = sha256(
            cache_path
        )
        write_json(self.evidence / "preflight.json", preflight)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "TRAVERSAL=OFF",
        ):
            verifier.verify_evidence(self.evidence)

    def test_independent_recall_gate_has_exact_boundary(self) -> None:
        records = self.records()
        target = next(
            record
            for record in records
            if record["treatment"] == "measured"
            and record["pair"] == "P1"
            and record["engine"] == "infinity"
        )
        point = next(
            point
            for point in target["audit"]["recall"]["points"]
            if point["k"] == 100 and point["ef"] == 128
        )
        set_recall_hits(point, point["possible_hits"] - 32)
        quality = verifier.recompute_audit_quality(records)["recall"]
        self.assertTrue(quality["all_points_within_deficit"])
        point_quality = next(
            item
            for item in quality["points"]
            if item["k"] == 100 and item["ef"] == 128
        )
        self.assertEqual(
            point_quality["maximum_infinity_deficit_exact"],
            {"numerator": 1, "denominator": 200},
        )

        set_recall_hits(point, point["possible_hits"] - 33)
        self.assertFalse(
            verifier.recompute_audit_quality(records)["recall"][
                "all_points_within_deficit"
            ]
        )

    def test_independent_recall_hit_count_records_are_strict(self) -> None:
        mutations = {
            "boolean hits": lambda point: point.update({"eligible_hits": True}),
            "negative hits": lambda point: point.update({"eligible_hits": -1}),
            "too many hits": lambda point: point.update(
                {"eligible_hits": point["possible_hits"] + 1}
            ),
            "wrong denominator": lambda point: point.update(
                {"possible_hits": point["possible_hits"] + 1}
            ),
            "float mismatch": lambda point: point.update({"recall": 0.5}),
            "extra key": lambda point: point.update({"unexpected": 1}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                record = self.records()[0]
                point = record["audit"]["recall"]["points"][0]
                mutate(point)
                with self.assertRaises(verifier.VerificationError):
                    verifier.recall_point_fraction(
                        record,
                        point["k"],
                        point["ef"],
                    )

    def test_independent_query_recall_gates_use_floor_and_absolute_gap(self) -> None:
        records = self.records()
        targets = [
            record
            for record in records
            if record["treatment"] == "measured" and record["pair"] == "P1"
        ]
        for record in targets:
            point = next(
                point
                for point in record["audit"]["recall"]["points"]
                if (point["k"], point["ef"])
                == (verifier.QUERY_K, verifier.QUERY_EF_SEARCH)
            )
            set_recall_hits(point, point["possible_hits"] - 6)
        metrics = verifier.recompute_metrics(records)["query"]
        self.assertTrue(metrics["recall_floor_met"])
        self.assertTrue(metrics["absolute_recall_gap_met"])

        faiss_point = next(
            point
            for point in targets[1]["audit"]["recall"]["points"]
            if (point["k"], point["ef"])
            == (verifier.QUERY_K, verifier.QUERY_EF_SEARCH)
        )
        set_recall_hits(faiss_point, faiss_point["possible_hits"])
        metrics = verifier.recompute_metrics(records)["query"]
        self.assertTrue(metrics["recall_floor_met"])
        self.assertFalse(metrics["absolute_recall_gap_met"])

        set_recall_hits(faiss_point, faiss_point["possible_hits"] - 7)
        metrics = verifier.recompute_metrics(records)["query"]
        self.assertFalse(metrics["recall_floor_met"])

    def test_valid_preflight_only_evidence_requires_explicit_mode(self) -> None:
        self.convert_to_preflight_only()

        result = verifier.verify_evidence(
            self.evidence,
            preflight_only=True,
        )

        self.assertEqual(result["status"], "PREFLIGHT_PASS")
        self.assertEqual(result["benchmark_members_executed"], 0)
        self.assertIs(result["campaign_complete"], False)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "verifier expects 'full-campaign'",
        ):
            verifier.verify_evidence(self.evidence)

    def test_full_campaign_is_rejected_in_preflight_mode(self) -> None:
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "verifier expects 'preflight-only'",
        ):
            verifier.verify_evidence(self.evidence, preflight_only=True)

    def test_preflight_only_rejects_campaign_member_files(self) -> None:
        self.convert_to_preflight_only()
        (self.evidence / "00-C1-infinity.stdout").write_text(
            "not permitted\n",
            encoding="utf-8",
        )
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "contains benchmark member files",
        ):
            verifier.verify_evidence(self.evidence, preflight_only=True)

    def test_preflight_only_result_cannot_claim_campaign_completion(self) -> None:
        self.convert_to_preflight_only()
        self.mutate_json(
            "preflight-result.json",
            lambda value: value.update(
                {
                    "benchmark_members_executed": 1,
                    "campaign_complete": True,
                }
            ),
        )
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "claims benchmark members",
        ):
            verifier.verify_evidence(self.evidence, preflight_only=True)

    def test_schema_8_is_rejected_explicitly(self) -> None:
        self.mutate_json(
            "preflight.json",
            lambda value: value.update({"evidence_schema_version": 8}),
        )
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            f"expected {verifier.EVIDENCE_SCHEMA_VERSION}, found 8",
        ):
            verifier.verify_evidence(self.evidence)

    def test_missing_audit_sidecar_is_rejected(self) -> None:
        record = self.records()[0]
        (self.evidence / record["audit_sidecar"]["path"]).unlink()
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "audit sidecar is missing or unreadable",
        ):
            verifier.verify_evidence(self.evidence)

    def test_symlink_audit_sidecar_is_rejected(self) -> None:
        records = self.records()
        first = self.evidence / records[0]["audit_sidecar"]["path"]
        target = records[1]["audit_sidecar"]["path"]
        first.unlink()
        first.symlink_to(target)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "symbolic link",
        ):
            verifier.verify_evidence(self.evidence)

    def test_audit_sidecar_hash_mismatch_is_rejected(self) -> None:
        self.mutate_first_sidecar(
            lambda data: data.__setitem__(-1, data[-1] ^ 1),
            synchronize_metadata=False,
        )

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "sidecar hashes disagree",
        ):
            verifier.verify_evidence(self.evidence)

    def test_audit_sidecar_size_mismatch_is_rejected(self) -> None:
        self.mutate_first_sidecar(
            lambda data: data.extend(b"\0"),
            synchronize_metadata=False,
        )

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "sidecar byte counts disagree",
        ):
            verifier.verify_evidence(self.evidence)

    def test_truncated_audit_sidecar_is_rejected_after_rehash(self) -> None:
        self.mutate_first_sidecar(
            lambda data: data.__delitem__(-1),
            synchronize_metadata=True,
        )

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "is truncated",
        ):
            verifier.verify_evidence(self.evidence)

    def test_audit_sidecar_trailing_bytes_are_rejected_after_rehash(self) -> None:
        self.mutate_first_sidecar(
            lambda data: data.extend(b"\0"),
            synchronize_metadata=True,
        )

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "trailing bytes",
        ):
            verifier.verify_evidence(self.evidence)

    def test_query_latency_sample_tampering_is_rejected_after_rehash(self) -> None:
        sample_offset = (
            struct.calcsize(AUDIT_BODY_HEADER_FORMAT)
            + QUERY_SCHEMA2_FIXED_BYTES
            + QUERY_SCHEMA2_PER_QUERY_BYTES
        )

        def corrupt(data: bytearray) -> None:
            (sample,) = struct.unpack_from("<Q", data, sample_offset)
            struct.pack_into("<Q", data, sample_offset, sample + 1)

        self.mutate_first_sidecar(corrupt, synchronize_metadata=True)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "normalized audit query latency_samples_ns",
        ):
            verifier.verify_evidence(self.evidence)

    def test_query_throughput_wall_tampering_is_rejected_after_rehash(self) -> None:
        query_header_offset = struct.calcsize(AUDIT_BODY_HEADER_FORMAT)
        wall_offset = query_header_offset + 128

        def corrupt(data: bytearray) -> None:
            (wall_ns,) = struct.unpack_from("<Q", data, wall_offset)
            struct.pack_into("<Q", data, wall_offset, wall_ns + 1)

        self.mutate_first_sidecar(corrupt, synchronize_metadata=True)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "query_throughput_wall_ns.*differs",
        ):
            verifier.verify_evidence(self.evidence)

    def test_query_result_checksum_tampering_is_rejected_after_rehash(self) -> None:
        query_header_offset = struct.calcsize(AUDIT_BODY_HEADER_FORMAT)
        checksum_offset = query_header_offset + 144

        def corrupt(data: bytearray) -> None:
            (checksum,) = struct.unpack_from("<Q", data, checksum_offset)
            struct.pack_into("<Q", data, checksum_offset, checksum ^ 1)

        self.mutate_first_sidecar(corrupt, synchronize_metadata=True)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "Query result checksum attestations differ",
        ):
            verifier.verify_evidence(self.evidence)

    def test_query_latency_sample_count_tampering_is_rejected(self) -> None:
        query_header_offset = struct.calcsize(AUDIT_BODY_HEADER_FORMAT)
        sample_count_offset = query_header_offset + 88

        def corrupt(data: bytearray) -> None:
            struct.pack_into(
                "<Q",
                data,
                sample_count_offset,
                verifier.QUERY_LATENCY_SAMPLE_COUNT - 1,
            )

        self.mutate_first_sidecar(corrupt, synchronize_metadata=True)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "sample or operation count differs",
        ):
            verifier.verify_evidence(self.evidence)

    def test_audit_sidecar_path_must_be_the_exact_member_basename(self) -> None:
        records = self.records()
        record = records[0]
        original = self.evidence / record["audit_sidecar"]["path"]
        forged_name = "forged.audit-v2.bin"
        original.rename(self.evidence / forged_name)
        record["audit_sidecar"]["path"] = forged_name
        self.write_records(records)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "audit sidecar path differs",
        ):
            verifier.verify_evidence(self.evidence)

    def test_graph_self_edge_is_rejected_after_rehash(self) -> None:
        header_bytes = (
            struct.calcsize(AUDIT_BODY_HEADER_FORMAT)
            + QUERY_SCHEMA2_FIXED_BYTES
            + QUERY_SCHEMA2_PER_QUERY_BYTES
            + verifier.QUERY_LATENCY_SAMPLE_COUNT * struct.calcsize("<Q")
        )
        first_neighbor = header_bytes + struct.calcsize("<iqII")

        def corrupt(data: bytearray) -> None:
            struct.pack_into("<i", data, first_neighbor, 0)

        self.mutate_first_sidecar(corrupt, synchronize_metadata=True)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "self edge",
        ):
            verifier.verify_evidence(self.evidence)

    def test_directed_level_zero_disconnect_is_rejected_after_rehash(self) -> None:
        header_bytes = (
            struct.calcsize(AUDIT_BODY_HEADER_FORMAT)
            + QUERY_SCHEMA2_FIXED_BYTES
            + QUERY_SCHEMA2_PER_QUERY_BYTES
            + verifier.QUERY_LATENCY_SAMPLE_COUNT * struct.calcsize("<Q")
        )
        first_neighbor = header_bytes + struct.calcsize("<iqII")

        def corrupt(data: bytearray) -> None:
            struct.pack_into("<i", data, first_neighbor, 2)

        self.mutate_first_sidecar(corrupt, synchronize_metadata=True)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "level-zero graph is disconnected",
        ):
            verifier.verify_evidence(self.evidence)

    def test_out_of_range_result_id_is_rejected_after_rehash(self) -> None:
        result_offset = (
            struct.calcsize(AUDIT_BODY_HEADER_FORMAT)
            + QUERY_SCHEMA2_FIXED_BYTES
            + QUERY_SCHEMA2_PER_QUERY_BYTES
            + verifier.QUERY_LATENCY_SAMPLE_COUNT * struct.calcsize("<Q")
            + VECTORS * struct.calcsize("<iqIIi")
            + struct.calcsize("<IIQ")
        )

        def corrupt(data: bytearray) -> None:
            struct.pack_into("<q", data, result_offset, VECTORS)

        self.mutate_first_sidecar(corrupt, synchronize_metadata=True)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "result label .* out of range",
        ):
            verifier.verify_evidence(self.evidence)

    def test_nan_result_distance_is_rejected_after_rehash(self) -> None:
        result_offset = (
            struct.calcsize(AUDIT_BODY_HEADER_FORMAT)
            + QUERY_SCHEMA2_FIXED_BYTES
            + QUERY_SCHEMA2_PER_QUERY_BYTES
            + verifier.QUERY_LATENCY_SAMPLE_COUNT * struct.calcsize("<Q")
            + VECTORS * struct.calcsize("<iqIIi")
            + struct.calcsize("<IIQ")
        )

        def corrupt(data: bytearray) -> None:
            struct.pack_into("<I", data, result_offset + 8, 0x7FC00000)

        self.mutate_first_sidecar(corrupt, synchronize_metadata=True)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "non-finite distance",
        ):
            verifier.verify_evidence(self.evidence)

    def test_unsorted_result_distances_are_rejected_after_rehash(self) -> None:
        result_offset = (
            struct.calcsize(AUDIT_BODY_HEADER_FORMAT)
            + QUERY_SCHEMA2_FIXED_BYTES
            + QUERY_SCHEMA2_PER_QUERY_BYTES
            + verifier.QUERY_LATENCY_SAMPLE_COUNT * struct.calcsize("<Q")
            + VECTORS * struct.calcsize("<iqIIi")
            + struct.calcsize("<IIQ")
        )

        def corrupt(data: bytearray) -> None:
            first = bytes(data[result_offset : result_offset + 12])
            second = bytes(data[result_offset + 12 : result_offset + 24])
            data[result_offset : result_offset + 12] = second
            data[result_offset + 12 : result_offset + 24] = first

        self.mutate_first_sidecar(corrupt, synchronize_metadata=True)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "distances are not sorted",
        ):
            verifier.verify_evidence(self.evidence)

    def test_forged_stdout_recall_is_rejected_against_sidecar(self) -> None:
        records = self.records()
        record = records[0]
        record["fields"]["infinity_recall_at_10_ef_32"] = "0.0"
        self.rewrite_stdout(record)
        self.write_records(records)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "stdout field infinity_recall_at_10_ef_32",
        ):
            verifier.verify_evidence(self.evidence)

    def test_forged_stdout_distance_hash_is_rejected_against_sidecar(self) -> None:
        records = self.records()
        record = records[0]
        name = "infinity_returned_distances_sha256_ef_32_k_10"
        record["fields"][name] = "0" * 64
        self.rewrite_stdout(record)
        self.write_records(records)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            f"stdout field {name}",
        ):
            verifier.verify_evidence(self.evidence)

    def test_heldout_manifest_hash_forgery_is_independently_rejected(self) -> None:
        heldout = self.mutate_json(
            verifier.HELDOUT_MANIFEST_FILENAME,
            lambda value: value.update({"queries_sha256": "0" * 64}),
        )
        preflight = json.loads(
            (self.evidence / "preflight.json").read_text(encoding="utf-8")
        )
        preflight["heldout"] = {
            **heldout,
            "record_path": verifier.HELDOUT_MANIFEST_FILENAME,
            "record_sha256": sha256(
                self.evidence / verifier.HELDOUT_MANIFEST_FILENAME
            ),
        }
        write_json(self.evidence / "preflight.json", preflight)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "queries_sha256 differs from independent reconstruction",
        ):
            verifier.verify_evidence(self.evidence)

    def test_preflight_mode_boolean_is_strict(self) -> None:
        self.mutate_json(
            "preflight.json",
            lambda value: value["runner"]["effective_arguments"].update(
                {"preflight_only": 0}
            ),
        )
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "preflight_only argument is not boolean",
        ):
            verifier.verify_evidence(self.evidence)

    def test_stale_raw_ninja_dependency_is_rejected_with_refreshed_hashes(
        self,
    ) -> None:
        deps_path = self.evidence / "infinity-ninja-deps.txt"
        deps_path.write_text(
            deps_path.read_text(encoding="utf-8").replace(
                "(VALID)",
                "(STALE)",
                1,
            ),
            encoding="utf-8",
        )
        closure_path = self.evidence / "infinity-build-input-closure.json"
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
        closure["ninja_deps"]["sha256"] = sha256(deps_path)
        write_json(closure_path, closure)
        preflight = json.loads(
            (self.evidence / "preflight.json").read_text(encoding="utf-8")
        )
        preflight["builds"]["infinity"]["build_input_closure"]["sha256"] = sha256(
            closure_path
        )
        write_json(self.evidence / "preflight.json", preflight)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "has stale deps",
        ):
            verifier.verify_evidence(self.evidence)

    def test_clean_output_list_must_match_authenticated_raw_plan(self) -> None:
        closure_path = self.evidence / "infinity-build-input-closure.json"
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
        clean_plan = closure["selected_target_graph"]["clean_plan"]
        stdout_path = self.evidence / clean_plan["stdout"]["captured_path"]
        stdout_path.write_text(
            stdout_path.read_text(encoding="utf-8").replace(
                "2 files.\n",
                "Remove injected-output.o\n3 files.\n",
                1,
            ),
            encoding="utf-8",
        )
        clean_plan["stdout"]["sha256"] = sha256(stdout_path)
        self.store_build_input_closure("infinity", closure)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "outputs differ from the raw capture",
        ):
            verifier.verify_evidence(self.evidence)

    def test_forged_clean_output_list_is_rejected_against_raw_plan(self) -> None:
        closure_path = self.evidence / "infinity-build-input-closure.json"
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
        closure["selected_target_graph"]["clean_plan"]["outputs"].append(
            "/fixture/build/injected-output.o"
        )
        self.store_build_input_closure("infinity", closure)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "outputs differ from the raw capture",
        ):
            verifier.verify_evidence(self.evidence)

    def test_authenticated_clean_stderr_must_be_empty(self) -> None:
        closure_path = self.evidence / "infinity-build-input-closure.json"
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
        clean_plan = closure["selected_target_graph"]["clean_plan"]
        stderr_path = self.evidence / clean_plan["stderr"]["captured_path"]
        stderr_path.write_text("unexpected warning\n", encoding="utf-8")
        clean_plan["stderr"]["sha256"] = sha256(stderr_path)
        self.store_build_input_closure("infinity", closure)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "clean plan emitted stderr",
        ):
            verifier.verify_evidence(self.evidence)

    def test_omitted_closure_file_is_rejected_with_refreshed_hashes(self) -> None:
        closure_path = self.evidence / "infinity-build-input-closure.json"
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
        closure["files"].pop()
        write_json(closure_path, closure)
        preflight = json.loads(
            (self.evidence / "preflight.json").read_text(encoding="utf-8")
        )
        reference = preflight["builds"]["infinity"]["build_input_closure"]
        reference["files"] -= 1
        reference["sha256"] = sha256(closure_path)
        write_json(self.evidence / "preflight.json", preflight)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "file set differs",
        ):
            verifier.verify_evidence(self.evidence)

    def test_manifest_detects_content_tampering(self) -> None:
        path = self.evidence / "06-P1-infinity.stdout"
        path.write_text(path.read_text(encoding="utf-8") + "tampered=1\n")
        with self.assertRaisesRegex(verifier.VerificationError, "hash mismatch"):
            verifier.verify_evidence(self.evidence)

    def test_macho_dynamic_paths_are_not_response_files(self) -> None:
        arguments = [
            "clang++",
            "@rpath/libfaiss.dylib",
            "@loader_path/libomp.dylib",
            "@executable_path/libsupport.dylib",
        ]
        used_response_files: set[tuple[str, str]] = set()

        expanded = verifier.expand_response_arguments(
            arguments,
            working_directory=Path("/fixture/build"),
            response_files={},
            used_response_files=used_response_files,
            context="fixture",
        )

        self.assertEqual(expanded, arguments)
        self.assertEqual(used_response_files, set())

    def test_missing_run_member_is_rejected_with_refreshed_manifest(self) -> None:
        self.mutate_json("runs.json", lambda records: records.pop())
        write_manifest(self.evidence)
        with self.assertRaisesRegex(verifier.VerificationError, "all 18"):
            verifier.verify_evidence(self.evidence)

    def test_incorrect_summary_is_rejected_with_refreshed_manifest(self) -> None:
        def corrupt(summary: dict[str, Any]) -> None:
            summary["ingestion"]["stratified_geometric_mean_ratio"] += 1e-8

        self.mutate_json("summary.json", corrupt)
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "summary ingestion",
        ):
            verifier.verify_evidence(self.evidence)

    def test_forged_query_qps_summary_is_rejected(self) -> None:
        self.mutate_json(
            "summary.json",
            lambda summary: summary["query"]["pairs"][0]["infinity"].update(
                {"qps": summary["query"]["pairs"][0]["infinity"]["qps"] + 1.0}
            ),
        )
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "summary query",
        ):
            verifier.verify_evidence(self.evidence)

    def test_recall_acceptance_must_match_independent_recomputation(self) -> None:
        self.mutate_json(
            "summary.json",
            lambda summary: summary["acceptance"].update(
                {"paired_recall_deficit_at_most_0_005": False}
            ),
        )
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "acceptance gates differ from independent recomputation",
        ):
            verifier.verify_evidence(self.evidence)

    def test_manifest_rejects_unlisted_file_and_incorrect_hash(self) -> None:
        (self.evidence / "unlisted.txt").write_text("extra\n", encoding="utf-8")
        with self.assertRaisesRegex(verifier.VerificationError, "unlisted"):
            verifier.verify_evidence(self.evidence)

        (self.evidence / "unlisted.txt").unlink()
        manifest_path = self.evidence / "MANIFEST.sha256"
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        lines[0] = "0" * 64 + lines[0][64:]
        manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(verifier.VerificationError, "hash mismatch"):
            verifier.verify_evidence(self.evidence)

    def test_manifest_covered_orphan_content_blob_is_rejected(self) -> None:
        write_content_blob(self.evidence, b"unreferenced provenance payload\n")
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "Orphan content-addressed blobs",
        ):
            verifier.verify_evidence(self.evidence)

    def test_failure_only_evidence_is_rejected(self) -> None:
        (self.evidence / "runs.json").unlink()
        (self.evidence / "summary.json").unlink()
        write_json(
            self.evidence / "failure.json",
            {"status": "fail", "error": "synthetic failure"},
        )
        write_manifest(self.evidence)
        with self.assertRaisesRegex(verifier.VerificationError, "failure.json"):
            verifier.verify_evidence(self.evidence)

    def test_binary_inconsistency_is_rejected(self) -> None:
        def corrupt(records: list[dict[str, Any]]) -> None:
            records[0]["binary"]["sha256"] = "3" * 64

        records = self.mutate_json("runs.json", corrupt)
        write_json(self.evidence / "00-C1-infinity.json", records[0])
        write_manifest(self.evidence)
        with self.assertRaisesRegex(verifier.VerificationError, "binary identity"):
            verifier.verify_evidence(self.evidence)

    def test_idle_to_launch_gate_is_rejected(self) -> None:
        def corrupt(records: list[dict[str, Any]]) -> None:
            records[0]["idle_to_launch_ns"] = 1_000_000_001

        records = self.mutate_json("runs.json", corrupt)
        write_json(self.evidence / "00-C1-infinity.json", records[0])
        write_manifest(self.evidence)
        with self.assertRaisesRegex(verifier.VerificationError, "exceeds one second"):
            verifier.verify_evidence(self.evidence)

    def test_raw_idle_tampering_is_rejected_with_refreshed_manifest(self) -> None:
        raw_path = self.evidence / "00-C1-infinity.idle-attempt-01.top.txt"
        raw_path.write_text(
            "[stdout]\n"
            + "".join("CPU usage: 2.0% user, 4.0% sys, 94.0% idle\n" for _ in range(16))
            + "\n[stderr]\n",
            encoding="utf-8",
        )
        write_manifest(self.evidence)
        with self.assertRaisesRegex(verifier.VerificationError, "sample 1"):
            verifier.verify_evidence(self.evidence)

    def test_failed_idle_stdout_cannot_be_overridden_by_stderr(self) -> None:
        records = json.loads((self.evidence / "runs.json").read_text(encoding="utf-8"))
        record = records[0]
        raw_path = self.evidence / record["idle"]["attempts"][0]["raw_path"]
        raw_path.write_text(
            "[stdout]\n"
            + "".join(
                "CPU usage: 50.0% user, 50.0% sys, 0.0% idle\n" for _ in range(16)
            )
            + "\n[stderr]\n"
            + "".join(
                "CPU usage: 2.0% user, 2.0% sys, 96.0% idle\n" for _ in range(16)
            ),
            encoding="utf-8",
        )
        record["idle"]["attempts"][0].update(
            {
                "idle_percentages": [96.0] * 16,
                "minimum_idle_percent": 96.0,
            }
        )
        self.write_records(records)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "stderr contains CPU-idle samples",
        ):
            verifier.verify_evidence(self.evidence)

    def test_duplicate_process_swap_counter_is_rejected(self) -> None:
        stderr_path = self.evidence / "00-C1-infinity.stderr"
        stderr_path.write_text(
            stderr_path.read_text(encoding="utf-8") + "7 swaps\n",
            encoding="utf-8",
        )
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "exactly one process swap count",
        ):
            verifier.verify_evidence(self.evidence)

    def test_warning_plus_nominal_thermal_capture_is_rejected(self) -> None:
        path = self.evidence / "00-C1-infinity.host-preflight.txt"
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace(
                "[thermal]\n",
                "[thermal]\nThermal warning level: 1\n",
                1,
            ),
            encoding="utf-8",
        )
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "exactly one thermal and performance status",
        ):
            verifier.verify_evidence(self.evidence)

    def test_raw_host_telemetry_tampering_is_rejected_with_refreshed_manifest(
        self,
    ) -> None:
        mutations = {
            "battery": lambda text: text.replace("100%;", "99%;", 1),
            "low-power": lambda text: text.replace(
                "AC Power:\n lowpowermode 0",
                "AC Power:\n lowpowermode 1",
                1,
            ),
            "thermal": lambda text: text.replace(
                "No thermal warning level has been recorded",
                "Thermal warning level: 1",
                1,
            ),
            "swap": lambda text: text.replace(
                "used = 0.00M",
                "used = 1.00M",
                1,
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                evidence = Path(temporary) / "evidence"
                shutil.copytree(self.base, evidence)
                raw_path = evidence / "00-C1-infinity.host-preflight.txt"
                raw_path.write_text(
                    mutate(raw_path.read_text(encoding="utf-8")),
                    encoding="utf-8",
                )
                write_manifest(evidence)
                with self.assertRaisesRegex(
                    verifier.VerificationError,
                    "differs from raw capture",
                ):
                    verifier.verify_evidence(evidence)

    def test_same_basename_alternate_dataset_path_is_rejected(self) -> None:
        def corrupt(records: list[dict[str, Any]]) -> None:
            records[0]["command"][3] = f"/tmp/alternate/{DATASET_NAME}"

        records = self.mutate_json("runs.json", corrupt)
        write_json(self.evidence / "00-C1-infinity.json", records[0])
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "evidence dataset exactly",
        ):
            verifier.verify_evidence(self.evidence)

    def test_archived_runner_tampering_is_rejected_with_refreshed_manifest(
        self,
    ) -> None:
        runner = self.evidence / verifier.RUNNER_CAPTURE_FILENAME
        runner.write_text(
            runner.read_text(encoding="utf-8") + "# replaced\n",
            encoding="utf-8",
        )
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "runner hash differs",
        ):
            verifier.verify_evidence(self.evidence)

    def test_build_capture_tampering_is_rejected_with_refreshed_manifest(
        self,
    ) -> None:
        compile_database = self.evidence / "infinity-compile_commands.json"
        entries = json.loads(compile_database.read_text(encoding="utf-8"))
        entries[0]["command"] += " -DFORGED"
        write_json(compile_database, entries)
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "captured-file hash differs",
        ):
            verifier.verify_evidence(self.evidence)

    def test_trailing_o0_is_rejected_with_all_hashes_refreshed(self) -> None:
        compile_path = self.evidence / "infinity-compile_commands.json"
        entries = json.loads(compile_path.read_text(encoding="utf-8"))
        entries[0]["command"] += " -O0"
        write_json(compile_path, entries)
        preflight = json.loads(
            (self.evidence / "preflight.json").read_text(encoding="utf-8")
        )
        preflight["builds"]["infinity"]["compile_commands"]["sha256"] = sha256(
            compile_path
        )
        write_json(self.evidence / "preflight.json", preflight)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "effective optimization is not -O3",
        ):
            verifier.verify_evidence(self.evidence)

    def test_hidden_response_file_o0_is_rejected_with_refreshed_hashes(
        self,
    ) -> None:
        response = {
            "path": "/fixture/build/hidden.rsp",
            "invocation_working_directory": "/fixture/build",
            **write_content_blob(self.evidence, b"-O0\n"),
        }
        compile_path = self.evidence / "infinity-compile_commands.json"
        entries = json.loads(compile_path.read_text(encoding="utf-8"))
        entries[0]["command"] += " @hidden.rsp"
        write_json(compile_path, entries)

        commands_path = self.evidence / "infinity-ninja-commands.txt"
        commands = commands_path.read_text(encoding="utf-8").splitlines()
        commands[0] += " @hidden.rsp"
        commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")

        preflight = json.loads(
            (self.evidence / "preflight.json").read_text(encoding="utf-8")
        )
        build = preflight["builds"]["infinity"]
        build["response_files"] = [response]
        build["compile_commands"]["sha256"] = sha256(compile_path)
        build["ninja_commands"]["sha256"] = sha256(commands_path)
        write_json(self.evidence / "preflight.json", preflight)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "effective optimization is not -O3",
        ):
            verifier.verify_evidence(self.evidence)

    def test_unreferenced_response_file_record_is_rejected(self) -> None:
        response = {
            "path": "/fixture/build/unused.rsp",
            "invocation_working_directory": "/fixture/build",
            **write_content_blob(self.evidence, b"-DUNUSED\n"),
        }
        preflight = json.loads(
            (self.evidence / "preflight.json").read_text(encoding="utf-8")
        )
        preflight["builds"]["infinity"]["response_files"] = [response]
        write_json(self.evidence / "preflight.json", preflight)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "unreferenced response-file records",
        ):
            verifier.verify_evidence(self.evidence)

    def test_same_output_compile_command_substitution_is_rejected(self) -> None:
        compile_path = self.evidence / "infinity-compile_commands.json"
        entries = json.loads(compile_path.read_text(encoding="utf-8"))
        entries[0]["command"] += " -DFORGED"
        write_json(compile_path, entries)
        preflight = json.loads(
            (self.evidence / "preflight.json").read_text(encoding="utf-8")
        )
        preflight["builds"]["infinity"]["compile_commands"]["sha256"] = sha256(
            compile_path
        )
        write_json(self.evidence / "preflight.json", preflight)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "no exact compiler invocation",
        ):
            verifier.verify_evidence(self.evidence)

    def test_compiled_source_cross_record_tampering_is_rejected(self) -> None:
        records = self.mutate_json(
            "compiled-source-hashes.json",
            lambda value: value["infinity"][0].update(
                {"absolute_path": "/fixture/repo/alternate.cpp"}
            ),
        )
        del records
        preflight = self.mutate_json(
            "preflight.json",
            lambda value: value["source_provenance"].update(
                {
                    "compiled_source_hashes_sha256": sha256(
                        self.evidence / "compiled-source-hashes.json"
                    )
                }
            ),
        )
        del preflight
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "compiled-source records differ",
        ):
            verifier.verify_evidence(self.evidence)

    def test_same_source_path_with_two_contents_is_rejected(self) -> None:
        source_path = self.evidence / "compiled-source-hashes.json"
        records = json.loads(source_path.read_text(encoding="utf-8"))
        records["infinity"][0].update(
            write_content_blob(
                self.evidence,
                b"// coherently substituted compiled source\n",
            )
        )
        write_json(source_path, records)
        preflight = json.loads(
            (self.evidence / "preflight.json").read_text(encoding="utf-8")
        )
        preflight["source_provenance"]["compiled_source_hashes_sha256"] = sha256(
            source_path
        )
        write_json(self.evidence / "preflight.json", preflight)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "has different content",
        ):
            verifier.verify_evidence(self.evidence)

    def test_compiled_source_must_match_its_build_closure(self) -> None:
        source_path = self.evidence / "compiled-source-hashes.json"
        records = json.loads(source_path.read_text(encoding="utf-8"))
        records["faiss_library"][0].update(
            write_content_blob(
                self.evidence,
                b"// source diverging from closure\n",
            )
        )
        write_json(source_path, records)
        preflight = json.loads(
            (self.evidence / "preflight.json").read_text(encoding="utf-8")
        )
        preflight["source_provenance"]["compiled_source_hashes_sha256"] = sha256(
            source_path
        )
        write_json(self.evidence / "preflight.json", preflight)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "compiled-source content differs from the build closure",
        ):
            verifier.verify_evidence(self.evidence)

    def test_faiss_compiled_source_must_match_derivative_tree(self) -> None:
        source_path = self.evidence / "compiled-source-hashes.json"
        records = json.loads(source_path.read_text(encoding="utf-8"))
        substituted = write_content_blob(
            self.evidence,
            b"// substituted FAISS source\n",
        )
        records["faiss_library"][0].update(substituted)
        write_json(source_path, records)

        closure_path = self.evidence / "faiss-library-build-input-closure.json"
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
        absolute = records["faiss_library"][0]["absolute_path"]
        closure_source = next(
            record
            for record in closure["files"]
            if record["path"] == absolute
            and "compile-source" in record["roles"]
        )
        closure_source.update(substituted)
        write_json(closure_path, closure)

        preflight = json.loads(
            (self.evidence / "preflight.json").read_text(encoding="utf-8")
        )
        preflight["source_provenance"]["compiled_source_hashes_sha256"] = sha256(
            source_path
        )
        preflight["linked_faiss_build"]["build_input_closure"]["sha256"] = sha256(
            closure_path
        )
        write_json(self.evidence / "preflight.json", preflight)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "pristine source .* differs from HEAD|compiled source .* differs",
        ):
            verifier.verify_evidence(self.evidence)

    def test_closure_schema_rejects_json_boolean(self) -> None:
        preflight = json.loads(
            (self.evidence / "preflight.json").read_text(encoding="utf-8")
        )
        preflight["builds"]["infinity"]["build_input_closure"][
            "closure_schema_version"
        ] = True
        write_json(self.evidence / "preflight.json", preflight)
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "reference schema must be an integer",
        ):
            verifier.verify_evidence(self.evidence)

    def test_omitted_faiss_dependency_is_rejected_against_macho(self) -> None:
        dependencies_path = self.evidence / "faiss-dependencies.json"
        dependency_manifest = json.loads(dependencies_path.read_text(encoding="utf-8"))
        dependency_manifest["dependencies"] = [
            dependency
            for dependency in dependency_manifest["dependencies"]
            if dependency["resolved_path"] != LIBFAISS_PATH
        ]
        write_json(dependencies_path, dependency_manifest)
        preflight = json.loads(
            (self.evidence / "preflight.json").read_text(encoding="utf-8")
        )
        reference = preflight["builds"]["faiss"]["dependencies"]
        reference["sha256"] = sha256(dependencies_path)
        reference["count"] = len(dependency_manifest["dependencies"])
        write_json(self.evidence / "preflight.json", preflight)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "otool outputs do not cover|dependency records differ",
        ):
            verifier.verify_evidence(self.evidence)

    def test_runner_argv_must_reproduce_effective_arguments(self) -> None:
        preflight = self.mutate_json(
            "preflight.json",
            lambda value: value["runner"]["argv"].extend(
                ["--idle-timeout-seconds", "1"]
            ),
        )
        del preflight
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "effective arguments differ from argv",
        ):
            verifier.verify_evidence(self.evidence)

    def test_compile_source_path_does_not_follow_live_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.cpp"
            second = root / "second.cpp"
            link = root / "source.cpp"
            first.write_text("first", encoding="ascii")
            second.write_text("second", encoding="ascii")
            link.symlink_to(first.name)
            entry = {"directory": str(root), "file": link.name}
            self.assertEqual(
                verifier.compile_source_paths([entry], "fixture"),
                {str(link)},
            )
            link.unlink()
            link.symlink_to(second.name)
            self.assertEqual(
                verifier.compile_source_paths([entry], "fixture"),
                {str(link)},
            )

    def test_verifier_mode_flag_rejects_abbreviation_and_repetition(self) -> None:
        for arguments in (
            ["/tmp/evidence", "--preflight"],
            ["/tmp/evidence", "--preflight-only", "--preflight-only"],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit):
                    verifier.parse_args(arguments)

    def test_compiler_must_be_present_in_archived_runtime_artifacts(self) -> None:
        preflight = json.loads(
            (self.evidence / "preflight.json").read_text(encoding="utf-8")
        )
        compiler_path = TOOL_METADATA["compiler"]["resolved_path"]
        del preflight["runtime_artifact_hashes"][compiler_path]
        preflight["runtime_artifact_snapshots"] = [
            snapshot
            for snapshot in preflight["runtime_artifact_snapshots"]
            if snapshot["original_path"] != compiler_path
        ]
        write_json(self.evidence / "preflight.json", preflight)

        records = json.loads((self.evidence / "runs.json").read_text(encoding="utf-8"))
        for record in records:
            for key in (
                "runtime_artifacts_before",
                "runtime_artifacts_after",
            ):
                record[key]["artifacts"] = [
                    artifact
                    for artifact in record[key]["artifacts"]
                    if artifact["path"] != compiler_path
                ]
        self.write_records(records)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "compiler is absent from runtime artifacts",
        ):
            verifier.verify_evidence(self.evidence)

    def test_coherent_noncanonical_dataset_substitution_is_rejected(self) -> None:
        replacement = b"\0" * DATASET_BYTES
        dataset_path = self.evidence / DATASET_NAME
        dataset_path.chmod(0o644)
        dataset_path.write_bytes(replacement)
        dataset_path.chmod(0o444)
        replacement_sha = hashlib.sha256(replacement).hexdigest()
        replacement_fnv = fnv1a64(replacement)

        dataset = json.loads(
            (self.evidence / "dataset.json").read_text(encoding="utf-8")
        )
        dataset.update({"sha256": replacement_sha, "fnv1a64": replacement_fnv})
        write_json(self.evidence / "dataset.json", dataset)

        preflight = json.loads(
            (self.evidence / "preflight.json").read_text(encoding="utf-8")
        )
        preflight["dataset"].update(
            {
                "sha256": replacement_sha,
                "fnv1a64": replacement_fnv,
                "record_sha256": sha256(self.evidence / "dataset.json"),
            }
        )
        write_json(self.evidence / "preflight.json", preflight)

        records = json.loads((self.evidence / "runs.json").read_text(encoding="utf-8"))
        for record in records:
            for key in ("dataset_before", "dataset_after"):
                record[key].update(
                    {
                        "sha256": replacement_sha,
                        "fnv1a64": replacement_fnv,
                    }
                )
            record["fields"].update(
                {
                    "data_sha256": replacement_sha,
                    "data_fnv1a64": str(replacement_fnv),
                }
            )
            (self.evidence / record["stdout_path"]).write_text(
                "".join(f"{key}={value}\n" for key, value in record["fields"].items()),
                encoding="utf-8",
            )
        self.write_records(records)

        summary = json.loads(
            (self.evidence / "summary.json").read_text(encoding="utf-8")
        )
        summary["workload"].update(
            {
                "data_sha256": replacement_sha,
                "data_fnv1a64": replacement_fnv,
            }
        )
        write_json(self.evidence / "summary.json", summary)
        write_manifest(self.evidence)

        with self.assertRaisesRegex(
            verifier.VerificationError,
            "frozen canonical dataset",
        ):
            verifier.verify_evidence(self.evidence)

    def test_development_idle_override_is_independently_verified(self) -> None:
        def use_development_idle(value: dict[str, Any]) -> None:
            value["protocol"].update(
                {
                    "idle_minimum_percent": 25.0,
                    "idle_policy": "development-only-override",
                }
            )
            value["runner"]["effective_arguments"].update(
                {"idle_minimum_percent": 25.0}
            )
            value["runner"]["argv"].extend(["--development-idle-minimum-percent", "25"])

        preflight = self.mutate_json(
            "preflight.json",
            use_development_idle,
        )
        self.assertEqual(preflight["protocol"]["idle_minimum_percent"], 25.0)
        records = self.mutate_json(
            "runs.json",
            lambda values: [
                record["idle"].update({"required_minimum_idle_percent": 25.0})
                for record in values
            ],
        )
        for record in records:
            prefix = f"{record['sequence']:02d}-{record['pair']}-{record['engine']}"
            raw_name = record["idle"]["attempts"][0]["raw_path"]
            (self.evidence / raw_name).write_text(
                "[stdout]\n"
                + "".join(
                    "CPU usage: 34.0% user, 40.0% sys, 26.0% idle\n" for _ in range(16)
                )
                + "\n[stderr]\n",
                encoding="utf-8",
            )
            record["idle"]["attempts"][0].update(
                {
                    "idle_percentages": [26.0] * 16,
                    "minimum_idle_percent": 26.0,
                }
            )
            write_json(self.evidence / f"{prefix}.json", record)
        write_json(self.evidence / "runs.json", records)
        self.mutate_json(
            "summary.json",
            lambda value: value.update(
                {
                    "idle": {
                        "minimum_percent": 25.0,
                        "window_seconds": 15,
                        "policy": "development-only-override",
                    },
                    "limitations": value["limitations"]
                    + [
                        "A development-only 25% CPU-idle threshold replaced "
                        "the default 95% threshold; this campaign is suitable "
                        "for iteration, not qualification."
                    ],
                }
            ),
        )
        write_manifest(self.evidence)

        result = verifier.verify_evidence(self.evidence)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["idle"]["minimum_percent"], 25.0)

    def test_idle_policy_threshold_mismatch_is_rejected(self) -> None:
        self.mutate_json(
            "preflight.json",
            lambda value: value["protocol"].update(
                {"idle_policy": "development-only-override"}
            ),
        )
        write_manifest(self.evidence)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "idle policy disagrees",
        ):
            verifier.verify_evidence(self.evidence)

    def test_data_inconsistency_is_rejected(self) -> None:
        def corrupt(records: list[dict[str, Any]]) -> None:
            records[0]["fields"]["data_fnv1a64"] = "0"

        records = self.mutate_json("runs.json", corrupt)
        write_json(self.evidence / "00-C1-infinity.json", records[0])
        fields = records[0]["fields"]
        (self.evidence / "00-C1-infinity.stdout").write_text(
            "".join(f"{key}={value}\n" for key, value in fields.items()),
            encoding="utf-8",
        )
        write_manifest(self.evidence)
        with self.assertRaisesRegex(verifier.VerificationError, "data_fnv1a64"):
            verifier.verify_evidence(self.evidence)


if __name__ == "__main__":
    unittest.main()
