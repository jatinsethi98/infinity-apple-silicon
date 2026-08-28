from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import shutil
import struct
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

from tools.apple_silicon import verify_native_hnsw_exactness as verifier

VECTOR_COUNT = 12_288
DIMENSION = 128
M = 32
EF_CONSTRUCTION = 200
CHUNK_SIZE = 8_192
MAX_CHUNKS = 2
WORKERS = 1
MMAX0 = 64
MMAX = 32
DATASET_BYTES = VECTOR_COUNT * DIMENSION * 4
DATASET_NAME = "d0-f32le-n12288-d128-seed0.bin"
DATASET_SHA256 = "f2e29c0f1a64d48a81e2adba0213c53d3015f459ef9936668a7dabb6a025f32a"
GRAPH_HEADER_BYTES = 184
LEVEL0_RECORD_BYTES = 280
UPPER_RECORD_BYTES = 132

GRAPH_HEADER = struct.Struct("<8sII13Qii2Q32sQ")
GRAPH_VERTEX = struct.Struct("<IiiI")
GRAPH_LAYER = struct.Struct("<II")
SAVE_HEADER = struct.Struct("<6Qii")
WITNESS = struct.Struct("<8sII5I")
WITNESS_V2 = struct.Struct("<8sII10I")


def witness_bits(value: int) -> dict[str, int]:
    return {field: value for field in verifier.WITNESS_FIELD_NAMES}


THRESHOLD_ONLY_EXPECTED_WITNESSES = {
    "control": {
        **{
            field: 1
            for field in verifier.WITNESS_FIELD_NAMES[:5]
        },
        **{
            field: 0
            for field in verifier.WITNESS_FIELD_NAMES[5:]
        },
    },
    "treatment": witness_bits(1),
}
COMBINED_EXPECTED_WITNESSES = {
    "control": witness_bits(0),
    "treatment": witness_bits(1),
}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def artifact(path: str, content: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": sha256(content), "bytes": len(content)}


def lifecycle(pid: int, base_time: int) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "leader_pid": pid,
        "terminal_observation": {
            "observed_at_monotonic_ns": base_time,
            "pid": pid,
            "idtype": 1,
            "options": 0x00000001 | 0x00000004 | 0x00000020,
            "code": 1,
            "status": 0,
        },
        "pre_reap_quiescence": {
            "observed_at_monotonic_ns": base_time + 1,
            "process_group": pid,
            "leader_pid": pid,
            "member_pids": [pid],
            "no_members_except_leader": True,
        },
        "reap": {
            "completed_at_monotonic_ns": base_time + 2,
            "requested_pid": pid,
            "options": 0,
            "returned_pid": pid,
            "raw_status": 0,
            "status_validated": True,
            "returncode": 0,
        },
    }


def directory_identity(value: os.stat_result) -> tuple[int, ...]:
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


def frozen_dataset() -> bytes:
    generator = random.Random(0)
    content = bytearray(DATASET_BYTES)
    float32 = struct.Struct("<f")
    for offset in range(0, DATASET_BYTES, float32.size):
        float32.pack_into(content, offset, generator.random())
    result = bytes(content)
    if sha256(result) != DATASET_SHA256:
        raise AssertionError(
            "test-local dataset encoder differs from the frozen corpus"
        )
    return result


def level_for(vertex: int) -> int:
    return 1 if vertex % 2 == 0 else 0


def neighbors_for(vertex: int, layer: int) -> tuple[int, int]:
    step = 1 if layer == 0 else 2
    return ((vertex - step) % VECTOR_COUNT, (vertex + step) % VECTOR_COUNT)


def canonical_graph() -> tuple[bytes, dict[str, Any]]:
    header = GRAPH_HEADER.pack(
        b"IFHXGR01",
        1,
        GRAPH_HEADER_BYTES,
        VECTOR_COUNT,
        DIMENSION,
        M,
        EF_CONSTRUCTION,
        CHUNK_SIZE,
        MAX_CHUNKS,
        WORKERS,
        0,
        VECTOR_COUNT,
        1,
        VECTOR_COUNT,
        MMAX0,
        MMAX,
        1,
        0,
        VECTOR_COUNT,
        1,
        bytes.fromhex(DATASET_SHA256),
        GRAPH_HEADER_BYTES,
    )
    parts: list[bytes] = [header]
    offset = len(header)
    layout: dict[str, Any] = {
        "vertex": [],
        "layer": {},
        "degree": {},
        "neighbor": {},
    }
    for vertex in range(VECTOR_COUNT):
        level = level_for(vertex)
        layout["vertex"].append(offset)
        parts.append(GRAPH_VERTEX.pack(vertex, vertex, level, level + 1))
        offset += GRAPH_VERTEX.size
        for layer in range(level + 1):
            layout["layer"][vertex, layer] = offset
            layout["degree"][vertex, layer] = offset + 4
            parts.append(GRAPH_LAYER.pack(layer, 2))
            offset += GRAPH_LAYER.size
            for index, neighbor in enumerate(neighbors_for(vertex, layer)):
                layout["neighbor"][vertex, layer, index] = offset
                parts.append(struct.pack("<i", neighbor))
                offset += 4
    result = b"".join(parts)
    if len(result) != offset:
        raise AssertionError("test-local graph layout accounting differs")
    return result, layout


def save_to_ptr(dataset: bytes) -> tuple[bytes, dict[str, Any]]:
    parts: list[bytes] = [
        SAVE_HEADER.pack(
            M,
            EF_CONSTRUCTION,
            VECTOR_COUNT,
            DIMENSION,
            MMAX0,
            MMAX,
            1,
            0,
        ),
        dataset,
    ]
    layer_sum = sum(level_for(vertex) for vertex in range(VECTOR_COUNT))
    layer_sum_offset = SAVE_HEADER.size + len(dataset)
    parts.append(struct.pack("<Q", layer_sum))
    level_zero_offset = layer_sum_offset + 8
    layout: dict[str, Any] = {
        "vectors": SAVE_HEADER.size,
        "layer_sum": layer_sum_offset,
        "level_zero": [],
        "upper": {},
        "labels": 0,
    }
    upper_ordinal = 0
    for vertex in range(VECTOR_COUNT):
        level = level_for(vertex)
        record = bytearray(LEVEL0_RECORD_BYTES)
        struct.pack_into("<i", record, 0, level)
        struct.pack_into(
            "<Q",
            record,
            8,
            upper_ordinal * UPPER_RECORD_BYTES if level else 0,
        )
        struct.pack_into("<i", record, 16, 2)
        struct.pack_into("<2i", record, 20, *neighbors_for(vertex, 0))
        layout["level_zero"].append(level_zero_offset + vertex * LEVEL0_RECORD_BYTES)
        parts.append(bytes(record))
        upper_ordinal += level
    if upper_ordinal != layer_sum:
        raise AssertionError("test-local upper-layer accounting differs")
    upper_offset = level_zero_offset + VECTOR_COUNT * LEVEL0_RECORD_BYTES
    upper_ordinal = 0
    for vertex in range(VECTOR_COUNT):
        for layer in range(1, level_for(vertex) + 1):
            record = bytearray(UPPER_RECORD_BYTES)
            struct.pack_into("<i", record, 0, 2)
            struct.pack_into("<2i", record, 4, *neighbors_for(vertex, layer))
            layout["upper"][vertex, layer] = (
                upper_offset + upper_ordinal * UPPER_RECORD_BYTES
            )
            parts.append(bytes(record))
            upper_ordinal += 1
    labels_offset = upper_offset + layer_sum * UPPER_RECORD_BYTES
    layout["labels"] = labels_offset
    parts.append(struct.pack(f"<{VECTOR_COUNT}i", *range(VECTOR_COUNT)))
    result = b"".join(parts)
    if labels_offset + VECTOR_COUNT * 4 != len(result):
        raise AssertionError("test-local SaveToPtr layout accounting differs")
    return result, layout


def stdout_json(graph: bytes, pointer_image: bytes) -> bytes:
    value = {
        "build": {
            "end": VECTOR_COUNT,
            "memory_delta_positive": True,
            "start": 0,
            "submitted_tasks": 1,
        },
        "configuration": {
            "chunk_size": CHUNK_SIZE,
            "dimension": DIMENSION,
            "ef_construction": EF_CONSTRUCTION,
            "m": M,
            "max_chunks": MAX_CHUNKS,
            "optimize": True,
            "vectors": VECTOR_COUNT,
            "workers": WORKERS,
        },
        "dataset": {"bytes": DATASET_BYTES, "sha256": DATASET_SHA256},
        "graph": {
            "bytes": len(graph),
            "entry_point": 0,
            "header_bytes": GRAPH_HEADER_BYTES,
            "max_level": 1,
            "mmax": MMAX,
            "mmax0": MMAX0,
            "sha256": sha256(graph),
        },
        "save_to_ptr": {
            "bytes": len(pointer_image),
            "roundtrip_graph_equal": True,
            "sha256": sha256(pointer_image),
        },
        "schema_version": 1,
        "status": "PASS",
    }
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    )


class VerifyNativeHnswExactnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = frozen_dataset()
        cls.graph, cls.graph_layout = canonical_graph()
        cls.pointer_image, cls.save_layout = save_to_ptr(cls.dataset)
        cls.stdout = stdout_json(cls.graph, cls.pointer_image)
        cls.control_witness = WITNESS.pack(b"IFHXWT01", 1, 36, 0, 0, 0, 0, 0)
        cls.treatment_witness = WITNESS.pack(b"IFHXWT01", 1, 36, 1, 1, 1, 1, 1)

        cls._base_temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls._base_temporary.name)
        (cls.base / DATASET_NAME).write_bytes(cls.dataset)
        roles: dict[str, Any] = {}
        for role_index, (role, witness) in enumerate(
            (
                ("control", cls.control_witness),
                ("treatment", cls.treatment_witness),
            )
        ):
            runs = []
            for ordinal in (1, 2):
                directory = cls.base / "exactness" / role / f"run-{ordinal}"
                directory.mkdir(parents=True)
                contents = {
                    "stdout": cls.stdout,
                    "stderr": b"",
                    "graph": cls.graph,
                    "save_to_ptr": cls.pointer_image,
                    "witness": witness,
                }
                pid = 1_000 + role_index * 10 + ordinal
                run: dict[str, Any] = {
                    "ordinal": ordinal,
                    "lifecycle": lifecycle(
                        pid,
                        1_000_000 + role_index * 100 + ordinal * 10,
                    ),
                }
                for name, content in contents.items():
                    relative = (
                        Path("exactness") / role / f"run-{ordinal}" / f"{name}.bin"
                    ).as_posix()
                    (cls.base / relative).write_bytes(content)
                    run[name] = artifact(relative, content)
                runs.append(run)
            roles[role] = {"runs": runs}
        cls.base_record = {
            "schema_version": 2,
            "dataset": artifact(DATASET_NAME, cls.dataset),
            "roles": roles,
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls._base_temporary.cleanup()

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.evidence_dir = Path(self._temporary.name) / "evidence"
        shutil.copytree(self.base, self.evidence_dir, copy_function=shutil.copy2)
        self.record = copy.deepcopy(self.base_record)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def run_record(
        self,
        record: dict[str, Any] | None = None,
        *,
        expected_roles: dict[str, int] | None = None,
        expected_witnesses: dict[str, dict[str, int]] | None = None,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        if expected_roles is not None:
            arguments["expected_roles"] = expected_roles
        if expected_witnesses is not None:
            arguments["expected_witnesses"] = expected_witnesses
        return verifier.validate_exactness_gate(
            self.evidence_dir,
            self.record if record is None else record,
            **arguments,
        )

    def assert_rejected(
        self,
        detail: str,
        *,
        record: dict[str, Any] | None = None,
        expected_roles: dict[str, int] | None = None,
        expected_witnesses: dict[str, dict[str, int]] | None = None,
    ) -> None:
        with self.assertRaisesRegex(
            verifier.ExactnessVerificationError,
            re.escape(detail),
        ):
            self.run_record(
                record,
                expected_roles=expected_roles,
                expected_witnesses=expected_witnesses,
            )

    def artifact_record(self, role: str, ordinal: int, name: str) -> dict[str, Any]:
        return self.record["roles"][role]["runs"][ordinal - 1][name]

    def replace_artifact(
        self,
        role: str,
        ordinal: int,
        name: str,
        content: bytes,
    ) -> None:
        record = self.artifact_record(role, ordinal, name)
        path = self.evidence_dir / record["path"]
        replacement = path.with_name(path.name + ".replacement")
        replacement.write_bytes(content)
        os.replace(replacement, path)
        record["sha256"] = sha256(content)
        record["bytes"] = len(content)

    def restore_artifact(self, role: str, ordinal: int, name: str) -> None:
        if self.record["schema_version"] == 3:
            witness = self.current_witness(
                self.record["expected_witnesses"],
                role,
            )
        else:
            witness = (
                self.control_witness
                if role == "control"
                else self.treatment_witness
            )
        content = {
            "graph": self.graph,
            "save_to_ptr": self.pointer_image,
            "stdout": self.stdout,
            "stderr": b"",
            "witness": witness,
        }[name]
        self.replace_artifact(role, ordinal, name, content)

    @staticmethod
    def current_witness(
        expected_witnesses: dict[str, dict[str, int]],
        role: str,
    ) -> bytes:
        return WITNESS_V2.pack(
            b"IFHXWT01",
            2,
            56,
            *(
                expected_witnesses[role][field]
                for field in verifier.WITNESS_FIELD_NAMES
            ),
        )

    def install_schema3(
        self,
        expected_witnesses: dict[str, dict[str, int]],
    ) -> None:
        self.record["schema_version"] = 3
        self.record["expected_witnesses"] = copy.deepcopy(
            expected_witnesses
        )
        for role in expected_witnesses:
            witness = self.current_witness(expected_witnesses, role)
            for ordinal in (1, 2):
                self.replace_artifact(role, ordinal, "witness", witness)

    def replace_dataset(self, content: bytes) -> None:
        path = self.evidence_dir / self.record["dataset"]["path"]
        replacement = path.with_name(path.name + ".replacement")
        replacement.write_bytes(content)
        os.replace(replacement, path)
        self.record["dataset"]["sha256"] = sha256(content)
        self.record["dataset"]["bytes"] = len(content)

    @staticmethod
    def patched(
        original: bytes,
        offset: int,
        layout: str,
        value: Any,
    ) -> bytes:
        candidate = bytearray(original)
        struct.pack_into(layout, candidate, offset, value)
        return bytes(candidate)

    def install_consistent_alternate(
        self,
        role: str,
        ordinal: int,
    ) -> None:
        graph = bytearray(self.graph)
        first = self.graph_layout["neighbor"][0, 0, 0]
        second = self.graph_layout["neighbor"][0, 0, 1]
        graph[first : first + 4], graph[second : second + 4] = (
            graph[second : second + 4],
            graph[first : first + 4],
        )
        pointer_image = bytearray(self.pointer_image)
        record = self.save_layout["level_zero"][0]
        (
            pointer_image[record + 20 : record + 24],
            pointer_image[record + 24 : record + 28],
        ) = (
            pointer_image[record + 24 : record + 28],
            pointer_image[record + 20 : record + 24],
        )
        graph_bytes = bytes(graph)
        pointer_bytes = bytes(pointer_image)
        self.replace_artifact(role, ordinal, "graph", graph_bytes)
        self.replace_artifact(role, ordinal, "save_to_ptr", pointer_bytes)
        self.replace_artifact(
            role,
            ordinal,
            "stdout",
            stdout_json(graph_bytes, pointer_bytes),
        )

    def test_valid_gate_is_read_only_and_returns_normalized_summary(self) -> None:
        before = {
            path.relative_to(self.evidence_dir): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
                path.stat().st_ctime_ns,
            )
            for path in self.evidence_dir.rglob("*")
            if path.is_file()
        }
        summary = self.run_record()
        after = {
            path.relative_to(self.evidence_dir): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
                path.stat().st_ctime_ns,
            )
            for path in self.evidence_dir.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["dataset"]["sha256"], DATASET_SHA256)
        self.assertEqual(summary["configuration"]["vectors"], VECTOR_COUNT)
        self.assertEqual(summary["configuration"]["dimension"], DIMENSION)
        self.assertEqual(summary["configuration"]["m"], M)
        self.assertEqual(summary["configuration"]["ef_construction"], EF_CONSTRUCTION)
        self.assertEqual(summary["configuration"]["workers"], 1)
        self.assertEqual(summary["graph"]["sha256"], sha256(self.graph))
        self.assertEqual(
            summary["save_to_ptr"]["sha256"],
            sha256(self.pointer_image),
        )
        self.assertEqual(
            set(summary["roles"]["control"]["witness"].values()),
            {0},
        )
        self.assertEqual(
            set(summary["roles"]["treatment"]["witness"].values()),
            {1},
        )
        self.assertIs(summary["within_role_deterministic"], True)
        self.assertIs(summary["cross_role_equal"], True)

    def test_schema3_threshold_only_matrix_is_authenticated(self) -> None:
        self.install_schema3(THRESHOLD_ONLY_EXPECTED_WITNESSES)

        summary, producer_semantics = (
            verifier.validate_exactness_gate_with_producer_semantics(
                self.evidence_dir,
                self.record,
                expected_witnesses=THRESHOLD_ONLY_EXPECTED_WITNESSES,
            )
        )

        self.assertEqual(summary["schema_version"], 3)
        self.assertEqual(
            summary["expected_witnesses"],
            THRESHOLD_ONLY_EXPECTED_WITNESSES,
        )
        self.assertFalse(summary["cross_role_witness_byte_equal"])
        self.assertEqual(
            summary["roles"]["control"]["witness"],
            THRESHOLD_ONLY_EXPECTED_WITNESSES["control"],
        )
        self.assertEqual(
            producer_semantics["treatment"][0]["witness"][
                "threshold_surviving_lane_observed"
            ],
            1,
        )
        self.assertEqual(
            producer_semantics["control"][0]["witness"]["schema_version"],
            2,
        )
        self.assertEqual(
            producer_semantics["control"][0]["witness"]["bytes"],
            56,
        )

    def test_schema3_combined_matrix_is_authenticated(self) -> None:
        self.install_schema3(COMBINED_EXPECTED_WITNESSES)

        summary = self.run_record(
            expected_witnesses=COMBINED_EXPECTED_WITNESSES,
        )

        self.assertEqual(
            summary["roles"]["control"]["witness"],
            witness_bits(0),
        )
        self.assertEqual(
            summary["roles"]["treatment"]["witness"],
            witness_bits(1),
        )

    def test_schema3_allows_equal_explicit_witness_vectors(self) -> None:
        equal = {
            "control": witness_bits(1),
            "treatment": witness_bits(1),
        }
        self.install_schema3(equal)

        summary = self.run_record(expected_witnesses=equal)

        self.assertTrue(summary["cross_role_witness_byte_equal"])

    def test_schema3_requires_and_authenticates_external_matrix(self) -> None:
        self.install_schema3(THRESHOLD_ONLY_EXPECTED_WITNESSES)

        self.assert_rejected("schema 3 exactness requires expected_witnesses")
        self.assert_rejected(
            "differ from authenticated expectations",
            expected_witnesses=COMBINED_EXPECTED_WITNESSES,
        )
        self.assert_rejected(
            "schema 3 exactness does not accept expected_roles",
            expected_roles={"control": 0, "treatment": 1},
            expected_witnesses=THRESHOLD_ONLY_EXPECTED_WITNESSES,
        )

    def test_schema3_requires_exact_control_and_treatment_roles(self) -> None:
        self.install_schema3(THRESHOLD_ONLY_EXPECTED_WITNESSES)
        renamed = {
            "baseline": copy.deepcopy(
                THRESHOLD_ONLY_EXPECTED_WITNESSES["control"]
            ),
            "optimized": copy.deepcopy(
                THRESHOLD_ONLY_EXPECTED_WITNESSES["treatment"]
            ),
        }

        self.assert_rejected(
            "must contain exactly control and treatment",
            expected_witnesses=renamed,
        )
        candidate = copy.deepcopy(self.record)
        candidate["expected_witnesses"] = renamed
        self.assert_rejected(
            "must contain exactly control and treatment",
            record=candidate,
            expected_witnesses=THRESHOLD_ONLY_EXPECTED_WITNESSES,
        )

    def test_schema3_rejects_malformed_witness_mappings(self) -> None:
        self.install_schema3(THRESHOLD_ONLY_EXPECTED_WITNESSES)
        malformed: dict[str, dict[str, dict[str, Any]]] = {}

        nonboolean = copy.deepcopy(THRESHOLD_ONLY_EXPECTED_WITNESSES)
        nonboolean["control"][verifier.WITNESS_FIELD_NAMES[0]] = True
        malformed["nonboolean"] = nonboolean

        missing = copy.deepcopy(THRESHOLD_ONLY_EXPECTED_WITNESSES)
        missing["control"].pop(verifier.WITNESS_FIELD_NAMES[0])
        malformed["missing"] = missing

        extra = copy.deepcopy(THRESHOLD_ONLY_EXPECTED_WITNESSES)
        extra["control"]["unexpected"] = 0
        malformed["extra"] = extra

        for name, matrix in malformed.items():
            with self.subTest(source="external", mutation=name):
                with self.assertRaises(
                    verifier.ExactnessVerificationError
                ):
                    self.run_record(expected_witnesses=matrix)
            with self.subTest(source="record", mutation=name):
                candidate = copy.deepcopy(self.record)
                candidate["expected_witnesses"] = matrix
                with self.assertRaises(
                    verifier.ExactnessVerificationError
                ):
                    self.run_record(
                        candidate,
                        expected_witnesses=(
                            THRESHOLD_ONLY_EXPECTED_WITNESSES
                        ),
                    )

    def test_schema3_rejects_bool_for_every_witness_bit(self) -> None:
        self.install_schema3(THRESHOLD_ONLY_EXPECTED_WITNESSES)
        for role in ("control", "treatment"):
            for field in verifier.WITNESS_FIELD_NAMES:
                with self.subTest(role=role, field=field):
                    candidate = copy.deepcopy(
                        THRESHOLD_ONLY_EXPECTED_WITNESSES
                    )
                    candidate[role][field] = True
                    with self.assertRaises(
                        verifier.ExactnessVerificationError
                    ):
                        self.run_record(expected_witnesses=candidate)

    def test_schema3_rejects_wrong_witness_size_and_schema(self) -> None:
        self.install_schema3(THRESHOLD_ONLY_EXPECTED_WITNESSES)
        control = self.current_witness(
            THRESHOLD_ONLY_EXPECTED_WITNESSES,
            "control",
        )
        cases = (
            ("size", control[:-1], "recorded size differs from 56"),
            (
                "schema",
                self.patched(control, 8, "<I", 1),
                "schema version differs",
            ),
        )
        for name, content, detail in cases:
            with self.subTest(name=name):
                self.replace_artifact("control", 1, "witness", content)
                self.assert_rejected(
                    detail,
                    expected_witnesses=(
                        THRESHOLD_ONLY_EXPECTED_WITNESSES
                    ),
                )
                self.restore_artifact("control", 1, "witness")

    def test_record_schema_is_exact_and_typed(self) -> None:
        mutations: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
            (
                "extra top-level key",
                lambda value: value.__setitem__("extra", None),
                "exactness gate keys differ",
            ),
            (
                "missing roles",
                lambda value: value.pop("roles"),
                "exactness gate keys differ",
            ),
            (
                "boolean schema",
                lambda value: value.__setitem__("schema_version", True),
                "schema version must be an integer",
            ),
            (
                "wrong schema",
                lambda value: value.__setitem__("schema_version", 1),
                "unsupported exactness gate schema version",
            ),
            (
                "extra role",
                lambda value: value["roles"].__setitem__(
                    "other", copy.deepcopy(value["roles"]["control"])
                ),
                "exactness roles keys differ",
            ),
            (
                "missing role",
                lambda value: value["roles"].pop("control"),
                "exactness roles keys differ",
            ),
            (
                "extra role key",
                lambda value: value["roles"]["control"].__setitem__("extra", None),
                "control exactness role keys differ",
            ),
            (
                "one run",
                lambda value: value["roles"]["control"]["runs"].pop(),
                "control must contain exactly two exactness runs",
            ),
            (
                "three runs",
                lambda value: value["roles"]["control"]["runs"].append(
                    copy.deepcopy(value["roles"]["control"]["runs"][0])
                ),
                "control must contain exactly two exactness runs",
            ),
            (
                "wrong ordinal",
                lambda value: value["roles"]["control"]["runs"][0].__setitem__(
                    "ordinal", 2
                ),
                "control exactness run 1 ordinal differs",
            ),
            (
                "boolean ordinal",
                lambda value: value["roles"]["control"]["runs"][0].__setitem__(
                    "ordinal", True
                ),
                "ordinal must be an integer",
            ),
            (
                "failed process",
                lambda value: value["roles"]["control"]["runs"][0]["lifecycle"][
                    "terminal_observation"
                ].__setitem__("status", 1),
                "does not prove successful exit",
            ),
            (
                "extra run key",
                lambda value: value["roles"]["control"]["runs"][0].__setitem__(
                    "extra", None
                ),
                "control exactness run 1 keys differ",
            ),
        ]
        for name, mutate, detail in mutations:
            with self.subTest(name=name):
                candidate = copy.deepcopy(self.base_record)
                mutate(candidate)
                self.assert_rejected(detail, record=candidate)

    def test_lifecycle_evidence_is_exact_typed_and_successful(self) -> None:
        def first_lifecycle(value: dict[str, Any]) -> dict[str, Any]:
            return value["roles"]["control"]["runs"][0]["lifecycle"]

        mutations: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
            (
                "legacy lifecycle schema",
                lambda value: first_lifecycle(value).__setitem__(
                    "schema_version", 1
                ),
                "lifecycle schema version differs",
            ),
            (
                "boolean leader PID",
                lambda value: first_lifecycle(value).__setitem__(
                    "leader_pid", True
                ),
                "leader PID must be an integer",
            ),
            (
                "missing terminal field",
                lambda value: first_lifecycle(value)["terminal_observation"].pop(
                    "idtype"
                ),
                "terminal observation keys differ",
            ),
            (
                "wrong terminal PID",
                lambda value: first_lifecycle(value)["terminal_observation"].__setitem__(
                    "pid", 9_999
                ),
                "terminal observation PID differs from the leader",
            ),
            (
                "wrong terminal idtype",
                lambda value: first_lifecycle(value)["terminal_observation"].__setitem__(
                    "idtype", 2
                ),
                "terminal observation idtype differs",
            ),
            (
                "missing WNOWAIT",
                lambda value: first_lifecycle(value)["terminal_observation"].__setitem__(
                    "options", 0x00000001 | 0x00000004
                ),
                "terminal observation options differ",
            ),
            (
                "signaled terminal code",
                lambda value: first_lifecycle(value)["terminal_observation"].__setitem__(
                    "code", 2
                ),
                "terminal observation does not prove successful exit",
            ),
            (
                "nonzero terminal status",
                lambda value: first_lifecycle(value)["terminal_observation"].__setitem__(
                    "status", 7
                ),
                "terminal observation does not prove successful exit",
            ),
            (
                "pre-reap time before terminal",
                lambda value: first_lifecycle(value)[
                    "pre_reap_quiescence"
                ].__setitem__(
                    "observed_at_monotonic_ns",
                    first_lifecycle(value)["terminal_observation"][
                        "observed_at_monotonic_ns"
                    ]
                    - 1,
                ),
                "pre-reap quiescence time is below",
            ),
            (
                "empty process group",
                lambda value: first_lifecycle(value)[
                    "pre_reap_quiescence"
                ].__setitem__("member_pids", []),
                "does not prove exact leader-only membership",
            ),
            (
                "surviving process-group member",
                lambda value: first_lifecycle(value)[
                    "pre_reap_quiescence"
                ].__setitem__(
                    "member_pids",
                    [
                        first_lifecycle(value)["leader_pid"],
                        first_lifecycle(value)["leader_pid"] + 1,
                    ],
                ),
                "does not prove exact leader-only membership",
            ),
            (
                "wrong process group",
                lambda value: first_lifecycle(value)[
                    "pre_reap_quiescence"
                ].__setitem__("process_group", 9_999),
                "does not prove exact leader-only membership",
            ),
            (
                "boolean process-group member",
                lambda value: first_lifecycle(value)[
                    "pre_reap_quiescence"
                ].__setitem__("member_pids", [True]),
                "member PID 0 must be an integer",
            ),
            (
                "false leader-only claim",
                lambda value: first_lifecycle(value)[
                    "pre_reap_quiescence"
                ].__setitem__("no_members_except_leader", False),
                "does not prove exact leader-only membership",
            ),
            (
                "reap time before quiescence",
                lambda value: first_lifecycle(value)["reap"].__setitem__(
                    "completed_at_monotonic_ns",
                    first_lifecycle(value)["pre_reap_quiescence"][
                        "observed_at_monotonic_ns"
                    ]
                    - 1,
                ),
                "reap completion time is below",
            ),
            (
                "wrong reap PID",
                lambda value: first_lifecycle(value)["reap"].__setitem__(
                    "returned_pid", 9_999
                ),
                "reap PID or options differ",
            ),
            (
                "nonzero reap options",
                lambda value: first_lifecycle(value)["reap"].__setitem__(
                    "options", 1
                ),
                "reap PID or options differ",
            ),
            (
                "boolean raw status",
                lambda value: first_lifecycle(value)["reap"].__setitem__(
                    "raw_status", True
                ),
                "reap raw status must be an integer",
            ),
            (
                "nonzero decoded exit status",
                lambda value: first_lifecycle(value)["reap"].__setitem__(
                    "raw_status", 7 << 8
                ),
                "raw status does not prove successful exit",
            ),
            (
                "unvalidated status",
                lambda value: first_lifecycle(value)["reap"].__setitem__(
                    "status_validated", False
                ),
                "does not prove successful validated reap",
            ),
            (
                "nonzero return code",
                lambda value: first_lifecycle(value)["reap"].__setitem__(
                    "returncode", 1
                ),
                "does not prove successful validated reap",
            ),
        ]
        for name, mutate, detail in mutations:
            with self.subTest(name=name):
                candidate = copy.deepcopy(self.base_record)
                mutate(candidate)
                self.assert_rejected(detail, record=candidate)

    def test_expected_roles_are_explicit_and_customizable(self) -> None:
        renamed = copy.deepcopy(self.record)
        renamed["roles"] = {
            "baseline": renamed["roles"]["control"],
            "optimized": renamed["roles"]["treatment"],
        }
        summary = self.run_record(
            renamed,
            expected_roles={"baseline": 0, "optimized": 1},
        )
        self.assertEqual(set(summary["roles"]), {"baseline", "optimized"})

        invalid_values = (
            {},
            {"control": 0},
            {"control": 0, "treatment": 0},
            {"control": 1, "treatment": 1},
            {"control": False, "treatment": 1},
            {"Control": 0, "treatment": 1},
            {"control": 0, "treatment": 1, "other": 1},
        )
        for expected_roles in invalid_values:
            with self.subTest(expected_roles=expected_roles):
                with self.assertRaises(verifier.ExactnessVerificationError):
                    self.run_record(expected_roles=expected_roles)

    def test_artifact_records_paths_hashes_and_aliases_are_rejected(self) -> None:
        mutations: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
            (
                "dataset hash",
                lambda value: value["dataset"].__setitem__("sha256", "0" * 64),
                "dataset SHA-256 differs",
            ),
            (
                "uppercase hash",
                lambda value: value["dataset"].__setitem__(
                    "sha256", DATASET_SHA256.upper()
                ),
                "is not lowercase SHA-256",
            ),
            (
                "boolean size",
                lambda value: value["dataset"].__setitem__("bytes", True),
                "bytes must be an integer",
            ),
            (
                "wrong size",
                lambda value: value["dataset"].__setitem__("bytes", DATASET_BYTES - 1),
                "recorded size differs",
            ),
            (
                "extra artifact key",
                lambda value: value["dataset"].__setitem__("extra", None),
                "exactness dataset keys differ",
            ),
            (
                "absolute path",
                lambda value: value["dataset"].__setitem__(
                    "path", str((self.evidence_dir / DATASET_NAME).resolve())
                ),
                "path must be relative",
            ),
            (
                "parent traversal",
                lambda value: value["dataset"].__setitem__(
                    "path", f"../{DATASET_NAME}"
                ),
                "path is not canonical",
            ),
            (
                "dot path",
                lambda value: value["dataset"].__setitem__("path", f"./{DATASET_NAME}"),
                "path is not canonical",
            ),
            (
                "backslash path",
                lambda value: value["dataset"].__setitem__(
                    "path", f"dir\\{DATASET_NAME}"
                ),
                "path contains a backslash",
            ),
            (
                "unfrozen path",
                lambda value: value["dataset"].__setitem__(
                    "path",
                    value["roles"]["control"]["runs"][0]["graph"]["path"],
                ),
                "file size differs",
            ),
            (
                "reused output",
                lambda value: value["roles"]["control"]["runs"][1].__setitem__(
                    "graph",
                    copy.deepcopy(value["roles"]["control"]["runs"][0]["graph"]),
                ),
                "reuses an artifact path",
            ),
        ]
        for name, mutate, detail in mutations:
            with self.subTest(name=name):
                candidate = copy.deepcopy(self.base_record)
                mutate(candidate)
                self.assert_rejected(detail, record=candidate)

    def test_symlink_artifacts_are_rejected(self) -> None:
        graph_record = self.artifact_record("control", 1, "graph")
        graph_path = self.evidence_dir / graph_record["path"]
        graph_path.unlink()
        graph_path.symlink_to(self.evidence_dir / DATASET_NAME)
        self.assert_rejected("cannot open control exactness run 1 graph")

    def test_nonregular_artifacts_are_rejected(self) -> None:
        graph_record = self.artifact_record("control", 1, "graph")
        graph_path = self.evidence_dir / graph_record["path"]
        graph_path.unlink()
        graph_path.mkdir()
        self.assert_rejected("is not a regular file")

    def test_missing_hard_linked_and_fifo_artifacts_are_rejected(self) -> None:
        graph_record = self.artifact_record("control", 1, "graph")
        graph_path = self.evidence_dir / graph_record["path"]
        graph_path.unlink()
        self.assert_rejected("cannot open control exactness run 1 graph")

        self.restore_artifact("control", 1, "graph")
        graph_path.unlink()
        os.link(self.base / graph_record["path"], graph_path)
        self.assert_rejected("does not have exactly one hard link")

        graph_path.unlink()
        os.mkfifo(graph_path)
        self.assert_rejected("is not a regular file")

    def test_evidence_directory_symlink_is_rejected(self) -> None:
        alias = self.evidence_dir.with_name("evidence-alias")
        alias.symlink_to(self.evidence_dir, target_is_directory=True)
        with self.assertRaisesRegex(
            verifier.ExactnessVerificationError,
            "evidence directory must not be a symlink",
        ):
            verifier.validate_exactness_gate(alias, self.record)

    def test_authenticated_directory_descriptor_is_borrowed_and_bound(self) -> None:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptor = os.open(self.evidence_dir, flags)
        identity = directory_identity(os.fstat(descriptor))
        try:
            summary = verifier.validate_exactness_gate(
                self.evidence_dir,
                self.record,
                directory_descriptor=descriptor,
                directory_identity=identity,
            )
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(directory_identity(os.fstat(descriptor)), identity)

            wrong_identity = list(identity)
            wrong_identity[1] += 1
            with self.assertRaisesRegex(
                verifier.ExactnessVerificationError,
                "differs from the authenticated root",
            ):
                verifier.validate_exactness_gate(
                    self.evidence_dir,
                    self.record,
                    directory_descriptor=descriptor,
                    directory_identity=tuple(wrong_identity),
                )
        finally:
            os.close(descriptor)

    def test_authenticated_directory_arguments_are_atomic(self) -> None:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptor = os.open(self.evidence_dir, flags)
        identity = directory_identity(os.fstat(descriptor))
        try:
            for arguments in (
                {"directory_descriptor": descriptor},
                {"directory_identity": identity},
            ):
                with self.subTest(arguments=sorted(arguments)):
                    with self.assertRaisesRegex(
                        verifier.ExactnessVerificationError,
                        "must be provided together",
                    ):
                        verifier.validate_exactness_gate(
                            self.evidence_dir,
                            self.record,
                            **arguments,
                        )
        finally:
            os.close(descriptor)

    def test_authenticated_reader_rejects_root_path_replacement(self) -> None:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptor = os.open(self.evidence_dir, flags)
        identity = directory_identity(os.fstat(descriptor))
        displaced = self.evidence_dir.with_name("evidence-displaced")
        original_read = verifier._EvidenceReader.read_artifact
        replaced = False

        def replace_then_read(
            reader: verifier._EvidenceReader,
            *args: Any,
            **kwargs: Any,
        ) -> tuple[bytes, dict[str, Any]]:
            nonlocal replaced
            if not replaced:
                replaced = True
                self.evidence_dir.rename(displaced)
                shutil.copytree(
                    self.base,
                    self.evidence_dir,
                    copy_function=shutil.copy2,
                )
            return original_read(reader, *args, **kwargs)

        try:
            with mock.patch.object(
                verifier._EvidenceReader,
                "read_artifact",
                new=replace_then_read,
            ):
                with self.assertRaisesRegex(
                    verifier.ExactnessVerificationError,
                    "evidence directory (descriptor|path) changed during verification",
                ):
                    verifier.validate_exactness_gate(
                        self.evidence_dir,
                        self.record,
                        directory_descriptor=descriptor,
                        directory_identity=identity,
                    )
            self.assertTrue(replaced)
            descriptor_status = os.fstat(descriptor)
            self.assertEqual(
                (descriptor_status.st_dev, descriptor_status.st_ino),
                identity[:2],
            )
        finally:
            os.close(descriptor)

    def test_dataset_bytes_are_frozen_and_cross_checked(self) -> None:
        candidate = bytearray(self.dataset)
        candidate[0] ^= 1
        self.replace_dataset(bytes(candidate))
        self.assert_rejected("dataset SHA-256 is not frozen")

    def test_stdout_must_be_exact_canonical_one_line_json(self) -> None:
        parsed = json.loads(self.stdout)
        mutations: list[tuple[str, bytes, str]] = [
            ("missing newline", self.stdout[:-1], "exactly one newline"),
            ("extra newline", self.stdout + b"\n", "exactly one newline"),
            (
                "pretty printed",
                json.dumps(parsed, indent=2, sort_keys=True).encode("ascii") + b"\n",
                "exactly one newline",
            ),
            (
                "duplicate key",
                b'{"status":"PASS",' + self.stdout[1:],
                "duplicate key",
            ),
            (
                "key reorder",
                json.dumps(
                    dict(reversed(list(parsed.items()))),
                    separators=(",", ":"),
                ).encode("ascii")
                + b"\n",
                "differs from canonical output",
            ),
            (
                "leading whitespace",
                b" " + self.stdout,
                "differs from canonical output",
            ),
            (
                "CRLF",
                self.stdout[:-1] + b"\r\n",
                "differs from canonical output",
            ),
            (
                "BOM",
                b"\xef\xbb\xbf" + self.stdout,
                "not canonical ASCII JSON",
            ),
            (
                "non-finite constant",
                self.stdout.replace(b'"workers":1', b'"workers":NaN'),
                "unsupported constant",
            ),
            ("non-ASCII", b"\xff\n", "not canonical ASCII JSON"),
        ]
        wrong_workers = copy.deepcopy(parsed)
        wrong_workers["configuration"]["workers"] = 2
        mutations.append(
            (
                "wrong constant",
                json.dumps(wrong_workers, sort_keys=True, separators=(",", ":")).encode(
                    "ascii"
                )
                + b"\n",
                "differs from canonical output",
            )
        )
        floating_workers = copy.deepcopy(parsed)
        floating_workers["configuration"]["workers"] = 1.0
        mutations.append(
            (
                "floating number",
                json.dumps(
                    floating_workers, sort_keys=True, separators=(",", ":")
                ).encode("ascii")
                + b"\n",
                "unsupported number",
            )
        )
        extra_key = copy.deepcopy(parsed)
        extra_key["extra"] = None
        mutations.append(
            (
                "extra key",
                json.dumps(extra_key, sort_keys=True, separators=(",", ":")).encode(
                    "ascii"
                )
                + b"\n",
                "differs from canonical output",
            )
        )
        wrong_hash = copy.deepcopy(parsed)
        wrong_hash["graph"]["sha256"] = "0" * 64
        mutations.append(
            (
                "wrong graph hash",
                json.dumps(wrong_hash, sort_keys=True, separators=(",", ":")).encode(
                    "ascii"
                )
                + b"\n",
                "differs from canonical output",
            )
        )

        for name, content, detail in mutations:
            with self.subTest(name=name):
                self.replace_artifact("control", 1, "stdout", content)
                self.assert_rejected(detail)
                self.restore_artifact("control", 1, "stdout")

    def test_witness_format_and_role_bits_are_exact(self) -> None:
        mutations: list[tuple[str, bytes, str]] = [
            (
                "magic",
                b"BROKEN!!" + self.control_witness[8:],
                "witness magic differs",
            ),
            (
                "schema",
                self.patched(self.control_witness, 8, "<I", 2),
                "schema version differs",
            ),
            (
                "declared size",
                self.patched(self.control_witness, 12, "<I", 35),
                "declared size differs",
            ),
            (
                "nonboolean flag",
                self.patched(self.control_witness, 16, "<I", 2),
                "non-boolean flag",
            ),
            (
                "control one bit",
                self.patched(self.control_witness, 16, "<I", 1),
                "witness differs from its role",
            ),
            ("truncated", self.control_witness[:-1], "recorded size differs"),
            ("trailing", self.control_witness + b"\0", "bytes exceeds 36"),
        ]
        for name, content, detail in mutations:
            with self.subTest(name=name):
                self.replace_artifact("control", 1, "witness", content)
                self.assert_rejected(detail)
                self.restore_artifact("control", 1, "witness")

        for flag in range(5):
            with self.subTest(role="control", flag=flag):
                content = self.patched(
                    self.control_witness,
                    16 + flag * 4,
                    "<I",
                    1,
                )
                self.replace_artifact("control", 1, "witness", content)
                self.assert_rejected("witness differs from its role")
                self.restore_artifact("control", 1, "witness")
            with self.subTest(role="treatment", flag=flag):
                content = self.patched(
                    self.treatment_witness,
                    16 + flag * 4,
                    "<I",
                    0,
                )
                self.replace_artifact("treatment", 1, "witness", content)
                self.assert_rejected("witness differs from its role")
                self.restore_artifact("treatment", 1, "witness")

    def test_stderr_must_be_empty_and_returncode_zero(self) -> None:
        self.replace_artifact("control", 1, "stderr", b"warning\n")
        self.assert_rejected("recorded size differs from 0")

    def test_graph_header_fields_are_all_bound(self) -> None:
        mutations: list[tuple[str, Callable[[bytearray], None]]] = [
            ("magic", lambda value: value.__setitem__(slice(0, 8), b"BROKEN!!")),
            ("schema", lambda value: struct.pack_into("<I", value, 8, 2)),
            ("header bytes", lambda value: struct.pack_into("<I", value, 12, 180)),
            ("vector count", lambda value: struct.pack_into("<Q", value, 16, 1)),
            ("dimension", lambda value: struct.pack_into("<Q", value, 24, 1)),
            ("M", lambda value: struct.pack_into("<Q", value, 32, 31)),
            ("ef construction", lambda value: struct.pack_into("<Q", value, 40, 199)),
            ("chunk size", lambda value: struct.pack_into("<Q", value, 48, 4096)),
            ("max chunks", lambda value: struct.pack_into("<Q", value, 56, 1)),
            ("workers", lambda value: struct.pack_into("<Q", value, 64, 2)),
            ("build start", lambda value: struct.pack_into("<Q", value, 72, 1)),
            ("build end", lambda value: struct.pack_into("<Q", value, 80, 1)),
            ("tasks", lambda value: struct.pack_into("<Q", value, 88, 2)),
            ("index count", lambda value: struct.pack_into("<Q", value, 96, 1)),
            ("mmax0", lambda value: struct.pack_into("<Q", value, 104, 63)),
            ("mmax", lambda value: struct.pack_into("<Q", value, 112, 31)),
            ("max level range", lambda value: struct.pack_into("<i", value, 120, 65)),
            ("entry range", lambda value: struct.pack_into("<i", value, 124, -1)),
            ("vertex records", lambda value: struct.pack_into("<Q", value, 128, 1)),
            ("label encoding", lambda value: struct.pack_into("<Q", value, 136, 0)),
            ("dataset digest", lambda value: value.__setitem__(144, value[144] ^ 1)),
            ("body offset", lambda value: struct.pack_into("<Q", value, 176, 0)),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                candidate = bytearray(self.graph)
                mutate(candidate)
                self.replace_artifact("control", 1, "graph", bytes(candidate))
                self.assert_rejected("header differs from the exact format")
                self.restore_artifact("control", 1, "graph")

    def test_graph_body_rejects_structural_and_ordering_mutations(self) -> None:
        vertex0 = self.graph_layout["vertex"][0]
        layer0 = self.graph_layout["layer"][0, 0]
        first0 = self.graph_layout["neighbor"][0, 0, 0]
        second0 = self.graph_layout["neighbor"][0, 0, 1]
        first_upper = self.graph_layout["neighbor"][0, 1, 0]
        mutations: list[tuple[str, Callable[[bytearray], None], str]] = [
            (
                "ordinal",
                lambda value: struct.pack_into("<I", value, vertex0, 1),
                "ordinal 1 is out of sequence",
            ),
            (
                "label",
                lambda value: struct.pack_into("<i", value, vertex0 + 4, 1),
                "label differs from its ordinal",
            ),
            (
                "level",
                lambda value: struct.pack_into("<i", value, vertex0 + 8, 2),
                "has an invalid level",
            ),
            (
                "layer count",
                lambda value: struct.pack_into("<I", value, vertex0 + 12, 1),
                "invalid layer count",
            ),
            (
                "layer ordinal",
                lambda value: struct.pack_into("<I", value, layer0, 1),
                "layer ordinal differs",
            ),
            (
                "degree",
                lambda value: struct.pack_into("<I", value, layer0 + 4, 65),
                "degree exceeds capacity",
            ),
            (
                "out of range",
                lambda value: struct.pack_into("<i", value, first0, VECTOR_COUNT),
                "out-of-range neighbor",
            ),
            (
                "self edge",
                lambda value: struct.pack_into("<i", value, first0, 0),
                "self edge",
            ),
            (
                "duplicate",
                lambda value: struct.pack_into(
                    "<i", value, second0, (0 - 1) % VECTOR_COUNT
                ),
                "duplicate neighbors",
            ),
            (
                "upper target lower",
                lambda value: struct.pack_into("<i", value, first_upper, 1),
                "upper-layer edge targets a lower-level vertex",
            ),
        ]
        for name, mutate, detail in mutations:
            with self.subTest(name=name):
                candidate = bytearray(self.graph)
                mutate(candidate)
                self.replace_artifact("control", 1, "graph", bytes(candidate))
                self.assert_rejected(detail)
                self.restore_artifact("control", 1, "graph")

        self.replace_artifact("control", 1, "graph", self.graph[:-1])
        self.assert_rejected("is truncated")
        self.restore_artifact("control", 1, "graph")
        self.replace_artifact("control", 1, "graph", self.graph + b"\0")
        self.assert_rejected("contains trailing bytes")

    def test_graph_requires_entry_at_observed_maximum_and_reachability(self) -> None:
        candidate = bytearray(self.graph)
        struct.pack_into("<i", candidate, 124, 1)
        self.replace_artifact("control", 1, "graph", bytes(candidate))
        self.assert_rejected("entry point is not on the maximum level")
        self.restore_artifact("control", 1, "graph")

        candidate = bytearray(self.graph)
        struct.pack_into("<i", candidate, 120, 2)
        self.replace_artifact("control", 1, "graph", bytes(candidate))
        self.assert_rejected("maximum level differs from its vertices")
        self.restore_artifact("control", 1, "graph")

        candidate = bytearray(self.graph)
        for vertex in range(VECTOR_COUNT):
            for index, neighbor in enumerate(
                ((vertex - 2) % VECTOR_COUNT, (vertex + 2) % VECTOR_COUNT)
            ):
                struct.pack_into(
                    "<i",
                    candidate,
                    self.graph_layout["neighbor"][vertex, 0, index],
                    neighbor,
                )
        self.replace_artifact("control", 1, "graph", bytes(candidate))
        self.assert_rejected("not reachable from its entry point")

    def test_save_to_ptr_metadata_fields_are_all_bound(self) -> None:
        mutations: list[tuple[str, int, str, int]] = [
            ("M", 0, "<Q", 31),
            ("ef construction", 8, "<Q", 199),
            ("vector count", 16, "<Q", 1),
            ("dimension", 24, "<Q", 1),
            ("mmax0", 32, "<Q", 63),
            ("mmax", 40, "<Q", 31),
            ("max level", 48, "<i", 65),
            ("entry point", 52, "<i", -1),
        ]
        for name, offset, layout, value in mutations:
            with self.subTest(name=name):
                content = self.patched(self.pointer_image, offset, layout, value)
                self.replace_artifact("control", 1, "save_to_ptr", content)
                self.assert_rejected("metadata differs from the exact plain-L2 format")
                self.restore_artifact("control", 1, "save_to_ptr")

    def test_save_to_ptr_sections_reject_all_structural_mutations(self) -> None:
        record0 = self.save_layout["level_zero"][0]
        record2 = self.save_layout["level_zero"][2]
        upper0 = self.save_layout["upper"][0, 1]
        mutations: list[tuple[str, Callable[[bytearray], None], str]] = [
            (
                "vector byte",
                lambda value: value.__setitem__(
                    self.save_layout["vectors"],
                    value[self.save_layout["vectors"]] ^ 1,
                ),
                "vectors differ from the frozen dataset",
            ),
            (
                "layer sum",
                lambda value: struct.pack_into(
                    "<Q",
                    value,
                    self.save_layout["layer_sum"],
                    VECTOR_COUNT,
                ),
                "upper-layer count differs from vertex levels",
            ),
            (
                "level",
                lambda value: struct.pack_into("<i", value, record0, 2),
                "has an invalid level",
            ),
            (
                "prefix padding",
                lambda value: value.__setitem__(record0 + 4, 1),
                "nonzero level-zero prefix padding",
            ),
            (
                "pointer offset",
                lambda value: struct.pack_into("<Q", value, record2 + 8, 0),
                "invalid upper-layer offset",
            ),
            (
                "level-zero degree",
                lambda value: struct.pack_into("<i", value, record0 + 16, 65),
                "level-zero degree exceeds capacity",
            ),
            (
                "level-zero out of range",
                lambda value: struct.pack_into("<i", value, record0 + 20, VECTOR_COUNT),
                "out-of-range neighbor",
            ),
            (
                "level-zero self edge",
                lambda value: struct.pack_into("<i", value, record0 + 20, 0),
                "self edge",
            ),
            (
                "level-zero duplicate",
                lambda value: struct.pack_into(
                    "<i", value, record0 + 24, (0 - 1) % VECTOR_COUNT
                ),
                "duplicate neighbors",
            ),
            (
                "tail padding",
                lambda value: value.__setitem__(record0 + 276, 1),
                "nonzero level-zero tail padding",
            ),
            (
                "upper degree",
                lambda value: struct.pack_into("<i", value, upper0, 33),
                "upper-layer degree exceeds capacity",
            ),
            (
                "upper out of range",
                lambda value: struct.pack_into("<i", value, upper0 + 4, VECTOR_COUNT),
                "out-of-range neighbor",
            ),
            (
                "upper self edge",
                lambda value: struct.pack_into("<i", value, upper0 + 4, 0),
                "self edge",
            ),
            (
                "upper duplicate",
                lambda value: struct.pack_into(
                    "<i", value, upper0 + 8, (0 - 2) % VECTOR_COUNT
                ),
                "duplicate neighbors",
            ),
            (
                "upper target lower",
                lambda value: struct.pack_into("<i", value, upper0 + 4, 1),
                "upper-layer edge targets a lower-level vertex",
            ),
            (
                "label",
                lambda value: struct.pack_into(
                    "<i", value, self.save_layout["labels"], 1
                ),
                "label 0 differs from its ordinal",
            ),
        ]
        for name, mutate, detail in mutations:
            with self.subTest(name=name):
                candidate = bytearray(self.pointer_image)
                mutate(candidate)
                self.replace_artifact("control", 1, "save_to_ptr", bytes(candidate))
                self.assert_rejected(detail)
                self.restore_artifact("control", 1, "save_to_ptr")

        self.replace_artifact("control", 1, "save_to_ptr", self.pointer_image[:-1])
        self.assert_rejected("is truncated")
        self.restore_artifact("control", 1, "save_to_ptr")
        self.replace_artifact("control", 1, "save_to_ptr", self.pointer_image + b"\0")
        self.assert_rejected("contains trailing bytes")

    def test_graph_and_save_to_ptr_ordered_adjacency_are_cross_checked(self) -> None:
        candidate = bytearray(self.pointer_image)
        record = self.save_layout["level_zero"][0]
        candidate[record + 20 : record + 24], candidate[record + 24 : record + 28] = (
            candidate[record + 24 : record + 28],
            candidate[record + 20 : record + 24],
        )
        self.replace_artifact("control", 1, "save_to_ptr", bytes(candidate))
        self.assert_rejected("graph and SaveToPtr topology differ")

    def test_within_role_determinism_requires_exact_bytes(self) -> None:
        self.install_consistent_alternate("control", 2)
        self.assert_rejected("control graph is not deterministic across two runs")

    def test_cross_role_equality_requires_exact_ordered_images(self) -> None:
        self.install_consistent_alternate("treatment", 1)
        self.install_consistent_alternate("treatment", 2)
        self.assert_rejected("cross-role graph bytes differ")

    def test_cross_role_save_equality_covers_inactive_capacity_bytes(self) -> None:
        for ordinal in (1, 2):
            candidate = bytearray(self.pointer_image)
            record = self.save_layout["level_zero"][0]
            struct.pack_into("<i", candidate, record + 28, 7)
            pointer_image = bytes(candidate)
            self.replace_artifact(
                "treatment",
                ordinal,
                "save_to_ptr",
                pointer_image,
            )
            self.replace_artifact(
                "treatment",
                ordinal,
                "stdout",
                stdout_json(self.graph, pointer_image),
            )
        self.assert_rejected("cross-role save_to_ptr bytes differ")

    def test_within_role_witness_determinism_is_not_inferred_from_role(self) -> None:
        candidate = bytearray(self.control_witness)
        struct.pack_into("<I", candidate, 16, 1)
        self.replace_artifact("control", 2, "witness", bytes(candidate))
        self.assert_rejected("witness differs from its role")

    def test_exact_eof_is_enforced_even_when_recorded_hashes_match(self) -> None:
        cases = (
            ("graph", self.graph + b"x", "contains trailing bytes"),
            (
                "save_to_ptr",
                self.pointer_image + b"x",
                "contains trailing bytes",
            ),
            ("witness", self.control_witness + b"x", "bytes exceeds 36"),
            ("stdout", self.stdout + b"x", "exactly one newline"),
        )
        for name, content, detail in cases:
            with self.subTest(name=name):
                self.replace_artifact("control", 1, name, content)
                self.assert_rejected(detail)
                self.restore_artifact("control", 1, name)


if __name__ == "__main__":
    unittest.main()
