from __future__ import annotations

import contextlib
import hashlib
import inspect
import struct
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tools.apple_silicon import native_hnsw_d0 as native
from tools.apple_silicon import verify_native_hnsw_d0 as verifier


MAGIC = b"IFD0AUD4"
SCHEMA = 4
VECTORS = 4
DIMENSIONS = 2
GRAPH_M = 2
EF_CONSTRUCTION = 4
QUERY_COUNT = 2
RECALL_POINTS = ((2, 2), (3, 3))
QUERY_K = 2
QUERY_EF_SEARCH = 2
LATENCY_SAMPLES = (101, 102, 103, 104)
THROUGHPUT_OPERATIONS = 4
THROUGHPUT_WALL_NS = 100_000

CAMPAIGN_NONCE = "11" * 32
BINARY_SHA256 = "22" * 32
DATASET_SHA256 = "33" * 32
QUERIES_SHA256 = "44" * 32
TRUTH_SHA256 = "55" * 32
SCHEDULE_SEQUENCE = 7

BINDING_STDOUT_FIELDS = frozenset(
    {
        "campaign_nonce",
        "schedule_sequence",
        "role_id",
        "binary_sha256",
        "dataset_sha256",
    }
)

PREFIX_OFFSETS = {
    "magic": (0, 8),
    "schema": (8, 12),
    "engine": (12, 16),
    "campaign_nonce": (16, 48),
    "schedule_sequence": (48, 56),
    "role_id": (56, 60),
    "execution_witness": (60, 64),
    "binary_sha256": (64, 96),
    "dataset_sha256": (96, 128),
}
QUERY_SECTION_OFFSET = 192
QUERY_TIMED_CORPUS_OFFSET = QUERY_SECTION_OFFSET + 56
QUERY_LATENCY_VALIDATED_OPERATIONS_OFFSET = QUERY_SECTION_OFFSET + 96
QUERY_LATENCY_VALIDATED_CHECKSUM_OFFSET = QUERY_SECTION_OFFSET + 104
QUERY_THROUGHPUT_VALIDATED_CHECKSUM_OFFSET = QUERY_SECTION_OFFSET + 136
QUERY_RESULT_CHECKSUM_OFFSET = QUERY_SECTION_OFFSET + 144
QUERY_PER_QUERY_CHECKSUMS_OFFSET = QUERY_SECTION_OFFSET + 160


def _double_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _truth(module: Any) -> dict[str, Any]:
    distances = (
        (0.0, 1.0, 4.0, 9.0),
        (9.0, 4.0, 1.0, 0.0),
    )
    common = {
        "queries_sha256": QUERIES_SHA256,
        "truth_sha256": TRUTH_SHA256,
        "tie_counts_at_10": [2, 2],
        "tie_counts_at_100": [3, 3],
    }
    if module is native:
        return {
            **common,
            "_distances": distances,
            "_cutoffs_at_10": [100.0, 100.0],
            "_cutoffs_at_100": [100.0, 100.0],
        }
    return {
        **common,
        "distances": distances,
        "cutoffs_at_10": [100.0, 100.0],
        "cutoffs_at_100": [100.0, 100.0],
    }


def _top_labels(query_index: int, k: int) -> tuple[int, ...]:
    return ((0, 1, 2, 3), (3, 2, 1, 0))[query_index][:k]


def _query_checksums(module: Any) -> list[int]:
    checksums: list[int] = []
    distances = _truth(verifier)["distances"]
    for query_index in range(QUERY_COUNT):
        labels = _top_labels(query_index, QUERY_K)
        distance_bits = [
            _float_bits(distances[query_index][label]) for label in labels
        ]
        checksums.append(
            module.query_result_checksum(query_index, labels, distance_bits)
        )
    return checksums


def _query_result_checksum(module: Any) -> int:
    return module.query_benchmark_checksum(_query_checksums(module))


def _execution_witness(module: Any, role_id: int) -> dict[str, int]:
    enabled = int(role_id in (2, 3))
    return {name: enabled for name in module.EXECUTION_WITNESS_FIELDS}


def _execution_witness_bits(module: Any, role_id: int) -> int:
    witness = _execution_witness(module, role_id)
    return sum(
        witness[name] << index
        for index, name in enumerate(module.EXECUTION_WITNESS_FIELDS)
    )


def _binding_prefix(
    module: Any,
    *,
    engine: str,
    role_id: int,
    campaign_nonce: str = CAMPAIGN_NONCE,
    schedule_sequence: int = SCHEDULE_SEQUENCE,
    binary_sha256: str = BINARY_SHA256,
    dataset_sha256: str = DATASET_SHA256,
) -> bytes:
    return struct.pack(
        "<8sII32sQII32s32s",
        MAGIC,
        SCHEMA,
        {"infinity": 1, "faiss": 2}[engine],
        bytes.fromhex(campaign_nonce),
        schedule_sequence,
        role_id,
        _execution_witness_bits(module, role_id),
        bytes.fromhex(binary_sha256),
        bytes.fromhex(dataset_sha256),
    )


def _body(
    module: Any,
    *,
    engine: str,
    role_id: int,
) -> tuple[bytes, dict[str, str]]:
    result_checksum = _query_result_checksum(module)
    per_query_checksums = _query_checksums(module)
    body = bytearray(
        struct.pack(
            "<QQIIiiIIQQII",
            VECTORS,
            DIMENSIONS,
            GRAPH_M,
            EF_CONSTRUCTION,
            0,
            0,
            2 * GRAPH_M,
            GRAPH_M,
            QUERY_COUNT,
            module.HELDOUT_QUERY_SEED,
            len(RECALL_POINTS),
            0,
        )
    )
    body.extend(
        struct.pack(
            "<IIIIIIIIIIQQ",
            module.QUERY_BENCHMARK_SCHEMA_VERSION,
            QUERY_COUNT,
            QUERY_K,
            QUERY_EF_SEARCH,
            1,
            2,
            module.QUERY_LATENCY_CONCURRENCY,
            module.QUERY_THROUGHPUT_CONCURRENCY,
            module.QUERY_PERCENTILE_METHOD_ID,
            module.QUERY_TRANSACTION_DEFINITION_ID,
            _double_bits(module.QUERY_RECALL_FLOOR),
            _double_bits(module.QUERY_MAXIMUM_ABSOLUTE_RECALL_GAP),
        )
    )
    body.extend(bytes.fromhex(QUERIES_SHA256))
    body.extend(
        struct.pack(
            "<QQQQQQQQQ",
            len(LATENCY_SAMPLES),
            len(LATENCY_SAMPLES),
            result_checksum,
            THROUGHPUT_OPERATIONS,
            THROUGHPUT_OPERATIONS,
            THROUGHPUT_WALL_NS,
            result_checksum,
            result_checksum,
            QUERY_COUNT,
        )
    )
    body.extend(struct.pack("<" + "Q" * QUERY_COUNT, *per_query_checksums))
    body.extend(struct.pack("<4Q", *LATENCY_SAMPLES))

    levels_digest = hashlib.sha256()
    levels_digest.update(b"infinity-hnsw-levels-v1")
    levels_digest.update(struct.pack("<Q", VECTORS))
    graph_digest = hashlib.sha256()
    graph_digest.update(b"infinity-hnsw-graph-v1")
    graph_digest.update(
        struct.pack("<QIIQQ", VECTORS, 0, 0, 2 * GRAPH_M, GRAPH_M)
    )
    for vertex in range(VECTORS):
        neighbor = (vertex + 1) % VECTORS
        body.extend(
            struct.pack("<iqIIi", 0, vertex, 2 * GRAPH_M, 1, neighbor)
        )
        levels_digest.update(struct.pack("<II", vertex, 0))
        graph_digest.update(struct.pack("<IIQ", vertex, 0, vertex))
        graph_digest.update(struct.pack("<IQI", 0, 1, neighbor))

    fields = {
        "audit_sidecar_schema": str(SCHEMA),
        "campaign_nonce": CAMPAIGN_NONCE,
        "schedule_sequence": str(SCHEDULE_SEQUENCE),
        "binary_sha256": BINARY_SHA256,
        "dataset_sha256": DATASET_SHA256,
        "query_benchmark_schema": str(
            module.QUERY_BENCHMARK_SCHEMA_VERSION
        ),
        "query_unique_queries": str(QUERY_COUNT),
        "query_k": str(QUERY_K),
        "query_ef_search": str(QUERY_EF_SEARCH),
        "query_warmup_passes": "1",
        "query_measured_passes": "2",
        "query_latency_concurrency": str(
            module.QUERY_LATENCY_CONCURRENCY
        ),
        "query_throughput_concurrency": str(
            module.QUERY_THROUGHPUT_CONCURRENCY
        ),
        "query_timed_corpus_sha256": QUERIES_SHA256,
        f"{engine}_query_latency_sample_count": str(len(LATENCY_SAMPLES)),
        f"{engine}_query_latency_validated_operations": str(
            len(LATENCY_SAMPLES)
        ),
        f"{engine}_query_latency_validated_result_checksum": str(
            result_checksum
        ),
        f"{engine}_query_throughput_operations": str(
            THROUGHPUT_OPERATIONS
        ),
        f"{engine}_query_throughput_validated_operations": str(
            THROUGHPUT_OPERATIONS
        ),
        f"{engine}_query_throughput_wall_ns": str(THROUGHPUT_WALL_NS),
        f"{engine}_query_throughput_validated_result_checksum": str(
            result_checksum
        ),
        f"{engine}_query_result_checksum": str(result_checksum),
        f"{engine}_query_per_query_checksum_count": str(QUERY_COUNT),
        f"{engine}_query_per_query_checksums": ",".join(
            str(value) for value in per_query_checksums
        ),
        f"{engine}_graph_vertex_count": str(VECTORS),
        f"{engine}_graph_reachable_count": str(VECTORS),
        f"{engine}_graph_directed_edges": str(VECTORS),
        f"{engine}_graph_level0_directed_edges": str(VECTORS),
        f"{engine}_graph_max_level": "0",
        f"{engine}_graph_entry_point": "0",
        f"{engine}_graph_level0_capacity": str(2 * GRAPH_M),
        f"{engine}_graph_upper_capacity": str(GRAPH_M),
        f"{engine}_graph_levels_sha256": levels_digest.hexdigest(),
        f"{engine}_graph_sha256": graph_digest.hexdigest(),
        f"{engine}_graph_level_histogram": f"0:{VECTORS}",
        f"{engine}_graph_degree_histograms": (
            "L0:" + ",".join(("0", str(VECTORS), "0", "0", "0"))
        ),
        "heldout_queries_sha256": QUERIES_SHA256,
        "heldout_truth_sha256": TRUTH_SHA256,
    }
    fields.update(
        {
            f"{engine}_{name}": str(value)
            for name, value in _execution_witness(module, role_id).items()
        }
    )

    distances = _truth(verifier)["distances"]
    for k, ef in RECALL_POINTS:
        body.extend(struct.pack("<IIQ", k, ef, QUERY_COUNT * k))
        returned_ids = []
        distance_digest = hashlib.sha256()
        for query_index in range(QUERY_COUNT):
            for label in _top_labels(query_index, k):
                encoded_distance = struct.pack(
                    "<f", distances[query_index][label]
                )
                body.extend(
                    struct.pack(
                        "<qI",
                        label,
                        struct.unpack("<I", encoded_distance)[0],
                    )
                )
                returned_ids.append(str(label))
                distance_digest.update(encoded_distance)
        fields[f"{engine}_recall_at_{k}_ef_{ef}"] = "1.0"
        fields[f"{engine}_returned_ids_ef_{ef}_k_{k}"] = ",".join(
            returned_ids
        )
        fields[
            f"{engine}_returned_distances_sha256_ef_{ef}_k_{k}"
        ] = distance_digest.hexdigest()
    return bytes(body), fields


def _fixture(
    module: Any,
    *,
    engine: str,
    role_id: int,
    schedule_sequence: int = SCHEDULE_SEQUENCE,
) -> tuple[bytes, dict[str, str]]:
    body, fields = _body(module, engine=engine, role_id=role_id)
    sidecar = _binding_prefix(
        module,
        engine=engine,
        role_id=role_id,
        schedule_sequence=schedule_sequence,
    ) + body
    fields.update(
        {
            "role_id": str(role_id),
            "schedule_sequence": str(schedule_sequence),
            "audit_sidecar_bytes": str(len(sidecar)),
            "audit_sidecar_sha256": hashlib.sha256(sidecar).hexdigest(),
        }
    )
    return sidecar, fields


def _refresh_sidecar_fields(
    fields: dict[str, str], sidecar: bytes
) -> dict[str, str]:
    updated = dict(fields)
    updated["audit_sidecar_bytes"] = str(len(sidecar))
    updated["audit_sidecar_sha256"] = hashlib.sha256(sidecar).hexdigest()
    return updated


def _mutate(
    sidecar: bytes,
    field: str,
    replacement: bytes,
) -> bytes:
    start, end = PREFIX_OFFSETS[field]
    if len(replacement) != end - start:
        raise AssertionError(f"Invalid {field} replacement length")
    result = bytearray(sidecar)
    result[start:end] = replacement
    return bytes(result)


@contextlib.contextmanager
def _small_protocol(module: Any):
    values = {
        "VECTORS": VECTORS,
        "DIMENSIONS": DIMENSIONS,
        "M": GRAPH_M,
        "EF_CONSTRUCTION": EF_CONSTRUCTION,
        "HELDOUT_QUERY_COUNT": QUERY_COUNT,
        "RECALL_POINTS": RECALL_POINTS,
        "QUERY_K": QUERY_K,
        "QUERY_EF_SEARCH": QUERY_EF_SEARCH,
        "QUERY_WARMUP_PASSES": 1,
        "QUERY_MEASURED_PASSES": 2,
        "QUERY_LATENCY_SAMPLE_COUNT": len(LATENCY_SAMPLES),
        "QUERY_THROUGHPUT_OPERATIONS": THROUGHPUT_OPERATIONS,
    }
    with contextlib.ExitStack() as stack:
        for name, value in values.items():
            stack.enter_context(patch.object(module, name, value))
        yield


def _expected_kwargs(role_id: int) -> dict[str, Any]:
    return {
        "expected_campaign_nonce": CAMPAIGN_NONCE,
        "expected_schedule_sequence": SCHEDULE_SEQUENCE,
        "expected_role_id": role_id,
        "expected_binary_sha256": BINARY_SHA256,
        "expected_dataset_sha256": DATASET_SHA256,
    }


def _parse(
    module: Any,
    sidecar: bytes,
    fields: dict[str, str],
    *,
    engine: str,
    role_id: int,
    expected_overrides: dict[str, Any] | None = None,
) -> Any:
    expected = _expected_kwargs(role_id)
    if expected_overrides:
        expected.update(expected_overrides)
    if module is native:
        return module.parse_audit_sidecar(
            Path("fixture.audit-v4.bin"),
            engine=engine,
            fields=fields,
            dataset={"sha256": DATASET_SHA256},
            truth=_truth(module),
            data=sidecar,
            **expected,
        )
    return module.parse_audit_sidecar(
        sidecar,
        engine=engine,
        fields=fields,
        truth=_truth(module),
        context="fixture audit sidecar",
        **expected,
    )


def _failure_type(module: Any) -> type[RuntimeError]:
    return native.D0Failure if module is native else verifier.VerificationError


class AuditSidecarV4ContractTests(unittest.TestCase):
    modules = (native, verifier)

    def test_v4_constants_and_required_stdout_schema(self) -> None:
        for module in self.modules:
            with self.subTest(module=module.__name__):
                self.assertEqual(module.AUDIT_SIDECAR_MAGIC, MAGIC)
                self.assertEqual(module.AUDIT_SIDECAR_SCHEMA_VERSION, SCHEMA)
                self.assertEqual(module.AUDIT_SIDECAR_SUFFIX, ".audit-v4.bin")
                self.assertTrue(
                    BINDING_STDOUT_FIELDS.issubset(module.AUDIT_DRIVER_FIELDS)
                )
                for engine in ("infinity", "faiss"):
                    if module is native:
                        names = module.expected_driver_fields(engine)
                    else:
                        names = module.expected_driver_field_names(engine)
                    self.assertTrue(BINDING_STDOUT_FIELDS.issubset(names))

    def test_parsers_expose_explicit_expected_binding_keywords(self) -> None:
        required = set(_expected_kwargs(1))
        for module in self.modules:
            with self.subTest(module=module.__name__):
                parameters = inspect.signature(
                    module.parse_audit_sidecar
                ).parameters
                self.assertTrue(required.issubset(parameters))
                self.assertIn("fields", parameters)

    def test_valid_control_treatment_and_standalone_roles(self) -> None:
        cases = (
            ("causal-control", "infinity", 1),
            ("causal-treatment", "infinity", 2),
            ("standalone-infinity", "infinity", 3),
            ("standalone-faiss", "faiss", 4),
        )
        for module in self.modules:
            with _small_protocol(module):
                for name, engine, role_id in cases:
                    with self.subTest(
                        module=module.__name__,
                        case=name,
                    ):
                        sidecar, fields = _fixture(
                            module,
                            engine=engine,
                            role_id=role_id,
                        )
                        parsed = _parse(
                            module,
                            sidecar,
                            fields,
                            engine=engine,
                            role_id=role_id,
                        )
                        audit = parsed[0] if module is verifier else parsed
                        self.assertEqual(audit["schema_version"], SCHEMA)
                        self.assertEqual(
                            audit["execution_witness"],
                            _execution_witness(module, role_id),
                        )
                        self.assertEqual(
                            audit["query"]["timed_query_corpus_sha256"],
                            QUERIES_SHA256,
                        )
                        self.assertEqual(
                            audit["query"]["latency_validated_operations"],
                            len(LATENCY_SAMPLES),
                        )
                        self.assertEqual(
                            audit["query"]["throughput_validated_operations"],
                            THROUGHPUT_OPERATIONS,
                        )
                        self.assertEqual(
                            audit["query"]["per_query_checksums"],
                            _query_checksums(module),
                        )

    def test_each_prefix_field_mutation_or_downgrade_is_rejected(self) -> None:
        mutations = {
            "magic": b"IFD0AUD3",
            "schema": struct.pack("<I", 3),
            "engine": struct.pack("<I", 2),
            "campaign_nonce": bytes.fromhex("aa" * 32),
            "schedule_sequence": struct.pack("<Q", SCHEDULE_SEQUENCE + 1),
            "role_id": struct.pack("<I", 2),
            "execution_witness": struct.pack("<I", 1 << 10),
            "binary_sha256": bytes.fromhex("bb" * 32),
            "dataset_sha256": bytes.fromhex("cc" * 32),
        }
        for module in self.modules:
            with _small_protocol(module):
                sidecar, fields = _fixture(
                    module,
                    engine="infinity",
                    role_id=1,
                )
                for field, replacement in mutations.items():
                    with self.subTest(
                        module=module.__name__,
                        field=field,
                    ):
                        mutated = _mutate(sidecar, field, replacement)
                        mutated_fields = _refresh_sidecar_fields(
                            fields, mutated
                        )
                        with self.assertRaises(_failure_type(module)):
                            _parse(
                                module,
                                mutated,
                                mutated_fields,
                                engine="infinity",
                                role_id=1,
                            )

    def test_each_execution_witness_bit_must_match_stdout(self) -> None:
        for module in self.modules:
            with _small_protocol(module):
                sidecar, fields = _fixture(
                    module,
                    engine="infinity",
                    role_id=2,
                )
                original_bits = _execution_witness_bits(module, 2)
                for index, name in enumerate(module.EXECUTION_WITNESS_FIELDS):
                    with self.subTest(module=module.__name__, field=name):
                        mutated = _mutate(
                            sidecar,
                            "execution_witness",
                            struct.pack("<I", original_bits ^ (1 << index)),
                        )
                        with self.assertRaises(_failure_type(module)):
                            _parse(
                                module,
                                mutated,
                                _refresh_sidecar_fields(fields, mutated),
                                engine="infinity",
                                role_id=2,
                            )

    def test_query_corpus_substitution_is_rejected_when_stdout_matches(self) -> None:
        for module in self.modules:
            with _small_protocol(module):
                sidecar, fields = _fixture(
                    module,
                    engine="infinity",
                    role_id=2,
                )
                mutated = bytearray(sidecar)
                mutated[
                    QUERY_TIMED_CORPUS_OFFSET : QUERY_TIMED_CORPUS_OFFSET + 32
                ] = bytes.fromhex("aa" * 32)
                mutated_fields = _refresh_sidecar_fields(fields, bytes(mutated))
                mutated_fields["query_timed_corpus_sha256"] = "aa" * 32
                with self.assertRaises(_failure_type(module)):
                    _parse(
                        module,
                        bytes(mutated),
                        mutated_fields,
                        engine="infinity",
                        role_id=2,
                    )

    def test_query_validated_operation_count_mismatch_is_rejected(self) -> None:
        for module in self.modules:
            with _small_protocol(module):
                sidecar, fields = _fixture(
                    module,
                    engine="infinity",
                    role_id=2,
                )
                mutated = bytearray(sidecar)
                struct.pack_into(
                    "<Q",
                    mutated,
                    QUERY_LATENCY_VALIDATED_OPERATIONS_OFFSET,
                    len(LATENCY_SAMPLES) - 1,
                )
                with self.assertRaises(_failure_type(module)):
                    _parse(
                        module,
                        bytes(mutated),
                        _refresh_sidecar_fields(fields, bytes(mutated)),
                        engine="infinity",
                        role_id=2,
                    )

    def test_consistently_rehashed_query_checksums_must_match_replay(self) -> None:
        for module in self.modules:
            with _small_protocol(module):
                sidecar, fields = _fixture(
                    module,
                    engine="infinity",
                    role_id=2,
                )
                per_query = _query_checksums(module)
                per_query[0] ^= 1
                aggregate = module.query_benchmark_checksum(per_query)
                mutated = bytearray(sidecar)
                for offset in (
                    QUERY_LATENCY_VALIDATED_CHECKSUM_OFFSET,
                    QUERY_THROUGHPUT_VALIDATED_CHECKSUM_OFFSET,
                    QUERY_RESULT_CHECKSUM_OFFSET,
                ):
                    struct.pack_into("<Q", mutated, offset, aggregate)
                struct.pack_into(
                    "<" + "Q" * QUERY_COUNT,
                    mutated,
                    QUERY_PER_QUERY_CHECKSUMS_OFFSET,
                    *per_query,
                )
                mutated_fields = _refresh_sidecar_fields(fields, bytes(mutated))
                mutated_fields[
                    "infinity_query_latency_validated_result_checksum"
                ] = str(aggregate)
                mutated_fields[
                    "infinity_query_throughput_validated_result_checksum"
                ] = str(aggregate)
                mutated_fields["infinity_query_result_checksum"] = str(aggregate)
                mutated_fields["infinity_query_per_query_checksums"] = ",".join(
                    str(value) for value in per_query
                )
                with self.assertRaises(_failure_type(module)):
                    _parse(
                        module,
                        bytes(mutated),
                        mutated_fields,
                        engine="infinity",
                        role_id=2,
                    )

    def test_role_engine_incompatibility_is_rejected(self) -> None:
        cases = (("infinity", 4), ("faiss", 1), ("faiss", 2), ("faiss", 3))
        for module in self.modules:
            with _small_protocol(module):
                for engine, role_id in cases:
                    with self.subTest(
                        module=module.__name__,
                        engine=engine,
                        role_id=role_id,
                    ):
                        sidecar, fields = _fixture(
                            module,
                            engine=engine,
                            role_id=role_id,
                        )
                        with self.assertRaises(_failure_type(module)):
                            _parse(
                                module,
                                sidecar,
                                fields,
                                engine=engine,
                                role_id=role_id,
                            )

    def test_binding_replay_against_changed_expectations_is_rejected(self) -> None:
        replays = {
            "expected_campaign_nonce": "66" * 32,
            "expected_schedule_sequence": SCHEDULE_SEQUENCE + 1,
            "expected_role_id": 2,
            "expected_binary_sha256": "77" * 32,
            "expected_dataset_sha256": "88" * 32,
        }
        for module in self.modules:
            with _small_protocol(module):
                sidecar, fields = _fixture(
                    module,
                    engine="infinity",
                    role_id=1,
                )
                for argument, value in replays.items():
                    with self.subTest(
                        module=module.__name__,
                        argument=argument,
                    ):
                        with self.assertRaises(_failure_type(module)):
                            _parse(
                                module,
                                sidecar,
                                fields,
                                engine="infinity",
                                role_id=1,
                                expected_overrides={argument: value},
                            )

    def test_every_binding_stdout_cross_check_is_rejected(self) -> None:
        corruptions = {
            "campaign_nonce": "99" * 32,
            "schedule_sequence": str(SCHEDULE_SEQUENCE + 1),
            "role_id": "2",
            "binary_sha256": "aa" * 32,
            "dataset_sha256": "bb" * 32,
        }
        for module in self.modules:
            with _small_protocol(module):
                sidecar, fields = _fixture(
                    module,
                    engine="infinity",
                    role_id=1,
                )
                for field, value in corruptions.items():
                    with self.subTest(
                        module=module.__name__,
                        field=field,
                    ):
                        corrupted_fields = dict(fields)
                        corrupted_fields[field] = value
                        with self.assertRaises(_failure_type(module)):
                            _parse(
                                module,
                                sidecar,
                                corrupted_fields,
                                engine="infinity",
                                role_id=1,
                            )


if __name__ == "__main__":
    unittest.main()
