#!/usr/bin/env python3
"""Deterministic end-to-end HNSW smoke for a native Infinity server."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import struct
import time
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

import infinity
from infinity import index
from infinity.common import ConflictType, InfinityException, NetworkAddress
from infinity.errors import ErrorCode


DATABASE_NAME = "default_db"
TABLE_NAME = "apple_native_hnsw_smoke_v1"
INDEX_NAME = "idx_embedding_hnsw"
SEED = 20260819
TARGET_ROW = 731
TOP_K = 10
FLOAT32 = struct.Struct("<f")
UINT64 = struct.Struct("<Q")

T = TypeVar("T")


class SmokeFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def timed(timings: dict[str, float], name: str, operation: Callable[[], T]) -> T:
    started = time.perf_counter_ns()
    result = operation()
    timings[name] = (time.perf_counter_ns() - started) / 1_000_000_000
    return result


def vectors(row_count: int, dimensions: int) -> list[list[float]]:
    generator = random.Random(SEED)
    return [
        [
            FLOAT32.unpack(FLOAT32.pack(generator.uniform(-1.0, 1.0)))[0]
            for _ in range(dimensions)
        ]
        for _ in range(row_count)
    ]


def response_fields(response: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name, value in vars(response).items():
        if value is None or isinstance(value, (bool, float, int, str)):
            fields[name] = value
        elif isinstance(value, list) and all(
            item is None or isinstance(item, (bool, float, int, str))
            for item in value
        ):
            fields[name] = value
        else:
            fields[name] = repr(value)
    return fields


def result_column(data: dict[str, list[Any]], expected_name: str) -> list[Any]:
    normalized = expected_name.lower()
    for name, values in data.items():
        if name.lower() == normalized:
            return values
    raise SmokeFailure(
        f"Missing result column {expected_name!r}; returned columns: {list(data)}"
    )


def dataset_digest(rows: list[tuple[int, list[float]]]) -> str:
    digest = hashlib.sha256()
    dimensions = len(rows[0][1]) if rows else 0
    digest.update(UINT64.pack(len(rows)))
    digest.update(UINT64.pack(dimensions))
    for row_id, vector in rows:
        digest.update(UINT64.pack(row_id))
        require(
            len(vector) == dimensions,
            f"Row {row_id} has {len(vector)} dimensions, expected {dimensions}",
        )
        for value in vector:
            digest.update(FLOAT32.pack(value))
    return digest.hexdigest()


def validate_complete_dataset(
    table: Any,
    expected_vectors: list[list[float]],
    timings: dict[str, float],
) -> dict[str, Any]:
    data, _, _ = timed(
        timings,
        "complete_dataset_scan_seconds",
        lambda: table.output(["id", "embedding"]).to_result(),
    )
    ids = [int(value) for value in result_column(data, "id")]
    returned_vectors = [
        [float(value) for value in vector]
        for vector in result_column(data, "embedding")
    ]
    require(
        len(ids) == len(returned_vectors),
        f"ID/vector column length mismatch: {len(ids)} != "
        f"{len(returned_vectors)}",
    )
    require(
        len(ids) == len(expected_vectors),
        f"Complete scan returned {len(ids)} rows, expected "
        f"{len(expected_vectors)}",
    )
    require(
        len(set(ids)) == len(ids),
        "Complete scan returned duplicate row IDs",
    )

    returned_rows = sorted(zip(ids, returned_vectors), key=lambda row: row[0])
    expected_ids = list(range(len(expected_vectors)))
    returned_ids = [row_id for row_id, _ in returned_rows]
    require(
        returned_ids == expected_ids,
        "Complete scan did not return the exact expected ID set",
    )

    for (row_id, returned), expected in zip(
        returned_rows,
        expected_vectors,
        strict=True,
    ):
        require(
            len(returned) == len(expected),
            f"Row {row_id} has {len(returned)} dimensions, expected "
            f"{len(expected)}",
        )
        require(
            b"".join(FLOAT32.pack(value) for value in returned)
            == b"".join(FLOAT32.pack(value) for value in expected),
            f"Row {row_id} vector bytes differ from the inserted float32 data",
        )

    expected_rows = list(enumerate(expected_vectors))
    expected_digest = dataset_digest(expected_rows)
    returned_digest = dataset_digest(returned_rows)
    require(
        returned_digest == expected_digest,
        "Complete dataset digest differs from generated input",
    )
    return {
        "row_count": len(returned_rows),
        "dimensions": len(expected_vectors[0]) if expected_vectors else 0,
        "id_min": returned_ids[0] if returned_ids else None,
        "id_max": returned_ids[-1] if returned_ids else None,
        "canonical_float32_sha256": returned_digest,
        "expected_float32_sha256": expected_digest,
    }


def validate_table(
    table: Any,
    expected_vectors: list[list[float]],
    run_id: str,
    timings: dict[str, float],
) -> dict[str, Any]:
    count_data, _, _ = timed(
        timings,
        "count_seconds",
        lambda: table.output(["count(*)"]).to_result(),
    )
    row_count = int(result_column(count_data, "count(star)")[0])
    require(
        row_count == len(expected_vectors),
        f"Expected {len(expected_vectors)} persisted rows, found {row_count}"
    )

    identity_data, _, _ = timed(
        timings,
        "run_identity_count_seconds",
        lambda: (
            table.output(["count(*)"])
            .filter(f"run_id = '{run_id}'")
            .to_result()
        ),
    )
    identity_count = int(result_column(identity_data, "count(star)")[0])
    require(
        identity_count == len(expected_vectors),
        f"Expected run ID {run_id!r} on every row, found {identity_count}",
    )

    indexes = timed(timings, "list_indexes_seconds", table.list_indexes)
    require(
        INDEX_NAME in indexes.index_names,
        f"Expected {INDEX_NAME!r} in index list {indexes.index_names!r}"
    )

    shown_index = timed(
        timings,
        "show_index_seconds",
        lambda: table.show_index(INDEX_NAME),
    )
    require(
        shown_index.error_code == ErrorCode.OK,
        f"show_index failed: {shown_index.error_code}: {shown_index.error_msg}",
    )
    require(
        shown_index.index_type.upper() == "HNSW",
        f"Expected HNSW index type, got {shown_index.index_type!r}"
    )

    segments = timed(timings, "show_segments_seconds", table.show_segments)
    segment_rows = segments.to_dicts()
    segment_count = len(segment_rows)
    require(segment_count > 0, "Expected at least one table segment")
    require(
        int(shown_index.segment_index_count) == segment_count,
        f"Expected index coverage for all {segment_count} table segments, got "
        f"{shown_index.segment_index_count}",
    )

    query = expected_vectors[TARGET_ROW]
    exact = sorted(
        (
            math.fsum((left - right) ** 2 for left, right in zip(vector, query)),
            row_id,
        )
        for row_id, vector in enumerate(expected_vectors)
    )
    exact_ids = [row_id for _, row_id in exact[:TOP_K]]

    missing_index_name = f"{INDEX_NAME}_missing"
    negative_control_started = time.perf_counter_ns()
    try:
        (
            table.output(["id"])
            .match_dense(
                "embedding",
                query,
                "float",
                "l2",
                TOP_K,
                {"index_name": missing_index_name},
            )
            .to_result()
        )
    except InfinityException as error:
        timings["missing_index_control_seconds"] = (
            time.perf_counter_ns() - negative_control_started
        ) / 1_000_000_000
        require(
            error.error_code == ErrorCode.INDEX_NOT_EXIST,
            "Expected INDEX_NOT_EXIST for the missing named index, got "
            f"{error.error_code}: {error.error_msg}"
        )
    else:
        raise SmokeFailure(
            f"Server ignored missing index_name {missing_index_name!r}"
        )

    query_data, _, _ = timed(
        timings,
        "forced_hnsw_query_seconds",
        lambda: (
            table.output(["id", "run_id", "_distance"])
            .match_dense(
                "embedding",
                query,
                "float",
                "l2",
                TOP_K,
                {"index_name": INDEX_NAME, "ef": "128"},
            )
            .to_result()
        ),
    )
    result_ids = [int(value) for value in result_column(query_data, "id")]
    result_run_ids = [
        str(value) for value in result_column(query_data, "run_id")
    ]
    distances = [
        float(value) for value in result_column(query_data, "DISTANCE")
    ]
    recall = len(set(result_ids) & set(exact_ids)) / TOP_K

    require(
        len(result_ids) == TOP_K,
        f"Expected {TOP_K} HNSW results, got {len(result_ids)}"
    )
    require(
        len(result_run_ids) == TOP_K and len(distances) == TOP_K,
        "HNSW result columns have inconsistent lengths",
    )
    require(
        len(set(result_ids)) == TOP_K,
        f"HNSW returned duplicate IDs: {result_ids!r}",
    )
    require(
        all(math.isfinite(distance) for distance in distances),
        f"HNSW returned a non-finite distance: {distances!r}",
    )
    require(
        all(
            left <= right
            for left, right in zip(distances, distances[1:])
        ),
        f"HNSW distances are not sorted ascending: {distances!r}",
    )
    require(
        result_run_ids == [run_id] * TOP_K,
        f"Query returned rows outside run ID {run_id!r}: {result_run_ids!r}",
    )
    require(
        result_ids[0] == TARGET_ROW,
        f"Expected exact target row {TARGET_ROW} first, got {result_ids[0]}"
    )
    require(
        abs(distances[0]) <= 1e-6,
        f"Expected zero distance for exact target, got {distances[0]}"
    )
    require(
        recall >= 0.8,
        f"Expected recall@{TOP_K} >= 0.8, got {recall}",
    )
    complete_dataset = validate_complete_dataset(
        table,
        expected_vectors,
        timings,
    )

    return {
        "run_id": run_id,
        "row_count": row_count,
        "run_identity_count": identity_count,
        "segments": segment_rows,
        "index_names": list(indexes.index_names),
        "show_index": response_fields(shown_index),
        "complete_dataset": complete_dataset,
        "query": {
            "forced_index_name": INDEX_NAME,
            "missing_index_negative_control": {
                "index_name": missing_index_name,
                "error_code": int(ErrorCode.INDEX_NOT_EXIST),
            },
            "target_row": TARGET_ROW,
            "result_ids": result_ids,
            "result_run_ids": result_run_ids,
            "distances": distances,
            "exact_ids": exact_ids,
            f"recall_at_{TOP_K}": recall,
        },
    }


def initialize(
    connection: Any,
    expected_vectors: list[list[float]],
    run_id: str,
    dimensions: int,
    batch_size: int,
    timings: dict[str, float],
) -> dict[str, Any]:
    database = connection.get_database(DATABASE_NAME)
    timed(
        timings,
        "drop_old_table_seconds",
        lambda: database.drop_table(TABLE_NAME, ConflictType.Ignore),
    )
    table = timed(
        timings,
        "create_table_seconds",
        lambda: database.create_table(
            TABLE_NAME,
            {
                "id": {"type": "integer"},
                "run_id": {"type": "varchar"},
                "embedding": {"type": f"vector,{dimensions},float"},
            },
            ConflictType.Error,
        ),
    )

    started = time.perf_counter_ns()
    for first_row in range(0, len(expected_vectors), batch_size):
        batch = [
            {
                "id": row_id,
                "run_id": run_id,
                "embedding": expected_vectors[row_id],
            }
            for row_id in range(
                first_row,
                min(first_row + batch_size, len(expected_vectors)),
            )
        ]
        response = table.insert(batch)
        require(
            response.error_code == ErrorCode.OK,
            f"Insert failed: {response.error_code}: {response.error_msg}",
        )
    timings["insert_seconds"] = (
        time.perf_counter_ns() - started
    ) / 1_000_000_000

    flush_response = timed(
        timings,
        "pre_index_flush_seconds",
        connection.flush_data,
    )
    require(
        flush_response.error_code == ErrorCode.OK,
        f"Pre-index flush failed: {flush_response.error_code}: "
        f"{flush_response.error_msg}",
    )
    create_response = timed(
        timings,
        "create_hnsw_index_seconds",
        lambda: table.create_index(
            INDEX_NAME,
            index.IndexInfo(
                "embedding",
                index.IndexType.Hnsw,
                {
                    "M": "16",
                    "ef_construction": "128",
                    "metric": "l2",
                },
            ),
            ConflictType.Error,
        ),
    )
    require(
        create_response.error_code == ErrorCode.OK,
        f"Index creation failed: {create_response.error_code}: "
        f"{create_response.error_msg}",
    )
    return validate_table(table, expected_vectors, run_id, timings)


def verify(
    connection: Any,
    expected_vectors: list[list[float]],
    run_id: str,
    timings: dict[str, float],
) -> dict[str, Any]:
    database = connection.get_database(DATABASE_NAME)
    table = timed(
        timings,
        "get_persisted_table_seconds",
        lambda: database.get_table(TABLE_NAME),
    )
    return validate_table(table, expected_vectors, run_id, timings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("initialize", "verify"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23871)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--dimensions", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows <= TARGET_ROW:
        parser.error(f"--rows must be greater than {TARGET_ROW}")
    if args.dimensions <= 0:
        parser.error("--dimensions must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    try:
        args.run_id = str(uuid.UUID(args.run_id))
    except ValueError:
        parser.error("--run-id must be a UUID")
    return args


def main() -> None:
    args = parse_args()
    expected_vectors = vectors(args.rows, args.dimensions)
    timings: dict[str, float] = {}
    started = time.perf_counter_ns()
    connection = timed(
        timings,
        "connect_seconds",
        lambda: infinity.connect(NetworkAddress(args.host, args.port)),
    )
    primary_error: BaseException | None = None
    try:
        if args.phase == "initialize":
            validation = initialize(
                connection,
                expected_vectors,
                args.run_id,
                args.dimensions,
                args.batch_size,
                timings,
            )
        else:
            validation = verify(
                connection,
                expected_vectors,
                args.run_id,
                timings,
            )
        if args.phase == "initialize":
            flush_response = timed(
                timings,
                "final_flush_seconds",
                connection.flush_data,
            )
            require(
                flush_response.error_code == ErrorCode.OK,
                f"Final flush failed: {flush_response.error_code}: "
                f"{flush_response.error_msg}",
            )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            timed(timings, "disconnect_seconds", connection.disconnect)
        except Exception:
            if primary_error is None:
                raise

    timings["total_seconds"] = (
        time.perf_counter_ns() - started
    ) / 1_000_000_000
    evidence = {
        "status": "pass",
        "phase": args.phase,
        "run_id": args.run_id,
        "endpoint": f"{args.host}:{args.port}",
        "platform": {
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "sdk_module": str(Path(infinity.__file__).resolve()),
        },
        "workload": {
            "rows": args.rows,
            "dimensions": args.dimensions,
            "batch_size": args.batch_size,
            "seed": SEED,
            "metric": "l2",
            "M": 16,
            "ef_construction": 128,
            "ef_search": 128,
        },
        "timings": timings,
        "validation": validation,
    }
    rendered = json.dumps(evidence, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
