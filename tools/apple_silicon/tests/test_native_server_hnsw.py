from __future__ import annotations

import struct
import unittest
from pathlib import Path

from tools.apple_silicon import native_server_hnsw_cycle as cycle
from tools.apple_silicon import native_server_hnsw_smoke as smoke


REPO = Path(__file__).resolve().parents[3]
V10 = (
    REPO
    / "docs/apple_silicon/checkpoints/native-server-hnsw-v2/evidence/cycle-v10"
)


class NativeServerLogProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.initialize_log = (V10 / "server-initialize.log").read_text(
            encoding="utf-8"
        )
        cls.verify_log = (V10 / "server-verify.log").read_text(
            encoding="utf-8"
        )
        cls.read_only_verify_log = "\n".join(
            line
            for line in cls.verify_log.splitlines()
            if "THRIFT: Flush Type:" not in line
        )

    def test_v10_object_flow_is_correlated_after_known_flush_is_removed(
        self,
    ) -> None:
        proof = cycle.correlate_index_proof(
            self.initialize_log,
            self.read_only_verify_log,
        )

        expected_key = "01a02383-9da8-70c0-be77-fcc12d9eafa2"
        self.assertEqual(proof["persisted"]["object_key"], expected_key)
        self.assertEqual(
            proof["checkpoint_mapping"]["object_key"],
            expected_key,
        )
        self.assertEqual(
            proof["verify_query"]["object_key"],
            expected_key,
        )
        self.assertEqual(proof["persisted"]["part_size"], 295368)

    def test_missing_index_prefix_is_not_counted_as_forced_query(self) -> None:
        only_missing = self.initialize_log.replace(
            "Use index: idx_embedding_hnsw\n",
            "Use index: another_index\n",
        )

        with self.assertRaisesRegex(cycle.CycleFailure, "found 0"):
            cycle.parse_forced_query_object(only_missing)

    def test_verify_index_persist_is_rejected(self) -> None:
        mutation = (
            self.verify_log
            + "\nPersist local path "
            + "db_1/tbl_0/idx_1/seg_0/chunk_0.idx to composed ObjAddr "
            + "(changed, 0, 1)\n"
        )

        with self.assertRaisesRegex(
            cycle.CycleFailure,
            "index_object_persist",
        ):
            cycle.reject_verify_index_mutation(mutation)

    def test_v10_verify_flush_is_rejected(self) -> None:
        with self.assertRaisesRegex(cycle.CycleFailure, "flush_rpc"):
            cycle.reject_verify_index_mutation(self.verify_log)

    def test_checkpoint_offset_drift_is_rejected(self) -> None:
        drifted = self.read_only_verify_log.replace(
            '"part_offset":0,"part_size":295368',
            '"part_offset":8,"part_size":295368',
            1,
        )

        with self.assertRaisesRegex(
            cycle.CycleFailure,
            "Checkpoint mapping differs",
        ):
            cycle.correlate_index_proof(self.initialize_log, drifted)


class NativeServerDatasetTests(unittest.TestCase):
    def test_generated_vectors_are_exact_float32(self) -> None:
        generated = smoke.vectors(8, 7)

        for vector in generated:
            for value in vector:
                self.assertEqual(
                    value,
                    struct.unpack("<f", struct.pack("<f", value))[0],
                )

    def test_dataset_digest_is_order_and_byte_sensitive(self) -> None:
        rows = [
            (0, [0.0, 1.0]),
            (1, [2.0, 3.0]),
        ]
        baseline = smoke.dataset_digest(rows)

        self.assertNotEqual(baseline, smoke.dataset_digest(list(reversed(rows))))
        self.assertNotEqual(
            baseline,
            smoke.dataset_digest([(0, [0.0, 1.0]), (1, [2.0, 3.5])]),
        )


if __name__ == "__main__":
    unittest.main()
