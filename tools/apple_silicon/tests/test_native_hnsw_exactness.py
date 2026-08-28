from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import pathlib
import struct
import tempfile
import unittest
from unittest import mock

from tools.apple_silicon import native_hnsw_exactness as exact


def witness_bits(value: int) -> dict[str, int]:
    return {field: value for field in exact.WITNESS_FIELD_NAMES}


THRESHOLD_ONLY_EXPECTED_WITNESSES = {
    "control": {
        **{
            field: 1
            for field in exact.WITNESS_FIELD_NAMES[:5]
        },
        **{
            field: 0
            for field in exact.WITNESS_FIELD_NAMES[5:]
        },
    },
    "treatment": witness_bits(1),
}
COMBINED_EXPECTED_WITNESSES = {
    "control": witness_bits(0),
    "treatment": witness_bits(1),
}


class ExactnessArtifactTests(unittest.TestCase):
    @staticmethod
    def fixture(
        role: str = "control",
        *,
        expected_witnesses: dict[str, dict[str, int]] | None = None,
    ) -> dict[str, object]:
        if expected_witnesses is None:
            expected_witnesses = THRESHOLD_ONLY_EXPECTED_WITNESSES
        vectors = 2
        dimension = 2
        dataset = struct.pack("<4f", 0.0, 1.0, 2.0, 3.0)
        dataset_sha256 = hashlib.sha256(dataset).hexdigest()

        graph = bytearray()
        graph.extend(exact.GRAPH_MAGIC)
        graph.extend(struct.pack("<II", exact.GRAPH_SCHEMA_VERSION, exact.GRAPH_HEADER_BYTES))
        graph.extend(
            struct.pack(
                "<13Qii2Q",
                vectors,
                dimension,
                exact.M,
                exact.EF_CONSTRUCTION,
                vectors,
                1,
                exact.WORKERS,
                0,
                vectors,
                1,
                vectors,
                2 * exact.M,
                exact.M,
                0,
                0,
                vectors,
                1,
            )
        )
        graph.extend(bytes.fromhex(dataset_sha256))
        graph.extend(struct.pack("<Q", exact.GRAPH_HEADER_BYTES))
        for ordinal, neighbor in ((0, 1), (1, 0)):
            graph.extend(struct.pack("<IiiI", ordinal, ordinal, 0, 1))
            graph.extend(struct.pack("<IIi", 0, 1, neighbor))

        save_to_ptr = bytearray(
            struct.pack(
                "<6Qii",
                exact.M,
                exact.EF_CONSTRUCTION,
                vectors,
                dimension,
                2 * exact.M,
                exact.M,
                0,
                0,
            )
        )
        save_to_ptr.extend(dataset)
        save_to_ptr.extend(struct.pack("<Q", 0))
        for neighbor in (1, 0):
            record = bytearray(exact.LEVEL_ZERO_STRIDE)
            struct.pack_into("<i", record, 0, 0)
            struct.pack_into("<Q", record, 8, 0)
            struct.pack_into("<i", record, 16, 1)
            struct.pack_into("<i", record, 20, neighbor)
            save_to_ptr.extend(record)
        save_to_ptr.extend(struct.pack("<2i", 0, 1))

        stdout_value = {
            "build": {
                "end": vectors,
                "memory_delta_positive": True,
                "start": 0,
                "submitted_tasks": 1,
            },
            "configuration": {
                "chunk_size": vectors,
                "dimension": dimension,
                "ef_construction": exact.EF_CONSTRUCTION,
                "m": exact.M,
                "max_chunks": 1,
                "optimize": True,
                "vectors": vectors,
                "workers": exact.WORKERS,
            },
            "dataset": {
                "bytes": len(dataset),
                "sha256": dataset_sha256,
            },
            "graph": {
                "bytes": len(graph),
                "entry_point": 0,
                "header_bytes": exact.GRAPH_HEADER_BYTES,
                "max_level": 0,
                "mmax": exact.M,
                "mmax0": 2 * exact.M,
                "sha256": hashlib.sha256(graph).hexdigest(),
            },
            "save_to_ptr": {
                "bytes": len(save_to_ptr),
                "roundtrip_graph_equal": True,
                "sha256": hashlib.sha256(save_to_ptr).hexdigest(),
            },
            "schema_version": 1,
            "status": "PASS",
        }
        stdout = (
            json.dumps(stdout_value, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("ascii")

        witness = exact.WITNESS_MAGIC + struct.pack(
            "<12I",
            exact.WITNESS_SCHEMA_VERSION,
            exact.WITNESS_BYTES,
            *(
                expected_witnesses[role][field]
                for field in exact.WITNESS_FIELD_NAMES
            ),
        )
        return {
            "expected_witnesses": expected_witnesses,
            "dataset": dataset,
            "stdout": stdout,
            "stderr": b"",
            "graph": bytes(graph),
            "save_to_ptr": bytes(save_to_ptr),
            "witness": witness,
        }

    @staticmethod
    def patches(fixture: dict[str, object]) -> tuple[mock._patch, ...]:
        return (
            mock.patch.object(exact, "VECTOR_COUNT", 2),
            mock.patch.object(exact, "DIMENSION", 2),
            mock.patch.object(exact, "CHUNK_SIZE", 2),
            mock.patch.object(exact, "MAX_CHUNKS", 1),
            mock.patch.object(
                exact,
                "DATASET_BYTES",
                len(fixture["dataset"]),
            ),
            mock.patch.object(
                exact,
                "DATASET_SHA256",
                hashlib.sha256(fixture["dataset"]).hexdigest(),
            ),
        )

    def parse(
        self,
        role: str = "control",
        *,
        expected_witnesses: dict[str, dict[str, int]] | None = None,
    ) -> dict:
        fixture = self.fixture(
            role,
            expected_witnesses=expected_witnesses,
        )
        with (
            self.patches(fixture)[0],
            self.patches(fixture)[1],
            self.patches(fixture)[2],
            self.patches(fixture)[3],
            self.patches(fixture)[4],
            self.patches(fixture)[5],
        ):
            return exact.parse_exactness_artifacts(role=role, **fixture)

    def test_parses_graph_pointer_image_and_role_witness(self) -> None:
        control = self.parse("control")
        treatment = self.parse("treatment")

        self.assertTrue(
            control["save_to_ptr"]["topology_matches_canonical_graph"]
        )
        self.assertTrue(control["save_to_ptr"]["labels_match_ordinals"])
        self.assertEqual(
            control["witness"]["incremental_treatment_compiled"],
            1,
        )
        self.assertEqual(
            control["witness"]["threshold_treatment_compiled"],
            0,
        )
        self.assertEqual(
            treatment["witness"]["threshold_treatment_compiled"],
            1,
        )

    def test_parses_combined_witness_matrix(self) -> None:
        control = self.parse(
            "control",
            expected_witnesses=COMBINED_EXPECTED_WITNESSES,
        )
        treatment = self.parse(
            "treatment",
            expected_witnesses=COMBINED_EXPECTED_WITNESSES,
        )

        for field in exact.WITNESS_FIELD_NAMES:
            self.assertEqual(control["witness"][field], 0)
            self.assertEqual(treatment["witness"][field], 1)

    def test_rejects_pointer_topology_that_differs_from_graph(self) -> None:
        fixture = self.fixture()
        pointer = bytearray(fixture["save_to_ptr"])
        vector_offset = struct.calcsize("<6Qii")
        graph_offset = vector_offset + len(fixture["dataset"]) + 8
        struct.pack_into("<i", pointer, graph_offset + 20, 0)
        fixture["save_to_ptr"] = bytes(pointer)

        patches = self.patches(fixture)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            self.assertRaisesRegex(exact.ExactnessFailure, "topology differs"),
        ):
            exact.parse_exactness_artifacts(role="control", **fixture)

    def test_rejects_noncanonical_stdout(self) -> None:
        fixture = self.fixture()
        fixture["stdout"] = fixture["stdout"].replace(b'{"build":', b'{ "build":', 1)
        patches = self.patches(fixture)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            self.assertRaisesRegex(exact.ExactnessFailure, "canonically serialized"),
        ):
            exact.parse_exactness_artifacts(role="control", **fixture)

    def test_rejects_boolean_stdout_schema_version(self) -> None:
        fixture = self.fixture()
        value = json.loads(fixture["stdout"])
        value["schema_version"] = True
        fixture["stdout"] = (
            json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("ascii")
        patches = self.patches(fixture)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            self.assertRaisesRegex(
                exact.ExactnessFailure, "status or schema differs"
            ),
        ):
            exact.parse_exactness_artifacts(role="control", **fixture)

    def test_rejects_witness_that_differs_from_expected_vector(self) -> None:
        fixture = self.fixture("treatment")
        fixture["witness"] = exact.WITNESS_MAGIC + struct.pack(
            "<12I",
            exact.WITNESS_SCHEMA_VERSION,
            exact.WITNESS_BYTES,
            *([0] * 10),
        )
        patches = self.patches(fixture)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            self.assertRaisesRegex(
                exact.ExactnessFailure,
                "differs from expected_witnesses",
            ),
        ):
            exact.parse_exactness_artifacts(role="treatment", **fixture)

    def test_rejects_malformed_expected_witness_mappings(self) -> None:
        mutations = {}

        nonboolean = copy.deepcopy(THRESHOLD_ONLY_EXPECTED_WITNESSES)
        nonboolean["control"][exact.WITNESS_FIELD_NAMES[0]] = True
        mutations["nonboolean"] = nonboolean

        missing = copy.deepcopy(THRESHOLD_ONLY_EXPECTED_WITNESSES)
        missing["control"].pop(exact.WITNESS_FIELD_NAMES[0])
        mutations["missing"] = missing

        extra = copy.deepcopy(THRESHOLD_ONLY_EXPECTED_WITNESSES)
        extra["control"]["unexpected"] = 0
        mutations["extra"] = extra

        for name, expected_witnesses in mutations.items():
            with self.subTest(name=name):
                fixture = self.fixture()
                fixture["expected_witnesses"] = expected_witnesses
                patches = self.patches(fixture)
                with (
                    patches[0],
                    patches[1],
                    patches[2],
                    patches[3],
                    patches[4],
                    patches[5],
                    self.assertRaises(exact.ExactnessFailure),
                ):
                    exact.parse_exactness_artifacts(
                        role="control",
                        **fixture,
                    )

    def test_rejects_wrong_witness_size_and_schema(self) -> None:
        fixture = self.fixture()
        cases = (
            ("size", fixture["witness"][:-1], "must be 56 bytes"),
            (
                "schema",
                exact.WITNESS_MAGIC
                + struct.pack(
                    "<12I",
                    1,
                    exact.WITNESS_BYTES,
                    *([0] * 10),
                ),
                "schema differs",
            ),
        )
        for name, witness, detail in cases:
            with self.subTest(name=name):
                candidate = dict(fixture)
                candidate["witness"] = witness
                patches = self.patches(candidate)
                with (
                    patches[0],
                    patches[1],
                    patches[2],
                    patches[3],
                    patches[4],
                    patches[5],
                    self.assertRaisesRegex(
                        exact.ExactnessFailure,
                        detail,
                    ),
                ):
                    exact.parse_exactness_artifacts(
                        role="control",
                        **candidate,
                    )

    @staticmethod
    def completed_runs(
        expected_witnesses: dict[str, dict[str, int]],
    ) -> list[exact._CompletedRun]:
        completed = []
        for sequence, (role, repetition) in enumerate(
            exact.EXACTNESS_RUN_ORDER
        ):
            witness = exact.WITNESS_MAGIC + struct.pack(
                "<12I",
                exact.WITNESS_SCHEMA_VERSION,
                exact.WITNESS_BYTES,
                *(
                    expected_witnesses[role][field]
                    for field in exact.WITNESS_FIELD_NAMES
                ),
            )
            artifacts = {
                name: {
                    "captured_path": (
                        f"exactness-{sequence:02d}-{role}-{repetition}-{name}"
                    ),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                }
                for name, content in {
                    "stdout": b"same stdout",
                    "stderr": b"",
                    "graph": b"same graph",
                    "save_to_ptr": b"same pointer",
                    "witness": witness,
                }.items()
            }
            completed.append(
                exact._CompletedRun(
                    record={
                        "role": role,
                        "repetition": repetition,
                        "process_evidence": {"lifecycle": {"sequence": sequence}},
                        "artifacts": artifacts,
                    },
                    parsed=exact._ParsedRun(
                        stdout=b"same stdout",
                        graph=b"same graph",
                        save_to_ptr=b"same pointer",
                        witness=witness,
                        verification={},
                    ),
                )
            )
        return completed

    def test_comparisons_follow_explicit_witness_vector_relation(self) -> None:
        threshold = exact._validate_gate_comparisons(
            self.completed_runs(THRESHOLD_ONLY_EXPECTED_WITNESSES),
            expected_witnesses=THRESHOLD_ONLY_EXPECTED_WITNESSES,
        )
        self.assertFalse(threshold["cross_role_witness_byte_equal"])

        equal = {
            "control": witness_bits(1),
            "treatment": witness_bits(1),
        }
        equal_result = exact._validate_gate_comparisons(
            self.completed_runs(equal),
            expected_witnesses=equal,
        )
        self.assertTrue(equal_result["cross_role_witness_byte_equal"])

    def test_independent_record_binds_expected_witnesses(self) -> None:
        record = exact._independent_verifier_record(
            self.completed_runs(THRESHOLD_ONLY_EXPECTED_WITNESSES),
            dataset_path=pathlib.Path(exact.d0.DATASET_FILENAME),
            expected_witnesses=THRESHOLD_ONLY_EXPECTED_WITNESSES,
        )

        self.assertEqual(record["schema_version"], 3)
        self.assertEqual(
            record["expected_witnesses"],
            THRESHOLD_ONLY_EXPECTED_WITNESSES,
        )

    @staticmethod
    def expected_executable() -> dict:
        return {
            "path": "/private/tmp/exactness-emitter",
            "device": 11,
            "inode": 22,
            "size": 33,
        }

    @classmethod
    def process_record(cls, pid: int, *, status: str = "stopped") -> dict:
        expected = cls.expected_executable()
        return {
            "pid": pid,
            "parent_pid": exact.os.getpid(),
            "process_group": pid,
            "status": status,
            "executable_path": expected["path"],
            "executable_device": expected["device"],
            "executable_inode": expected["inode"],
            "executable_size": expected["size"],
        }

    @classmethod
    def owned_child(
        cls,
        *,
        pid: int = 1234,
        lifecycle=exact._ChildLifecycle.RUNNING,
        group_state=exact._ChildGroupState.VERIFIED,
    ):
        return exact._SpawnedChildState(
            pid_cell=ctypes.c_int(pid),
            expected_executable=cls.expected_executable(),
            ownership=exact._ChildOwnership.OWNED,
            spawn_result=0,
            pid=pid,
            lifecycle=lifecycle,
            group_state=group_state,
        )

    def test_cleanup_kills_group_observes_terminal_then_reaps(self) -> None:
        child = self.owned_child()
        events = []

        def killpg(pid, signal_number):
            events.append(("killpg", pid, signal_number))

        def observe(pid, *, include_stopped):
            events.append(("observe", pid, include_stopped))
            return exact._WaitObservation(
                pid,
                exact.CLD_KILLED,
                exact.signal.SIGKILL,
            )

        def members(pid):
            events.append(("members", pid))
            return [pid]

        def waitpid(pid, options):
            events.append(("waitpid", pid, options))
            return pid, exact.signal.SIGKILL

        with (
            mock.patch.object(exact.os, "killpg", side_effect=killpg),
            mock.patch.object(
                exact, "_observe_child_nonconsuming", side_effect=observe
            ),
            mock.patch.object(exact, "_group_member_pids", side_effect=members),
            mock.patch.object(exact.os, "waitpid", side_effect=waitpid),
        ):
            self.assertEqual(exact._cleanup_child(child), [])

        self.assertEqual(
            events,
            [
                ("killpg", 1234, exact.signal.SIGKILL),
                ("observe", 1234, False),
                ("members", 1234),
                ("waitpid", 1234, 0),
            ],
        )
        self.assertEqual(child.lifecycle, exact._ChildLifecycle.REAPED)
        self.assertEqual(child.group_state, exact._ChildGroupState.EMPTY)

    def test_reaped_child_never_triggers_process_or_group_syscalls(self) -> None:
        child = self.owned_child(
            lifecycle=exact._ChildLifecycle.REAPED,
            group_state=exact._ChildGroupState.UNRESOLVED,
        )
        with (
            mock.patch.object(exact.os, "killpg") as killpg,
            mock.patch.object(exact.os, "kill") as kill,
            mock.patch.object(exact.os, "waitpid") as waitpid,
            mock.patch.object(exact, "_group_member_pids") as group_members,
        ):
            errors = exact._cleanup_child(child)
        self.assertEqual(
            errors,
            ["normalized interrupted exactness reap publication"],
        )
        self.assertEqual(child.group_state, exact._ChildGroupState.EMPTY)
        killpg.assert_not_called()
        kill.assert_not_called()
        waitpid.assert_not_called()
        group_members.assert_not_called()

    def test_group_snapshot_failure_leaves_leader_unreaped_for_retry(self) -> None:
        child = self.owned_child(
            lifecycle=exact._ChildLifecycle.TERMINAL_OBSERVED,
        )
        child.terminal_code = exact.CLD_EXITED
        child.terminal_status = 0
        with (
            mock.patch.object(exact.os, "killpg") as killpg,
            mock.patch.object(
                exact,
                "_group_member_pids",
                side_effect=RuntimeError("inspection failed"),
            ),
            mock.patch.object(exact.os, "waitpid") as waitpid,
        ):
            errors = exact._cleanup_child(child)
        self.assertEqual(
            errors,
            ["settle child: RuntimeError: inspection failed"],
        )
        killpg.assert_called_once_with(1234, exact.signal.SIGKILL)
        waitpid.assert_not_called()
        self.assertEqual(
            child.lifecycle,
            exact._ChildLifecycle.TERMINAL_OBSERVED,
        )
        self.assertEqual(child.group_state, exact._ChildGroupState.KILL_SENT)

    def test_reap_interruption_retries_without_losing_terminal_status(self) -> None:
        child = self.owned_child(
            lifecycle=exact._ChildLifecycle.TERMINAL_OBSERVED,
            group_state=exact._ChildGroupState.QUIESCENT,
        )
        child.terminal_code = exact.CLD_EXITED
        child.terminal_status = 7
        with mock.patch.object(
            exact.os,
            "waitpid",
            side_effect=[KeyboardInterrupt(), ChildProcessError()],
        ) as waitpid:
            with self.assertRaises(KeyboardInterrupt):
                exact._reap_observed_child(child)
            self.assertEqual(
                child.lifecycle,
                exact._ChildLifecycle.REAP_IN_PROGRESS,
            )
            with self.assertRaisesRegex(
                exact.ExactnessFailure,
                "already reaped",
            ):
                exact._reap_observed_child(child)
        self.assertEqual(waitpid.call_count, 2)
        self.assertEqual(child.lifecycle, exact._ChildLifecycle.REAPED)
        self.assertEqual(child.group_state, exact._ChildGroupState.EMPTY)
        self.assertFalse(child.reap_status_validated)

    def test_initial_echild_is_not_accepted_as_a_completed_reap(self) -> None:
        child = self.owned_child(
            lifecycle=exact._ChildLifecycle.TERMINAL_OBSERVED,
            group_state=exact._ChildGroupState.QUIESCENT,
        )
        child.terminal_code = exact.CLD_EXITED
        child.terminal_status = 0
        with (
            mock.patch.object(
                exact.os,
                "waitpid",
                side_effect=ChildProcessError(),
            ),
            self.assertRaisesRegex(
                exact.ExactnessFailure,
                "already reaped",
            ),
        ):
            exact._reap_observed_child(child)
        self.assertEqual(
            child.lifecycle,
            exact._ChildLifecycle.REAPED,
        )
        self.assertEqual(child.group_state, exact._ChildGroupState.EMPTY)
        self.assertFalse(child.reap_status_validated)

    def test_cleanup_retries_reap_without_any_process_or_group_lookup(
        self,
    ) -> None:
        child = self.owned_child(
            lifecycle=exact._ChildLifecycle.REAP_IN_PROGRESS,
            group_state=exact._ChildGroupState.QUIESCENT,
        )
        child.terminal_code = exact.CLD_EXITED
        child.terminal_status = 0
        with (
            mock.patch.object(
                exact.os,
                "waitpid",
                return_value=(1234, 0),
            ) as waitpid,
            mock.patch.object(exact.os, "killpg") as killpg,
            mock.patch.object(exact.os, "kill") as kill,
            mock.patch.object(exact, "_group_member_pids") as group_members,
        ):
            self.assertEqual(exact._cleanup_child(child), [])
        waitpid.assert_called_once_with(1234, 0)
        killpg.assert_not_called()
        kill.assert_not_called()
        group_members.assert_not_called()
        self.assertEqual(child.lifecycle, exact._ChildLifecycle.REAPED)
        self.assertEqual(child.group_state, exact._ChildGroupState.EMPTY)
        self.assertTrue(child.reap_status_validated)

    def test_unverified_never_resumed_child_is_directly_killed_and_reaped(
        self,
    ) -> None:
        child = self.owned_child(
            lifecycle=exact._ChildLifecycle.STOPPED,
            group_state=exact._ChildGroupState.UNVERIFIED,
        )
        events = []

        def kill(pid, signal_number):
            events.append(("kill", pid, signal_number))

        def observe(pid, *, include_stopped):
            events.append(("observe", pid, include_stopped))
            return exact._WaitObservation(
                pid,
                exact.CLD_KILLED,
                exact.signal.SIGKILL,
            )

        def members(pid):
            events.append(("members", pid))
            return [pid]

        def waitpid(pid, options):
            events.append(("waitpid", pid, options))
            return pid, exact.signal.SIGKILL

        with (
            mock.patch.object(
                exact.d0,
                "capture_process_identity",
                side_effect=RuntimeError("identity unavailable"),
            ),
            mock.patch.object(exact.os, "kill", side_effect=kill),
            mock.patch.object(exact.os, "killpg") as killpg,
            mock.patch.object(
                exact, "_observe_child_nonconsuming", side_effect=observe
            ),
            mock.patch.object(exact, "_group_member_pids", side_effect=members),
            mock.patch.object(exact.os, "waitpid", side_effect=waitpid),
        ):
            errors = exact._cleanup_child(child)
        self.assertEqual(
            errors,
            ["verify process group: RuntimeError: identity unavailable"],
        )
        self.assertEqual(
            events,
            [
                ("kill", 1234, exact.signal.SIGKILL),
                ("observe", 1234, False),
                ("members", 1234),
                ("waitpid", 1234, 0),
            ],
        )
        killpg.assert_not_called()
        self.assertEqual(child.lifecycle, exact._ChildLifecycle.REAPED)
        self.assertEqual(child.group_state, exact._ChildGroupState.EMPTY)
        self.assertTrue(child.reap_status_validated)

    @staticmethod
    def spawn_library(
        *,
        pid: int = 1234,
        spawn_result: int = 0,
        actions_destroy_result: int = 0,
        attributes_destroy_result: int = 0,
    ) -> mock.Mock:
        library = mock.Mock()
        library.posix_spawnattr_init.return_value = 0
        library.posix_spawnattr_setflags.return_value = 0

        def getflags(_attributes, flags_pointer):
            flags_pointer._obj.value = exact.SPAWN_FLAGS
            return 0

        library.posix_spawnattr_getflags.side_effect = getflags
        library.posix_spawn_file_actions_init.return_value = 0
        library.posix_spawn_file_actions_adddup2.return_value = 0
        library.posix_spawn_file_actions_addclose.return_value = 0

        def spawn(pid_pointer, *_args):
            pid_pointer._obj.value = pid
            return spawn_result

        library.posix_spawn.side_effect = spawn
        library.posix_spawn_file_actions_destroy.return_value = (
            actions_destroy_result
        )
        library.posix_spawnattr_destroy.return_value = (
            attributes_destroy_result
        )
        return library

    def test_spawn_destroy_failure_cleans_spawned_child_and_attempts_both(
        self,
    ) -> None:
        library = self.spawn_library(actions_destroy_result=22)
        child = exact._SpawnedChildState(pid_cell=ctypes.c_int(0))
        with (
            mock.patch.object(
                exact, "_configure_spawn_api", return_value=library
            ),
            mock.patch.object(
                exact,
                "_cleanup_child_after_failure",
                return_value=exact._CleanupOutcome((), None),
            ) as cleanup_child,
            self.assertRaisesRegex(
                exact.ExactnessFailure,
                "posix_spawn_file_actions_destroy failed: 22",
            ),
        ):
            exact._spawn_suspended(
                pathlib.Path("/private/tmp/exactness-emitter"),
                [],
                {},
                {0: 64},
                child,
            )
        cleanup_child.assert_called_once_with(child)
        self.assertEqual(child.pid, 1234)
        self.assertEqual(child.ownership, exact._ChildOwnership.OWNED)
        library.posix_spawn_file_actions_destroy.assert_called_once()
        library.posix_spawnattr_destroy.assert_called_once()

    def test_spawn_attempts_both_destructors_when_both_fail(self) -> None:
        library = self.spawn_library(attributes_destroy_result=17)
        library.posix_spawn_file_actions_destroy.side_effect = RuntimeError(
            "actions destroy interrupted"
        )
        child = exact._SpawnedChildState(pid_cell=ctypes.c_int(0))
        with (
            mock.patch.object(
                exact, "_configure_spawn_api", return_value=library
            ),
            mock.patch.object(
                exact,
                "_cleanup_child_after_failure",
                return_value=exact._CleanupOutcome((), None),
            ) as cleanup_child,
            self.assertRaisesRegex(
                exact.ExactnessFailure,
                "actions destroy interrupted.*posix_spawnattr_destroy failed: 17",
            ),
        ):
            exact._spawn_suspended(
                pathlib.Path("/private/tmp/exactness-emitter"),
                [],
                {},
                {0: 64},
                child,
            )
        cleanup_child.assert_called_once_with(child)
        library.posix_spawn_file_actions_destroy.assert_called_once()
        library.posix_spawnattr_destroy.assert_called_once()

    def test_destructor_keyboard_interrupt_is_preserved(self) -> None:
        library = self.spawn_library()
        interruption = KeyboardInterrupt("destructor interrupt")
        library.posix_spawn_file_actions_destroy.side_effect = interruption
        child = exact._SpawnedChildState(pid_cell=ctypes.c_int(0))
        with (
            mock.patch.object(
                exact, "_configure_spawn_api", return_value=library
            ),
            mock.patch.object(
                exact,
                "_cleanup_child_after_failure",
                return_value=exact._CleanupOutcome((), None),
            ),
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            exact._spawn_suspended(
                pathlib.Path("/private/tmp/exactness-emitter"),
                [],
                {},
                {0: 64},
                child,
            )
        self.assertIs(raised.exception, interruption)
        library.posix_spawnattr_destroy.assert_called_once()

    def test_spawn_failure_does_not_adopt_positive_undefined_pid(self) -> None:
        library = self.spawn_library(spawn_result=13)
        child = exact._SpawnedChildState(pid_cell=ctypes.c_int(0))
        with (
            mock.patch.object(
                exact, "_configure_spawn_api", return_value=library
            ),
            mock.patch.object(
                exact, "_observe_child_nonconsuming"
            ) as observe_child,
            mock.patch.object(exact.os, "killpg") as killpg,
            mock.patch.object(exact.os, "kill") as kill,
            self.assertRaisesRegex(exact.ExactnessFailure, "Permission denied"),
        ):
            exact._spawn_suspended(
                pathlib.Path("/private/tmp/exactness-emitter"),
                [],
                {},
                {0: 64},
                child,
            )
        self.assertEqual(child.spawn_result, 13)
        self.assertEqual(child.pid_cell.value, 1234)
        self.assertEqual(child.pid, -1)
        self.assertEqual(child.ownership, exact._ChildOwnership.NO_CHILD)
        observe_child.assert_not_called()
        killpg.assert_not_called()
        kill.assert_not_called()
        library.posix_spawn_file_actions_destroy.assert_called_once()
        library.posix_spawnattr_destroy.assert_called_once()

    def test_spawn_rejects_a_multithreaded_supervisor_before_setup(self) -> None:
        child = exact._SpawnedChildState(pid_cell=ctypes.c_int(0))
        with (
            mock.patch.object(exact.threading, "active_count", return_value=2),
            mock.patch.object(exact, "_configure_spawn_api") as configure,
            self.assertRaisesRegex(
                exact.ExactnessFailure,
                "single-threaded supervisor",
            ),
        ):
            exact._spawn_suspended(
                pathlib.Path("/usr/bin/true"),
                [],
                {},
                {},
                child,
            )
        configure.assert_not_called()
        self.assertEqual(child.ownership, exact._ChildOwnership.UNSTARTED)

    def test_spawn_rejects_preexisting_direct_children(self) -> None:
        library = self.spawn_library()
        child = exact._SpawnedChildState(pid_cell=ctypes.c_int(0))
        with (
            mock.patch.object(
                exact, "_configure_spawn_api", return_value=library
            ),
            mock.patch.object(
                exact.d0,
                "process_child_pids",
                return_value=[4321],
            ),
            self.assertRaisesRegex(
                exact.ExactnessFailure,
                "already owns child processes",
            ),
        ):
            exact._spawn_suspended(
                pathlib.Path("/usr/bin/true"),
                [],
                {},
                {},
                child,
            )
        library.posix_spawn.assert_not_called()
        self.assertEqual(child.ownership, exact._ChildOwnership.NO_CHILD)

    def test_pid_written_then_spawn_interrupted_is_recovered_and_cleaned(
        self,
    ) -> None:
        pid = 1234
        library = self.spawn_library(pid=pid)

        def interrupted_spawn(pid_pointer, *_args):
            pid_pointer._obj.value = pid
            raise KeyboardInterrupt()

        library.posix_spawn.side_effect = interrupted_spawn
        child = exact._SpawnedChildState(
            pid_cell=ctypes.c_int(0),
            expected_executable=self.expected_executable(),
        )
        with (
            mock.patch.object(
                exact, "_configure_spawn_api", return_value=library
            ),
            mock.patch.object(
                exact,
                "_observe_child_nonconsuming",
                side_effect=[
                    exact._WaitObservation(pid, exact.CLD_STOPPED, 19),
                    exact._WaitObservation(
                        pid,
                        exact.CLD_KILLED,
                        exact.signal.SIGKILL,
                    ),
                ],
            ),
            mock.patch.object(
                exact.d0,
                "capture_process_identity",
                return_value=self.process_record(pid),
            ),
            mock.patch.object(exact.os, "getpgid", return_value=pid),
            mock.patch.object(exact.os, "getsid", return_value=pid),
            mock.patch.object(exact.os, "killpg") as killpg,
            mock.patch.object(exact, "_group_member_pids", return_value=[pid]),
            mock.patch.object(
                exact.os,
                "waitpid",
                return_value=(pid, exact.signal.SIGKILL),
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            exact._spawn_suspended(
                pathlib.Path("/private/tmp/exactness-emitter"),
                [],
                {},
                {0: 64},
                child,
            )
        killpg.assert_called_once_with(pid, exact.signal.SIGKILL)
        self.assertEqual(child.ownership, exact._ChildOwnership.OWNED)
        self.assertEqual(child.lifecycle, exact._ChildLifecycle.REAPED)
        self.assertEqual(child.group_state, exact._ChildGroupState.EMPTY)

    def test_interrupt_before_pid_assignment_recovers_known_success(
        self,
    ) -> None:
        pid = 1234
        library = self.spawn_library(pid=pid)
        child = exact._SpawnedChildState(
            pid_cell=ctypes.c_int(0),
            expected_executable=self.expected_executable(),
        )
        original_spawn_call = exact._spawn_call

        def interrupt_after_result(result, operation):
            if operation == "posix_spawn":
                raise KeyboardInterrupt()
            original_spawn_call(result, operation)

        with (
            mock.patch.object(
                exact, "_configure_spawn_api", return_value=library
            ),
            mock.patch.object(
                exact, "_spawn_call", side_effect=interrupt_after_result
            ),
            mock.patch.object(
                exact,
                "_observe_child_nonconsuming",
                return_value=exact._WaitObservation(
                    pid,
                    exact.CLD_KILLED,
                    exact.signal.SIGKILL,
                ),
            ),
            mock.patch.object(
                exact.d0,
                "capture_process_identity",
                return_value=self.process_record(pid),
            ),
            mock.patch.object(exact.os, "getpgid", return_value=pid),
            mock.patch.object(exact.os, "getsid", return_value=pid),
            mock.patch.object(exact.os, "killpg") as killpg,
            mock.patch.object(exact, "_group_member_pids", return_value=[pid]),
            mock.patch.object(
                exact.os,
                "waitpid",
                return_value=(pid, exact.signal.SIGKILL),
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            exact._spawn_suspended(
                pathlib.Path("/private/tmp/exactness-emitter"),
                [],
                {},
                {0: 64},
                child,
            )
        self.assertEqual(child.spawn_result, 0)
        self.assertEqual(child.pid, pid)
        killpg.assert_called_once_with(pid, exact.signal.SIGKILL)
        self.assertEqual(child.lifecycle, exact._ChildLifecycle.REAPED)
        self.assertEqual(child.group_state, exact._ChildGroupState.EMPTY)

    def test_indeterminate_spawn_identity_mismatch_is_never_signaled(
        self,
    ) -> None:
        child = exact._SpawnedChildState(
            pid_cell=ctypes.c_int(1234),
            expected_executable=self.expected_executable(),
            ownership=exact._ChildOwnership.SPAWN_INDETERMINATE,
        )
        mismatched = self.process_record(1234)
        mismatched["executable_inode"] += 1
        with (
            mock.patch.object(
                exact,
                "_observe_child_nonconsuming",
                return_value=exact._WaitObservation(
                    1234,
                    exact.CLD_STOPPED,
                    19,
                ),
            ),
            mock.patch.object(
                exact.d0,
                "capture_process_identity",
                return_value=mismatched,
            ),
            mock.patch.object(exact.os, "getpgid", return_value=1234),
            mock.patch.object(exact.os, "getsid", return_value=1234),
            mock.patch.object(exact.os, "killpg") as killpg,
            mock.patch.object(exact.os, "kill") as kill,
        ):
            errors = exact._cleanup_child(child)
        self.assertIn("candidate identity differs", "; ".join(errors))
        self.assertEqual(child.ownership, exact._ChildOwnership.UNRESOLVED)
        killpg.assert_not_called()
        kill.assert_not_called()

    def test_indeterminate_spawn_never_adopts_a_preexisting_child(self) -> None:
        child = exact._SpawnedChildState(
            pid_cell=ctypes.c_int(1234),
            expected_executable=self.expected_executable(),
            preexisting_child_pids=(1234,),
            ownership=exact._ChildOwnership.SPAWN_INDETERMINATE,
        )
        with (
            mock.patch.object(
                exact,
                "_observe_child_nonconsuming",
                return_value=exact._WaitObservation(
                    1234,
                    exact.CLD_STOPPED,
                    19,
                ),
            ),
            mock.patch.object(
                exact.d0,
                "capture_process_identity",
                return_value=self.process_record(1234),
            ),
            mock.patch.object(exact.os, "killpg") as killpg,
            mock.patch.object(exact.os, "kill") as kill,
        ):
            errors = exact._cleanup_child(child)
        self.assertIn("candidate identity differs", "; ".join(errors))
        self.assertEqual(child.ownership, exact._ChildOwnership.UNRESOLVED)
        killpg.assert_not_called()
        kill.assert_not_called()

    def test_indeterminate_spawn_echild_proves_no_owned_child(self) -> None:
        child = exact._SpawnedChildState(
            pid_cell=ctypes.c_int(1234),
            expected_executable=self.expected_executable(),
            ownership=exact._ChildOwnership.SPAWN_INDETERMINATE,
        )
        with (
            mock.patch.object(
                exact,
                "_observe_child_nonconsuming",
                side_effect=ChildProcessError(),
            ),
            mock.patch.object(exact.os, "killpg") as killpg,
            mock.patch.object(exact.os, "kill") as kill,
        ):
            self.assertEqual(exact._cleanup_child(child), [])
        self.assertEqual(child.ownership, exact._ChildOwnership.NO_CHILD)
        killpg.assert_not_called()
        kill.assert_not_called()

    def test_wait_for_stopped_publishes_terminal_before_failure(
        self,
    ) -> None:
        child = self.owned_child(lifecycle=exact._ChildLifecycle.UNOBSERVED)
        with (
            mock.patch.object(
                exact,
                "_observe_child_nonconsuming",
                return_value=exact._WaitObservation(
                    1234,
                    exact.CLD_EXITED,
                    0,
                ),
            ),
            self.assertRaisesRegex(
                exact.ExactnessFailure,
                "terminated before its suspended attestation",
            ),
        ):
            exact._wait_for_stopped(child, deadline=exact.time.monotonic() + 1)
        self.assertEqual(
            child.lifecycle,
            exact._ChildLifecycle.TERMINAL_OBSERVED,
        )
        self.assertEqual(child.terminal_code, exact.CLD_EXITED)
        self.assertEqual(child.terminal_status, 0)

    def test_wait_for_exit_publishes_signal_before_failure(
        self,
    ) -> None:
        child = self.owned_child()
        with (
            mock.patch.object(
                exact,
                "_observe_child_nonconsuming",
                return_value=exact._WaitObservation(
                    1234,
                    exact.CLD_KILLED,
                    exact.signal.SIGKILL,
                ),
            ),
            self.assertRaisesRegex(
                exact.ExactnessFailure,
                "terminated by a signal",
            ),
        ):
            exact._wait_for_exit(child, deadline=exact.time.monotonic() + 1)
        self.assertEqual(
            child.lifecycle,
            exact._ChildLifecycle.TERMINAL_OBSERVED,
        )
        self.assertEqual(child.terminal_code, exact.CLD_KILLED)

    def test_cleanup_retry_preserves_interruption_after_success(self) -> None:
        child = self.owned_child()
        attempt = 0
        interruption = KeyboardInterrupt("cleanup interrupt")

        def cleanup(state):
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise interruption
            state.lifecycle = exact._ChildLifecycle.REAPED
            state.group_state = exact._ChildGroupState.EMPTY
            return []

        with mock.patch.object(exact, "_cleanup_child", side_effect=cleanup):
            outcome = exact._cleanup_child_after_failure(child)
        self.assertEqual(attempt, 2)
        self.assertEqual(outcome.errors, ())
        self.assertIs(outcome.interruption, interruption)
        self.assertEqual(child.lifecycle, exact._ChildLifecycle.REAPED)
        self.assertEqual(child.group_state, exact._ChildGroupState.EMPTY)

    def test_run_one_retries_internal_cleanup_with_same_state(self) -> None:
        pid = 1234
        dataset = b"frozen dataset"
        emitter = b"Mach-O fixture"
        captured_states = []
        cleanup_states = []
        primary_interrupt = KeyboardInterrupt("spawn interrupt")
        cleanup_interrupt = KeyboardInterrupt("cleanup interrupt")
        library = self.spawn_library(pid=pid)

        def interrupted_spawn(pid_pointer, *_args):
            pid_pointer._obj.value = pid
            raise primary_interrupt

        def cleanup(child):
            cleanup_states.append(child)
            if len(cleanup_states) == 1:
                captured_states.append(child)
                raise cleanup_interrupt
            if len(cleanup_states) == 2:
                return ["internal cleanup incomplete"]
            child.ownership = exact._ChildOwnership.OWNED
            child.pid = pid
            child.lifecycle = exact._ChildLifecycle.REAPED
            child.group_state = exact._ChildGroupState.EMPTY
            return []

        library.posix_spawn.side_effect = interrupted_spawn
        with tempfile.TemporaryDirectory() as directory_text:
            directory = pathlib.Path(directory_text)
            executable = directory / "exactness-emitter"
            executable.write_bytes(emitter)
            dataset_path = directory / "dataset.bin"
            dataset_path.write_bytes(dataset)
            output_dir = directory / "evidence"
            output_dir.mkdir()
            directory_descriptor = exact.os.open(
                output_dir,
                exact.os.O_RDONLY | exact.os.O_DIRECTORY,
            )
            try:
                with (
                    mock.patch.object(exact, "DATASET_BYTES", len(dataset)),
                    mock.patch.object(
                        exact,
                        "DATASET_SHA256",
                        hashlib.sha256(dataset).hexdigest(),
                    ),
                    mock.patch.object(
                        exact, "_configure_spawn_api", return_value=library
                    ),
                    mock.patch.object(
                        exact, "_cleanup_child", side_effect=cleanup
                    ),
                    self.assertRaises(KeyboardInterrupt) as raised,
                ):
                    exact._run_one(
                        sequence=0,
                        role="control",
                        repetition=0,
                        executable=executable,
                        dataset_path=dataset_path,
                        dataset=dataset,
                        output_dir=output_dir,
                        directory_descriptor=directory_descriptor,
                        timeout_seconds=1.0,
                        expected_emitter={
                            "path": str(executable),
                            "sha256": hashlib.sha256(emitter).hexdigest(),
                            "bytes": len(emitter),
                        },
                        expected_witnesses=(
                            THRESHOLD_ONLY_EXPECTED_WITNESSES
                        ),
                    )
            finally:
                exact.os.close(directory_descriptor)

        self.assertIs(raised.exception, primary_interrupt)
        notes = (
            *getattr(raised.exception, "__notes__", ()),
            *getattr(raised.exception, "_infinity_notes", ()),
        )
        self.assertTrue(
            any(
                "cleanup raised KeyboardInterrupt" in note
                for note in notes
            )
        )
        self.assertEqual(len(captured_states), 1)
        self.assertEqual(
            cleanup_states,
            [captured_states[0], captured_states[0], captured_states[0]],
        )
        self.assertEqual(
            captured_states[0].lifecycle,
            exact._ChildLifecycle.REAPED,
        )
        self.assertEqual(
            captured_states[0].group_state,
            exact._ChildGroupState.EMPTY,
        )

    def test_darwin_siginfo_layout_matches_frozen_abi(self) -> None:
        self.assertEqual(ctypes.sizeof(exact._DarwinSiginfo), 104)
        self.assertEqual(exact._DarwinSiginfo.si_pid.offset, 12)
        self.assertEqual(exact._DarwinSiginfo.si_status.offset, 20)


if __name__ == "__main__":
    unittest.main()
