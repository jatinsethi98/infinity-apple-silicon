from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from unittest import mock

from tools.apple_silicon import check_apple_execution_policy as gate


class FakeSystem:
    """A complete gate backend that never invokes a host command."""

    def __init__(self) -> None:
        self.system = "Darwin"
        self.architecture = "arm64"
        self.now = 0.0
        self.resolved_path = "/Applications/Xcode.app/Contents/Developer"
        self.resolve_error: Optional[BaseException] = None
        self.command_results: Dict[
            Tuple[str, ...],
            List[Any],
        ] = {}
        self.command_calls: List[Tuple[Tuple[str, ...], float]] = []
        self.sleep_calls: List[float] = []
        self.create_error: Optional[BaseException] = None
        self.execute_error: Optional[BaseException] = None
        self.cleanup_error: Optional[BaseException] = None
        self.cleanup_calls = 0
        digest = "a" * 64
        self.material = gate.CanaryMaterial(
            directory=Path("/private/tmp/fake-private-directory"),
            executable=Path("/private/tmp/fake-private-directory/true-canary"),
            directory_device=1,
            directory_inode=2,
            evidence={
                "bytes": 4096,
                "copy_mode": "0700",
                "copy_name": gate.CANARY_FILENAME,
                "copy_sha256": digest,
                "private_directory_mode": "0700",
                "source_mode": "0755",
                "source_path": str(gate.TRUE_PATH),
                "source_sha256": digest,
                "verified_byte_equal": True,
            },
        )
        empty_digest = gate._sha256(b"")
        self.execution: Dict[str, Any] = {
            "new_process_group": True,
            "new_session": True,
            "process_group_leader_pid": 4242,
            "process_group_member_pids_before_reap": [4242],
            "process_group_quiescent_before_reap": True,
            "process_group_required_forced_cleanup": False,
            "reaped": True,
            "returncode": 0,
            "stderr_bytes": 0,
            "stderr_sha256": empty_digest,
            "stdout_bytes": 0,
            "stdout_sha256": empty_digest,
            "timed_out": False,
            "timeout_milliseconds": 5000,
        }
        self._install_valid_commands()

    @staticmethod
    def result(
        command: Sequence[str],
        *,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> gate.CommandResult:
        return gate.CommandResult(tuple(command), returncode, stdout, stderr)

    def set_results(
        self,
        command: Sequence[str],
        results: Sequence[Any],
    ) -> None:
        self.command_results[tuple(command)] = list(results)

    def set_result(
        self,
        command: Sequence[str],
        *,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        self.set_results(
            command,
            [
                self.result(
                    command,
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                )
            ],
        )

    def set_cpu_samples(
        self,
        values: Sequence[str],
        *,
        pid: int = 771,
        command_name: str = "/usr/libexec/syspolicyd",
    ) -> None:
        command = (
            "/bin/ps",
            "-p",
            str(pid),
            "-o",
            "pid=,%cpu=,comm=",
        )
        self.set_results(
            command,
            [
                self.result(
                    command,
                    stdout=(
                        "%d %s %s\n" % (pid, value, command_name)
                    ).encode("ascii"),
                )
                for value in values
            ],
        )

    def _install_valid_commands(self) -> None:
        self.set_result(
            gate.SYSCTL_COMMAND,
            stdout=(
                "{ sec = %d, usec = 1 } Sat Aug 22 02:52:11 2026\n"
                % gate.BOOT_CUTOFF_SECONDS
            ).encode("ascii"),
        )
        self.set_result(
            gate.GATEKEEPER_COMMAND,
            stdout=b"assessments enabled\n",
        )
        self.set_result(
            gate.SIP_COMMAND,
            stdout=b"System Integrity Protection status: enabled.\n",
        )
        self.set_result(
            gate.XCODE_SELECT_COMMAND,
            stdout=b"/Applications/Xcode.app/Contents/Developer\n",
        )
        self.set_result(gate.XCODE_FIRST_LAUNCH_COMMAND)
        self.set_result(gate.XCODE_LICENSE_COMMAND)
        self.set_result(
            gate.SYSPOLICY_DISCOVERY_COMMAND,
            stdout=b"771 /usr/libexec/syspolicyd\n",
        )
        self.set_cpu_samples(["12.0", "13.0", "14.0", "15.0", "16.0"])
        self.set_result(
            gate.SYSPOLICY_LOG_COMMAND,
            stdout=b"Timestamp                       Thread     Type\n",
        )

    def system_name(self) -> str:
        return self.system

    def machine(self) -> str:
        return self.architecture

    def run_command(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
    ) -> gate.CommandResult:
        key = tuple(argv)
        self.command_calls.append((key, timeout_seconds))
        queue = self.command_results.get(key)
        if not queue:
            raise AssertionError("Unexpected or exhausted command: %r" % (key,))
        result = queue.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def resolve_directory(self, path: str) -> str:
        if self.resolve_error is not None:
            raise self.resolve_error
        if path != "/Applications/Xcode.app/Contents/Developer":
            raise AssertionError("Unexpected Xcode path: %s" % path)
        return self.resolved_path

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds

    def create_verified_canary(self) -> gate.CanaryMaterial:
        if self.create_error is not None:
            raise self.create_error
        return self.material

    def execute_verified_canary(
        self,
        material: gate.CanaryMaterial,
        *,
        timeout_seconds: float,
    ) -> Dict[str, Any]:
        self.assert_material(material)
        if timeout_seconds != gate.CANARY_TIMEOUT_SECONDS:
            raise AssertionError("Unexpected canary timeout")
        if self.execute_error is not None:
            raise self.execute_error
        return dict(self.execution)

    def cleanup_verified_canary(
        self,
        material: gate.CanaryMaterial,
    ) -> Dict[str, Any]:
        self.assert_material(material)
        self.cleanup_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error
        return {"private_directory_removed": True}

    def assert_material(self, material: gate.CanaryMaterial) -> None:
        if material != self.material:
            raise AssertionError("Unexpected canary material")


class SuccessfulGateTests(unittest.TestCase):
    def test_complete_gate_passes_with_canonical_deterministic_json(self) -> None:
        first_stdout = io.BytesIO()
        first_stderr = io.BytesIO()
        first = FakeSystem()

        returncode = gate.main(
            [],
            operations=first,
            stdout=first_stdout,
            stderr=first_stderr,
        )

        self.assertEqual(returncode, 0)
        self.assertEqual(first_stderr.getvalue(), b"")
        record = json.loads(first_stdout.getvalue())
        self.assertEqual(record["status"], "PASS")
        self.assertTrue(record["read_only"])
        self.assertEqual(record["schema_version"], 1)
        self.assertTrue(record["checks"]["boot_time"]["strictly_after_cutoff"])
        self.assertEqual(
            record["checks"]["syspolicyd_cpu"]["samples_percent"],
            ["12", "13", "14", "15", "16"],
        )
        self.assertTrue(
            record["checks"]["canary_execution"][
                "process_group_quiescent_before_reap"
            ]
        )
        self.assertEqual(
            first_stdout.getvalue(),
            gate.canonical_json_bytes(record),
        )

        second_stdout = io.BytesIO()
        self.assertEqual(
            gate.main(
                [],
                operations=FakeSystem(),
                stdout=second_stdout,
                stderr=io.BytesIO(),
            ),
            0,
        )
        self.assertEqual(second_stdout.getvalue(), first_stdout.getvalue())

    def test_only_fixed_read_only_commands_are_requested(self) -> None:
        fake = FakeSystem()
        gate.run_gate(fake)

        commands = [call[0] for call in fake.command_calls]
        expected_prefix = [
            gate.SYSCTL_COMMAND,
            gate.GATEKEEPER_COMMAND,
            gate.SIP_COMMAND,
            gate.XCODE_SELECT_COMMAND,
            gate.XCODE_FIRST_LAUNCH_COMMAND,
            gate.XCODE_LICENSE_COMMAND,
            gate.SYSPOLICY_DISCOVERY_COMMAND,
        ]
        self.assertEqual(commands[: len(expected_prefix)], expected_prefix)
        self.assertEqual(commands[-1], gate.SYSPOLICY_LOG_COMMAND)
        self.assertEqual(
            commands[len(expected_prefix) : -1],
            [
                (
                    "/bin/ps",
                    "-p",
                    "771",
                    "-o",
                    "pid=,%cpu=,comm=",
                )
            ]
            * gate.CPU_SAMPLE_COUNT,
        )
        flattened = "\n".join(" ".join(command) for command in commands)
        for forbidden in (
            "DevToolsSecurity",
            "developer-mode",
            "disable",
            "kickstart",
            "launchctl",
            "xattr",
            "systemextensionsctl",
        ):
            self.assertNotIn(forbidden, flattened)

    def test_one_sub_threshold_cpu_sample_is_not_persistent(self) -> None:
        fake = FakeSystem()
        fake.set_cpu_samples(["99.0", "100.0", "89.9", "101.0", "99.0"])

        record = gate.run_gate(fake)

        self.assertFalse(
            record["checks"]["syspolicyd_cpu"][
                "persistent_at_or_above_threshold"
            ]
        )
        self.assertEqual(fake.sleep_calls, [1.0, 1.0, 1.0, 1.0])

    def test_discovery_accepts_unrelated_process_names_with_spaces(self) -> None:
        fake = FakeSystem()
        fake.set_result(
            gate.SYSPOLICY_DISCOVERY_COMMAND,
            stdout=(
                b"781 Core Audio Driver (Example.driver)\n"
                b"771 /usr/libexec/syspolicyd\n"
                b"798 /Library/Application Support/Example/Agent\n"
            ),
        )

        record = gate.run_gate(fake)

        self.assertEqual(record["checks"]["syspolicyd_discovery"]["pid"], 771)


class GateFailureTests(unittest.TestCase):
    def assert_failure(
        self,
        fake: FakeSystem,
        expected_check: str,
    ) -> gate.GateFailure:
        with self.assertRaises(gate.GateFailure) as context:
            gate.run_gate(fake)
        self.assertEqual(context.exception.check, expected_check)
        self.assertEqual(
            context.exception.exit_code,
            gate.FAILURE_EXIT_CODES[expected_check],
        )
        return context.exception

    def test_rejects_non_macos_and_non_arm64(self) -> None:
        for system, machine in (("Linux", "arm64"), ("Darwin", "x86_64")):
            with self.subTest(system=system, machine=machine):
                fake = FakeSystem()
                fake.system = system
                fake.architecture = machine
                self.assert_failure(fake, "platform")
                self.assertEqual(fake.command_calls, [])

    def test_boot_time_must_be_strictly_after_cutoff(self) -> None:
        for seconds, microseconds in (
            (gate.BOOT_CUTOFF_SECONDS - 1, 999999),
            (gate.BOOT_CUTOFF_SECONDS, 0),
        ):
            with self.subTest(seconds=seconds, microseconds=microseconds):
                fake = FakeSystem()
                fake.set_result(
                    gate.SYSCTL_COMMAND,
                    stdout=(
                        "{ sec = %d, usec = %d } cutoff\n"
                        % (seconds, microseconds)
                    ).encode("ascii"),
                )
                self.assert_failure(fake, "boot_time")

        fake = FakeSystem()
        fake.set_result(
            gate.SYSCTL_COMMAND,
            stdout=(
                "{ sec = %d, usec = 1 } after\n"
                % gate.BOOT_CUTOFF_SECONDS
            ).encode("ascii"),
        )
        self.assertEqual(gate.run_gate(fake)["status"], "PASS")

    def test_rejects_malformed_boot_time(self) -> None:
        for output in (
            b"",
            b"not a boot time\n",
            b"{ sec = 1, usec = 1000000 }\n",
            b"{ sec = 1, usec = 0 }\nsecond line\n",
        ):
            with self.subTest(output=output):
                fake = FakeSystem()
                fake.set_result(gate.SYSCTL_COMMAND, stdout=output)
                self.assert_failure(fake, "boot_time")

    def test_gatekeeper_must_report_exact_enabled_state(self) -> None:
        for output in (
            b"assessments disabled\n",
            b"assessments enabled\nextra\n",
            b"Assessments enabled\n",
        ):
            with self.subTest(output=output):
                fake = FakeSystem()
                fake.set_result(gate.GATEKEEPER_COMMAND, stdout=output)
                self.assert_failure(fake, "gatekeeper")

    def test_sip_must_report_exact_enabled_state(self) -> None:
        fake = FakeSystem()
        fake.set_result(
            gate.SIP_COMMAND,
            stdout=b"System Integrity Protection status: disabled.\n",
        )
        self.assert_failure(fake, "sip")

    def test_nonzero_and_timeout_commands_fail_their_own_gate(self) -> None:
        fake = FakeSystem()
        fake.set_result(gate.XCODE_FIRST_LAUNCH_COMMAND, returncode=1)
        self.assert_failure(fake, "xcode_first_launch")

        fake = FakeSystem()
        fake.set_results(
            gate.GATEKEEPER_COMMAND,
            [
                subprocess.TimeoutExpired(
                    list(gate.GATEKEEPER_COMMAND),
                    gate.COMMAND_TIMEOUT_SECONDS,
                )
            ],
        )
        self.assert_failure(fake, "gatekeeper")

    def test_xcode_select_must_be_absolute_existing_directory(self) -> None:
        fake = FakeSystem()
        fake.set_result(gate.XCODE_SELECT_COMMAND, stdout=b"relative/path\n")
        self.assert_failure(fake, "xcode_select")

        fake = FakeSystem()
        fake.resolve_error = FileNotFoundError("missing")
        self.assert_failure(fake, "xcode_select")

    def test_xcode_license_check_must_pass(self) -> None:
        fake = FakeSystem()
        fake.set_result(
            gate.XCODE_LICENSE_COMMAND,
            returncode=69,
            stderr=b"license not accepted\n",
        )
        self.assert_failure(fake, "xcode_license")

    def test_syspolicyd_discovery_requires_one_exact_process(self) -> None:
        outputs = (
            b"",
            b"771 /usr/libexec/syspolicyd\n772 syspolicyd\n",
            b"not-a-pid /usr/libexec/syspolicyd\n",
            b"771 /usr/libexec/not-syspolicyd\n",
        )
        for output in outputs:
            with self.subTest(output=output):
                fake = FakeSystem()
                fake.set_result(
                    gate.SYSPOLICY_DISCOVERY_COMMAND,
                    stdout=output,
                )
                self.assert_failure(fake, "syspolicyd_discovery")

    def test_persistent_ninety_percent_cpu_is_rejected(self) -> None:
        fake = FakeSystem()
        fake.set_cpu_samples(["90.0", "91.0", "105.5", "100.0", "99.0"])

        error = self.assert_failure(fake, "syspolicyd_cpu")

        cpu = error.checks["syspolicyd_cpu"]
        self.assertTrue(cpu["persistent_at_or_above_threshold"])
        self.assertEqual(cpu["threshold_percent"], "90")

    def test_cpu_identity_format_range_and_budget_fail_closed(self) -> None:
        cases = (
            ("identity", ["10"], 772, "/usr/libexec/syspolicyd"),
            ("name", ["10"], 771, "/usr/libexec/other"),
            ("range", ["1201"], 771, "/usr/libexec/syspolicyd"),
        )
        for label, values, pid, command_name in cases:
            with self.subTest(label=label):
                fake = FakeSystem()
                sample_command = (
                    "/bin/ps",
                    "-p",
                    "771",
                    "-o",
                    "pid=,%cpu=,comm=",
                )
                fake.set_results(
                    sample_command,
                    [
                        fake.result(
                            sample_command,
                            stdout=(
                                "%d %s %s\n"
                                % (pid, values[0], command_name)
                            ).encode("ascii"),
                        )
                    ],
                )
                self.assert_failure(fake, "syspolicyd_cpu")

        fake = FakeSystem()
        fake.now = 100.0

        def excessive_sleep(seconds: float) -> None:
            fake.sleep_calls.append(seconds)
            fake.now += 3.0

        fake.sleep = excessive_sleep  # type: ignore[assignment]
        self.assert_failure(fake, "syspolicyd_cpu")

    def test_each_forbidden_fresh_log_message_is_rejected(self) -> None:
        for message in gate.FORBIDDEN_LOG_MESSAGES:
            with self.subTest(message=message):
                fake = FakeSystem()
                fake.set_result(
                    gate.SYSPOLICY_LOG_COMMAND,
                    stdout=("timestamp " + message + "\n").encode("ascii"),
                )
                error = self.assert_failure(fake, "syspolicyd_logs")
                self.assertEqual(
                    error.checks["syspolicyd_logs"][
                        "forbidden_match_counts"
                    ][message],
                    1,
                )

    def test_forbidden_log_message_on_stderr_is_also_rejected(self) -> None:
        fake = FakeSystem()
        fake.set_result(
            gate.SYSPOLICY_LOG_COMMAND,
            stderr=b"Unable to initialize qtn_proc\n",
        )
        self.assert_failure(fake, "syspolicyd_logs")

    def test_canary_copy_must_be_verified_and_creation_is_fail_closed(self) -> None:
        fake = FakeSystem()
        evidence = dict(fake.material.evidence)
        evidence["copy_sha256"] = "b" * 64
        fake.material = gate.CanaryMaterial(
            fake.material.directory,
            fake.material.executable,
            fake.material.directory_device,
            fake.material.directory_inode,
            evidence,
        )
        self.assert_failure(fake, "canary_copy")
        self.assertEqual(fake.cleanup_calls, 1)

        fake = FakeSystem()
        fake.create_error = OSError("copy failed")
        self.assert_failure(fake, "canary_copy")
        self.assertEqual(fake.cleanup_calls, 0)

    def test_canary_timeout_nonzero_or_output_is_rejected_and_cleaned(self) -> None:
        mutations: Sequence[Mapping[str, Any]] = (
            {"timed_out": True},
            {"returncode": 1},
            {"stdout_bytes": 1},
            {"stderr_bytes": 1},
            {"timeout_milliseconds": 5001},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                fake = FakeSystem()
                fake.execution.update(mutation)
                self.assert_failure(fake, "canary_execution")
                self.assertEqual(fake.cleanup_calls, 1)

    def test_canary_requires_empty_unassisted_process_group(self) -> None:
        for mutation in (
            {"process_group_quiescent_before_reap": False},
            {"process_group_member_pids_before_reap": [4242, 4243]},
            {"process_group_required_forced_cleanup": True},
        ):
            with self.subTest(mutation=mutation):
                fake = FakeSystem()
                fake.execution.update(mutation)
                self.assert_failure(fake, "canary_process_group")
                self.assertEqual(fake.cleanup_calls, 1)

    def test_canary_execution_exception_is_rejected_and_cleaned(self) -> None:
        fake = FakeSystem()
        fake.execute_error = OSError("spawn failed")
        self.assert_failure(fake, "canary_execution")
        self.assertEqual(fake.cleanup_calls, 1)

    def test_cleanup_failure_is_distinct_and_dominates_prior_failure(self) -> None:
        fake = FakeSystem()
        fake.execution["returncode"] = 1
        fake.cleanup_error = OSError("cleanup failed")

        error = self.assert_failure(fake, "canary_cleanup")

        self.assertIn("earlier failure", str(error))
        self.assertEqual(fake.cleanup_calls, 1)


class CanaryFilesystemTests(unittest.TestCase):
    def test_synthetic_source_is_copied_byte_exactly_and_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source-true"
            source.write_bytes(b"synthetic Mach-O bytes")
            source.chmod(0o755)
            private_root = root / "private"
            private_root.mkdir()

            material = gate.create_verified_canary(
                source=source,
                temporary_root=private_root,
            )
            try:
                self.assertEqual(
                    material.executable.read_bytes(),
                    source.read_bytes(),
                )
                self.assertEqual(
                    material.evidence["source_sha256"],
                    material.evidence["copy_sha256"],
                )
                self.assertEqual(
                    material.evidence["private_directory_mode"],
                    "0700",
                )
            finally:
                cleanup = gate.cleanup_verified_canary(material)
            self.assertEqual(cleanup, {"private_directory_removed": True})
            self.assertFalse(os.path.lexists(str(material.directory)))

    def test_symlink_source_is_rejected_without_leaving_private_directory(self) -> None:
        if not hasattr(os, "O_NOFOLLOW"):
            self.skipTest("O_NOFOLLOW is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"target")
            target.chmod(0o755)
            source = root / "source"
            source.symlink_to(target)
            private_root = root / "private"
            private_root.mkdir()

            with self.assertRaises(OSError):
                gate.create_verified_canary(
                    source=source,
                    temporary_root=private_root,
                )

            self.assertEqual(list(private_root.iterdir()), [])

    def test_nonexecutable_source_is_rejected_before_temp_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(b"bytes")
            source.chmod(0o600)
            private_root = root / "private"
            private_root.mkdir()

            with self.assertRaisesRegex(RuntimeError, "not executable"):
                gate.create_verified_canary(
                    source=source,
                    temporary_root=private_root,
                )

            self.assertEqual(list(private_root.iterdir()), [])


class ProcessSupervisionTests(unittest.TestCase):
    class ImmediateProcess:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.pid = 4242
            self.returncode = None
            self.kill_calls = 0

        def kill(self) -> None:
            self.kill_calls += 1

    def test_bounded_child_requests_new_session_and_closed_descriptors(self) -> None:
        created: List[ProcessSupervisionTests.ImmediateProcess] = []

        def factory(*args: Any, **kwargs: Any) -> Any:
            del args
            process = self.ImmediateProcess(**kwargs)
            created.append(process)
            return process

        with (
            mock.patch.object(gate.subprocess, "Popen", side_effect=factory),
            mock.patch.object(gate.os, "getpgid", return_value=4242),
            mock.patch.object(
                gate,
                "_observe_terminal_nonconsuming",
                return_value=gate._TerminalObservation(
                    4242,
                    gate.CLD_EXITED,
                    0,
                ),
            ),
            mock.patch.object(
                gate,
                "_group_member_pids",
                return_value=[4242],
            ),
            mock.patch.object(gate.os, "waitpid", return_value=(4242, 0)),
        ):
            result = gate._bounded_popen(
                ("/fixture/true",),
                cwd="/fixture",
                timeout_seconds=5.0,
            )

        self.assertEqual(result, (0, b"", b"", False, False, [4242]))
        self.assertTrue(created[0].kwargs["start_new_session"])
        self.assertTrue(created[0].kwargs["close_fds"])
        self.assertIs(created[0].kwargs["stdin"], subprocess.DEVNULL)
        self.assertNotEqual(created[0].kwargs["stdout"], subprocess.PIPE)
        self.assertNotEqual(created[0].kwargs["stderr"], subprocess.PIPE)

    def test_timeout_kills_group_reaps_and_reports_timeout(self) -> None:
        process = self.ImmediateProcess()
        terminal_calls = 0

        def wait_for_terminal(
            child: gate._BoundedChild,
            *,
            deadline: float,
        ) -> bool:
            del deadline
            nonlocal terminal_calls
            terminal_calls += 1
            if terminal_calls == 1:
                return False
            child.terminal = gate._TerminalObservation(
                4242,
                gate.CLD_KILLED,
                gate.signal.SIGKILL,
            )
            return True

        with (
            mock.patch.object(gate.subprocess, "Popen", return_value=process),
            mock.patch.object(gate.os, "getpgid", return_value=4242),
            mock.patch.object(
                gate,
                "_wait_for_terminal",
                side_effect=wait_for_terminal,
            ),
            mock.patch.object(
                gate,
                "_group_member_pids",
                return_value=[4242],
            ),
            mock.patch.object(gate, "_kill_group") as kill_group,
            mock.patch.object(
                gate.os,
                "waitpid",
                return_value=(4242, gate.signal.SIGKILL),
            ),
        ):
            result = gate._bounded_popen(
                ("/fixture/true",),
                cwd="/fixture",
                timeout_seconds=5.0,
            )

        self.assertEqual(result, (-9, b"", b"", True, True, [4242]))
        kill_group.assert_called_once_with(4242)
        self.assertEqual(terminal_calls, 2)

    def test_interrupted_reap_never_performs_a_later_group_syscall(self) -> None:
        process = self.ImmediateProcess()
        with (
            mock.patch.object(gate.subprocess, "Popen", return_value=process),
            mock.patch.object(gate.os, "getpgid", return_value=4242) as getpgid,
            mock.patch.object(
                gate,
                "_observe_terminal_nonconsuming",
                return_value=gate._TerminalObservation(
                    4242,
                    gate.CLD_EXITED,
                    0,
                ),
            ),
            mock.patch.object(
                gate,
                "_group_member_pids",
                return_value=[4242],
            ) as group_members,
            mock.patch.object(
                gate.os,
                "waitpid",
                side_effect=[KeyboardInterrupt(), ChildProcessError()],
            ) as waitpid,
            mock.patch.object(gate, "_kill_group") as kill_group,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "cleanup also failed: retry reap",
            ):
                gate._bounded_popen(
                    ("/fixture/true",),
                    cwd="/fixture",
                    timeout_seconds=5.0,
                )

        getpgid.assert_called_once_with(4242)
        group_members.assert_called_once_with(4242)
        self.assertEqual(waitpid.call_count, 2)
        kill_group.assert_not_called()
        self.assertEqual(process.kill_calls, 0)

    def test_surviving_group_member_forces_cleanup_before_reap(self) -> None:
        process = self.ImmediateProcess()
        group_checks = 0

        def wait_for_group(
            child: gate._BoundedChild,
            *,
            deadline: float,
        ) -> bool:
            del deadline
            nonlocal group_checks
            group_checks += 1
            if group_checks == 1:
                return False
            child.pre_reap_member_pids = [4242]
            return True

        with (
            mock.patch.object(gate.subprocess, "Popen", return_value=process),
            mock.patch.object(gate.os, "getpgid", return_value=4242),
            mock.patch.object(
                gate,
                "_wait_for_terminal",
                side_effect=lambda child, deadline: (
                    setattr(
                        child,
                        "terminal",
                        gate._TerminalObservation(
                            4242,
                            gate.CLD_EXITED,
                            0,
                        ),
                    )
                    or True
                ),
            ),
            mock.patch.object(
                gate,
                "_wait_for_quiescent_group",
                side_effect=wait_for_group,
            ),
            mock.patch.object(gate, "_kill_group") as kill_group,
            mock.patch.object(gate.os, "waitpid", return_value=(4242, 0)),
        ):
            result = gate._bounded_popen(
                ("/fixture/true",),
                cwd="/fixture",
                timeout_seconds=5.0,
            )

        self.assertEqual(result, (0, b"", b"", False, True, [4242]))
        kill_group.assert_called_once_with(4242)
        self.assertEqual(group_checks, 2)


class OutputAndExitCodeTests(unittest.TestCase):
    def test_failure_json_is_canonical_and_uses_distinct_nonzero_code(self) -> None:
        fake = FakeSystem()
        fake.system = "Linux"
        stdout = io.BytesIO()
        stderr = io.BytesIO()

        returncode = gate.main(
            [],
            operations=fake,
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(returncode, gate.FAILURE_EXIT_CODES["platform"])
        self.assertNotEqual(returncode, 0)
        self.assertEqual(stdout.getvalue(), b"")
        record = json.loads(stderr.getvalue())
        self.assertEqual(record["failure"]["check"], "platform")
        self.assertEqual(stderr.getvalue(), gate.canonical_json_bytes(record))

    def test_failure_exit_codes_are_unique_and_nonzero(self) -> None:
        codes = list(gate.FAILURE_EXIT_CODES.values())
        self.assertEqual(len(codes), len(set(codes)))
        self.assertTrue(all(code != 0 for code in codes))

    def test_arguments_are_rejected_before_host_access(self) -> None:
        fake = FakeSystem()
        stderr = io.BytesIO()

        returncode = gate.main(
            ["--unexpected"],
            operations=fake,
            stdout=io.BytesIO(),
            stderr=stderr,
        )

        self.assertEqual(returncode, gate.FAILURE_EXIT_CODES["usage"])
        self.assertEqual(json.loads(stderr.getvalue())["status"], "FAIL")
        self.assertEqual(fake.command_calls, [])

    def test_malformed_backend_result_fails_closed(self) -> None:
        fake = FakeSystem()
        fake.command_results[gate.SYSCTL_COMMAND] = [object()]
        self.assertRaisesRegex(
            gate.GateFailure,
            "malformed",
            gate.run_gate,
            fake,
        )


if __name__ == "__main__":
    unittest.main()
