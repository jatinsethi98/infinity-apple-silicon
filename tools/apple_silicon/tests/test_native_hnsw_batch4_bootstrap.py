from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools.apple_silicon import (
    bootstrap_verify_native_hnsw_batch4_causal as bootstrap,
)


REPO = Path(__file__).resolve().parents[3]
SOURCE_DIR = REPO / "tools" / "apple_silicon"
BOOTSTRAP = SOURCE_DIR / "bootstrap_verify_native_hnsw_batch4_causal.py"
VERIFIER_NAME = "verifier-native_hnsw_batch4_causal.py"
HELPER_NAME = "module-verify_native_hnsw_d0.py"
EXACTNESS_HELPER_NAME = "module-verify_native_hnsw_exactness.py"
HELPER_GLOBAL = "_AUTHENTICATED_NATIVE_HNSW_D0_HELPER"
EXACTNESS_HELPER_GLOBAL = "_AUTHENTICATED_NATIVE_HNSW_EXACTNESS_HELPER"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verifier_source(label: str) -> bytes:
    return (
        "import json\n"
        "import sys\n"
        f"helper = globals()[{HELPER_GLOBAL!r}]\n"
        f"exactness_helper = globals()[{EXACTNESS_HELPER_GLOBAL!r}]\n"
        "print(json.dumps({"
        f"'label': {label!r}, "
        "'helper': helper.VALUE, "
        "'exactness_helper': exactness_helper.VALUE, "
        "'argv': sys.argv[1:], "
        "'sys_path': sys.path"
        "}, sort_keys=True))\n"
    ).encode("ascii")


def helper_source(value: str) -> bytes:
    return f"VALUE = {value!r}\n".encode("ascii")


class BootstrapTests(unittest.TestCase):
    def write_bundle(
        self,
        root: Path,
        *,
        verifier: bytes | None = None,
        helper: bytes | None = None,
        exactness_helper: bytes | None = None,
    ) -> tuple[bytes, bytes, bytes]:
        verifier_value = verifier or verifier_source("bundle-a")
        helper_value = helper or helper_source("helper-a")
        exactness_helper_value = exactness_helper or helper_source("exactness-a")
        (root / VERIFIER_NAME).write_bytes(verifier_value)
        (root / HELPER_NAME).write_bytes(helper_value)
        (root / EXACTNESS_HELPER_NAME).write_bytes(exactness_helper_value)
        return verifier_value, helper_value, exactness_helper_value

    def command(
        self,
        root: Path,
        verifier: bytes,
        helper: bytes,
        exactness_helper: bytes,
        *,
        verifier_size: int | None = None,
        helper_size: int | None = None,
        exactness_helper_size: int | None = None,
        preflight_only: bool = False,
    ) -> list[str]:
        command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(BOOTSTRAP),
            str(root),
            "--verifier-sha256",
            digest(verifier),
            "--verifier-bytes",
            str(len(verifier) if verifier_size is None else verifier_size),
            "--helper-sha256",
            digest(helper),
            "--helper-bytes",
            str(len(helper) if helper_size is None else helper_size),
            "--exactness-helper-sha256",
            digest(exactness_helper),
            "--exactness-helper-bytes",
            str(
                len(exactness_helper)
                if exactness_helper_size is None
                else exactness_helper_size
            ),
        ]
        if preflight_only:
            command.append("--preflight-only")
        return command

    def run_bootstrap(
        self,
        command: list[str],
        *,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd="/",
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_valid_bundle_executes_only_against_authenticated_root_and_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verifier, helper, exactness_helper = self.write_bundle(root)
            result = self.run_bootstrap(
                self.command(
                    root,
                    verifier,
                    helper,
                    exactness_helper,
                    preflight_only=True,
                )
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads(result.stdout)
            self.assertEqual(output["label"], "bundle-a")
            self.assertEqual(output["helper"], "helper-a")
            self.assertEqual(output["exactness_helper"], "exactness-a")
            self.assertEqual(
                output["argv"],
                [str(root.resolve()), "--preflight-only"],
            )
            self.assertNotIn(str(root), output["sys_path"])

    def test_second_evidence_root_is_rejected_before_verifier_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_a = root / "evidence-a"
            evidence_b = root / "evidence-b"
            evidence_a.mkdir()
            evidence_b.mkdir()
            marker = root / "verifier-executed"
            verifier = (
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).touch()\n"
            ).encode("ascii")
            helper = helper_source("helper-a")
            exactness_helper = helper_source("exactness-a")
            self.write_bundle(
                evidence_a,
                verifier=verifier,
                helper=helper,
                exactness_helper=exactness_helper,
            )
            command = self.command(
                evidence_a,
                verifier,
                helper,
                exactness_helper,
            )
            command.extend(["--", str(evidence_b)])
            result = self.run_bootstrap(command)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(marker.exists())

    def test_verifier_imports_normally_as_repo_package(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "from tools.apple_silicon import "
                    "verify_native_hnsw_batch4_causal as verifier; "
                    "print(verifier.__name__)"
                ),
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "tools.apple_silicon.verify_native_hnsw_batch4_causal",
        )

    def test_each_required_interpreter_effect_is_enforced(self) -> None:
        for flag in ("-I", "-S", "-B"):
            with self.subTest(flag=flag):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    verifier, helper, exactness_helper = self.write_bundle(root)
                    command = self.command(
                        root,
                        verifier,
                        helper,
                        exactness_helper,
                    )
                    command.remove(flag)
                    result = self.run_bootstrap(command)
                    if flag == "-B" and result.returncode == 0:
                        # Apple's isolated Python 3.9 suppresses bytecode by default.
                        self.assertNotEqual(result.stdout, "")
                        continue
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("requires python3 -I -S -B", result.stderr)
                    self.assertEqual(result.stdout, "")

    def test_substituted_trusted_source_is_rejected(self) -> None:
        for filename in (VERIFIER_NAME, HELPER_NAME, EXACTNESS_HELPER_NAME):
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    verifier, helper, exactness_helper = self.write_bundle(root)
                    (root / filename).write_bytes(
                        (root / filename).read_bytes()[:-1] + b"X"
                    )
                    result = self.run_bootstrap(
                        self.command(root, verifier, helper, exactness_helper)
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("SHA-256 mismatch", result.stderr)
                    self.assertEqual(result.stdout, "")

    def test_symlinked_trusted_source_is_rejected(self) -> None:
        for filename in (VERIFIER_NAME, HELPER_NAME, EXACTNESS_HELPER_NAME):
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    verifier, helper, exactness_helper = self.write_bundle(root)
                    source = root / f"{filename}.real"
                    path = root / filename
                    path.rename(source)
                    path.symlink_to(source.name)
                    result = self.run_bootstrap(
                        self.command(root, verifier, helper, exactness_helper)
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("non-symlink regular file", result.stderr)
                    self.assertEqual(result.stdout, "")

    def test_hard_linked_trusted_source_is_rejected(self) -> None:
        for filename in (VERIFIER_NAME, HELPER_NAME, EXACTNESS_HELPER_NAME):
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    verifier, helper, exactness_helper = self.write_bundle(root)
                    os.link(root / filename, root / f"{filename}.alias")
                    result = self.run_bootstrap(
                        self.command(root, verifier, helper, exactness_helper)
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("must not be hard-linked", result.stderr)
                    self.assertEqual(result.stdout, "")

    def test_symlinked_evidence_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence"
            evidence.mkdir()
            verifier, helper, exactness_helper = self.write_bundle(evidence)
            link = root / "evidence-link"
            link.symlink_to(evidence.name)
            result = self.run_bootstrap(
                self.command(link, verifier, helper, exactness_helper)
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("non-symlink directory", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_duplicate_option_is_rejected_before_verifier_execution(self) -> None:
        options = {
            "--verifier-sha256": lambda verifier, _helper, _exactness: digest(
                verifier
            ),
            "--verifier-bytes": lambda verifier, _helper, _exactness: str(
                len(verifier)
            ),
            "--helper-sha256": lambda _verifier, helper, _exactness: digest(
                helper
            ),
            "--helper-bytes": lambda _verifier, helper, _exactness: str(
                len(helper)
            ),
            "--exactness-helper-sha256": lambda _verifier, _helper, exactness: (
                digest(exactness)
            ),
            "--exactness-helper-bytes": lambda _verifier, _helper, exactness: (
                str(len(exactness))
            ),
            "--preflight-only": lambda _verifier, _helper, _exactness: None,
        }
        for option, value_for in options.items():
            with self.subTest(option=option):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    marker = root / "verifier-executed"
                    verifier = (
                        "from pathlib import Path\n"
                        f"Path({str(marker)!r}).touch()\n"
                    ).encode("ascii")
                    helper = helper_source("helper-a")
                    exactness_helper = helper_source("exactness-a")
                    self.write_bundle(
                        root,
                        verifier=verifier,
                        helper=helper,
                        exactness_helper=exactness_helper,
                    )
                    command = self.command(
                        root,
                        verifier,
                        helper,
                        exactness_helper,
                        preflight_only=option == "--preflight-only",
                    )
                    duplicate_value = value_for(
                        verifier,
                        helper,
                        exactness_helper,
                    )
                    command.append(option)
                    if duplicate_value is not None:
                        command.append(duplicate_value)
                    result = self.run_bootstrap(command)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("must not be repeated", result.stderr)
                    self.assertFalse(marker.exists())

    def test_unknown_option_is_rejected_before_verifier_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "verifier-executed"
            verifier = (
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).touch()\n"
            ).encode("ascii")
            helper = helper_source("helper-a")
            exactness_helper = helper_source("exactness-a")
            self.write_bundle(
                root,
                verifier=verifier,
                helper=helper,
                exactness_helper=exactness_helper,
            )
            command = self.command(root, verifier, helper, exactness_helper)
            command.append("--unknown-trust-option")
            result = self.run_bootstrap(command)
            self.assertEqual(result.returncode, 2)
            self.assertIn("unrecognized arguments", result.stderr)
            self.assertFalse(marker.exists())

    def test_root_replacement_after_verifier_execution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evidence"
            root.mkdir()
            verifier = (
                "from pathlib import Path\n"
                "import sys\n"
                "root = Path(sys.argv[1])\n"
                "root.rename(root.with_name('evidence-original'))\n"
                "root.mkdir()\n"
                "raise SystemExit(0)\n"
            ).encode("ascii")
            helper = helper_source("helper-a")
            exactness_helper = helper_source("exactness-a")
            self.write_bundle(
                root,
                verifier=verifier,
                helper=helper,
                exactness_helper=exactness_helper,
            )
            result = self.run_bootstrap(
                self.command(root, verifier, helper, exactness_helper)
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Evidence directory", result.stderr)
            self.assertIn("changed", result.stderr)

    def test_fstat_instability_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verifier, _, _ = self.write_bundle(root)
            trusted = bootstrap.TrustedSource(
                VERIFIER_NAME,
                digest(verifier),
                len(verifier),
            )
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
            directory_descriptor = os.open(root, flags)
            real_fstat = os.fstat
            calls = 0

            def unstable_fstat(descriptor: int) -> os.stat_result | SimpleNamespace:
                nonlocal calls
                value = real_fstat(descriptor)
                calls += 1
                if calls != 2:
                    return value
                return SimpleNamespace(
                    st_dev=value.st_dev,
                    st_ino=value.st_ino,
                    st_mode=value.st_mode,
                    st_nlink=value.st_nlink,
                    st_uid=value.st_uid,
                    st_gid=value.st_gid,
                    st_size=value.st_size,
                    st_mtime_ns=value.st_mtime_ns + 1,
                    st_ctime_ns=value.st_ctime_ns,
                )

            try:
                with mock.patch.object(
                    bootstrap.os,
                    "fstat",
                    side_effect=unstable_fstat,
                ):
                    with self.assertRaisesRegex(
                        bootstrap.BootstrapError,
                        "changed during authenticated read",
                    ):
                        bootstrap._read_source(directory_descriptor, trusted)
            finally:
                os.close(directory_descriptor)

    def test_bundle_is_fully_authenticated_before_any_source_executes(self) -> None:
        for changed in (VERIFIER_NAME, HELPER_NAME, EXACTNESS_HELPER_NAME):
            with self.subTest(changed=changed):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    verifier_marker = root / "verifier-executed"
                    helper_marker = root / "helper-executed"
                    exactness_marker = root / "exactness-helper-executed"
                    verifier = (
                        "from pathlib import Path\n"
                        f"Path({str(verifier_marker)!r}).touch()\n"
                    ).encode("ascii")
                    helper = (
                        "from pathlib import Path\n"
                        f"Path({str(helper_marker)!r}).touch()\n"
                    ).encode("ascii")
                    exactness_helper = (
                        "from pathlib import Path\n"
                        f"Path({str(exactness_marker)!r}).touch()\n"
                    ).encode("ascii")
                    self.write_bundle(
                        root,
                        verifier=verifier,
                        helper=helper,
                        exactness_helper=exactness_helper,
                    )
                    path = root / changed
                    path.write_bytes(path.read_bytes()[:-1] + b"X")
                    result = self.run_bootstrap(
                        self.command(
                            root,
                            verifier,
                            helper,
                            exactness_helper,
                        )
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertFalse(verifier_marker.exists())
                    self.assertFalse(helper_marker.exists())
                    self.assertFalse(exactness_marker.exists())

    def test_bundle_is_fully_authenticated_before_any_source_compiles(self) -> None:
        for changed in (VERIFIER_NAME, HELPER_NAME, EXACTNESS_HELPER_NAME):
            with self.subTest(changed=changed):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    verifier = b"this is not valid Python !\n"
                    helper = b"this is also not valid Python !\n"
                    exactness_helper = b"exactness is not valid Python !\n"
                    self.write_bundle(
                        root,
                        verifier=verifier,
                        helper=helper,
                        exactness_helper=exactness_helper,
                    )
                    path = root / changed
                    path.write_bytes(path.read_bytes()[:-1] + b"X")
                    result = self.run_bootstrap(
                        self.command(
                            root,
                            verifier,
                            helper,
                            exactness_helper,
                        )
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("mismatch", result.stderr)
                    self.assertNotIn("SyntaxError", result.stderr)

    def test_mixed_trusted_bundles_are_rejected(self) -> None:
        bundle_a = (
            verifier_source("bundle-a"),
            helper_source("helper-a"),
            helper_source("exactness-a"),
        )
        bundle_b = (
            verifier_source("bundle-b"),
            helper_source("helper-b"),
            helper_source("exactness-b"),
        )
        filenames = (VERIFIER_NAME, HELPER_NAME, EXACTNESS_HELPER_NAME)
        for changed_index, changed_name in enumerate(filenames):
            with self.subTest(changed=changed_name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    actual = list(bundle_a)
                    actual[changed_index] = bundle_b[changed_index]
                    self.write_bundle(
                        root,
                        verifier=actual[0],
                        helper=actual[1],
                        exactness_helper=actual[2],
                    )
                    result = self.run_bootstrap(
                        self.command(root, *bundle_a)
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(changed_name, result.stderr)
                    self.assertIn("mismatch", result.stderr)

    def test_hostile_pythonpath_and_adjacent_modules_are_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence"
            hostile = root / "hostile"
            evidence.mkdir()
            hostile.mkdir()
            shutil.copy2(
                SOURCE_DIR / "verify_native_hnsw_batch4_causal.py",
                evidence / VERIFIER_NAME,
            )
            shutil.copy2(
                SOURCE_DIR / "verify_native_hnsw_d0.py",
                evidence / HELPER_NAME,
            )
            shutil.copy2(
                SOURCE_DIR / "verify_native_hnsw_exactness.py",
                evidence / EXACTNESS_HELPER_NAME,
            )
            marker = root / "marker"
            payload = (
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('loaded', encoding='ascii')\n"
            )
            (evidence / "array.py").write_text(payload, encoding="ascii")
            (evidence / "verify_native_hnsw_d0.py").write_text(
                payload,
                encoding="ascii",
            )
            (evidence / "verify_native_hnsw_exactness.py").write_text(
                payload,
                encoding="ascii",
            )
            (hostile / "sitecustomize.py").write_text(payload, encoding="ascii")
            (hostile / "array.py").write_text(payload, encoding="ascii")
            verifier = (evidence / VERIFIER_NAME).read_bytes()
            helper = (evidence / HELPER_NAME).read_bytes()
            exactness_helper = (evidence / EXACTNESS_HELPER_NAME).read_bytes()
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(hostile)
            result = self.run_bootstrap(
                self.command(
                    evidence,
                    verifier,
                    helper,
                    exactness_helper,
                ),
                environment=environment,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn('"status": "FAIL"', result.stderr)
            self.assertFalse(marker.exists())

    def test_direct_captured_verifier_execution_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "marker"
            payload = (
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('loaded', encoding='ascii')\n"
            )
            shutil.copy2(
                SOURCE_DIR / "verify_native_hnsw_batch4_causal.py",
                root / VERIFIER_NAME,
            )
            shutil.copy2(
                SOURCE_DIR / "verify_native_hnsw_d0.py",
                root / HELPER_NAME,
            )
            shutil.copy2(
                SOURCE_DIR / "verify_native_hnsw_exactness.py",
                root / EXACTNESS_HELPER_NAME,
            )
            (root / "argparse.py").write_text(payload, encoding="ascii")
            hostile = root / "hostile"
            hostile.mkdir()
            (hostile / "hashlib.py").write_text(payload, encoding="ascii")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(hostile)
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(root / VERIFIER_NAME),
                    "--help",
                ],
                cwd="/",
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("authenticated Batch4 bootstrap", result.stderr)
            self.assertFalse(marker.exists())

    def test_caller_size_mismatch_is_rejected(self) -> None:
        for changed in ("verifier", "helper", "exactness_helper"):
            with self.subTest(changed=changed):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    verifier, helper, exactness_helper = self.write_bundle(root)
                    result = self.run_bootstrap(
                        self.command(
                            root,
                            verifier,
                            helper,
                            exactness_helper,
                            verifier_size=(
                                len(verifier) + 1
                                if changed == "verifier"
                                else None
                            ),
                            helper_size=(
                                len(helper) + 1
                                if changed == "helper"
                                else None
                            ),
                            exactness_helper_size=(
                                len(exactness_helper) + 1
                                if changed == "exactness_helper"
                                else None
                            ),
                        )
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("size mismatch", result.stderr)
                    self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
