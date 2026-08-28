from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.apple_silicon import native_hnsw_batch4_causal as causal
from tools.apple_silicon import native_hnsw_d0 as d0
from tools.apple_silicon import verify_native_hnsw_batch4_causal as causal_verifier
from tools.apple_silicon import verify_native_hnsw_d0 as verifier


def compile_entry(
    *,
    build_directory: Path,
    source: Path,
    output: Path,
    definition: str | None = None,
) -> dict[str, object]:
    arguments = ["/fixture/clang++"]
    if definition is not None:
        arguments.append(definition)
    arguments.extend(["-c", str(source), "-o", str(output)])
    return {
        "directory": str(build_directory),
        "file": str(source),
        "arguments": arguments,
        "output": str(output),
    }


def release_arguments() -> list[str]:
    return [
        "-O3",
        "-DNDEBUG",
        "-arch",
        "arm64",
        "-isysroot",
        "/fixture/MacOSX.sdk",
        "-mmacosx-version-min=14.0",
    ]


class SyntheticNinja:
    def __init__(
        self,
        *,
        build_directory: Path,
        products: list[dict[str, object]],
    ) -> None:
        self.build_directory = build_directory
        self.products = products
        self.product_units: dict[str, list[tuple[Path, Path]]] = {}
        units: dict[Path, Path] = {}
        for product in products:
            product_name = str(product["name"])
            product_entries = product["compile_entries"]
            if not isinstance(product_entries, list):
                raise AssertionError("product compile entries are not a list")
            selected: list[tuple[Path, Path]] = []
            for entry in product_entries:
                if not isinstance(entry, dict):
                    raise AssertionError("compile entry is not a mapping")
                source = Path(str(entry["file"]))
                output = Path(str(entry["output"]))
                previous = units.get(output)
                if previous is not None and previous != source:
                    raise AssertionError(f"conflicting unit: {output}")
                units.setdefault(output, source)
                selected.append((output, source))
            self.product_units[product_name] = selected
        self.units = sorted(
            units.items(),
            key=lambda item: str(item[0]).encode("utf-8"),
        )

    def relative_output(self, output: Path) -> str:
        return output.relative_to(self.build_directory).as_posix()

    def __call__(
        self,
        command: list[str],
        *,
        cwd: Path,
        **_: object,
    ) -> SimpleNamespace:
        self.assert_canonical_cwd(cwd)
        if command[-2:] == ["-t", "deps"]:
            stdout = "".join(
                f"{self.relative_output(output)}: "
                "#deps 1, deps mtime 1 (VALID)\n"
                f"    {source}\n\n"
                for output, source in self.units
            )
        elif command[-3:] == ["-t", "targets", "all"]:
            stdout = (
                "".join(
                    f"{self.relative_output(output)}: "
                    "CXX_COMPILER__fixture_Release\n"
                    for output, _ in self.units
                )
                + "".join(
                    f"{product['requested_target']}: "
                    "CXX_EXECUTABLE_LINKER__fixture_Release\n"
                    for product in self.products
                )
            )
        elif "-t" in command and command[command.index("-t") + 1] == "query":
            stdout = "".join(
                f"{product['requested_target']}:\n"
                "  input: CXX_EXECUTABLE_LINKER__fixture_Release\n"
                + "".join(
                    f"    {self.relative_output(output)}\n"
                    for output, _ in self.product_units[str(product["name"])]
                )
                + "  outputs:\n"
                for product in self.products
            )
            stdout += "".join(
                f"{self.relative_output(output)}:\n"
                "  input: CXX_COMPILER__fixture_Release\n"
                f"    {source}\n"
                "  outputs:\n"
                for output, source in self.units
            )
        elif "-t" in command and command[command.index("-t") + 1] == "clean":
            removals = [
                self.relative_output(output)
                for output, _ in self.units
            ]
            removals.extend(
                str(product["requested_target"]) for product in self.products
            )
            stdout = (
                "Cleaning...\n"
                + "".join(
                    f"Target {product['requested_target']}\n"
                    for product in self.products
                )
                + "".join(f"Remove {path}\n" for path in removals)
                + f"{len(removals)} files.\n"
            )
        else:
            stdout = "ninja: no work to do.\n"
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    def assert_canonical_cwd(self, cwd: Path) -> None:
        if cwd != self.build_directory:
            raise AssertionError(f"unexpected cwd: {cwd}")

    def bytes_call(
        self,
        command: list[str],
        *,
        cwd: Path,
        **kwargs: object,
    ) -> SimpleNamespace:
        completed = self(command, cwd=cwd, **kwargs)
        return SimpleNamespace(
            stdout=completed.stdout.encode("utf-8"),
            stderr=completed.stderr.encode("utf-8"),
            returncode=completed.returncode,
        )


class SettlementFailureEvidenceTests(unittest.TestCase):
    def fixture(self) -> tuple[Path, Path, list[str]]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        evidence = root / "evidence"
        build = root / "build"
        evidence.mkdir()
        build.mkdir()
        command = [
            "/fixture/ninja",
            "-C",
            str(build),
            "benchmark-bin",
            "exactness-bin",
        ]
        return evidence, build, command

    def read_record(self, evidence: Path) -> dict[str, object]:
        return json.loads(
            (evidence / "fixture-closure-settlement.json").read_text(
                encoding="utf-8"
            )
        )

    def test_success_writes_no_failure_evidence(self) -> None:
        evidence, build, command = self.fixture()
        completed = SimpleNamespace(
            returncode=0,
            stdout=b"ninja: no work to do.\n",
            stderr=b"",
        )
        with patch.object(d0, "run_bytes", return_value=completed) as run:
            d0.require_build_capture_settled(
                output_dir=evidence,
                capture_prefix="fixture",
                command=command,
                cwd=build,
                label="fixture",
            )

        run.assert_called_once_with(command, cwd=build, check=False)
        self.assertEqual(list(evidence.iterdir()), [])

    def test_stale_output_preserves_transcript_and_target_order(self) -> None:
        evidence, build, command = self.fixture()
        stdout = b"[1/2] Building benchmark\r\n[2/2] Linking exactness\r\n\xff"
        stderr = b"ninja: no work to do.\r\n\xfe"
        completed = SimpleNamespace(
            returncode=0,
            stdout=stdout,
            stderr=stderr,
        )
        with (
            patch.object(d0, "run_bytes", return_value=completed),
            self.assertRaisesRegex(
                d0.D0Failure,
                "fixture became stale during capture",
            ),
        ):
            d0.require_build_capture_settled(
                output_dir=evidence,
                capture_prefix="fixture",
                command=command,
                cwd=build,
                label="fixture",
            )

        record = self.read_record(evidence)
        self.assertEqual(record["command"], command)
        self.assertEqual(record["working_directory"], str(build))
        self.assertEqual(record["returncode"], 0)
        self.assertEqual(record["reason"], "missing-no-work-marker")
        self.assertEqual(
            record["settlement_failure_schema_version"],
            d0.NINJA_SETTLEMENT_FAILURE_SCHEMA_VERSION,
        )
        for stream_name, stream_bytes in (
            ("stdout", stdout),
            ("stderr", stderr),
        ):
            stream_record = record[stream_name]
            self.assertIsInstance(stream_record, dict)
            self.assertEqual(stream_record["encoding"], "base64")
            self.assertEqual(
                base64.b64decode(stream_record["content_base64"], validate=True),
                stream_bytes,
            )
            self.assertIsNone(stream_record["utf8_text"])
            self.assertEqual(stream_record["bytes"], len(stream_bytes))
            self.assertEqual(
                stream_record["sha256"],
                hashlib.sha256(stream_bytes).hexdigest(),
            )

        d0.write_manifest(evidence)
        manifest = (evidence / "MANIFEST.sha256").read_text(encoding="utf-8")
        self.assertIn("fixture-closure-settlement.json", manifest)
        self.assertNotIn("fixture-closure-settlement.stdout", manifest)
        self.assertNotIn("fixture-closure-settlement.stderr", manifest)

    def test_nonzero_exit_is_rejected_even_with_no_work_marker(self) -> None:
        evidence, build, command = self.fixture()
        completed = SimpleNamespace(
            returncode=7,
            stdout=b"ninja: no work to do.\n",
            stderr=b"fatal build error\n",
        )
        with (
            patch.object(d0, "run_bytes", return_value=completed),
            self.assertRaisesRegex(
                d0.D0Failure,
                "settlement command failed with exit status 7",
            ),
        ):
            d0.require_build_capture_settled(
                output_dir=evidence,
                capture_prefix="fixture",
                command=command,
                cwd=build,
                label="fixture",
            )

        record = self.read_record(evidence)
        self.assertEqual(record["returncode"], 7)
        self.assertEqual(record["reason"], "nonzero-returncode")
        self.assertEqual(
            base64.b64decode(
                record["stderr"]["content_base64"],
                validate=True,
            ),
            b"fatal build error\n",
        )
        self.assertEqual(record["stderr"]["utf8_text"], "fatal build error\n")

    def test_timeout_preserves_partial_raw_streams_and_null_returncode(self) -> None:
        evidence, build, command = self.fixture()
        timeout = subprocess.TimeoutExpired(
            command,
            timeout=60,
            output=b"partial stdout\xff",
            stderr=b"partial stderr\xfe",
        )
        with (
            patch.object(d0, "run_bytes", side_effect=timeout),
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            d0.require_build_capture_settled(
                output_dir=evidence,
                capture_prefix="fixture",
                command=command,
                cwd=build,
                label="fixture",
            )

        record = self.read_record(evidence)
        self.assertIsNone(record["returncode"])
        self.assertEqual(record["reason"], "execution-exception")
        self.assertEqual(
            record["execution_error"],
            {
                "type": "TimeoutExpired",
                "message": str(timeout),
            },
        )
        self.assertEqual(
            base64.b64decode(
                record["stdout"]["content_base64"],
                validate=True,
            ),
            b"partial stdout\xff",
        )
        self.assertEqual(
            base64.b64decode(
                record["stderr"]["content_base64"],
                validate=True,
            ),
            b"partial stderr\xfe",
        )

    def test_existing_failure_evidence_is_not_overwritten(self) -> None:
        evidence, build, command = self.fixture()
        record_path = evidence / "fixture-closure-settlement.json"
        record_path.write_bytes(b"sentinel")
        completed = SimpleNamespace(returncode=0, stdout=b"rebuilt\n", stderr=b"")
        with (
            patch.object(d0, "run_bytes", return_value=completed),
            self.assertRaisesRegex(
                d0.D0Failure,
                "fixture became stale during capture; could not preserve "
                "Ninja settlement failure evidence",
            ),
        ):
            d0.require_build_capture_settled(
                output_dir=evidence,
                capture_prefix="fixture",
                command=command,
                cwd=build,
                label="fixture",
            )

        self.assertEqual(record_path.read_bytes(), b"sentinel")
        self.assertEqual(list(evidence.iterdir()), [record_path])

    def test_atomic_link_failure_leaves_no_partial_record(self) -> None:
        evidence, build, command = self.fixture()
        completed = SimpleNamespace(returncode=0, stdout=b"rebuilt\n", stderr=b"")
        with (
            patch.object(d0, "run_bytes", return_value=completed),
            patch.object(d0.os, "link", side_effect=OSError("link fault")),
            self.assertRaisesRegex(
                d0.D0Failure,
                r"fixture became stale during capture; could not preserve "
                r"Ninja settlement failure evidence \(OSError\): link fault",
            ),
        ):
            d0.require_build_capture_settled(
                output_dir=evidence,
                capture_prefix="fixture",
                command=command,
                cwd=build,
                label="fixture",
            )

        self.assertEqual(list(evidence.iterdir()), [])
        self.assertEqual(
            [
                path
                for path in evidence.parent.iterdir()
                if path.name.startswith(
                    f".{evidence.name}.fixture-closure-settlement.json."
                )
            ],
            [],
        )

    def test_execution_and_publication_failures_are_both_reported(self) -> None:
        evidence, build, command = self.fixture()
        record_path = evidence / "fixture-closure-settlement.json"
        record_path.write_bytes(b"sentinel")
        timeout = subprocess.TimeoutExpired(
            command,
            timeout=60,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )
        with (
            patch.object(d0, "run_bytes", side_effect=timeout),
            self.assertRaisesRegex(
                d0.D0Failure,
                "timed out after 60 seconds; could not preserve "
                "Ninja settlement failure evidence",
            ),
        ):
            d0.require_build_capture_settled(
                output_dir=evidence,
                capture_prefix="fixture",
                command=command,
                cwd=build,
                label="fixture",
            )

        self.assertEqual(record_path.read_bytes(), b"sentinel")


class ClosureSchemaV4Tests(unittest.TestCase):
    def capture_fixture(
        self,
        *,
        product_count: int,
        legacy: bool,
        shared_compile: bool = True,
    ) -> tuple[
        Path,
        dict[str, object],
        list[dict[str, object]],
        dict[str, str],
        Path,
    ]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        build_directory = root / "build"
        evidence_directory = root / "evidence"
        build_directory.mkdir()
        evidence_directory.mkdir()
        shared_entry: dict[str, object] | None = None
        if shared_compile:
            source = root / "shared.cpp"
            source.write_text("int shared() { return 1; }\n", encoding="utf-8")
            object_path = build_directory / "shared.o"
            object_path.write_bytes(b"synthetic object\n")
            shared_entry = compile_entry(
                build_directory=build_directory,
                source=source,
                output=object_path,
            )
        products: list[dict[str, object]] = []
        names = ["benchmark", "exactness-emitter"]
        targets = ["benchmark-bin", "exactness-bin"]
        for index in range(product_count):
            if shared_entry is None:
                source = root / f"{names[index]}.cpp"
                source.write_text(
                    f"int product_{index}() {{ return {index}; }}\n",
                    encoding="utf-8",
                )
                object_path = build_directory / f"{names[index]}.o"
                object_path.write_bytes(f"synthetic object {index}\n".encode())
                entry = compile_entry(
                    build_directory=build_directory,
                    source=source,
                    output=object_path,
                )
            else:
                entry = shared_entry
                object_path = Path(str(entry["output"]))
            binary = build_directory / targets[index]
            binary.write_bytes(f"binary {index}\n".encode())
            products.append(
                {
                    "name": names[index],
                    "requested_target": targets[index],
                    "expected_output": binary,
                    "compile_entries": [entry],
                    "link_arguments": [
                        "/fixture/clang++",
                        str(object_path),
                        "-o",
                        str(binary),
                    ],
                }
            )
        cache = {"CMAKE_OSX_SYSROOT": "/fixture/MacOSX.sdk"}
        build_tool = {"resolved_path": "/fixture/ninja"}
        runner = SyntheticNinja(
            build_directory=build_directory,
            products=products,
        )
        capture_arguments: dict[str, object] = {
            "output_dir": evidence_directory,
            "capture_prefix": "fixture",
            "build_directory": build_directory,
            "build_tool": build_tool,
            "cache": cache,
            "response_references": set(),
            "label": "fixture",
        }
        if legacy:
            product = products[0]
            capture_arguments.update(
                {
                    "requested_target": product["requested_target"],
                    "expected_output": product["expected_output"],
                    "compile_entries": product["compile_entries"],
                    "link_arguments": product["link_arguments"],
                }
            )
        else:
            capture_arguments["products"] = products
        with (
            patch.object(d0, "run_text", side_effect=runner),
            patch.object(d0, "run_bytes", side_effect=runner.bytes_call),
        ):
            reference = d0.capture_build_input_closure(**capture_arguments)
        return evidence_directory, reference, products, cache, build_directory

    def validate_two_product_fixture(
        self,
        evidence: Path,
        reference: dict[str, object],
        products: list[dict[str, object]],
        cache: dict[str, str],
        build_directory: Path,
    ) -> dict[str, object]:
        return verifier.validate_build_input_closure(
            evidence,
            reference,
            capture_prefix="fixture",
            products=products,
            build_directory=build_directory,
            build_tool_path="/fixture/ninja",
            cache=cache,
            response_files={},
            used_response_files=set(),
            context="fixture closure",
        )

    def test_single_product_legacy_wrapper_emits_and_verifies_schema_4(self) -> None:
        evidence, reference, products, cache, build_directory = self.capture_fixture(
            product_count=1,
            legacy=True,
        )
        product = products[0]
        closure = verifier.validate_build_input_closure(
            evidence,
            reference,
            capture_prefix="fixture",
            requested_target=str(product["requested_target"]),
            expected_output=str(product["expected_output"]),
            build_directory=build_directory,
            build_tool_path="/fixture/ninja",
            cache=cache,
            compile_entries=list(product["compile_entries"]),
            link_arguments=list(product["link_arguments"]),
            response_files={},
            used_response_files=set(),
            context="fixture closure",
        )
        self.assertEqual(reference["closure_schema_version"], 4)
        self.assertEqual(reference["products"], 1)
        self.assertEqual(
            [product["name"] for product in closure["products"]],
            ["benchmark"],
        )
        self.assertEqual(len(closure["compile_units"]), 1)

    def test_two_products_share_one_canonical_compile_union(self) -> None:
        evidence, reference, products, cache, build_directory = self.capture_fixture(
            product_count=2,
            legacy=False,
        )
        closure = verifier.validate_build_input_closure(
            evidence,
            reference,
            capture_prefix="fixture",
            products=products,
            build_directory=build_directory,
            build_tool_path="/fixture/ninja",
            cache=cache,
            response_files={},
            used_response_files=set(),
            context="fixture closure",
        )
        self.assertEqual(reference["products"], 2)
        self.assertEqual(
            [product["name"] for product in closure["products"]],
            ["benchmark", "exactness-emitter"],
        )
        self.assertEqual(len(closure["compile_units"]), 1)
        self.assertEqual(len(closure["dependency_graph"]), 1)
        self.assertEqual(
            closure["selected_target_graph"]["selected_outputs"],
            [str(product["expected_output"]) for product in products],
        )
        clean_command = closure["selected_target_graph"]["clean_plan"]["command"]
        self.assertEqual(
            clean_command[-2:],
            [str(product["requested_target"]) for product in products],
        )

    def test_two_products_preserve_independent_compile_sets_in_union(self) -> None:
        evidence, reference, products, cache, build_directory = self.capture_fixture(
            product_count=2,
            legacy=False,
            shared_compile=False,
        )
        closure = self.validate_two_product_fixture(
            evidence,
            reference,
            products,
            cache,
            build_directory,
        )
        expected_outputs = [
            [str(product["compile_entries"][0]["output"])]
            for product in products
        ]
        self.assertEqual(
            [product["compile_outputs"] for product in closure["products"]],
            expected_outputs,
        )
        self.assertEqual(
            [unit["output"] for unit in closure["compile_units"]],
            sorted(
                (output for outputs in expected_outputs for output in outputs),
                key=lambda value: value.encode("utf-8"),
            ),
        )

        closure_path = evidence / str(reference["captured_path"])
        recorded = json.loads(closure_path.read_text(encoding="utf-8"))
        recorded["products"][1]["compile_outputs"] = expected_outputs[0]
        d0.write_json(closure_path, recorded)
        reference["sha256"] = d0.sha256(closure_path)
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "ordered products differ",
        ):
            self.validate_two_product_fixture(
                evidence,
                reference,
                products,
                cache,
                build_directory,
            )

    def test_product_compile_union_must_equal_combined_capture(self) -> None:
        build = Path("/fixture/build")
        products = []
        combined = []
        for name in ("benchmark", "exactness"):
            entry = compile_entry(
                build_directory=build,
                source=Path(f"/fixture/{name}.cpp"),
                output=build / f"{name}.o",
            )
            products.append({"name": name, "compile_entries": [entry]})
            combined.append(entry)

        producer_union = d0.canonical_compile_entry_union(
            products,
            combined_entries=combined,
            label="fixture",
            response_references=set(),
        )
        verifier_union = verifier.canonical_compile_entry_union(
            products,
            combined_entries=combined,
            context="fixture",
            response_files={},
            used_response_files=set(),
        )
        self.assertEqual(producer_union, combined)
        self.assertEqual(verifier_union, combined)

        extra = compile_entry(
            build_directory=build,
            source=Path("/fixture/unrelated.c"),
            output=build / "unrelated.o",
        )
        for name, invalid_combined in (
            ("omission", combined[:1]),
            ("addition", [*combined, extra]),
        ):
            with self.subTest(implementation="producer", mutation=name):
                with self.assertRaisesRegex(
                    d0.D0Failure,
                    "canonical product union differs from combined capture",
                ):
                    d0.canonical_compile_entry_union(
                        products,
                        combined_entries=invalid_combined,
                        label="fixture",
                        response_references=set(),
                    )
            with self.subTest(implementation="verifier", mutation=name):
                with self.assertRaisesRegex(
                    verifier.VerificationError,
                    "canonical product union differs from combined capture",
                ):
                    verifier.canonical_compile_entry_union(
                        products,
                        combined_entries=invalid_combined,
                        context="fixture",
                        response_files={},
                        used_response_files=set(),
                    )

    def test_product_omission_duplication_and_reordering_are_rejected(self) -> None:
        mutations = {
            "omitted": lambda value: value["products"].pop(),
            "duplicated": lambda value: value["products"].append(
                value["products"][0]
            ),
            "reordered": lambda value: value["products"].reverse(),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                (
                    evidence,
                    reference,
                    products,
                    cache,
                    build_directory,
                ) = self.capture_fixture(product_count=2, legacy=False)
                closure_path = evidence / str(reference["captured_path"])
                closure = json.loads(closure_path.read_text(encoding="utf-8"))
                mutate(closure)
                d0.write_json(closure_path, closure)
                reference["sha256"] = d0.sha256(closure_path)
                with self.assertRaisesRegex(
                    verifier.VerificationError,
                    "ordered products differ",
                ):
                    self.validate_two_product_fixture(
                        evidence,
                        reference,
                        products,
                        cache,
                        build_directory,
                    )

    def test_selected_output_omission_duplication_and_reordering_are_rejected(
        self,
    ) -> None:
        mutations = {
            "omitted": lambda values: values.pop(),
            "duplicated": lambda values: values.append(values[0]),
            "reordered": lambda values: values.reverse(),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                (
                    evidence,
                    reference,
                    products,
                    cache,
                    build_directory,
                ) = self.capture_fixture(product_count=2, legacy=False)
                closure_path = evidence / str(reference["captured_path"])
                closure = json.loads(closure_path.read_text(encoding="utf-8"))
                mutate(
                    closure["selected_target_graph"]["selected_outputs"]
                )
                d0.write_json(closure_path, closure)
                reference["sha256"] = d0.sha256(closure_path)
                with self.assertRaisesRegex(
                    verifier.VerificationError,
                    "selected graph outputs differ",
                ):
                    self.validate_two_product_fixture(
                        evidence,
                        reference,
                        products,
                        cache,
                        build_directory,
                    )

    def test_causal_rebuild_contract_binds_both_ordered_products(self) -> None:
        build_directory = Path("/fixture/build")
        benchmark = build_directory / causal.BUILD_PRODUCT_TARGETS[0][1]
        exactness = build_directory / causal.BUILD_PRODUCT_TARGETS[1][1]
        closure = {
            "products": [
                {
                    "name": "benchmark",
                    "expected_output": str(benchmark),
                },
                {
                    "name": "exactness-emitter",
                    "expected_output": str(exactness),
                },
            ],
            "compile_units": [],
            "files": [],
            "selected_target_graph": {
                "selected_outputs": [str(benchmark), str(exactness)],
                "material_outputs": [str(benchmark), str(exactness)],
                "clean_plan": {
                    "outputs": [str(benchmark), str(exactness)],
                },
                "derived_link_outputs": [],
            },
        }
        runner_freshness, runner_exact, _ = causal.rebuild_output_contract(
            closure,
            build_directory=build_directory,
            expected_output=benchmark,
        )
        verifier_freshness, verifier_exact, _ = (
            causal_verifier.rebuild_output_contract(
                closure,
                build_directory=build_directory,
                binary_path=str(benchmark),
            )
        )
        self.assertEqual(runner_exact, {benchmark, exactness})
        self.assertEqual(verifier_exact, {str(benchmark), str(exactness)})
        self.assertEqual(runner_freshness, {benchmark, exactness})
        self.assertEqual(
            verifier_freshness,
            {str(benchmark), str(exactness)},
        )

        closure["selected_target_graph"]["selected_outputs"].reverse()
        with self.assertRaisesRegex(
            causal.CausalFailure,
            "ordered build products",
        ):
            causal.rebuild_output_contract(
                closure,
                build_directory=build_directory,
                expected_output=benchmark,
            )
        with self.assertRaisesRegex(
            causal_verifier.CausalVerificationError,
            "ordered build products",
        ):
            causal_verifier.rebuild_output_contract(
                closure,
                build_directory=build_directory,
                binary_path=str(benchmark),
            )

    def test_duplicate_output_and_conflicting_shared_compile_are_rejected(self) -> None:
        build = Path("/fixture/build")
        source = Path("/fixture/source.cpp")
        shared = build / "shared.o"
        left = compile_entry(
            build_directory=build,
            source=source,
            output=shared,
            definition="-DROLE=1",
        )
        right = compile_entry(
            build_directory=build,
            source=source,
            output=shared,
            definition="-DROLE=2",
        )
        duplicate_output_products = [
            {
                "name": "benchmark",
                "expected_output": build / "same",
                "compile_entries": [left],
            },
            {
                "name": "exactness-emitter",
                "expected_output": build / "same",
                "compile_entries": [left],
            },
        ]
        with self.assertRaisesRegex(d0.D0Failure, "duplicates expected output"):
            d0.validate_ninja_commands(
                "",
                cache={
                    "CMAKE_CXX_COMPILER": "/fixture/clang++",
                    "CMAKE_OSX_SYSROOT": "/fixture/MacOSX.sdk",
                    "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0",
                },
                build_directory=build,
                label="fixture",
                response_references=set(),
                products=duplicate_output_products,
            )

        conflicting_products = [
            {
                "name": "benchmark",
                "expected_output": build / "benchmark",
                "compile_entries": [left],
            },
            {
                "name": "exactness-emitter",
                "expected_output": build / "exactness",
                "compile_entries": [right],
            },
        ]
        with self.assertRaisesRegex(d0.D0Failure, "conflicting compile commands"):
            d0.validate_ninja_commands(
                "",
                cache={
                    "CMAKE_CXX_COMPILER": "/fixture/clang++",
                    "CMAKE_OSX_SYSROOT": "/fixture/MacOSX.sdk",
                    "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0",
                },
                build_directory=build,
                label="fixture",
                response_references=set(),
                products=conflicting_products,
            )

    def test_missing_and_duplicate_product_links_are_rejected(self) -> None:
        build = Path("/fixture/build")
        source = Path("/fixture/source.cpp")
        common = release_arguments()
        entries = []
        products = []
        command_lines = []
        for name in ("benchmark", "exactness"):
            object_path = build / f"{name}.o"
            entry = {
                "directory": str(build),
                "file": str(source),
                "arguments": [
                    "/fixture/clang++",
                    *common,
                    "-c",
                    str(source),
                    "-o",
                    str(object_path),
                ],
            }
            entries.append(entry)
            products.append(
                {
                    "name": name,
                    "expected_output": build / name,
                    "compile_entries": [entry],
                }
            )
            command_lines.append(" ".join(entry["arguments"]))
        links = [
            " ".join(
                [
                    "/fixture/clang++",
                    *common,
                    str(build / f"{name}.o"),
                    "-o",
                    str(build / name),
                ]
            )
            for name in ("benchmark", "exactness")
        ]
        cache = {
            "CMAKE_CXX_COMPILER": "/fixture/clang++",
            "CMAKE_OSX_SYSROOT": "/fixture/MacOSX.sdk",
            "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0",
        }
        with self.assertRaisesRegex(d0.D0Failure, "exactly one expected link"):
            d0.validate_ninja_commands(
                "\n".join([*command_lines, links[0]]),
                cache=cache,
                build_directory=build,
                label="fixture",
                response_references=set(),
                products=products,
            )
        with self.assertRaisesRegex(d0.D0Failure, "exactly one expected link"):
            d0.validate_ninja_commands(
                "\n".join([*command_lines, *links, links[1]]),
                cache=cache,
                build_directory=build,
                label="fixture",
                response_references=set(),
                products=products,
            )

        observed = verifier.validate_ninja_commands(
            "\n".join([*command_lines, *links]),
            cache=cache,
            build_directory=build,
            context="fixture",
            response_files={},
            used_response_files=set(),
            products=products,
        )
        self.assertEqual(len(observed), 2)

    def test_clean_plan_binds_ordered_targets(self) -> None:
        text = (
            "Cleaning...\n"
            "Target benchmark\n"
            "Target exactness\n"
            "Remove benchmark\n"
            "Remove exactness\n"
            "2 files.\n"
        )
        self.assertEqual(
            d0.parse_ninja_clean_plan(
                text,
                build_directory=Path("/fixture/build"),
                requested_targets=["benchmark", "exactness"],
                label="fixture",
            ),
            ["/fixture/build/benchmark", "/fixture/build/exactness"],
        )
        with self.assertRaisesRegex(d0.D0Failure, "target list differs"):
            d0.parse_ninja_clean_plan(
                text,
                build_directory=Path("/fixture/build"),
                requested_targets=["exactness", "benchmark"],
                label="fixture",
            )

    def test_schema_3_reference_is_rejected(self) -> None:
        reference = {
            "closure_schema_version": 3,
            "captured_path": "fixture-build-input-closure.json",
            "sha256": "0" * 64,
            "products": 1,
            "compile_units": 1,
            "dependency_outputs": 1,
            "files": 1,
        }
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "reference schema is unsupported",
        ):
            verifier.validate_build_input_closure(
                Path("/fixture/evidence"),
                reference,
                capture_prefix="fixture",
                requested_target="benchmark",
                expected_output="/fixture/build/benchmark",
                build_directory=Path("/fixture/build"),
                build_tool_path="/fixture/ninja",
                cache={"CMAKE_OSX_SYSROOT": "/fixture/MacOSX.sdk"},
                compile_entries=[
                    compile_entry(
                        build_directory=Path("/fixture/build"),
                        source=Path("/fixture/source.cpp"),
                        output=Path("/fixture/build/benchmark.o"),
                    )
                ],
                link_arguments=[
                    "/fixture/clang++",
                    "/fixture/build/benchmark.o",
                    "-o",
                    "/fixture/build/benchmark",
                ],
                response_files={},
                used_response_files=set(),
                context="fixture closure",
            )


if __name__ == "__main__":
    unittest.main()
