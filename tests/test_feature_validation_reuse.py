"""Real Git replay preserves per-feature checks while reusing exact source prefixes."""

from collections import Counter
import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import validate_feature_cherry_picks as validator


class FlushedOutput(io.StringIO):
    def flush(self):
        self.flushed = self.getvalue()


class FeatureValidationReuseTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.git("init", "--quiet")
        for key, value in (
            ("user.name", "Fixture"),
            ("user.email", "fixture@localhost"),
            ("commit.gpgSign", "false"),
            ("core.autocrlf", "false"),
        ):
            self.git("config", key, value)
        hooks = self.root / "disabled-hooks"
        hooks.mkdir()
        self.git("config", "core.hooksPath", str(hooks))
        self.base = self.commit(
            "Base",
            {
                "ruinedfooocus/__init__.py": "",
                "ruinedfooocus/core/__init__.py": "",
                "ruinedfooocus/core/paths.py": "from pathlib import Path\ndef resource_path(name): return Path(__file__).parents[1] / 'resources' / name\n",
                "ruinedfooocus/consumer.py": "from ruinedfooocus.core.paths import resource_path\nVALUE = resource_path('shared.txt').read_text()\n",
                "ruinedfooocus/resources/shared.txt": "shared",
                "conflict.txt": "base\n",
                "tests/test_fixture.py": """import os
from pathlib import Path
import unittest
from ruinedfooocus.consumer import VALUE
class Fixture(unittest.TestCase):
    def test_resource_and_execution(self):
        self.assertEqual(VALUE, 'shared')
        (Path(os.environ['RF_DATA_DIR']) / 'executed.txt').write_text('ran')
""",
            },
        )
        self.features = []
        self.shas = {}

    def git(self, *args):
        return subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True, check=True
        ).stdout.strip()

    def commit(self, subject, files):
        for name, text in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
        self.git("add", "--all")
        self.git("commit", "--quiet", "-m", subject)
        return self.git("rev-parse", "HEAD")

    def feature(self, key, files, requires=()):
        self.shas[key] = self.commit(key, files)
        self.features.append(
            dict(id=key, subject=key, requires=list(requires), tests=["test_fixture.py"])
        )

    def native_source_fixture(self, sources):
        vendor = self.root / "repositories/ComfyUI"
        vendor.mkdir(parents=True)
        subprocess.run(["git", "init", "--quiet", str(vendor)], check=True)
        for name, text in sources.items():
            path = vendor / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        subprocess.run(["git", "-C", str(vendor), "add", "--all"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(vendor),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@localhost",
                "-c",
                "commit.gpgSign=false",
                "commit",
                "--quiet",
                "-m",
                "Pinned native fixture",
            ],
            check=True,
        )
        return subprocess.check_output(
            ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
        ).strip()

    def replay(self, selected, *, limit=4096, tests=False, check_before=None):
        output = FlushedOutput()
        output.flushed = ""
        picks, checked_trees, executed = [], [], []
        batches = []
        run, contracts = subprocess.run, validator.python_contracts

        def command(args, **kwargs):
            if (
                args[0] == "git"
                and len(args) > 1
                and args[1] == "cherry-pick"
                and "--allow-empty" in args
            ):
                batches.append(args[3:])
                picks.extend(args[3:])
                if check_before:
                    for sha in args[3:]:
                        check_before(sha, output.flushed)
            result = run(args, **kwargs)
            if args[0] != "git" and len(args) > 1 and args[1] == "-c":
                executed.append(result.returncode)
                # A later feature must reset tracked changes made by prior tests.
                (Path(kwargs["cwd"]) / "ruinedfooocus/resources/shared.txt").write_text("dirty")
            return result

        def check(folder, **options):
            checked_trees.append(
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD^{tree}"], cwd=folder, text=True
                ).strip()
            )
            return contracts(folder, **options)

        cache = validator.AppliedPrefixes(limit=limit)
        with (
            patch.object(validator, "AppliedPrefixes", return_value=cache),
            patch.object(validator.subprocess, "run", side_effect=command),
            patch.object(validator, "python_contracts", side_effect=check),
            contextlib.redirect_stdout(output),
        ):
            report = validator.run(
                self.root, dict(base=self.base, features=self.features), selected, tests=tests
            )
        self.last_cherry_pick_batches = batches
        return report, picks, checked_trees, executed, output.getvalue(), cache

    def test_batches_preserve_commit_order_trees_and_individual_source_identities(self):
        previous = []
        for index in range(35):
            key = f"step-{index}"
            self.feature(key, {f"steps/{index}.txt": str(index)}, previous)
            previous = [key]
        with patch.object(validator, "CHERRY_PICK_BATCH_SIZE", 1):
            serial = self.replay(previous, limit=0)
        serial_batches = self.last_cherry_pick_batches
        batched = self.replay(previous, limit=0)
        self.assertEqual(len(serial_batches), 35)
        self.assertEqual([len(batch) for batch in self.last_cherry_pick_batches], [32, 3])
        self.assertEqual(serial[1], batched[1])
        self.assertEqual(serial[2], batched[2])
        for result in (serial, batched):
            record = result[0]["features"][0]
            self.assertEqual(record["status"], "passed", record)
            self.assertEqual(record["commits"], [self.shas[f"step-{index}"] for index in range(35)])
            count = subprocess.check_output(
                ["git", "rev-list", "--count", self.base + "..HEAD"],
                cwd=result[0]["scratch"],
                text=True,
            )
            self.assertEqual(int(count), 35)
            messages = subprocess.check_output(
                ["git", "log", "--reverse", "--format=%s", self.base + "..HEAD"],
                cwd=result[0]["scratch"],
                text=True,
            ).splitlines()
            self.assertEqual(messages, [f"step-{index}" for index in range(35)])

    def test_conflict_inside_batch_preserves_only_successful_prefix_and_clears_sequencer(self):
        self.feature("common", {"common.txt": "shared"})
        self.commit("Omitted source context", {"conflict.txt": "omitted\n"})
        self.feature("bad", {"conflict.txt": "changed\n"}, ["common"])
        self.feature("after", {"must-not-exist.txt": "later"}, ["bad"])
        report, _, _, _, _, cache = self.replay(["after"])
        record = report["features"][0]
        self.assertEqual(record["status"], "failed")
        self.assertIn("Cherry-pick bad failed", record["error"])
        self.assertEqual(record["commits"], [self.shas["common"]])
        length, head = cache.longest([self.shas["common"], self.shas["bad"]])
        self.assertEqual(length, 1)
        scratch = Path(report["scratch"])
        self.assertEqual(
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=scratch, text=True).strip(),
            head,
        )
        self.assertFalse((scratch / "must-not-exist.txt").exists())
        self.assertFalse((scratch / ".git/CHERRY_PICK_HEAD").exists())
        self.assertFalse((scratch / ".git/sequencer").exists())

    def test_source_ref_is_pinned_before_commit_subject_lookup(self):
        self.feature("selected", {"selected.txt": "selected"})
        expected = self.git("rev-parse", "HEAD")
        check_output = subprocess.check_output
        histories = []

        def output(args, **kwargs):
            if args[:2] == ["git", "log"]:
                self.commit("Later source update", {"later.txt": "outside snapshot"})
                histories.append(args[-1])
            return check_output(args, **kwargs)

        with patch.object(validator.subprocess, "check_output", side_effect=output):
            report, _, _, _, _, _ = self.replay(["selected"])
        self.assertEqual(report["source_ref_sha"], expected)
        self.assertEqual(histories, [self.base + ".." + expected])
        self.assertNotEqual(self.git("rev-parse", "HEAD"), expected)
        self.assertEqual(report["features"][0]["status"], "passed")

    def test_isolated_closure_reports_deferred_integration_without_hiding_eager_failures(self):
        self.feature(
            "optional",
            {
                "ruinedfooocus/optional.py": (
                    "def invoke():\n    from ruinedfooocus.integration import execute\n"
                    "    return execute()\n"
                )
            },
        )
        self.feature("eager", {"ruinedfooocus/eager.py": "import ruinedfooocus.integration\n"})
        report, _, _, _, _, _ = self.replay(["optional", "eager"])
        optional, eager = report["features"]
        self.assertEqual(optional["status"], "passed", optional)
        self.assertEqual(
            optional["deferred_imports"],
            [
                dict(
                    path="ruinedfooocus/optional.py",
                    line=2,
                    scope="invoke",
                    error="missing ruinedfooocus.integration",
                )
            ],
        )
        self.assertEqual(eager["status"], "failed", eager)
        self.assertIn("missing ruinedfooocus.integration", eager["error"])

    def test_selected_runtime_callers_still_fail_for_unavailable_deferred_dependencies(self):
        self.feature(
            "optional",
            {
                "ruinedfooocus/optional.py": (
                    "def invoke():\n    from ruinedfooocus.integration import execute\n"
                    "    return execute()\n"
                ),
                "tests/test_fixture.py": (
                    "import unittest\nfrom ruinedfooocus.optional import invoke\n"
                    "class Fixture(unittest.TestCase):\n"
                    "    def test_runtime_dependency(self): invoke()\n"
                ),
            },
        )
        report, _, _, executed, _, _ = self.replay(["optional"], tests=True)
        record = report["features"][0]
        self.assertEqual(executed, [1])
        self.assertEqual(record["status"], "failed", record)
        self.assertEqual(len(record["deferred_imports"]), 1)
        self.assertIn("ModuleNotFoundError", record["error"])

    def test_exact_prefix_reuse_reduces_real_picks_with_identical_trees_lists_and_all_checks(self):
        self.feature("common", {"common.txt": "shared"})
        self.feature("a", {"a.txt": "A"}, ["common"])
        self.feature("b", {"b.txt": "B"}, ["common"])
        self.feature("a2", {"a2.txt": "A2"}, ["a"])
        baseline = self.replay(["a", "b", "a2"], limit=0, tests=True)
        cached = self.replay(["a", "b", "a2"], tests=True)
        self.assertEqual(len(baseline[1]), 7)
        self.assertEqual(len(cached[1]), 4)
        self.assertEqual(baseline[2], cached[2])
        for result in (baseline, cached):
            self.assertEqual(result[3], [0, 0, 0])
            self.assertEqual(len(result[2]), 3)
            for feature in result[0]["features"]:
                self.assertEqual(feature["status"], "passed", feature)
                self.assertIn("executed=1", feature["test_output"])
                self.assertTrue(
                    (
                        Path(result[0]["scratch"]) / "tmp" / feature["id"] / "data/executed.txt"
                    ).is_file()
                )
        fields = ("id", "closure", "commits", "status", "python_files", "tests")
        self.assertEqual(
            [{key: row[key] for key in fields} for row in baseline[0]["features"]],
            [{key: row[key] for key in fields} for row in cached[0]["features"]],
        )

    def test_failed_pick_is_not_cached_and_its_actual_conflict_flushes_before_next_feature(self):
        self.feature("common", {"common.txt": "shared"})
        self.commit("Omitted source context", {"conflict.txt": "omitted\n"})
        self.feature("bad", {"conflict.txt": "changed\n"}, ["common"])
        self.feature("good", {"good.txt": "good"}, ["common"])
        observed = []

        def before(sha, flushed):
            if sha == self.shas["good"]:
                observed.append(flushed)

        report, picks, trees, _, text, _ = self.replay(["bad", "good", "bad"], check_before=before)
        self.assertEqual(
            Counter(picks),
            Counter({self.shas["common"]: 1, self.shas["bad"]: 2, self.shas["good"]: 1}),
        )
        self.assertEqual(
            [row["status"] for row in report["features"]], ["failed", "passed", "failed"]
        )
        self.assertEqual(len(trees), 1)
        self.assertTrue(observed)
        self.assertIn("bad: failed", observed[0])
        self.assertIn("CONFLICT (content): Merge conflict in conflict.txt", observed[0])
        self.assertNotIn("Auto-merging", observed[0])
        self.assertNotIn("bad: failed [", text)
        for row in (report["features"][0], report["features"][2]):
            self.assertEqual(row["commits"], [self.shas["common"]])
            self.assertIn("Auto-merging", row["error"])
            self.assertIn("conflict.txt", row["error"])

    def test_source_order_and_cache_bound_never_substitute_an_unmatched_prefix(self):
        self.feature("common", {"common.txt": "shared"})
        self.feature("x", {"x.txt": "X"})
        self.feature("y", {"y.txt": "Y"})
        self.feature("xy", {"xy.txt": "XY"}, ["common", "x", "y"])
        self.feature("yx", {"yx.txt": "YX"}, ["common", "y", "x"])
        report, picks, trees, _, _, cache = self.replay(["xy", "yx"], limit=2)
        self.assertEqual(Counter(picks)[self.shas["common"]], 1)
        self.assertEqual(Counter(picks)[self.shas["x"]], 2)
        self.assertEqual(Counter(picks)[self.shas["y"]], 2)
        self.assertEqual(cache.size, 2)
        self.assertEqual(len(trees), 2)
        self.assertTrue(all(row["status"] == "passed" for row in report["features"]))
        self.assertEqual(
            report["features"][1]["commits"], [self.shas[key] for key in ("common", "y", "x", "yx")]
        )

    def test_resource_contracts_recheck_each_feature_even_with_cached_applied_sources(self):
        self.feature(
            "broken-resource",
            {
                "ruinedfooocus/broken.py": "from ruinedfooocus.core.paths import resource_path\nMISSING = resource_path('absent.txt')\n"
            },
        )
        self.feature("a", {"a.txt": "A"}, ["broken-resource"])
        self.feature("b", {"b.txt": "B"}, ["broken-resource"])
        report, picks, trees, executed, _, _ = self.replay(["a", "b"], tests=True)
        self.assertEqual(Counter(picks)[self.shas["broken-resource"]], 1)
        self.assertEqual(len(trees), 2)
        self.assertEqual(executed, [])
        self.assertTrue(
            all(
                row["status"] == "failed" and "missing bundled resource" in row["error"]
                for row in report["features"]
            )
        )

    def test_guided_edit_suites_reuse_the_existing_bounded_native_sources(self):
        (self.root / ".git/info/exclude").write_text("repositories/\n")
        sources = {
            "nodes.py": "NATIVE_CONTROL_AND_VAE = True\n",
            "comfy_extras/nodes_mask.py": "NATIVE_MASK_INVERT = True\n",
        }
        native_revision = self.native_source_fixture(sources)
        unneeded = self.root / "repositories/ComfyUI/comfy_extras/unselected_private.py"
        unneeded.write_text("MUST_NOT_COPY = True\n")
        for key, pattern in (
            ("owned-guide", "test_guide_handoffs.py"),
            ("uploaded-guide", "test_uploaded_guide_handoffs.py"),
            ("guided-pipeline", "test_guided_pipeline_contract.py"),
        ):
            self.assertEqual(validator.native_cpu_test_files([pattern]), [])
            self.feature(
                key,
                {
                    "tests/" + pattern: "from pathlib import Path\nimport unittest\n"
                    "class Fixture(unittest.TestCase):\n"
                    "    def test_required_common_native_sources(self):\n"
                    f"        for name, expected in {sources!r}.items():\n"
                    "            self.assertEqual((Path('repositories/ComfyUI') / name).read_text(), expected)\n"
                    "        self.assertFalse(Path('repositories/ComfyUI/comfy_extras/unselected_private.py').exists())\n"
                },
            )
            self.features[-1].update(
                tests=[pattern], native_comfy=True, native_comfy_revision=native_revision
            )
            report, _, _, executed, _, _ = self.replay([key], tests=True)
            self.assertEqual(executed, [0])
            self.assertEqual(report["features"][0]["status"], "passed", report["features"][0])
            self.assertEqual(report["features"][0]["native_comfy_revision"], native_revision)

    def test_undeclared_native_sources_are_not_copied_from_mutable_vendor_files(self):
        (self.root / ".git/info/exclude").write_text("repositories/\n")
        source = self.root / "repositories/ComfyUI/folder_paths.py"
        source.parent.mkdir(parents=True)
        source.write_text("UNDECLARED_MUTABLE_SOURCE = True\n", encoding="utf-8")
        self.feature(
            "ordinary",
            {
                "tests/test_fixture.py": "from pathlib import Path\nimport unittest\n"
                "class Fixture(unittest.TestCase):\n"
                "    def test_no_undeclared_native_source(self):\n"
                "        self.assertFalse(Path('repositories/ComfyUI/folder_paths.py').exists())\n"
            },
        )
        report, _, _, executed, _, _ = self.replay(["ordinary"], tests=True)
        self.assertEqual(executed, [0])
        self.assertEqual(report["features"][0]["status"], "passed", report["features"][0])
        self.assertNotIn("native_comfy_revision", report["features"][0])

    def test_native_geometry_and_color_sources_are_staged_after_wildcard_expansion(self):
        (self.root / ".git/info/exclude").write_text("repositories/\n")
        sources = {
            "comfy_extras/nodes_post_processing.py": "POST_PROCESSING = True\n",
            "comfy_extras/color_util.py": "COLOR_UTIL = True\n",
            "node_helpers.py": "NODE_HELPERS = True\n",
            "comfy_extras/nodes_text_overlay.py": "TEXT_OVERLAY = True\n",
            "comfy_extras/nodes_morphology.py": "MORPHOLOGY = True\n",
        }
        geometry = {
            name: text
            for name, text in sources.items()
            if not name.endswith(("nodes_text_overlay.py", "nodes_morphology.py"))
        }
        color = {name: text for name, text in sources.items() if not name.endswith("color_util.py")}
        native_revision = self.native_source_fixture(sources)
        (self.root / "repositories/ComfyUI/node_helpers.py").write_text(
            "UNCOMMITTED_NATIVE_SOURCE_MUST_NOT_BE_USED = True\n", encoding="utf-8"
        )
        for key, pattern, expected in (
            ("geometry", "test_image_geometry*.py", geometry),
            ("integration", "test_image_adjustment_integration.py", sources),
            ("callers", "test_image_plan_callers.py", sources),
            ("residual", "test_image_residual_ui_api.py", sources),
            ("crop", "test_image_crop_to_mask.py", geometry),
            ("finishing", "test_image_color_finishing.py", color),
            ("color", "test_image_color_tools.py", color),
        ):
            test = "test_image_geometry_tools.py" if key == "geometry" else pattern
            self.feature(
                key,
                {
                    "tests/" + test: "from pathlib import Path\nimport unittest\n"
                    "class Fixture(unittest.TestCase):\n"
                    "    def test_selected_native_sources(self):\n"
                    f"        for name, expected in {expected!r}.items():\n"
                    "            self.assertEqual((Path('repositories/ComfyUI') / name).read_text(), expected)\n"
                },
            )
            self.features[-1].update(
                tests=[pattern], native_comfy=True, native_comfy_revision=native_revision
            )
            report, _, _, executed, _, _ = self.replay([key], tests=True)
            self.assertEqual(executed, [0])
            self.assertEqual(report["features"][0]["status"], "passed", report["features"][0])
            self.assertEqual(report["features"][0]["native_comfy_revision"], native_revision)
            for name in sources.keys() - expected.keys():
                self.assertFalse((Path(report["scratch"]) / "repositories/ComfyUI" / name).exists())

    def test_failure_summary_keeps_the_python_root_cause_after_a_long_traceback(self):
        output = FlushedOutput()
        error = (
            "CPU tests failed:\n"
            + "\n".join(f"  frame {index}" for index in range(20))
            + "\nAssertionError: expected captured source\n"
        )
        with contextlib.redirect_stdout(output):
            validator.print_failure("cpu-fixture", error)
        self.assertIn("AssertionError: expected captured source", output.flushed)
        self.assertLess(len(output.flushed.splitlines()), 5)


if __name__ == "__main__":
    unittest.main()
