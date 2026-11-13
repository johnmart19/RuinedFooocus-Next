"""Cherry-pick features and their declared prerequisites onto original upstream.

Uses a separate temporary Git repository with read-only access to source objects.
Does not move the application branch, install packages, launch the app, or run video
tests. Use the application's Python environment for the optional CPU tests.
"""

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tarfile
import re
import time

CHERRY_PICK_BATCH_SIZE = 32


class CherryPickFailure(RuntimeError):
    def __init__(self, message, applied_head=None):
        super().__init__(message)
        self.applied_head = applied_head


def cherry_pick_suffix(git, base, dependencies, by_id, subjects, applied):
    """Keep individual commits and bound each Git argument list to 32 source SHAs."""
    for start in range(0, len(dependencies), CHERRY_PICK_BATCH_SIZE):
        batch = dependencies[start : start + CHERRY_PICK_BATCH_SIZE]
        shas = [subjects[by_id[key]["subject"]] for key in batch]
        result = git("cherry-pick", "--allow-empty", *shas)
        if result.returncode:
            progress = git("rev-list", "--first-parent", "--reverse", base + "..HEAD")
            snapshots = progress.stdout.splitlines()
            completed = len(snapshots) - len(applied)
            head = None
            failed = "batch " + ", ".join(batch)
            if progress.returncode == 0 and 0 <= completed < len(batch):
                applied.extend(shas[:completed])
                head = snapshots[-1] if snapshots else base
                failed = batch[completed]
            raise CherryPickFailure(
                f"Cherry-pick {failed} failed:\n{result.stdout}\n{result.stderr}", head
            )
        applied.extend(shas)


class AppliedPrefixes:
    """Invocation-local exact source prefixes; at most 4096 committed snapshots."""

    def __init__(self, limit=4096):
        self.root = {}
        self.limit = limit
        self.size = 0

    def longest(self, commits):
        children, length, head = self.root, 0, None
        for sha in commits:
            entry = children.get(sha)
            if entry is None:
                break
            head, children = entry
            length += 1
        return length, head

    def remember(self, commits, snapshots):
        children = self.root
        for sha, head in zip(commits, snapshots, strict=True):
            entry = children.get(sha)
            if entry is None:
                if self.size >= self.limit:
                    return
                entry = (head, {})
                children[sha] = entry
                self.size += 1
            _, children = entry


def remember_applied(prefixes, git, base, commits):
    if not commits or prefixes.size >= prefixes.limit:
        return
    result = git("rev-list", "--first-parent", "--reverse", base + "..HEAD")
    snapshots = result.stdout.splitlines()
    # An uncertain snapshot is a cache miss, never a reason to skip validation.
    if result.returncode == 0 and len(snapshots) == len(commits):
        prefixes.remember(commits, snapshots)


def print_failure(key, error):
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    causes = [
        line
        for line in lines[1:]
        if line.startswith(("CONFLICT", "error:", "fatal:"))
        or re.match(r"(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception):", line)
    ]
    summary = list(dict.fromkeys(lines[:1] + (causes or lines[1:])))[:8]
    print(key + ": failed\n" + "\n".join("  " + line[:1000] for line in summary), flush=True)


def native_comfy_snapshot(root, scratch, revision=None):
    """Export the consumer's exact pin; never import a mutable vendor checkout."""
    return native_snapshot(root, scratch, "ComfyUI", "ComfyUI", revision=revision)


def native_snapshot(root, scratch, repository, folder, revision=None):
    def exact_pin(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value)

    if revision is not None and not exact_pin(revision):
        raise ValueError(f"{repository} validation requires an exact commit pin")
    bootstrap = scratch / "ruinedfooocus/app/bootstrap.py"
    if bootstrap.is_file():
        tree = ast.parse(bootstrap.read_text(encoding="utf-8"))
        value = next(
            (
                node.value
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "git_repos"
                    for target in node.targets
                )
            ),
            None,
        )
        pins = ast.literal_eval(value) if value is not None else []
        consumer_pin = next((row["hash"] for row in pins if row["name"] == repository), None)
        if not exact_pin(consumer_pin):
            raise ValueError(f"{repository} consumer requires an exact commit pin")
        if revision is not None and revision != consumer_pin:
            raise ValueError(f"{repository} declared pin does not match the consumer bootstrap pin")
        revision = consumer_pin
    if not exact_pin(revision):
        raise ValueError(f"{repository} validation requires an exact commit pin")
    source = root / "repositories" / folder
    object_type = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(source), "cat-file", "-t", revision],
        capture_output=True,
        text=True,
    )
    if object_type.returncode:
        raise RuntimeError(f"Pinned {repository} source is unavailable: {object_type.stderr}")
    if object_type.stdout.strip() != "commit":
        raise ValueError(f"{repository} validation requires an exact commit object")
    parent = scratch / "tmp"
    parent.mkdir(exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix=f"native-{folder.lower()}-", dir=parent)).resolve()
    if not destination.is_relative_to(scratch.resolve()):
        raise ValueError("Native snapshot must stay inside the validation checkout")
    with tempfile.TemporaryFile() as stream:
        result = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(source), "archive", "--format=tar", revision],
            stdout=stream,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"Pinned {repository} source is unavailable: {result.stderr}")
        stream.seek(0)
        with tarfile.open(fileobj=stream) as archive:
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if not target.is_relative_to(destination) or not (
                    member.isfile() or member.isdir()
                ):
                    raise ValueError("Native source archive contains an unsafe entry")
            archive.extractall(destination, filter="data")
    return destination, revision


def load_manifest(path):
    """Read monolithic manifests or independently cherry-pickable registrations."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    features = list(manifest.get("features", []))
    directory = manifest.get("feature_directory")
    if directory:
        folder = (path.parent / directory).resolve()
        if not folder.is_relative_to(path.parent.resolve()) or not folder.is_dir():
            raise ValueError("Feature directory must exist inside the manifest directory")
        for registration in sorted(folder.glob("*.json")):
            item = json.loads(registration.read_text(encoding="utf-8"))
            if item["id"] != registration.stem:
                raise ValueError(f"Feature ID must match its filename: {registration.name}")
            features.append(item)
    ids = [item["id"] for item in features]
    subjects = [item["subject"] for item in features]
    if len(ids) != len(set(ids)) or len(subjects) != len(set(subjects)):
        raise ValueError("Feature IDs and subjects must be unique")
    for item in features:
        for key in ("native_comfy_revision", "native_gguf_revision"):
            if key in item and not (
                isinstance(item[key], str) and re.fullmatch(r"[0-9a-f]{40}", item[key])
            ):
                raise ValueError(f"{item['id']}: {key} requires an exact commit pin")
    dependency_order(features, ids)
    return {**manifest, "features": features}


def dependency_order(features, selected):
    by_id = {item["id"]: item for item in features}
    result, active, visited = [], set(), set()

    def visit(key):
        if key not in by_id:
            raise ValueError(f"Unknown feature: {key}")
        if key in active:
            raise ValueError(f"Cyclic feature dependency: {key}")
        if key in visited:
            return
        active.add(key)
        for prerequisite in by_id[key].get("requires", []):
            visit(prerequisite)
        active.remove(key)
        visited.add(key)
        result.append(key)

    for key in selected:
        visit(key)
    return result


def module_exports(tree):
    """Collect module bindings without mistaking function locals for exports."""
    exports = set()

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exports.add(node.name)
            return
        if isinstance(
            node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            exports.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            exports.update(a.asname or a.name.split(".")[0] for a in node.names)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return exports


def cpu_test_code(patterns):
    return f"""import os
import sys
import unittest
from collections import Counter
os.environ.update(CUDA_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1')
sys.argv = [sys.argv[0], '--cpu']
import torch
if torch.cuda.is_initialized():
    raise RuntimeError('CPU feature checks must not initialize a GPU')
def prohibit_gpu(*args, **kwargs):
    raise RuntimeError('GPU initialization prohibited in CPU feature checks')
torch.cuda._lazy_init = prohibit_gpu
for name in ('init', 'is_available', 'device_count', 'current_device',
             'get_device_properties', 'get_device_name', 'get_device_capability', 'mem_get_info'):
    setattr(torch.cuda, name, prohibit_gpu)
patterns = {patterns!r}
suites = []
for pattern in patterns:
    suite = unittest.defaultTestLoader.discover('tests', pattern=pattern)
    if suite.countTestCases() == 0:
        raise ValueError('No CPU tests discovered for ' + pattern)
    suites.append(suite)
def leaf_cases(suite):
    for case in suite:
        if isinstance(case, unittest.TestSuite):
            yield from leaf_cases(case)
        else:
            yield case

discovered_cases = [case for suite in suites for case in leaf_cases(suite)]
expected_ids = Counter(case.id() for case in discovered_cases)
expected = sum(expected_ids.values())
fixture_groups = {{}}
for case in discovered_cases:
    module = type(case).__module__
    classname = module + '.' + type(case).__qualname__
    for target in ('setUpClass (' + classname + ')', 'setUpModule (' + module + ')'):
        fixture_groups.setdefault(target, Counter())[case.id()] += 1
del discovered_cases, case

class ExecutionResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed_ids = []
        self.leaf_skips = Counter()
        self.fixture_skips = Counter()
        self.fixture_events = 0
        self.subtest_skips = 0
        self.invalid_skips = []
    def startTest(self, case):
        self.executed_ids.append(case.id())
        super().startTest(case)
    def addSkip(self, case, reason):
        identity = case.id()
        if isinstance(case, unittest.case._SubTest):
            if case.test_case.id() in self.executed_ids:
                self.subtest_skips += 1
            else:
                self.invalid_skips.append(identity)
        elif isinstance(case, unittest.suite._ErrorHolder):
            group = fixture_groups.get(identity)
            if group is not None:
                remaining = group - Counter(self.executed_ids) - self.fixture_skips
                if remaining:
                    self.fixture_skips.update(remaining)
                    self.fixture_events += 1
                else:
                    self.invalid_skips.append(identity)
            elif identity.replace('tearDownClass (', 'setUpClass (', 1).replace('tearDownModule (', 'setUpModule (', 1) in fixture_groups:
                self.fixture_events += 1
            else:
                self.invalid_skips.append(identity)
        elif identity in expected_ids:
            self.leaf_skips[identity] += 1
        else:
            self.invalid_skips.append(identity)
        super().addSkip(case, reason)

result = unittest.TextTestRunner(verbosity=1, resultclass=ExecutionResult).run(unittest.TestSuite(suites))
if torch.cuda.is_initialized():
    raise RuntimeError('CPU feature checks initialized a GPU')
started_ids = Counter(result.executed_ids)
disposed_ids = started_ids + result.fixture_skips
executed = sum((started_ids - result.leaf_skips).values())
skipped = sum(result.leaf_skips.values()) + sum(result.fixture_skips.values())
print(f'CPU dispositions: executed={{executed}}, skipped={{skipped}}, discovered={{expected}}, fixture_skip_events={{result.fixture_events}}, subtest_skips={{result.subtest_skips}}')
if result.testsRun != sum(started_ids.values()) or disposed_ids != expected_ids or result.invalid_skips or any(count > started_ids[identity] for identity, count in result.leaf_skips.items()):
    raise RuntimeError(f'CPU runner executed {{executed}} and explicitly skipped {{skipped}} of {{expected}} discovered tests with mismatched leaf dispositions; inspect TestCase.run overrides and custom suites')
raise SystemExit(not result.wasSuccessful())
"""


def cpu_test_patterns(folder, patterns):
    """Expand patterns before discovery so excluded modules are never imported."""
    selected = set()
    for pattern in patterns:
        if Path(pattern).name != pattern:
            raise ValueError("CPU test patterns must be filenames inside tests/")
        matches = {
            path.name
            for path in (folder / "tests").glob(pattern)
            if path.is_file()
            and path.suffix == ".py"
            and not path.name.startswith(
                ("test_video_", "test_native_video_", "test_ltx_", "test_hunyuan_", "test_wan_")
            )
        }
        if not matches:
            raise ValueError(f"No non-video CPU tests matched {pattern}")
        selected.update(matches)
    return sorted(selected)


def native_cpu_test_files(test_patterns):
    """Additional source contracts for the concrete CPU suites selected for replay."""
    geometry = {
        "test_image_geometry_tools.py",
        "test_image_adjustment_integration.py",
        "test_image_plan_callers.py",
        "test_image_crop_to_mask.py",
        "test_image_residual_ui_api.py",
    }
    color = {
        "test_image_color_tools.py",
        "test_image_color_finishing.py",
        "test_image_adjustment_integration.py",
        "test_image_plan_callers.py",
        "test_image_residual_ui_api.py",
    }
    selected = set(test_patterns)
    files = []
    if selected & (geometry | color | {"test_image_filters.py"}):
        files.append("comfy_extras/nodes_post_processing.py")
    if selected & geometry:
        files.append("comfy_extras/color_util.py")
    if selected & color:
        files.extend(["comfy_extras/nodes_text_overlay.py", "comfy_extras/nodes_morphology.py"])
    return files


def import_scopes(tree):
    """Class bodies execute in their enclosing scope; function bodies are deferred."""
    scopes = {}

    def visit(node, deferred=False, scope=()):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            scopes[id(node)] = (deferred, ".".join(scope) or "<module>")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Decorators, defaults and annotations execute in the enclosing scope.
            for child in ast.iter_child_nodes(node):
                if child not in node.body:
                    visit(child, deferred, scope)
            for child in node.body:
                visit(child, True, (*scope, node.name))
            return
        if isinstance(node, ast.ClassDef):
            # Bases and decorators execute before the class body, without making
            # imports in a module-level class into optional runtime references.
            for child in ast.iter_child_nodes(node):
                if child not in node.body:
                    visit(child, deferred, scope)
            for child in node.body:
                visit(child, deferred, (*scope, node.name))
            return
        for child in ast.iter_child_nodes(node):
            visit(child, deferred, scope)

    visit(tree)
    return scopes


def local_imports(node, path):
    """Resolve the repository's absolute and package-relative import statements."""
    roots = {"ruinedfooocus", "modules", "tools"}
    if isinstance(node, ast.Import):
        return [(alias.name, []) for alias in node.names if alias.name.split(".")[0] in roots]
    if not isinstance(node, ast.ImportFrom):
        return []
    module = node.module or ""
    if node.level:
        package = path.split("/")[:-1]
        if not package or package[0] not in roots:
            return []
        if node.level > len(package):
            raise ValueError("relative import beyond local package")
        module = ".".join([*package[: len(package) - node.level + 1], *module.split(".")]).rstrip(
            "."
        )
    return [(module, node.names)] if module.split(".")[0] in roots else []


def dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return parent + "." + node.attr if parent else None
    return None


def python_contracts(folder, *, allow_deferred=False, deferred_imports=None):
    """Check all local imports, or report deferred misses for an isolated closure.

    Full-tree callers remain strict by default. A closure can explicitly defer
    function-body imports only when its caller collects the unresolved references;
    its selected runtime tests still own the dependencies of exercised functions.
    """
    if allow_deferred and deferred_imports is None:
        raise ValueError("Deferred import checks require a report collector")
    trees = {}
    excluded = {
        ".git",
        "repositories",
        "venv",
        "tmp",
        "cache",
        "models",
        "outputs",
        ".codex",
        ".claude",
        "__pycache__",
    }
    for directory, children, files in os.walk(folder):
        children[:] = [
            name
            for name in children
            if name not in {"__pycache__", ".venv", "venv", "node_modules"}
            and (Path(directory) != folder or name not in excluded)
        ]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = Path(directory) / name
            rel = path.relative_to(folder).as_posix()
            trees[rel] = ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel)
    failures = []
    for path, tree in trees.items():
        scopes = import_scopes(tree)

        def missing_import(node, message):
            deferred, scope = scopes[id(node)]
            if allow_deferred and deferred:
                deferred_imports.append(
                    dict(path=path, line=node.lineno, scope=scope, error=message)
                )
            else:
                failures.append(f"{path}:{node.lineno}: {message}")

        resolved_imports = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                try:
                    resolved_imports[id(node)] = local_imports(node, path)
                except ValueError as error:
                    missing_import(node, str(error))
        resource_bindings = {}
        if path.startswith("ruinedfooocus/"):
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "ruinedfooocus.core.paths":
                            resource_bindings[alias.asname or "ruinedfooocus"] = (
                                alias.name if alias.asname else "ruinedfooocus"
                            )
                for module, aliases in resolved_imports.get(id(node), []):
                    for alias in aliases:
                        name = module + "." + alias.name
                        if name in {
                            "ruinedfooocus.core.paths",
                            "ruinedfooocus.core.paths.asset_path",
                            "ruinedfooocus.core.paths.resource_path",
                        }:
                            resource_bindings[alias.asname or alias.name] = name
        for node in ast.walk(tree):
            call_name = dotted_name(node.func) if isinstance(node, ast.Call) else None
            if call_name:
                first, *attributes = call_name.split(".")
                call_name = (
                    ".".join([resource_bindings[first], *attributes])
                    if first in resource_bindings
                    else None
                )
            resource_helper = (
                call_name.rsplit(".", 1)[-1]
                if call_name
                in {
                    "ruinedfooocus.core.paths.asset_path",
                    "ruinedfooocus.core.paths.resource_path",
                }
                else None
            )
            if (
                isinstance(node, ast.Call)
                and resource_helper
                and node.args
                and all(
                    isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    for arg in node.args
                )
            ):
                directory = "ui/assets" if resource_helper == "asset_path" else "resources"
                resource = folder / "ruinedfooocus" / directory
                resource = resource.joinpath(*(arg.value for arg in node.args))
                if not resource.exists():
                    failures.append(
                        f"{path}:{node.lineno}: missing bundled resource {resource.relative_to(folder)}"
                    )
            for name, imports in resolved_imports.get(id(node), []):
                stem = name.replace(".", "/")
                initializer = stem + "/__init__.py"
                target = initializer if initializer in trees else stem + ".py"
                package = initializer in trees or (target not in trees and (folder / stem).is_dir())
                if target not in trees and not package:
                    missing_import(node, f"missing {name}")
                    continue
                exports = module_exports(trees[target]) if target in trees else set()
                for alias in imports:
                    submodule = name.replace(".", "/") + "/" + alias.name
                    if (
                        alias.name != "*"
                        and alias.name not in exports
                        and not (
                            package
                            and (submodule + ".py" in trees or (folder / submodule).is_dir())
                        )
                    ):
                        missing_import(node, f"{name} has no {alias.name}")
    if failures:
        raise ValueError("\n".join(failures))
    return len(trees)


def run(root, manifest, selected, ref="HEAD", tests=False):
    root = root.resolve()
    source_ref_sha = subprocess.check_output(
        ["git", "rev-parse", "--verify", ref + "^{commit}"], cwd=root, text=True
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_ref_sha):
        raise ValueError("Feature validation requires an exact source commit")
    temp_root = root / "tmp"
    temp_root.mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="feature-picks-", dir=temp_root)).resolve()
    if scratch.parent != temp_root.resolve():
        raise ValueError("Validation checkout must remain under repository tmp/")
    objects = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--git-path", "objects"], cwd=root, text=True
        ).strip()
    )
    if not objects.is_absolute():
        objects = root / objects
    subprocess.run(["git", "init", "--quiet", str(scratch)], check=True)
    (scratch / ".git/objects/info/alternates").write_bytes(
        (objects.resolve().as_posix() + "\n").encode("utf-8")
    )

    def git(*args):
        return subprocess.run(["git", *args], cwd=scratch, capture_output=True, text=True)

    for key, value in [
        ("user.name", "RuinedFooocus validation"),
        ("user.email", "validation@localhost"),
        ("commit.gpgSign", "false"),
        ("core.autocrlf", "false"),
    ]:
        assert git("config", key, value).returncode == 0
    hooks = scratch / ".git/disabled-hooks"
    hooks.mkdir()
    assert git("config", "core.hooksPath", str(hooks)).returncode == 0
    checkout = git("checkout", "--detach", manifest["base"])
    if checkout.returncode:
        raise RuntimeError(checkout.stderr)
    history = subprocess.check_output(
        ["git", "log", "--format=%H%x09%s", manifest["base"] + ".." + source_ref_sha],
        cwd=root,
        text=True,
    )
    subjects = {}
    for row in history.splitlines():
        sha, subject = row.split("\t", 1)
        if subject in subjects:
            raise ValueError(f"Ambiguous commit subject: {subject}")
        subjects[subject] = sha
    by_id = {item["id"]: item for item in manifest["features"]}
    # Only native source contracts are copied. No backend or package is installed.
    native_files = [
        "folder_paths.py",
        "cuda_malloc.py",
        "node_helpers.py",
        "nodes.py",
        "comfy/sd.py",
        "comfy/sd1_tokenizer/vocab.json",
        "comfy/sd1_tokenizer/merges.txt",
        "comfy/sd1_tokenizer/tokenizer_config.json",
        "comfy/sd1_tokenizer/special_tokens_map.json",
        "comfy/cli_args.py",
        "comfy/model_detection.py",
        "comfy/supported_models.py",
        "comfy/supported_models_base.py",
        "comfy/model_base.py",
        "comfy/model_sampling.py",
        "comfy/pixel_space_convert.py",
        "comfy/samplers.py",
        "comfy/utils.py",
        "comfy/text_encoders/qwen35.py",
        "comfy/text_encoders/sam3_clip.py",
        "comfy/ldm/rt_detr/rtdetr_v4.py",
        "comfy_extras/nodes_fibo2.py",
        "comfy_extras/nodes_moss_image.py",
        "comfy_extras/nodes_textgen.py",
        "comfy_extras/nodes_cfg.py",
        "comfy_extras/nodes_slg.py",
        "comfy_extras/nodes_hidream_o1.py",
        "comfy_extras/nodes_qwen.py",
        "comfy_extras/nodes_latent.py",
        "comfy_extras/nodes_flux.py",
        "comfy_extras/nodes_ideogram4.py",
        "comfy_extras/nodes_edit_model.py",
        "comfy_extras/nodes_mask.py",
        "comfy_extras/nodes_dataset.py",
        "comfy_extras/nodes_compositing.py",
        "comfy_extras/nodes_chroma_radiance.py",
        "comfy_extras/nodes_sd3.py",
        "comfy_extras/nodes_model_advanced.py",
        "comfy_extras/nodes_lotus.py",
        "comfy_extras/nodes_moge.py",
        "comfy_extras/nodes_mediapipe.py",
        "comfy_extras/nodes_sdpose.py",
        "comfy_extras/nodes_marigold.py",
        "comfy_extras/nodes_cond.py",
        "comfy_extras/pose/keypoint_draw.py",
        "comfy_extras/nodes_images.py",
        "comfy_execution/preview_tonemap.py",
        "comfy_extras/nodes_custom_sampler.py",
        "comfy_extras/nodes_advanced_samplers.py",
        "comfy/k_diffusion/sampling.py",
        "comfy_api/latest/_io.py",
        "comfy/sd1_clip.py",
    ]
    report = dict(
        base=manifest["base"],
        ref=ref,
        source_ref_sha=source_ref_sha,
        scratch=str(scratch),
        video_tests=False,
        features=[],
    )
    prefixes = AppliedPrefixes()
    for key in selected:
        item = by_id[key]
        closure = dependency_order(manifest["features"], [key])
        ordered = [subjects.get(by_id[dependency]["subject"]) for dependency in closure]
        reused, head = prefixes.longest(ordered)
        if git("reset", "--hard", head or manifest["base"]).returncode:
            raise RuntimeError("Could not reset isolated validation repository")
        # Test files are not left behind between features by reset; dependencies
        # reuse only exact ordered source SHAs already applied in this scratch.
        # Contracts and tests still execute separately for every selected feature.
        record = dict(id=key, closure=closure, commits=ordered[:reused], status="passed")
        started = time.perf_counter()
        print(f"{key}: replaying {len(closure) - reused} commits ({reused} reused)", flush=True)
        captured = False
        try:
            cherry_pick_suffix(
                git, manifest["base"], closure[reused:], by_id, subjects, record["commits"]
            )
            record["replay_seconds"] = round(time.perf_counter() - started, 3)
            remember_applied(prefixes, git, manifest["base"], record["commits"])
            captured = True
            record["deferred_imports"] = []
            record["python_files"] = python_contracts(
                scratch, allow_deferred=True, deferred_imports=record["deferred_imports"]
            )
            test_patterns = item.get("tests", [])
            if tests and test_patterns and not item.get("skip_tests"):
                test_patterns = cpu_test_patterns(scratch, test_patterns)
                (scratch / "tmp").mkdir(exist_ok=True)
                fixture_source = None
                if item.get("native_comfy"):
                    fixture_source, native_revision = native_comfy_snapshot(
                        root, scratch, revision=item.get("native_comfy_revision")
                    )
                    record["native_comfy_revision"] = native_revision
                for rel in (
                    [*native_files, *native_cpu_test_files(test_patterns)] if fixture_source else []
                ):
                    original = fixture_source / rel
                    if original.is_file():
                        destination = scratch / "repositories/ComfyUI" / rel
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(original, destination)
                env = os.environ.copy()
                env["RF_OFFLINE"] = "1"
                env["RF_DATA_DIR"] = str(scratch / "tmp" / key / "data")
                (Path(env["RF_DATA_DIR"]) / "tmp").mkdir(parents=True, exist_ok=True)
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                if item.get("native_comfy"):
                    env["RF_TEST_COMFY_SOURCE"] = str(fixture_source)
                if item.get("native_gguf"):
                    native_source, native_revision = native_snapshot(
                        root,
                        scratch,
                        "molbal/ComfyUI-GGUF",
                        "molbal_comfyui_gguf",
                        revision=item.get("native_gguf_revision"),
                    )
                    env["RF_TEST_GGUF_SOURCE"] = str(native_source)
                    record["native_gguf_revision"] = native_revision
                env["TEMP"] = env["TMP"] = str(temp_root)
                code = cpu_test_code(test_patterns)
                import sys

                result = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=scratch,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                record["test_output"] = result.stdout + result.stderr
                if result.returncode:
                    raise RuntimeError("CPU tests failed:\n" + record["test_output"])
            record["tests"] = test_patterns if tests and not item.get("skip_tests") else []
        except (KeyError, RuntimeError, ValueError, SyntaxError) as error:
            record.update(status="failed", error=str(error))
            print_failure(key, error)
            if not captured:
                # HEAD contains only successful picks; never cache the conflicted
                # index or the failed source SHA. Preserve useful earlier prefixes.
                remember_applied(prefixes, git, manifest["base"], record["commits"])
            git("cherry-pick", "--abort")
            if isinstance(error, CherryPickFailure) and error.applied_head is not None:
                if git("reset", "--hard", error.applied_head).returncode:
                    raise RuntimeError("Could not restore the successful replay prefix") from error
        record["total_seconds"] = round(time.perf_counter() - started, 3)
        report["features"].append(record)
        if record["status"] == "passed":
            print(
                f"{key}: passed ({len(closure)} commits, {reused} reused; "
                f"{record['total_seconds']:.3f}s)",
                flush=True,
            )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("feature_commits.json")
    )
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--feature", action="append", help="Feature ID; default: all features")
    parser.add_argument(
        "--tests",
        action="store_true",
        help="Run declared CPU tests; video tests are always excluded",
    )
    parser.add_argument("--output", type=Path, default=Path("tmp/feature-cherry-picks.json"))
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    selected = args.feature or [item["id"] for item in manifest["features"]]
    report = run(Path(__file__).resolve().parents[1], manifest, selected, args.ref, args.tests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return any(item["status"] != "passed" for item in report["features"])


if __name__ == "__main__":
    raise SystemExit(main())
