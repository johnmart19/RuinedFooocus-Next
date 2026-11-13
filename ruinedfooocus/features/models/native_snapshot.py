"""Immutable native image facts, loaded once from a reviewed local snapshot.

Source inspection belongs to ``tools/update_native_snapshot.py``. Runtime callers
only read this data; a recipe is provenance, never model admission or execution.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import RLock
from types import MappingProxyType

SCHEMA_VERSION = 1
PARSER_VERSION = "native-image-facts-1"
ACCEPTED_COMFY_PIN = "989d93d0d968e704c589ce0146bb18c77f6b50cc"
ACCEPTED_GGUF_PIN = "da5f435cd9a9e54ad27eed417ed72b1f2543ef41"
DEFAULT_SNAPSHOT = Path(__file__).with_name("native_snapshot.json")
_MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
_lock = RLock()
_loaded = {}
_last_good = {}


def plain_data(value):
    """Return an owned JSON value from frozen facts."""
    if isinstance(value, Mapping):
        return {key: plain_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain_data(item) for item in value]
    return value


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def content_digest(value):
    return hashlib.sha256(
        json.dumps(
            plain_data(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class NativeImageFamilyFacts:
    family_id: str
    native_classes: tuple
    configurations: tuple
    component_roles: tuple
    template_models: tuple
    examples: tuple
    example_recipe: Mapping | None
    unresolved: tuple

    def metadata(self):
        return plain_data(self.__dict__)


@dataclass(frozen=True)
class NativeCapabilitySnapshot:
    manifest: Mapping
    families: Mapping
    aliases: Mapping
    schemas: Mapping
    recipes: Mapping

    @property
    def snapshot_id(self):
        return self.manifest["snapshot_id"]

    def metadata(self):
        """Capture identity without copying the registry or all recipes."""
        source = self.manifest["source_manifest"]
        return plain_data(
            {
                "schema_version": self.manifest["schema_version"],
                "parser_version": self.manifest["parser_version"],
                "snapshot_id": self.snapshot_id,
                "snapshot_sha256": self.manifest["snapshot_sha256"],
                "source_manifest_sha256": self.manifest["source_manifest_sha256"],
                "native_pin": source["native_pin"],
                "gguf_pin": source["gguf_pin"],
                "template_identity": {
                    key: source["template_identity"].get(key)
                    for key in (
                        "manifest_sha256",
                        "packages",
                        "recipe_assets_match_manifest",
                        "all_json_assets_match_manifest",
                        "observed_json_manifest_sha256",
                    )
                },
            }
        )


def snapshot_document(snapshot):
    """Maintainer/export interface; generation callers capture metadata only."""
    return {
        "manifest": plain_data(snapshot.manifest),
        "facts": {
            "families": {key: row.metadata() for key, row in snapshot.families.items()},
            "aliases": plain_data(snapshot.aliases),
            "schemas": plain_data(snapshot.schemas),
            "recipes": plain_data(snapshot.recipes),
        },
    }


def parse_native_snapshot(document):
    """Validate a complete manifest/facts envelope before exposing any row."""
    if not isinstance(document, dict) or set(document) != {"manifest", "facts"}:
        raise ValueError("Invalid native snapshot envelope.")
    manifest, facts = document["manifest"], document["facts"]
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("parser_version") != PARSER_VERSION
    ):
        raise ValueError("Native snapshot parser/schema requires an explicit migration.")
    source = manifest["source_manifest"]
    if source["native_pin"] != ACCEPTED_COMFY_PIN or source["gguf_pin"] != ACCEPTED_GGUF_PIN:
        raise ValueError("Native snapshot differs from the accepted native pins.")
    if source["template_identity"].get("recipe_assets_match_manifest") is not True:
        raise ValueError("Native recipe bytes do not match the accepted template manifest.")
    if manifest["source_manifest_sha256"] != content_digest(source):
        raise ValueError("Native snapshot source manifest digest differs.")
    if manifest["snapshot_sha256"] != content_digest(facts):
        raise ValueError("Native snapshot facts digest differs.")
    identity = {key: value for key, value in manifest.items() if key != "snapshot_id"}
    if manifest["snapshot_id"] != content_digest(identity):
        raise ValueError("Native snapshot identity differs.")
    if set(facts) != {"families", "aliases", "schemas", "recipes"}:
        raise ValueError("Invalid native snapshot fact groups.")
    families = {}
    fields = set(NativeImageFamilyFacts.__dataclass_fields__)
    for key, row in facts["families"].items():
        if set(row) != fields or key != row["family_id"] or not row["native_classes"]:
            raise ValueError("Invalid native image family facts.")
        if any(
            item.get("source", {}).get("kind") != "native-workflow-example"
            for item in row["examples"]
        ):
            raise ValueError("Native examples require their own provenance.")
        if row["example_recipe"] is not None and row["example_recipe"] not in row["examples"]:
            raise ValueError("Selected native example is absent from family examples.")
        families[key] = NativeImageFamilyFacts(
            **{name: _freeze(value) for name, value in row.items()}
        )
    if any(target not in families for target in facts["aliases"].values()):
        raise ValueError("Native snapshot family alias has no facts.")
    required = manifest["required_contracts"]
    if (
        set(required["families"]) - families.keys()
        or set(required["schemas"]) - facts["schemas"].keys()
    ):
        raise ValueError("Required native snapshot contract is absent.")
    native_classes = {name for family in families.values() for name in family.native_classes}
    if set(required.get("native_classes", ())) - native_classes:
        raise ValueError("Required native configuration class is absent.")
    return NativeCapabilitySnapshot(
        manifest=_freeze(manifest),
        families=MappingProxyType(families),
        aliases=_freeze(facts["aliases"]),
        schemas=_freeze(facts["schemas"]),
        recipes=_freeze(facts["recipes"]),
    )


def _read_snapshot(path):
    with path.open("rb") as stream:
        raw = stream.read(_MAX_SNAPSHOT_BYTES + 1)
    if len(raw) > _MAX_SNAPSHOT_BYTES:
        raise ValueError("Native snapshot exceeds its bounded size.")
    return parse_native_snapshot(json.loads(raw))


def load_native_snapshot(path=None):
    """Read once per process/path; no source scan, stat polling or network work.

    An explicit promotion invalidates this cache. Invalid input retains the last
    accepted in-process snapshot; a first invalid/missing load returns ``None``.
    """
    key = str(Path(path) if path is not None else DEFAULT_SNAPSHOT)
    with _lock:
        if key not in _loaded:
            try:
                _last_good[key] = _read_snapshot(Path(key))
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                pass
            _loaded[key] = _last_good.get(key)
        return _loaded[key]


def get_image_family_facts(family, *, snapshot=None):
    snapshot = snapshot if snapshot is not None else load_native_snapshot()
    if snapshot is None or not isinstance(family, str):
        return None
    return snapshot.families.get(snapshot.aliases.get(family, family))


def native_sampler_choices(snapshot=None):
    """Published native choices from bundled facts; never import a sampler.

    The accepted source's public KSampler aliases are resolved at snapshot build
    time. A missing/legacy snapshot returns empty tuples for a labelled fallback.
    """
    snapshot = snapshot if snapshot is not None else load_native_snapshot()
    if snapshot is None:
        return (), ()
    schema = snapshot.schemas.get("KSampler", {})
    ports = {port["name"]: port for port in schema.get("inputs", ())}

    def choices(name):
        port = ports.get(name, {})
        values = port.get("options", {}).get("choices", ())
        if port.get("dynamic_choices", True) or not isinstance(values, (tuple, list)):
            return ()
        if not all(isinstance(value, str) and value for value in values):
            return ()
        return tuple(values)

    return choices("sampler_name"), choices("scheduler")


def _schema_breaks(previous, candidate):
    old_inputs = {row["name"]: row for row in previous["inputs"]}
    new_inputs = {row["name"]: row for row in candidate["inputs"]}
    return (
        previous["outputs"] != candidate["outputs"]
        or any(
            key not in new_inputs
            or old["type"] != new_inputs[key]["type"]
            or (not old["required"] and new_inputs[key]["required"])
            for key, old in old_inputs.items()
        )
        or any(row["required"] and key not in old_inputs for key, row in new_inputs.items())
        or any(
            set(row.get("options", {}).get("choices", ()))
            - set(new_inputs.get(key, {}).get("options", {}).get("choices", ()))
            for key, row in old_inputs.items()
        )
        or (not previous["unresolved"] and bool(candidate["unresolved"]))
    )


def native_snapshot_delta(previous, candidate):
    """Report semantic/schema, graph-role and affected-family upgrade changes."""
    before, after = snapshot_document(previous)["facts"], snapshot_document(candidate)["facts"]
    result = {}
    previous_choices, candidate_choices = (
        native_sampler_choices(previous),
        native_sampler_choices(candidate),
    )
    result["sampling_choices"] = {
        name: {
            "added": sorted(set(new) - set(old)),
            "removed": sorted(set(old) - set(new)),
            "order_changed": old != new,
        }
        for name, old, new in zip(("samplers", "schedulers"), previous_choices, candidate_choices)
    }
    for group in ("families", "schemas", "recipes"):
        old, new = before[group], after[group]
        result[group] = {
            "added": sorted(new.keys() - old.keys()),
            "removed": sorted(old.keys() - new.keys()),
            "changed": sorted(key for key in old.keys() & new.keys() if old[key] != new[key]),
            "unresolved": sorted(key for key, row in new.items() if row.get("unresolved")),
        }
    role_groups = []
    for facts in (before, after):
        role_groups.append(
            {
                key: {
                    "required_roles": row["required_roles"],
                    "optional_roles": row["optional_roles"],
                    "bindings": sorted(
                        {
                            (model["role"], model["node_id"], model["activity"])
                            for model in row["models"]
                        }
                    ),
                }
                for key, row in facts["recipes"].items()
            }
        )
    old_roles, new_roles = role_groups
    result["graph_roles"] = {
        "added": sorted(new_roles.keys() - old_roles.keys()),
        "removed": sorted(old_roles.keys() - new_roles.keys()),
        "changed": sorted(
            key for key in old_roles.keys() & new_roles.keys() if old_roles[key] != new_roles[key]
        ),
        "unresolved": result["recipes"]["unresolved"],
    }
    required = previous.manifest["required_contracts"]
    result["required_removed"] = [
        f"{group}:{key}"
        for group in ("families", "schemas")
        for key in required[group]
        if key not in after[group]
    ]
    old_classes = {
        name for family in before["families"].values() for name in family["native_classes"]
    }
    new_classes = {
        name for family in after["families"].values() for name in family["native_classes"]
    }
    result["required_removed"].extend(
        f"native_classes:{key}" for key in sorted(old_classes - new_classes)
    )
    result["configuration_semantics_changed"] = [
        key
        for key in before["families"].keys() & after["families"].keys()
        if {
            row["native_class"]: row["contract_sha256"]
            for row in before["families"][key]["configurations"]
        }
        != {
            row["native_class"]: row["contract_sha256"]
            for row in after["families"][key]["configurations"]
        }
    ]
    result["configuration_semantics_changed"].sort()
    result["required_changed"] = [
        f"schemas:{key}"
        for key in required["schemas"]
        if key in after["schemas"]
        and before["schemas"][key]["contract_sha256"] != after["schemas"][key]["contract_sha256"]
    ]
    result["semantic_changed"] = [
        key
        for key in result["schemas"]["changed"]
        if before["schemas"][key]["contract_sha256"] != after["schemas"][key]["contract_sha256"]
    ]
    result["required_breaking"] = [
        key
        for key in required["schemas"]
        if key in after["schemas"] and _schema_breaks(before["schemas"][key], after["schemas"][key])
    ]
    changed_nodes = set(result["schemas"]["changed"] + result["schemas"]["removed"])
    changed_recipes = set(result["recipes"]["changed"] + result["recipes"]["removed"])
    affected = set(result["families"]["changed"] + result["families"]["removed"])
    for family_id, family in {**before["families"], **after["families"]}.items():
        if any(
            example["recipe_id"] in changed_recipes
            or changed_nodes.intersection(example["node_ids"])
            for example in family["examples"]
        ):
            affected.add(family_id)
    result["affected_families"] = sorted(affected)
    result["identity_changed"] = previous.snapshot_id != candidate.snapshot_id
    return result


@dataclass(frozen=True)
class SnapshotRefreshResult:
    promoted: bool
    snapshot: NativeCapabilitySnapshot | None
    delta: Mapping | None = None
    error: str | None = None


def refresh_native_snapshot(candidate_path, *, cache_path=None, reviewed_semantics=False):
    """Explicitly promote an already generated candidate as one atomic envelope.

    Required removal always blocks promotion. Required schema changes require a
    caller's explicit semantic review. Generation never calls this maintainer API.
    """
    destination = Path(cache_path) if cache_path is not None else DEFAULT_SNAPSHOT
    key = str(destination)
    temporary = None
    with _lock:
        previous = load_native_snapshot(destination)
        delta = None
        try:
            candidate = _read_snapshot(Path(candidate_path))
            if previous is not None:
                delta = native_snapshot_delta(previous, candidate)
                if delta["required_removed"]:
                    raise ValueError("Required native contracts were removed; promotion blocked.")
                if delta["required_breaking"]:
                    raise ValueError("Required native schema semantics broke; promotion blocked.")
                if (
                    delta["semantic_changed"] or delta["configuration_semantics_changed"]
                ) and not reviewed_semantics:
                    raise ValueError("Changed consumed native semantics need explicit review.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            raw = (
                json.dumps(snapshot_document(candidate), indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=destination.parent,
                prefix=destination.name + ".",
                suffix=".candidate",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary = None
            _last_good[key] = _loaded[key] = candidate
            return SnapshotRefreshResult(True, candidate, _freeze(delta))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            return SnapshotRefreshResult(False, previous, _freeze(delta), str(error))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
