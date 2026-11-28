"""Local, signature-bound artifacts for native layout and attention guidance."""

from dataclasses import dataclass
from contextlib import ExitStack, contextmanager
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
import pickle
import copy
import os
import struct
import zipfile

from ruinedfooocus.core.model_formats import NATIVE_STATE_DICT_SUFFIXES, is_safetensors
from ruinedfooocus.features.models.model_identity import file_signature

MAX_BYTES = 2 * 1024**3
CATEGORIES = {"gligen": "gligen", "hypernetwork": "hypernetworks"}


@dataclass(frozen=True)
class AuxiliaryAsset:
    role: str
    name: str
    signature: tuple

    @property
    def public_signature(self):
        # Decimal strings retain nanoseconds through browser JSON number limits.
        return (self.name, self.signature[1], *(str(value) for value in self.signature[2:]))

    def metadata(self):
        return dict(name=self.name, signature=list(self.public_signature))


def model_name(value):
    if not isinstance(value, str) or not 0 < len(value) <= 1024:
        raise ValueError("Select a local guidance model from its configured folder.")
    name = PurePosixPath(value.replace("\\", "/"))
    if name.is_absolute() or ":" in value or ".." in name.parts:
        raise ValueError("Guidance models must be relative names inside configured folders.")
    if name.suffix.lower() not in NATIVE_STATE_DICT_SUFFIXES:
        raise ValueError("Choose a native state-dictionary guidance model.")
    return name.as_posix()


def validate_container(path):
    """Inspect container bounds only; native weights_only loading owns Torch data."""
    if not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("Guidance model files must be no larger than 2 GiB.")
    if is_safetensors(path):
        return
    if zipfile.is_zipfile(path):
        # Bound central-directory allocation before ZipFile constructs its entries.
        with path.open("rb") as source:
            source.seek(max(0, path.stat().st_size - 65557))
            tail = source.read(65557)
        offset = tail.rfind(b"PK\x05\x06")
        if offset < 0 or len(tail) - offset < 22:
            raise ValueError("The guidance archive has no valid ZIP directory.")
        directory = struct.unpack_from("<4s4H2LH", tail, offset)
        if directory[1] or directory[2] or directory[4] > 4096 or directory[5] > 4 * 1024**2:
            raise ValueError("The guidance archive exceeds its native directory bounds.")
        try:
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                if len(members) > 4096 or sum(item.file_size for item in members) > MAX_BYTES:
                    raise ValueError(
                        "The guidance archive exceeds its entry or expanded size limit."
                    )
                if any(item.flag_bits & 1 for item in members):
                    raise ValueError(
                        "Encrypted guidance archives are not supported by the native reader."
                    )
                if any(
                    item.filename.endswith("data.pkl") and item.file_size > 16 * 1024**2
                    for item in members
                ):
                    raise ValueError("The guidance archive has oversized Torch metadata.")
        except (zipfile.BadZipFile, NotImplementedError) as error:
            raise ValueError(
                "The guidance archive is not a supported native Torch container."
            ) from error


def safe_native_state(path):
    """Use the pinned weights-only reader with no permissive pickle fallback."""
    import comfy.utils

    try:
        state = comfy.utils.load_torch_file(str(path), safe_load=True)
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        RuntimeError,
        EOFError,
        pickle.UnpicklingError,
        RecursionError,
        zipfile.BadZipFile,
    ) as error:
        raise ValueError(
            "The guidance file is not a supported native weights-only state dictionary."
        ) from error
    if not isinstance(state, Mapping):
        raise ValueError("The guidance file must contain a native state dictionary.")
    return state


def require_ordinary_codec(vae):
    if any(
        getattr(vae, key, None) != value
        for key, value in (
            ("latent_channels", 4),
            ("downscale_ratio", 8),
            ("upscale_ratio", 8),
            ("output_channels", 3),
        )
    ):
        raise ValueError(
            "Native SD layout and attention guidance need a four-channel, eight-times RGB codec."
        )


def validate_identity(name, expected):
    if expected is not None and not (
        isinstance(expected, (list, tuple))
        and len(expected) == 4
        and expected[0] == name
        and type(expected[1]) is int
        and 0 <= expected[1] <= MAX_BYTES
        and all(
            isinstance(value, str) and value.isascii() and value.isdecimal() and len(value) <= 24
            for value in expected[2:]
        )
    ):
        raise ValueError("The saved guidance file identity is invalid. Select the model again.")


def selected_asset(manager, role, name, expected=None):
    name = model_name(name)
    validate_identity(name, expected)
    for folder in manager.get_model_folders(CATEGORIES[role]):
        root = Path(folder).resolve()
        supplied = root / name
        if not supplied.is_file():
            continue
        path = supplied.resolve()
        if not path.is_relative_to(root):
            raise ValueError("The guidance model resolves outside its configured folder.")
        before = tuple(file_signature(path))
        validate_container(path)
        if before != tuple(file_signature(path)):
            raise ValueError("The guidance file changed during inspection. Select it again.")
        asset = AuxiliaryAsset(role, name, before)
        if expected is not None and tuple(expected) != asset.public_signature:
            raise ValueError("The saved guidance model changed. Select it again before generating.")
        return asset
    raise ValueError(f"Install the selected {role} model in its configured folder: {name}.")


def replay_values(metadata):
    """Clear absent guidance and retain exact public source proofs for an explicit replay."""
    from ruinedfooocus.features.image_editing import gligen_layout, native_hypernetwork

    result = gligen_layout.RESETS | native_hypernetwork.RESETS
    for feature in (gligen_layout, native_hypernetwork):
        model_key, proof_key, _ = feature.FIELDS
        if not metadata.get(model_key):
            continue
        name = model_name(metadata[model_key])
        proof = metadata.get(proof_key)
        if proof is None:
            raise ValueError(
                "Saved guidance needs its original file identity. Select the model again."
            )
        validate_identity(name, proof)
        result.update(
            (key, copy.deepcopy(metadata[key])) for key in feature.FIELDS if key in metadata
        )
        if feature is gligen_layout:
            gligen_layout.parse_layout(result["gligen_layout"])
    return result


def artifact(asset):
    from ruinedfooocus.core.preparation import PreparationRequest
    from ruinedfooocus.features.generation.image_preparation import ImageArtifact

    path = Path(asset.signature[0])
    return ImageArtifact(asset.role, asset.name, PreparationRequest("", path), asset.signature)


def local_choices(manager, role):
    """Bounded, link-safe local discovery; browsing never deserializes Torch weights."""
    choices, seen = [], set()
    scanned = 0
    for folder in manager.get_model_folders(CATEGORIES[role]):
        root = Path(folder).resolve()
        pending, visited = [root], set()
        while pending:
            current = pending.pop()
            resolved = current.resolve()
            if not resolved.is_relative_to(root) or resolved in visited:
                continue
            visited.add(resolved)
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        scanned += 1
                        if scanned > 16384:
                            return sorted(choices, key=str.casefold)
                        path = Path(entry.path)
                        if entry.is_dir():
                            pending.append(path)
                        elif entry.is_file() and path.resolve().is_relative_to(root):
                            name = path.relative_to(root).as_posix()
                            if name in seen:
                                continue
                            seen.add(name)
                            if (
                                path.suffix.lower() in NATIVE_STATE_DICT_SUFFIXES
                                and not (role == "hypernetwork" and is_safetensors(path))
                                and 0 < entry.stat().st_size <= MAX_BYTES
                            ):
                                choices.append(name)
            except OSError:
                continue
    return sorted(choices, key=str.casefold)


def attention_layout(model):
    """The native UNet traversal indexes one GLIGEN fuser per SpatialTransformer."""
    diffusion = model.get_model_object("diffusion_model")
    if type(diffusion).__name__ != "UNetModel":
        raise ValueError("Layout and hypernetwork guidance require a compatible native SD UNet.")
    queries, contexts, key_widths = [], set(), set()
    for block in (*diffusion.input_blocks, diffusion.middle_block, *diffusion.output_blocks):
        for spatial in block:
            if type(spatial).__name__ != "SpatialTransformer":
                continue
            query = int(spatial.proj_in.weight.shape[0])
            queries.append(query)
            for attention in spatial.transformer_blocks:
                context = query
                if attention.attn2 is not None:
                    context = int(attention.attn2.to_k.weight.shape[1])
                    contexts.add(context)
                    key_widths.add(context)
                key_widths.add(context if attention.disable_self_attn else query)
    if not queries or not contexts:
        raise ValueError("The loaded UNet has no native layout/attention guidance consumer.")
    return tuple(queries), frozenset(contexts), frozenset(key_widths)


@contextmanager
def sampling_scope(owner, data, manager):
    """Hold only requested auxiliary source leases through native sampling."""
    from ruinedfooocus.features.generation.image_preparation import prepared_asset

    with ExitStack() as stack:
        captured = []
        for key in ("_native_gligen_layout", "_native_hypernetwork"):
            plan = data.get(key)
            if plan is not None:
                asset = plan.asset
                path = stack.enter_context(
                    prepared_asset(data, asset.role, asset.name, manager, CATEGORIES[asset.role])
                )
                captured.append((path, asset.signature))
        try:
            yield
            if any(tuple(file_signature(path)) != signature for path, signature in captured):
                raise ValueError(
                    "A guidance file changed during sampling. Submit the request again."
                )
        finally:
            if data.get("_native_hypernetwork") is not None:
                from ruinedfooocus.features.image_editing.native_hypernetwork import offload

                offload(owner)


@contextmanager
def apply_scope(owner, model, vae, data, manager, check_cancelled):
    """Keep auxiliary leases around the request-local native patch scope."""
    from ruinedfooocus.features.image_editing import model_patches

    with (
        sampling_scope(owner, data, manager),
        model_patches.apply_scope(owner, model, vae, data, manager, check_cancelled) as patched,
    ):
        yield patched
