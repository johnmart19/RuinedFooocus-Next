"""Capture local Diffusers weight bundles as data, without runtime or acquisition."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from ruinedfooocus.features.models.model_identity import file_signature

BUNDLE_KEY = "_local_diffusers_bundle"
# Exact file preference from pinned comfy/diffusers_load.py, checked by source test.
DIFFUSION_FILES = (
    "diffusion_pytorch_model.fp16.safetensors",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.fp16.bin",
    "diffusion_pytorch_model.bin",
)
TEXT_FILES = (
    "model.fp16.safetensors",
    "model.safetensors",
    "pytorch_model.fp16.bin",
    "pytorch_model.bin",
)


@dataclass(frozen=True)
class BundleFile:
    role: str
    source: Path
    path: Path
    signature: tuple

    def verify(self):
        if (
            self.source.resolve(strict=True) != self.path
            or tuple(file_signature(self.path)) != self.signature
        ):
            raise ValueError("A local Diffusers bundle file changed. Refresh and submit again.")


@dataclass(frozen=True)
class LocalDiffusersBundle:
    index: BundleFile
    diffusion: BundleFile
    encoders: tuple[BundleFile, ...]
    vae: BundleFile

    @property
    def files(self):
        return (self.index, self.diffusion, *self.encoders, self.vae)

    @property
    def identity(self):
        signatures = [item.signature for item in self.files]
        return hashlib.sha256(json.dumps(signatures).encode()).hexdigest()

    def verify(self, check_cancelled=None):
        for item in self.files:
            if check_cancelled is not None:
                check_cancelled()
            item.verify()


def is_diffusers_index(path):
    return Path(path).name == "model_index.json"


def _capture_file(role, source):
    source = Path(source).absolute()
    path = source.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Local Diffusers {role} must be a regular file.")
    result = BundleFile(role, source, path, tuple(file_signature(path)))
    result.verify()
    return result


def _first_file(root, names, role, required=True):
    for name in names:
        source = root / name
        if source.is_file():
            return _capture_file(role, source)
    if required:
        raise FileNotFoundError(
            f"Local Diffusers bundle is missing {role}. The pinned native directory path "
            "requires one of its supported single-file weight names."
        )
    return None


def capture_bundle(index, *, check_cancelled=None):
    """Snapshot only the local weight layout supported by the native directory path."""
    if not is_diffusers_index(index):
        raise ValueError("Select a local Diffusers model_index.json.")
    if check_cancelled is not None:
        check_cancelled()
    manifest = _capture_file("index", index)
    if manifest.signature[1] > 1024**2:
        raise ValueError("Local Diffusers model index is too large.")
    with manifest.path.open("rb") as stream:
        raw = stream.read(1024**2 + 1)
    if len(raw) > 1024**2:
        raise ValueError("Local Diffusers model index is too large.")
    metadata = json.loads(raw)
    manifest.verify()
    if not isinstance(metadata, dict):
        raise ValueError("Local Diffusers model index must be a JSON object.")
    # The native directory reader has two encoder slots; never silently drop a third.
    if metadata.get("text_encoder_3") not in (None, [None, None]):
        raise ValueError("The pinned native directory caller has no third text-encoder slot.")
    root = Path(index).absolute().parent
    diffusion = _first_file(root / "unet", DIFFUSION_FILES, "diffusion")
    first = _first_file(root / "text_encoder", TEXT_FILES, "text_encoder")
    second_required = metadata.get("text_encoder_2") not in (None, [None, None])
    second = _first_file(root / "text_encoder_2", TEXT_FILES, "text_encoder_2", second_required)
    vae = _first_file(root / "vae", DIFFUSION_FILES, "vae")
    result = LocalDiffusersBundle(manifest, diffusion, (first, second) if second else (first,), vae)
    result.verify(check_cancelled)
    return result


def discover_bundles(roots):
    """Local-only index discovery; malformed/incomplete bundles remain visible to explain."""
    seen = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for index in sorted(root.rglob("model_index.json")):
            if is_diffusers_index(index) and index.is_file():
                key = str(index.resolve())
                if key not in seen:
                    seen.add(key)
                    yield index.absolute()


def bundle_inventory(bundle, family="Unknown", choices=None):
    if any(value for value in (choices or {}).values()):
        raise ValueError("Local Diffusers bundle components are fixed. Clear component overrides.")
    rows = []
    for item in (*bundle.encoders, bundle.vae):
        rows.append(
            dict(
                key=item.role,
                label="VAE"
                if item.role == "vae"
                else "Text encoder 2"
                if item.role == "text_encoder_2"
                else "Text encoder",
                filename=item.source.relative_to(bundle.index.source.parent).as_posix(),
                category="vae" if item.role == "vae" else "clip",
                override=False,
                locked=True,
                status="Bundled local file",
                location=str(item.path),
                source="",
                bytes=item.signature[1],
                published_bytes=None,
                download_key=None,
                download_available=False,
            )
        )
    return dict(
        family=family,
        architecture=family if family != "Unknown" else "Local Diffusers bundle",
        rows=rows,
        missing_payload_bytes=0,
        bundle_identity=bundle.identity,
        note="Bundle files stay fixed. Native loading still validates architecture and compatibility.",
    )


def bundle_metadata(index, detector=None):
    """Local presentation metadata only; detector must already be initialized."""
    metadata = {
        "rf_display_name": Path(index).parent.name,
        "rf_source": "Local Diffusers bundle",
        "baseModel": "Unknown",
        "model": {"type": "Checkpoint"},
    }
    try:
        bundle = capture_bundle(index)
        metadata["_rf_file_signature"] = list(bundle.index.signature)
        metadata["_rf_bundle_identity"] = bundle.identity
        if detector is not None:
            metadata["baseModel"] = detector(bundle.diffusion.path) or "Unknown"
    except (OSError, ValueError) as error:
        metadata["_rf_bundle_error"] = str(error)
    return metadata
