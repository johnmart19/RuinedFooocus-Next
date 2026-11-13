"""Owned precomputed map handoffs; original edit pixels remain a separate input."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from ruinedfooocus.features.image_processing.image_geometry_tools import (
    FrozenPixels,
    NATIVE_REVISION,
    capture_source,
)

HANDOFF_KEY = "_editor_guide_handoff"
KEY = "_editor_guide_request"
DENOISE_FIELD = "control_guide_denoise"


@dataclass(frozen=True)
class GuideSpec:
    kind: str
    convention: str
    projection: str
    producer: str
    source_id: str
    guide_id: str
    width: int
    height: int


@dataclass(frozen=True)
class ResultGuideHandoff:
    session_id: str
    revision: int
    source_id: str
    guide_id: str
    source_frame_id: str
    guide_frame_id: str
    source_sha256: str
    guide_sha256: str
    source_signature: tuple
    guide_signature: tuple
    spec: GuideSpec

    def public_context(self):
        return dict(
            session_id=self.session_id,
            revision=self.revision,
            source_id=self.source_id,
            additional_image_ids=[self.guide_id],
        )

    def metadata(self):
        result = asdict(self)
        result.update(
            source_signature=list(self.source_signature[1:]),
            guide_signature=list(self.guide_signature[1:]),
            purpose="precomputed-control-guide",
            preprocessing="none",
        )
        return result


@dataclass(frozen=True)
class UploadedGuideHandoff:
    owner_key: str
    source_sha256: str
    guide_sha256: str
    spec: GuideSpec


def _technical(value, key=""):
    hidden = {
        "owner_key",
        "session_id",
        "revision",
        "source_id",
        "guide_id",
        "source_frame_id",
        "guide_frame_id",
        "frame_id",
    }
    if isinstance(value, dict):
        return {
            name: _technical(child, name)
            for name, child in value.items()
            if name not in hidden and name not in ("path", "canonical_path")
        }
    if isinstance(value, (list, tuple)):
        if (
            "signature" in key
            and value
            and isinstance(value[0], str)
            and Path(value[0]).is_absolute()
        ):
            value = value[1:]
        return [_technical(child, key) for child in value]
    if key == "model" and isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name
    return value


@dataclass(frozen=True)
class GuideRequest:
    handoff: ResultGuideHandoff | UploadedGuideHandoff
    source: FrozenPixels
    guide: FrozenPixels
    denoise: float
    controls_json: str
    compatibility: object
    conversion_signatures: tuple = ()
    conversion_sources: tuple = ()

    def controls(self):
        return json.loads(self.controls_json)

    def metadata(self):
        channels = tuple(getattr(self.compatibility, "invert_channels", ()))
        return _technical(
            dict(
                purpose="original-source-controlled-edit",
                provenance="caller-declared-upload"
                if isinstance(self.handoff, UploadedGuideHandoff)
                else "owned-utility-result",
                source_sha256=self.handoff.source_sha256,
                guide_sha256=self.handoff.guide_sha256,
                source_signature=list(getattr(self.handoff, "source_signature", ())[1:]),
                guide_signature=list(getattr(self.handoff, "guide_signature", ())[1:]),
                guide=asdict(self.handoff.spec),
                source_pixels=dict(
                    size=list(self.source.size),
                    dtype=self.source.dtype,
                    channels=self.source.channels,
                    color_space=self.source.color_space,
                ),
                guide_pixels=dict(
                    size=list(self.guide.size),
                    dtype=self.guide.dtype,
                    channels=self.guide.channels,
                    color_space=self.guide.color_space,
                ),
                denoise=self.denoise,
                source_pixel_sha256=self.source.source_sha256,
                guide_pixel_sha256=self.guide.source_sha256,
                alpha_policy="original source coverage restored after native decoding",
                control=self.compatibility.metadata(),
                controls=self.controls(),
                derived_guide=dict(
                    invert_channels=list(channels),
                    native_conversion="InvertMask per scalar channel before geometry"
                    if channels
                    else "none",
                    native_revision=NATIVE_REVISION if channels else None,
                    native_sources=[
                        dict(file=name, sha256=sha) for name, sha in self.conversion_sources
                    ],
                    applications="once per request-owned working guide" if channels else "none",
                    spatial_flip=False,
                    reestimate=False,
                ),
            )
        )


def captured(data):
    request = data.get(KEY)
    if request is None:
        return None
    if not isinstance(request, GuideRequest) or not isinstance(
        request.handoff, (ResultGuideHandoff, UploadedGuideHandoff)
    ):
        raise ValueError("Use a captured owned or uploaded guide request, not a raw mapping.")
    return request


def automatic_source_plan(data):
    """Retain a supported source frame; uniformly fit oversized sources on Auto."""
    request = captured(data)
    if (
        request is None
        or data.get("resolution_auto") is not True
        or data.get("image_resize_mode", "preserve") != "preserve"
        or data.get("_image_size_plan") is not None
    ):
        return None
    from ruinedfooocus.features.generation.size_plan import SizePlan

    canvas = SizePlan.create(request.source.size, 1, alignment=16).canvas
    if max(canvas) <= 2048 and math.prod(canvas) <= 2048 * 2048:
        return dict(target=request.source.size, mode="preserve")
    from ruinedfooocus.features.image_processing.image_geometry_tools import output_size, parameters

    # Reuse the maintained contain plan, including side bounds for panoramas;
    # area-only scaling can otherwise leave an elongated source over-wide.
    fitted = output_size(
        request.source.size,
        parameters(
            dict(
                transform_operation="resize",
                transform_width=2048,
                transform_height=2048,
                transform_fit="contain",
            )
        ),
    )
    target = tuple(max(64, side) for side in fitted)
    SizePlan.create(target, 1, alignment=16, maximum=2048)
    return dict(target=target, mode="pad")


def automatic_source_size(data):
    plan = automatic_source_plan(data)
    return plan["target"] if plan is not None else None


def classify_map_metadata(metadata):
    """Describe saved producer semantics only; this confers no ownership authority."""
    if not isinstance(metadata, dict):
        raise ValueError("Read the saved geometry result metadata before selecting its convention.")
    info = metadata.get("image_utility_source", metadata)
    if not isinstance(info, dict) or info.get("operation") not in ("moge-depth", "lotus-depth"):
        raise ValueError("Choose an owned depth or normal utility result, not an arbitrary image.")
    producer = info["operation"]
    width, height = info.get("width"), info.get("height")
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError("The saved guide has no valid source dimensions.")
    kind = info.get("moge_output", "depth") if producer == "moge-depth" else "depth"
    convention = info.get("convention")
    projection = info.get("projection", "perspective")
    if kind == "normal":
        if (
            convention not in ("native-opengl", "native-directx")
            or projection != "perspective"
            or info.get("normal_frame") != "native-camera"
            or info.get("normalization") != "native-normal*0.5+0.5"
        ):
            raise ValueError(
                "This normal result has no supported encoded camera-normal convention."
            )
    elif kind == "depth":
        if convention not in (
            "white-near",
            "white-near; invalid-black",
            "white-near; uncovered-native-far-fill",
        ):
            raise ValueError("This depth result has no supported white-near convention.")
        if projection not in ("perspective", "equirectangular"):
            raise ValueError("This depth result has no supported projection.")
    else:
        raise ValueError("Choose an explicit depth or normal map result.")
    return dict(
        kind=kind,
        convention=convention,
        projection=projection,
        producer=producer,
        width=width,
        height=height,
    )


def _step_spec(store, value, result_id):
    step = next((row for row in reversed(value["steps"]) if row["result_id"] == result_id), None)
    info = (step or {}).get("execution", {}).get("image_utility_source", {})
    facts = classify_map_metadata(info)
    if (step or {}).get("operation") != facts["producer"]:
        raise ValueError("Choose an owned depth or normal utility result, not an arbitrary image.")
    source = store._get(value, step["source_id"], kind="image")
    guide = store._get(value, result_id, kind="image")
    if source.frame.size != guide.frame.size or source.frame.orientation != guide.frame.orientation:
        raise ValueError("The guide must match its exact oriented original source frame.")
    if any(
        ref.frame.dtype != "uint8" or ref.frame.color_space != "sRGB" for ref in (source, guide)
    ):
        raise ValueError(
            "Use explicit 8-bit sRGB source and guide working images for controlled editing."
        )
    if (facts["width"], facts["height"]) != source.frame.size:
        raise ValueError("The saved guide dimensions do not match its source frame.")
    return (
        source,
        guide,
        GuideSpec(**facts, source_id=source.artifact_id, guide_id=guide.artifact_id),
    )


def describe_result_guide(store, session_id, revision, result_id, *, owner):
    """Read eligibility from owned execution provenance, never a filename or label."""
    with store._edit(session_id, owner) as value:
        store._revision(value, revision)
        if value["head_id"] != result_id or value.get("pending") is not None:
            raise ValueError(
                "The current guide result changed or another image operation is pending."
            )
        return _step_spec(store, value, result_id)[2]


def restore_result_guide(store, session_id, revision, result_id, *, owner):
    """Restore the retained source and bind the map within one Editor transaction."""
    with store._edit(session_id, owner) as value:
        store._revision(value, revision)
        if value["head_id"] != result_id or value.get("pending") is not None:
            raise ValueError(
                "The current guide result changed or another image operation is pending."
            )
        source, guide, spec = _step_spec(store, value, result_id)
        handoff = ResultGuideHandoff(
            session_id,
            revision + 1,
            source.artifact_id,
            guide.artifact_id,
            source.frame_id,
            guide.frame_id,
            source.sha256,
            guide.sha256,
            source.signature,
            guide.signature,
            spec,
        )
        value.update(
            head_id=source.artifact_id, revision=revision + 1, pending=None, redo=[result_id]
        )
        restored = store._snapshot(value)
        store._save(value)
    return restored, handoff


def _refs(store, handoff, owner):
    if not isinstance(handoff, ResultGuideHandoff):
        raise ValueError("Capture an owned map-for-editing handoff first.")
    with store._edit(handoff.session_id, owner) as value:
        store._revision(value, handoff.revision)
        if value["head_id"] != handoff.source_id:
            raise ValueError("The original source changed; adopt the current guide again.")
        source, guide, spec = _step_spec(store, value, handoff.guide_id)
        if (
            spec != handoff.spec
            or source.artifact_id != handoff.source_id
            or source.frame_id != handoff.source_frame_id
            or guide.frame_id != handoff.guide_frame_id
            or source.sha256 != handoff.source_sha256
            or guide.sha256 != handoff.guide_sha256
            or source.signature != handoff.source_signature
            or guide.signature != handoff.guide_signature
        ):
            raise ValueError("The captured source, guide, frame or convention changed.")
    return source, guide


def handoff_images(store, handoff, *, owner):
    """UI display copies; request capture independently rereads the owned originals."""
    import numpy as np
    from PIL import Image, ImageOps

    source, guide = _refs(store, handoff, owner)
    with Image.open(source.path) as image:
        original = ImageOps.exif_transpose(image).convert("RGBA").copy()
    with Image.open(guide.path) as image:
        control = ImageOps.exif_transpose(image).convert("RGB").copy()
    if original.size != control.size:
        raise ValueError("The guide and source must remain in one exact frame.")
    if handoff.spec.kind == "depth":
        pixels = np.asarray(control)
        if not np.array_equal(pixels[..., 0], pixels[..., 1]) or not np.array_equal(
            pixels[..., 0], pixels[..., 2]
        ):
            raise ValueError(
                "A depth guide must contain grayscale disparity, not a color visualization."
            )
    for ref in (source, guide):
        with open(ref.path, "rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != ref.sha256:
                raise ValueError("The original source or guide pixels changed after capture.")
    _refs(store, handoff, owner)
    return dict(main_view=original, input_image=control)


def capture_guide_inputs(store, handoff, *, owner, data, manager, family):
    """Freeze separate source/guide pixels and explicit edit intent before enqueue."""
    import numpy as np
    from ruinedfooocus.features.gallery.editor_session import _json
    from ruinedfooocus.features.image_editing.guide_control_compat import validate_guide_control

    denoise = data.get(DENOISE_FIELD)
    if type(denoise) not in (int, float) or not math.isfinite(denoise) or not 0 <= denoise <= 1:
        raise ValueError("Choose explicit edit denoise from zero through one.")
    images = handoff_images(store, handoff, owner=owner)
    controls = data.get("controlnet")
    if not isinstance(controls, dict):
        raise ValueError("Select the compatible guide control before editing.")
    if data.get("inpaint_toggle") or controls.get("loopback"):
        raise ValueError(
            "Controlled editing uses the retained original without another mask or loopback source."
        )
    compatibility = validate_guide_control(manager, family, dict(controls), handoff.spec)
    from ruinedfooocus.features.image_processing import mask_native

    conversion_signatures = (
        mask_native.source_signatures() if getattr(compatibility, "invert_channels", ()) else ()
    )
    conversion_sources = tuple(
        (name, hashlib.sha256(Path(signature[0]).read_bytes()).hexdigest())
        for name, signature in zip(mask_native.FILES, conversion_signatures)
    )
    if conversion_signatures and mask_native.source_signatures() != conversion_signatures:
        raise ValueError("The native conversion source changed during capture; submit again.")
    request = GuideRequest(
        handoff,
        capture_source(images["main_view"]),
        capture_source(np.asarray(images["input_image"]).copy()),
        float(denoise),
        _json(compatibility.controls),
        compatibility,
        conversion_signatures,
        conversion_sources,
    )
    result = dict(
        data,
        input_image=images["main_view"],
        main_view=images["main_view"].copy(),
        controlnet=request.controls(),
    )
    result[KEY] = request
    result.update(control_recipe=compatibility.recipe, control_union_type=compatibility.union_type)
    result.pop(HANDOFF_KEY, None)
    return result


def capture_request(data, *, store, owner, manager, family):
    """Shared capture hook; ordinary generation has no guide-specific changes."""
    if data.get(HANDOFF_KEY) is not None:
        return capture_guide_inputs(
            store, data[HANDOFF_KEY], owner=owner, data=data, manager=manager, family=family
        )
    if KEY not in data:
        return data
    request = validate_request(data, store=store, owner=owner, manager=manager, family=family)
    return dict(
        data,
        input_image=request_image(request.source),
        controlnet=request.controls(),
        control_recipe=request.compatibility.recipe,
        control_union_type=request.compatibility.union_type,
        **{DENOISE_FIELD: request.denoise},
    )


def capture_uploaded_guide_inputs(
    *, source, guide, kind, convention, denoise, data, manager, family, owner
):
    """Explicit API bytes have content ownership and never forge an Editor ticket."""
    import numpy as np
    from ruinedfooocus.features.gallery.editor_session import _json, _owner_key
    from ruinedfooocus.features.image_editing.guide_control_compat import validate_guide_control
    from ruinedfooocus.features.image_processing import mask_native

    if data.get("editor_context") is not None or data.get("_editor_operation") is not None:
        raise ValueError("Uploaded guides cannot borrow owned Editor tickets or context.")
    if type(denoise) not in (int, float) or not math.isfinite(denoise) or not 0 <= denoise <= 1:
        raise ValueError("Choose explicit edit denoise from zero through one.")
    declarations = {
        ("depth", "white-near"): "white-near",
        ("normal", "moge-opengl"): "native-opengl",
        ("normal", "moge-directx"): "native-directx",
        ("normal", "normal-bae"): "normalbae-scannet-camera",
    }
    if (kind, convention) not in declarations:
        raise ValueError(
            "Declare white-near depth, MoGe OpenGL/DirectX normals or NormalBae camera normals."
        )
    original, selected = capture_source(source), capture_source(guide)
    if (
        original.dtype != "uint8"
        or selected.dtype != "uint8"
        or original.color_space != "sRGB"
        or selected.color_space != "sRGB"
    ):
        raise ValueError("Controlled uploads require explicit 8-bit sRGB working pixels.")
    if original.size != selected.size:
        raise ValueError("The uploaded guide must match its original source frame exactly.")
    pixels = np.frombuffer(selected.pixels, dtype=np.uint8).reshape(
        selected.size[1], selected.size[0], selected.channels
    )
    if selected.channels == 1:
        pixels = np.repeat(pixels, 3, axis=-1)
    if kind == "depth" and (
        not np.array_equal(pixels[..., 0], pixels[..., 1])
        or not np.array_equal(pixels[..., 0], pixels[..., 2])
    ):
        raise ValueError(
            "Depth uploads must contain grayscale disparity, not a color visualization."
        )
    selected = capture_source(pixels[..., :3].copy())
    spec = GuideSpec(
        kind,
        declarations[(kind, convention)],
        "perspective",
        "explicit-upload",
        original.source_sha256,
        selected.source_sha256,
        *original.size,
    )
    binding = UploadedGuideHandoff(
        _owner_key(owner), original.source_sha256, selected.source_sha256, spec
    )
    controls = data.get("controlnet")
    if not isinstance(controls, dict) or data.get("inpaint_toggle") or controls.get("loopback"):
        raise ValueError(
            "Choose an explicit guide control without another mask or loopback source."
        )
    compatibility = validate_guide_control(manager, family, dict(controls), spec)
    signatures = mask_native.source_signatures() if compatibility.invert_channels else ()
    sources = tuple(
        (name, hashlib.sha256(Path(signature[0]).read_bytes()).hexdigest())
        for name, signature in zip(mask_native.FILES, signatures)
    )
    if signatures and mask_native.source_signatures() != signatures:
        raise ValueError("The native conversion source changed during capture; submit again.")
    request = GuideRequest(
        binding,
        original,
        selected,
        float(denoise),
        _json(compatibility.controls),
        compatibility,
        signatures,
        sources,
    )
    result = dict(
        data,
        input_image=request_image(original),
        controlnet=request.controls(),
        control_recipe=compatibility.recipe,
        control_union_type=compatibility.union_type,
        **{DENOISE_FIELD: float(denoise), KEY: request},
    )
    result.pop(HANDOFF_KEY, None)
    return result


def validate_request(data, *, store, owner, manager, family):
    request = captured(data)
    if request is None:
        raise ValueError("Capture the original source and its guide before controlled editing.")
    if isinstance(request.handoff, ResultGuideHandoff):
        _refs(store, request.handoff, owner)
    else:
        from ruinedfooocus.features.gallery.editor_session import _owner_key

        if request.handoff.owner_key != _owner_key(owner):
            raise PermissionError("This uploaded guide belongs to another request owner.")
        if (
            request.source.source_sha256 != request.handoff.source_sha256
            or request.guide.source_sha256 != request.handoff.guide_sha256
        ):
            raise ValueError("The uploaded source or guide content changed after capture.")
    size = (request.handoff.spec.width, request.handoff.spec.height)
    if request.source.size != size or request.guide.size != size or request.guide.channels != 3:
        raise ValueError("The captured original and guide must share the exact source frame.")
    if request.conversion_signatures:
        from ruinedfooocus.features.image_processing import mask_native

        if mask_native.source_signatures() != request.conversion_signatures:
            raise ValueError(
                "The native normal conversion source changed; capture the guide again."
            )
    request.compatibility.verify(manager, family, request.controls(), request.handoff.spec)
    return request


def request_image(pixels):
    import numpy as np
    from PIL import Image

    array = np.frombuffer(pixels.pixels, dtype=pixels.dtype).reshape(
        pixels.size[1], pixels.size[0], pixels.channels
    )
    return Image.fromarray(array[..., 0].copy() if pixels.channels == 1 else array.copy())


def consumer_guide_image(request):
    """Convert selected normal scalar channels with the pinned native mask op."""
    import numpy as np
    from PIL import Image
    import torch
    from ruinedfooocus.features.image_processing import mask_native

    image = request_image(request.guide)
    channels = tuple(getattr(request.compatibility, "invert_channels", ()))
    if not channels:
        return image
    if request.handoff.spec.kind != "normal" or channels not in ((0,), (0, 1)):
        raise ValueError("The captured normal guide has an unsupported channel conversion.")
    ops = mask_native.operations(request.conversion_signatures)
    with torch.device("cpu"), torch.inference_mode():
        pixels = torch.from_numpy(np.asarray(image).copy()).float().unsqueeze(0) / 255
        for channel in channels:
            pixels[..., channel] = ops.InvertMask(None, pixels[..., channel])[0]
        converted = (pixels[0] * 255).round().to(torch.uint8).numpy()
    if mask_native.source_signatures() != request.conversion_signatures:
        raise ValueError("The native normal conversion source changed; capture the guide again.")
    return Image.fromarray(converted, "RGB")


def prepare_reference_pair(request, target_size, mode="preserve", *, multiple=16):
    """Use existing image geometry for both inputs; no depth/normal inference runs."""
    from ruinedfooocus.features.image_editing.image_geometry import prepare_reference

    if not isinstance(request, GuideRequest):
        raise ValueError("Capture a controlled edit before preparing its image frame.")
    source, frame = prepare_reference(
        request_image(request.source), target_size, mode, multiple=multiple, preserve_alpha=True
    )
    guide, guide_frame = prepare_reference(
        consumer_guide_image(request), target_size, mode, multiple=multiple
    )
    if frame != guide_frame or source.shape[1:3] != guide.shape[1:3]:
        raise ValueError("The original source and control guide acquired different frames.")
    return source[..., :3], guide, frame, source[..., 3:]


def restore_source_alpha(decoded, alpha, frame):
    import torch
    from ruinedfooocus.features.image_editing.image_geometry import restore_reference_frame

    selected = restore_reference_frame(alpha, frame)
    if tuple(decoded.shape[:3]) != tuple(selected.shape[:3]):
        raise ValueError("The finished controlled edit no longer matches the source alpha frame.")
    return torch.cat(
        (decoded[..., :3], selected.to(device=decoded.device, dtype=decoded.dtype)), dim=-1
    )
