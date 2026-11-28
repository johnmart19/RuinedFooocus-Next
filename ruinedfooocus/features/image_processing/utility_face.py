"""Native MediaPipe facial landmark overlays and region masks for one still image."""

from ruinedfooocus.features.models.model_identity import cached_safetensors_header
from ruinedfooocus.features.image_processing.depth_utilities import (
    cached_model,
    clear_model,
    native_call,
)
from ruinedfooocus.features.image_processing.native_vision import _number


MASK_REGIONS = ("face_oval", "lips", "left_eye", "right_eye", "irises")
FACE_MASK_DEFAULTS = {
    "face_output": "overlay",
    "face_mask_regions": "all",
    **{"face_mask_" + region: region == "face_oval" for region in MASK_REGIONS},
}


def validate_model(path):
    header = cached_safetensors_header(path)
    expected = {
        "detector_short.stem.weight": [24, 3, 5, 5],
        "detector_full.stem.weight": [32, 3, 3, 3],
        "mesh.stem.weight": [16, 3, 3, 3],
        "canonical_vertices": [468, 3],
    }
    if (
        any(header.get(key, {}).get("shape") != shape for key, shape in expected.items())
        or not all(
            any(key.startswith(prefix) for key in header)
            for prefix in ("blendshapes.", "procrustes_indices", "procrustes_weights")
        )
        or not all(
            "topology." + part in header
            for part in (
                "face_oval",
                "left_eye",
                "right_eye",
                "left_eyebrow",
                "right_eyebrow",
                "lips",
                "irises",
                "nose",
            )
        )
    ):
        raise ValueError(
            "Select the native MediaPipe face landmarker bundle with detectors, landmarks and topology."
        )
    return "MediaPipeFaceLandmarker"


COMPONENTS = (("image_utility_model", "face-landmarker", ("detection",), validate_model),)


def parameters(data):
    variant = data.get("face_detector_variant", "short")
    if variant not in ("short", "full", "both"):
        raise ValueError("Choose short, full or both face detector ranges.")
    options = {key: data.get(key, default) for key, default in FACE_MASK_DEFAULTS.items()}
    if options["face_output"] not in ("overlay", "region-mask"):
        raise ValueError("Choose a facial landmark overlay or a face-region selection mask.")
    if options["face_mask_regions"] not in ("all", "custom"):
        raise ValueError("Choose all native face-mask regions or an explicit custom selection.")
    if any(type(options["face_mask_" + region]) is not bool for region in MASK_REGIONS):
        raise ValueError("Each face-mask region must be enabled or disabled explicitly.")
    if (
        options["face_output"] == "region-mask"
        and options["face_mask_regions"] == "custom"
        and not any(options["face_mask_" + region] for region in MASK_REGIONS)
    ):
        raise ValueError("Choose at least one native face-mask region.")
    return {
        "face_detector_variant": variant,
        "face_max_count": _number(
            data.get("face_max_count", 1), "Maximum faces", 1, 16, integer=True
        ),
        "face_min_confidence": _number(
            data.get("face_min_confidence", 0.5), "Face confidence", 0, 1
        ),
        **options,
    }


def render(owner, request, pixels, data, manager, check_cancelled):
    import torch
    from ruinedfooocus.features.image_processing.utility_device import run_utility

    def execute(policy):
        try:
            output, details = _render(owner, request, pixels, check_cancelled, policy)
            if not isinstance(output, torch.Tensor) or tuple(output.shape) != tuple(pixels.shape):
                raise ValueError("The native face utility returned invalid image dimensions.")
            # Synchronize the complete native result inside the retry boundary.
            output = output.detach().float().cpu()
            if not torch.isfinite(output).all():
                raise ValueError("The native face utility returned non-finite image values.")
            return output, details
        except BaseException:
            clear_model(owner)
            raise

    result = run_utility(
        "Face landmarks", execute, check_cancelled=check_cancelled, native_model_placement=True
    )
    output, details = result.value
    return output, {**details, "device_policy": result.metadata()}


def _render(owner, request, pixels, check_cancelled, policy):
    import folder_paths
    from comfy_extras.nodes_mediapipe import (
        LoadMediaPipeFaceLandmarker,
        MediaPipeFaceLandmarker,
        MediaPipeFaceMeshVisualize,
    )

    (item,) = request.files

    def load():
        folder_paths.add_model_folder_path("detection", str(item.path.parent), is_default=True)
        model = LoadMediaPipeFaceLandmarker.execute(item.path.name)[0]
        # Native initial placement is retained by the shared model policy. Only
        # explicit CPU or a supported failure retry changes fresh patchers;
        # offload placement, float32 weights and native loading remain intact.
        model.load_device = policy.device
        for patcher in model.patchers.values():
            patcher.load_device = policy.device
        return model, tuple(model.patchers.values())

    model = cached_model(
        owner, request, load, check_cancelled, source_identity=(str(policy.device),)
    )
    device_details = {
        "inference_device": str(model.load_device),
        "inference_dtype": "torch.float32",
        "raster_device": "cpu",
    }
    options = dict(request.parameters)
    landmarks = native_call(
        check_cancelled,
        MediaPipeFaceLandmarker.execute,
        model,
        pixels,
        detector_variant=options["face_detector_variant"],
        num_faces=options["face_max_count"],
        min_confidence=options["face_min_confidence"],
        missing_frame_fallback="empty",
    )[0]
    if (
        landmarks.get("image_size") != (pixels.shape[1], pixels.shape[2])
        or len(landmarks.get("frames", [])) != 1
    ):
        raise ValueError(
            "Native face landmarks changed the source image dimensions or frame count."
        )
    if options.get("face_output", "overlay") == "region-mask":
        import hashlib
        import torch
        from comfy_extras.nodes_mediapipe import MediaPipeFaceMask
        from comfy_extras.nodes_mask import MaskToImage

        regions = {"regions": options["face_mask_regions"]}
        if regions["regions"] == "custom":
            regions.update((region, options["face_mask_" + region]) for region in MASK_REGIONS)
        mask = native_call(check_cancelled, MediaPipeFaceMask.execute, landmarks, regions)[0]
        if (
            not isinstance(mask, torch.Tensor)
            or tuple(mask.shape) != (1, pixels.shape[1], pixels.shape[2])
            or not torch.isfinite(mask).all()
            or mask.min() < 0
            or mask.max() > 1
        ):
            raise ValueError("The native face-mask node returned invalid selection pixels.")
        empty = not bool(mask.any())
        face_count = len(landmarks["frames"][0])
        output = native_call(check_cancelled, MaskToImage.execute, mask)[0]
        return output, {
            **device_details,
            "native": "MediaPipeFaceLandmarker + MediaPipeFaceMask",
            "native_output": "FACE_LANDMARKS → MASK → IMAGE",
            "face_count": face_count,
            "result": "face-region-mask",
            "mask_role": "inpaint-selection",
            "mask_values": "white selects; black protects",
            "mask_channels": "equal RGB scalar selection; source alpha is coverage only",
            "source_sha256": hashlib.sha256(request.rgba).hexdigest(),
            "mask_frame": "exact oriented source frame; no crop or resize",
            "mask_empty": empty,
            "mask_empty_reason": "no-face"
            if not face_count
            else "no-region-pixels"
            if empty
            else None,
        }
    output = native_call(
        check_cancelled,
        MediaPipeFaceMeshVisualize.execute,
        landmarks,
        connections={"connections": "all"},
        color="#00ff00",
        thickness=1,
        point_size=2,
        image=pixels,
    )[0]
    return output, {
        **device_details,
        "native": "MediaPipeFaceLandmarker",
        "native_output": "FACE_LANDMARKS → IMAGE",
        "face_count": len(landmarks["frames"][0]),
        "result": "facial-landmark-overlay",
    }
