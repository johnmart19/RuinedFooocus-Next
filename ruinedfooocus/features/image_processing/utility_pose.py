"""Native SDPose whole-body keypoints rendered as a source-sized still map."""

from contextlib import ExitStack
from ruinedfooocus.features.image_editing import lotus_depth
from ruinedfooocus.features.models.model_identity import cached_safetensors_header
from ruinedfooocus.features.image_processing.depth_utilities import cached_model, native_call
from ruinedfooocus.features.image_processing.native_vision import _number


def validate_model(path):
    from ruinedfooocus.features.models.component_candidates import classify_shapes

    lotus_depth.validate_model(path)
    header = cached_safetensors_header(path)
    if not any(key.endswith("heatmap_head.conv_layers.0.weight") for key in header):
        raise ValueError("Select a native SDPose checkpoint with a whole-body heatmap head.")
    # CheckpointLoaderSimple in the official recipe consumes the bundled SD VAE.
    prefix = "first_stage_model."
    codec = {key[len(prefix) :]: value for key, value in header.items() if key.startswith(prefix)}
    component = classify_shapes(
        {key: tuple(entry.get("shape", ())) for key, entry in codec.items()}
    )
    if component.get("vae_channels") != 4 or component.get("vae_layout") != "planar":
        raise ValueError("Select an SDPose checkpoint with its bundled four-channel SD VAE.")
    return "SDPose"


COMPONENTS = (
    (
        "image_utility_model",
        "sdpose-checkpoint",
        ("checkpoints", "diffusion_models"),
        validate_model,
    ),
)


def parameters(data):
    return {
        "pose_score_threshold": _number(
            data.get("pose_score_threshold", 0.5), "Pose confidence", 0, 1
        )
    }


def render(owner, request, pixels, data, manager, check_cancelled):
    import comfy.sd
    from comfy_extras.nodes_sdpose import SDPoseKeypointExtractor, SDPoseDrawKeypoints

    (item,) = request.files

    def load():
        from comfy.model_management import unload_model_and_clones

        with ExitStack() as cleanup:
            model, _, vae, *_ = comfy.sd.load_checkpoint_guess_config(
                str(item.path), output_clip=False, output_vae=True
            )
            if model is not None:
                cleanup.callback(unload_model_and_clones, model)
            if vae is not None:
                cleanup.callback(unload_model_and_clones, vae.patcher)
            check_cancelled()
            if model is None or vae is None:
                raise ValueError(
                    "The native SDPose checkpoint loader did not return its model and bundled VAE."
                )
            vae.throw_exception_if_invalid()
            if (
                vae.latent_channels != 4
                or vae.downscale_ratio != 8
                or not hasattr(model.model.diffusion_model, "heatmap_head")
            ):
                raise ValueError(
                    "The native SDPose checkpoint has incompatible model or VAE components."
                )
            cleanup.pop_all()
        return (model, vae), (model, vae.patcher)

    model, vae = cached_model(owner, request, load, check_cancelled)
    keypoints = native_call(
        check_cancelled,
        SDPoseKeypointExtractor.execute,
        model,
        vae,
        pixels,
        batch_size=1,
        bboxes=None,
    )[0]
    if (
        len(keypoints) != 1
        or (keypoints[0].get("canvas_width"), keypoints[0].get("canvas_height")) != request.size
    ):
        raise ValueError(
            "Native SDPose keypoints changed the source image dimensions or frame count."
        )
    output = native_call(
        check_cancelled,
        SDPoseDrawKeypoints.execute,
        keypoints,
        draw_body=True,
        draw_hands=True,
        draw_face=True,
        draw_feet=True,
        stick_width=4,
        face_point_size=2,
        score_threshold=dict(request.parameters)["pose_score_threshold"],
        draw_head=True,
    )[0]
    return output, {
        "native": "SDPose",
        "native_output": "POSE_KEYPOINT → IMAGE",
        "people": len(keypoints[0]["people"]),
        "result": "whole-body-pose-map",
        "person_scope": "one-full-image-person",
    }
