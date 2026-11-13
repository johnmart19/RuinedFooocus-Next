"""Source archive preparation; platform runtime bundles are separate releases."""

import ast
from io import BytesIO
import json
from pathlib import Path
import subprocess
import zipfile
from ruinedfooocus.core.paths import source_checkout

PORTABLE_TARGETS = ("windows-cuda", "windows-rocm", "linux-cuda", "linux-rocm")
USER_DIRECTORIES = (
    "settings",
    "outputs",
    "models",
    "cache",
    "venv",
    "repositories",
    "tmp",
    "chatbots",
    "llamas",
    "assistants",
    "prompt_presets",
    "history-backups",
    ".codex",
    ".claude",
    ".agents",
)
WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".gguf", ".pth", ".pt", ".onnx", ".bin")
MAX_MODEL_PLACEMENT_BYTES = 16 * 1024
MODEL_PLACEMENT_FILES = (
    "models/README.md",
    "models/background_removal/put_birefnet_models_here",
    "models/checkpoints/put_checkpoints_here",
    "models/clip_vision/put_clip_vision_models_here",
    "models/controlnet/put_controlnets_and_t2i_here",
    "models/diffusers/put_local_diffusers_bundles_here",
    "models/diffusion_models/put_image_diffusion_models_here",
    "models/embeddings/put_textual_inversion_embeddings_here",
    "models/faceswap/put_gfpgan_face_restore_here",
    "models/geometry_estimation/put_depth_estimation_models_here",
    "models/inbox/Place_models_to_be_sorted_here",
    "models/llm/put_llama_cpp_models_here",
    "models/loras/put_loras_here",
    "models/photomaker/put_photomaker_models_here",
    "models/style_models/put_reference_style_models_here",
    "models/text_encoders/put_text_encoder_models_here",
    "models/upscale_models/put_esrgan_and_other_upscale_models_here",
    "models/vae/put_vae_here",
    "models/vae_approx/put_approximate_vae_decoders_here",
)


def git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments])


def _append_model_placement(root, commit, archive):
    # Match regular file entries at the selected revision, not working-tree paths.
    files = []
    entries = git(root, "ls-tree", "-rlz", commit, "--", *MODEL_PLACEMENT_FILES)
    for entry in entries.split(b"\0"):
        attributes, _, path = entry.partition(b"\t")
        if (
            attributes.startswith(b"100644 blob ")
            and int(attributes.rsplit(maxsplit=1)[1]) <= MAX_MODEL_PLACEMENT_BYTES
        ):
            name = path.decode("utf-8", errors="surrogateescape")
            if name in MODEL_PLACEMENT_FILES:
                files.append(name)
    if files:
        placement = git(
            root, "-c", "core.autocrlf=false", "archive", "--format=zip", commit, "--", *files
        )
        with zipfile.ZipFile(BytesIO(placement)) as source:
            for entry in source.infolist():
                archive.writestr(entry, source.read(entry))


def build_source_archive(root, output, ref="HEAD"):
    root, output = Path(root).resolve(), Path(output).resolve()
    commit = git(root, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
    bootstrap = git(root, "show", f"{commit}:ruinedfooocus/app/bootstrap.py").decode("utf-8")
    tree = ast.parse(bootstrap)
    repositories = ast.literal_eval(
        next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(name, ast.Name) and name.id == "git_repos" for name in node.targets)
        )
    )
    manifest = {
        "kind": "source-archive",
        "source_commit": commit,
        "backend_pins": {item["name"]: item["hash"] for item in repositories},
        "portable_targets": list(PORTABLE_TARGETS),
        "runtime_bundled": False,
        "models_bundled": False,
        "data_environment": "RF_DATA_DIR",
    }
    exclusions = [f":(exclude){name}" for name in USER_DIRECTORIES]
    exclusions += [f":(exclude,icase)*{suffix}" for suffix in WEIGHT_SUFFIXES]
    exclusions += [":(exclude)AGENTS.md", ":(exclude)CLAUDE.md", ":(exclude)distribution.json"]
    exclusions += [f":(exclude){name}" for name in ("reinstall", "reinstalltorch", "freezetorch")]
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is outside cleanup: an existing artifact is never removed.
    destination = output.open("x+b")
    try:
        with destination:
            subprocess.run(
                [
                    "git",
                    "-c",
                    "core.autocrlf=false",
                    "-C",
                    str(root),
                    "archive",
                    "--format=zip",
                    commit,
                    "--",
                    ".",
                    *exclusions,
                ],
                stdout=destination,
                check=True,
            )
            with zipfile.ZipFile(destination, "a", compression=zipfile.ZIP_DEFLATED) as archive:
                _append_model_placement(root, commit, archive)
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise ValueError("Archive contains duplicate paths")
                if any(name.lower().endswith(WEIGHT_SUFFIXES) for name in names):
                    raise ValueError("Source archives must not include model weights")
                if any(
                    name.startswith("models/")
                    and not name.endswith("/")
                    and name not in MODEL_PLACEMENT_FILES
                    for name in names
                ):
                    raise ValueError("Source archives can include only model placement files")
                entry = zipfile.ZipInfo(
                    "distribution.json", date_time=archive.infolist()[0].date_time
                )
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, json.dumps(manifest, indent=2) + "\n")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return manifest
