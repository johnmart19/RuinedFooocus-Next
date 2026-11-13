from ruinedfooocus.core.paths import data_path

"""Saved native compute options, applied before backend initialization."""
from ruinedfooocus.core.settings_schema import (
    COMPUTE_CHOICES,
    COMPUTE_MEMORY_FIELDS,
    validate_known,
)
import os


CHOICES = COMPUTE_CHOICES
ADVANCED_CONTROLS = (
    (
        "image_deterministic",
        "Deterministic algorithms",
        "Restart required. May reduce speed; native backend controls unsupported operations.",
    ),
    (
        "image_smart_memory",
        "Smart memory",
        "Auto retains native weight caching. Disabled unloads more aggressively.",
    ),
    (
        "image_pinned_memory",
        "Pinned memory",
        "Auto follows the backend. Disabled avoids pinned host-memory transfers.",
    ),
    (
        "image_mmap",
        "Memory-mapped model files",
        "Auto follows native loading. Disabled avoids mmap; Enabled requests mmap where supported.",
    ),
    (
        "image_allocator",
        "GPU allocator",
        "Auto follows native device policy. Async requires CUDA runtime support; CPU ignores this preference.",
    ),
    (
        "image_unet_precision",
        "Image network precision",
        "Auto follows the model and device. Forced precision may change quality or memory use.",
    ),
    (
        "image_text_precision",
        "Text encoder precision",
        "Auto preserves native encoder choices. CPU ignores saved FP16/BF16 overrides.",
    ),
    (
        "image_vae_precision",
        "VAE precision",
        "Auto follows the model. CPU ignores saved FP16/BF16 overrides.",
    ),
    (
        "image_attention",
        "Attention implementation",
        "Native PyTorch, split or quad attention. CUDA/ROCm-specific fused extensions remain explicit CLI options.",
    ),
    (
        "image_attention_upcast",
        "Attention upcasting",
        "Auto follows native stability safeguards. Disabling may change numerical quality.",
    ),
    (
        "image_compiler",
        "Native model compiler",
        "Auto permits native compilation when supported; Disabled opts out. Does not install a compiler.",
    ),
    (
        "image_cuda_graphs",
        "Native CUDA graphs",
        "Auto follows the backend; Disabled opts out. No effect on unsupported devices.",
    ),
)
MEMORY_FIELDS = COMPUTE_MEMORY_FIELDS


def validate_compute(values):
    result = {
        key: value
        for key, value in values.items()
        if key in CHOICES or key in MEMORY_FIELDS or key == "cpu_everywhere"
    }
    return validate_known(result)


def explicit_options(parser, argv):
    """Recognize argparse's long-option abbreviations and --option=value syntax."""
    known = {option for action in parser._actions for option in action.option_strings}
    result = set()
    for token in argv:
        if token == "--":
            break
        option = token.split("=", 1)[0]
        matches = (
            [name for name in known if name.startswith(option)] if option.startswith("--") else []
        )
        if option in known:
            result.add(option)
        elif len(matches) == 1:
            result.add(matches[0])
    return result


def apply_compute(args, values, options=()):
    """CLI choices win, including opposing flags and explicit zero values."""
    from ruinedfooocus.packaging.backend_preferences import saved_cpu_requested

    values = validate_compute(values)
    options = set(options)
    if (
        saved_cpu_requested(values)
        and not os.environ.get("TORCH_PLATFORM")
        and not options.intersection(
            (
                "--cpu",
                "--directml",
                "--gpu-device-id",
                "--cuda-device",
                "--default-device",
                "--oneapi-device-selector",
                "--gpu-only",
                "--highvram",
                "--lowvram",
                "--novram",
                "--normalvram",
                "--rocm10",
                "--legacy",
                "--cuda-nightly",
            )
        )
    ):
        args.cpu = True
    for key, flag in (
        ("image_deterministic", "deterministic"),
        ("image_smart_memory", "disable_smart_memory"),
        ("image_pinned_memory", "disable_pinned_memory"),
    ):
        if (
            values.get(key) in ("Enabled", "Disabled")
            and "--" + flag.replace("_", "-") not in options
        ):
            setattr(args, flag, True)
    mmap = values.get("image_mmap", "Auto")
    if mmap != "Auto" and not options.intersection(("--mmap-torch-files", "--disable-mmap")):
        args.mmap_torch_files = mmap == "Enabled"
        args.disable_mmap = mmap == "Disabled"
    allocator = values.get("image_allocator", "Auto")
    if (
        allocator != "Auto"
        and not options.intersection(("--cuda-malloc", "--disable-cuda-malloc"))
        and not getattr(args, "cpu", False)
        and os.environ.get("TORCH_PLATFORM") != "cpu"
    ):
        args.cuda_malloc = allocator == "Async"
        args.disable_cuda_malloc = allocator == "Native"
        args._saved_allocator_flags = (
            "cuda_malloc" if allocator == "Async" else "disable_cuda_malloc",
        )
    for key in MEMORY_FIELDS:
        flag = "--" + key.removeprefix("image_").replace("_", "-")
        if flag not in options and values.get(key) is not None:
            setattr(args, key.removeprefix("image_"), values[key])
    for key, enabled, disabled, conflicts in (
        (
            "image_dynamic_vram",
            "enable_dynamic_vram",
            "disable_dynamic_vram",
            (
                "--cpu",
                "--directml",
                "--gpu-only",
                "--highvram",
                "--lowvram",
                "--novram",
                "--normalvram",
            ),
        ),
        ("image_fast_disk", "fast_disk", "disable_fast_disk", ()),
    ):
        choice = values.get(key, "Auto")
        flags = {"--" + name.replace("_", "-") for name in (enabled, disabled)}
        if choice != "Auto" and not options.intersection(flags | set(conflicts)):
            setattr(args, enabled, choice == "Enabled")
            setattr(args, disabled, choice == "Disabled")
    offload = values.get("image_async_offload", "Auto")
    if offload != "Auto" and not options.intersection(
        ("--async-offload", "--disable-async-offload")
    ):
        args.disable_async_offload = offload == "Disabled"
        args.async_offload = None if offload == "Disabled" else int(offload)
    if values.get("image_vae_device") == "CPU" and "--gpu-only" not in options:
        args.cpu_vae = True
    if "image_preview_method" in values and "--preview-method" not in options:
        args.preview_method = values["image_preview_method"]
    for key, suffix in (
        ("image_unet_precision", "unet"),
        ("image_text_precision", "text_enc"),
        ("image_vae_precision", "vae"),
    ):
        flags = {
            f"--{precision}-{suffix.replace('_', '-')}"
            for precision in (
                "fp32",
                "fp16",
                "bf16",
                "fp64",
                "fp8_e4m3fn",
                "fp8_e5m2",
                "fp8_e8m0fnu",
            )
        }
        flags |= {"--force-fp32", "--force-fp16"}
        precision = values.get(key, "Auto")
        if precision != "Auto" and not options.intersection(flags):
            if (
                getattr(args, "cpu", False)
                or os.environ.get("TORCH_PLATFORM") == "cpu"
                or (suffix == "vae" and args.cpu_vae)
            ) and precision != "FP32":
                flag = precision.lower() + "_" + suffix
                args._saved_precision_suppressed = (
                    *getattr(args, "_saved_precision_suppressed", ()),
                    flag,
                )
                continue
            flag = precision.lower() + "_" + suffix
            setattr(args, flag, True)
            if precision in ("FP16", "BF16"):
                args._saved_precision_flags = (*getattr(args, "_saved_precision_flags", ()), flag)
    attention = values.get("image_attention", "Auto")
    attention_flags = {
        "--use-" + kind + "-attention"
        for kind in ("pytorch-cross", "split-cross", "quad-cross", "sage", "flash", "ck")
    }
    if attention != "Auto" and not options.intersection(attention_flags):
        setattr(
            args,
            "use_"
            + {"PyTorch": "pytorch", "Split": "split", "Quad": "quad"}[attention]
            + "_cross_attention",
            True,
        )
    upcast = values.get("image_attention_upcast", "Auto")
    if upcast != "Auto" and not options.intersection(
        ("--force-upcast-attention", "--dont-upcast-attention")
    ):
        args.force_upcast_attention = upcast == "Enabled"
        args.dont_upcast_attention = upcast == "Disabled"
    for key, flag in (
        ("image_compiler", "disable_comfy_compiler"),
        ("image_cuda_graphs", "disable_cuda_graphs"),
    ):
        if values.get(key) == "Disabled" and "--" + flag.replace("_", "-") not in options:
            setattr(args, flag, True)


def load_compute(args, parser, argv):
    from ruinedfooocus.core.profile_store import read_profile

    folder = (
        data_path("settings", args.settings)
        if args.settings not in (None, "", "default")
        else data_path("settings")
    )
    path = folder / "settings.json"
    values = read_profile(path)
    options = explicit_options(parser, argv)
    before = vars(args).copy()
    args._user_torch_platform = os.environ.get("TORCH_PLATFORM") or None
    apply_compute(args, values, options)
    from ruinedfooocus.packaging.backend_preferences import (
        cpu_everywhere,
        saved_cpu_requested,
        saved_image_backend,
    )

    args._compute_status = {
        "requested": {
            key: value
            for key, value in values.items()
            if key in CHOICES or key in MEMORY_FIELDS or key == "cpu_everywhere"
        },
        "cli": sorted(options),
        "backend": {
            "saved_image_backend": saved_image_backend(values),
            "cpu_everywhere": cpu_everywhere(values),
            "cpu_applied": bool(saved_cpu_requested(values) and args.cpu and not before.get("cpu")),
        },
        "profile_applied": {
            key: value
            for key, value in vars(args).items()
            if not key.startswith("_") and before.get(key) != value
        },
        "cpu_suppressed": [],
    }
    return values


def finalize_compute(args, *, cpu=None):
    """Apply CPU safeguards after environment/automatic device resolution.

    Only flags supplied by the profile are removed; explicit CLI choices are
    retained. No hardware is queried here.
    """
    cpu = (
        getattr(args, "cpu", False) or os.environ.get("TORCH_PLATFORM") == "cpu"
        if cpu is None
        else cpu
    )
    suppressed = list(getattr(args, "_saved_precision_suppressed", ()))
    if cpu:
        args.cpu = True
        for flag in (
            *getattr(args, "_saved_precision_flags", ()),
            *getattr(args, "_saved_allocator_flags", ()),
        ):
            setattr(args, flag, False)
            if flag not in suppressed:
                suppressed.append(flag)
    status = getattr(args, "_compute_status", None)
    if status is not None:
        from ruinedfooocus.backends.comfy.comfy_args import COMFY_DESTINATIONS

        status["cpu_suppressed"] = suppressed
        status["effective"] = {
            key: getattr(args, key, None)
            for key in (*COMFY_DESTINATIONS, "preview_method")
            if key != "fast"
        }
    return suppressed
