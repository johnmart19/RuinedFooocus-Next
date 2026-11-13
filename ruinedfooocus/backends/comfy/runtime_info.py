"""Describe the initialized image runtime without probing devices or kernels."""

from importlib.metadata import PackageNotFoundError, version

_PREVIEW_POLICIES = {
    "auto": "Auto (Fast RGB when the model supports it)",
    "none": "Off",
    "latent2rgb": "Fast RGB when the model supports it",
    "taesd": "Approximate VAE when available; otherwise Fast RGB when supported",
}


def runtime_details(torch, device, device_name, args, memory, package_version=version):
    if device.type == "cpu":
        backend = "CPU"
    elif args.directml is not None:
        backend = "DirectML (legacy PyTorch)"
    elif device.type == "cuda":
        backend = "ROCm / HIP" if torch.version.hip else "CUDA"
    else:
        backend = device.type.upper()
    rows = [
        ["Active image backend", backend],
        ["Selected device", device_name],
        ["PyTorch build", torch.__version__],
    ]
    if torch.version.hip:
        try:
            sdk = package_version("rocm")
        except PackageNotFoundError:
            sdk = "Not declared (system/custom build)"
        rows += [["Compiled HIP component", torch.version.hip], ["ROCm wheel SDK", sdk]]
    elif torch.version.cuda:
        rows.append(["Compiled CUDA runtime", torch.version.cuda])
    preview_method = getattr(args, "preview_method", None)
    preview = _PREVIEW_POLICIES.get(getattr(preview_method, "value", preview_method), "Unknown")
    rows += [
        ["Dynamic VRAM active", "Yes" if memory.aimdo_enabled else "No"],
        ["Sampling preview at launch", preview],
        [
            "Weight transfer streams requested",
            "Disabled"
            if args.disable_async_offload
            else str("Native automatic" if args.async_offload is None else args.async_offload),
        ],
        ["VAE target requested", "CPU" if args.cpu_vae else "Native automatic"],
        [
            "Fast optimizations requested",
            ", ".join(sorted(str(getattr(value, "value", value)) for value in args.fast)) or "None",
        ],
        [
            "Qualification",
            "Build and initialized-state information, not kernel, model-quality or driver-stability certification. Chat and ONNX tools use separate runtimes.",
        ],
    ]
    for label, flags in (
        (
            "Image precision requested",
            (
                "fp32_unet",
                "fp16_unet",
                "bf16_unet",
                "fp64_unet",
                "fp8_e4m3fn_unet",
                "fp8_e5m2_unet",
                "fp8_e8m0fnu_unet",
            ),
        ),
        (
            "Text precision requested",
            (
                "fp32_text_enc",
                "fp16_text_enc",
                "bf16_text_enc",
                "fp8_e4m3fn_text_enc",
                "fp8_e5m2_text_enc",
            ),
        ),
        ("VAE precision requested", ("fp32_vae", "fp16_vae", "bf16_vae")),
        (
            "Attention requested",
            (
                "use_pytorch_cross_attention",
                "use_split_cross_attention",
                "use_quad_cross_attention",
                "use_sage_attention",
                "use_flash_attention",
                "use_ck_attention",
            ),
        ),
    ):
        rows.append(
            [
                label,
                ", ".join(flag for flag in flags if getattr(args, flag, False))
                or "Native automatic",
            ]
        )
    rows += [
        [
            "Native compiler requested",
            "Disabled" if getattr(args, "disable_comfy_compiler", False) else "Native automatic",
        ],
        [
            "CUDA graphs requested",
            "Disabled" if getattr(args, "disable_cuda_graphs", False) else "Native automatic",
        ],
    ]
    return rows
