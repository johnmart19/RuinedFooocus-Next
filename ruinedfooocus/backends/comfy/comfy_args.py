"""Compute arguments matching the pinned ComfyUI parser; no backend imports at bootstrap."""

import enum


class PerformanceFeature(enum.Enum):
    Fp16Accumulation = "fp16_accumulation"
    Fp8MatrixMultiplication = "fp8_matrix_mult"
    CublasOps = "cublas_ops"
    AutoTune = "autotune"


COMFY_DESTINATIONS = (
    "force_fp32",
    "force_fp16",
    "fp32_unet",
    "fp64_unet",
    "bf16_unet",
    "fp16_unet",
    "fp8_e4m3fn_unet",
    "fp8_e5m2_unet",
    "fp8_e8m0fnu_unet",
    "fp16_vae",
    "fp32_vae",
    "bf16_vae",
    "cpu_vae",
    "fp8_e4m3fn_text_enc",
    "fp8_e5m2_text_enc",
    "fp16_text_enc",
    "fp32_text_enc",
    "bf16_text_enc",
    "fp16_intermediates",
    "force_channels_last",
    "directml",
    "supports_fp8_compute",
    "enable_triton_backend",
    "disable_triton_backend",
    "use_split_cross_attention",
    "use_quad_cross_attention",
    "use_pytorch_cross_attention",
    "use_sage_attention",
    "use_flash_attention",
    "use_ck_attention",
    "disable_xformers",
    "force_upcast_attention",
    "dont_upcast_attention",
    "gpu_only",
    "high_ram",
    "highvram",
    "lowvram",
    "novram",
    "cpu",
    "reserve_vram",
    "vram_headroom",
    "disable_nvml_pressure",
    "async_offload",
    "disable_async_offload",
    "disable_dynamic_vram",
    "enable_dynamic_vram",
    "fast_disk",
    "disable_fast_disk",
    "disable_cuda_graphs",
    "disable_comfy_compiler",
    "assert_graph_breaks",
    "force_non_blocking",
    "disable_smart_memory",
    "deterministic",
    "fast",
    "disable_pinned_memory",
    "mmap_torch_files",
    "disable_mmap",
    "cuda_malloc",
    "disable_cuda_malloc",
)


def add_comfy_arguments(parser):
    allocator = parser.add_mutually_exclusive_group()
    allocator.add_argument(
        "--cuda-malloc",
        action="store_true",
        help="Use the CUDA async allocator; requires runtime support.",
    )
    allocator.add_argument(
        "--disable-cuda-malloc",
        action="store_true",
        help="Use the native Torch allocator.",
    )
    attn_group = parser.add_mutually_exclusive_group()
    fp_group = parser.add_mutually_exclusive_group()
    fpte_group = parser.add_mutually_exclusive_group()
    fpunet_group = parser.add_mutually_exclusive_group()
    fpvae_group = parser.add_mutually_exclusive_group()
    upcast = parser.add_mutually_exclusive_group()
    vram_group = parser.add_mutually_exclusive_group()
    fp_group.add_argument(
        "--force-fp32",
        action="store_true",
        help="Force fp32 (If this makes your GPU work better please report it).",
    )
    fp_group.add_argument("--force-fp16", action="store_true", help="Force fp16.")
    fpunet_group.add_argument(
        "--fp32-unet", action="store_true", help="Run the diffusion model in fp32."
    )
    fpunet_group.add_argument(
        "--fp64-unet", action="store_true", help="Run the diffusion model in fp64."
    )
    fpunet_group.add_argument(
        "--bf16-unet", action="store_true", help="Run the diffusion model in bf16."
    )
    fpunet_group.add_argument(
        "--fp16-unet", action="store_true", help="Run the diffusion model in fp16"
    )
    fpunet_group.add_argument(
        "--fp8_e4m3fn-unet",
        action="store_true",
        help="Store unet weights in fp8_e4m3fn.",
    )
    fpunet_group.add_argument(
        "--fp8_e5m2-unet", action="store_true", help="Store unet weights in fp8_e5m2."
    )
    fpunet_group.add_argument(
        "--fp8_e8m0fnu-unet",
        action="store_true",
        help="Store unet weights in fp8_e8m0fnu.",
    )
    fpvae_group.add_argument(
        "--fp16-vae",
        action="store_true",
        help="Run the VAE in fp16, might cause black images.",
    )
    fpvae_group.add_argument(
        "--fp32-vae", action="store_true", help="Run the VAE in full precision fp32."
    )
    fpvae_group.add_argument("--bf16-vae", action="store_true", help="Run the VAE in bf16.")
    parser.add_argument("--cpu-vae", action="store_true", help="Run the VAE on the CPU.")
    fpte_group.add_argument(
        "--fp8_e4m3fn-text-enc",
        action="store_true",
        help="Store text encoder weights in fp8 (e4m3fn variant).",
    )
    fpte_group.add_argument(
        "--fp8_e5m2-text-enc",
        action="store_true",
        help="Store text encoder weights in fp8 (e5m2 variant).",
    )
    fpte_group.add_argument(
        "--fp16-text-enc",
        action="store_true",
        help="Store text encoder weights in fp16.",
    )
    fpte_group.add_argument(
        "--fp32-text-enc",
        action="store_true",
        help="Store text encoder weights in fp32.",
    )
    fpte_group.add_argument(
        "--bf16-text-enc",
        action="store_true",
        help="Store text encoder weights in bf16.",
    )
    parser.add_argument(
        "--fp16-intermediates",
        action="store_true",
        help="Experimental: Use fp16 for intermediate tensors between nodes instead of fp32.",
    )
    parser.add_argument(
        "--force-channels-last",
        action="store_true",
        help="Force channels last format when inferencing the models.",
    )
    parser.add_argument(
        "--directml",
        type=int,
        nargs="?",
        metavar="DIRECTML_DEVICE",
        const=-1,
        help="Use torch-directml.",
    )
    parser.add_argument(
        "--supports-fp8-compute",
        action="store_true",
        help="ComfyUI will act like if the device supports fp8 compute.",
    )
    parser.add_argument(
        "--enable-triton-backend",
        action="store_true",
        help="ComfyUI will enable the use of Triton backend in comfy-kitchen. Is disabled at launch by default.",
    )
    parser.add_argument(
        "--disable-triton-backend",
        action="store_true",
        help="Force-disable the comfy-kitchen Triton backend, overriding --enable-triton-backend.",
    )
    attn_group.add_argument(
        "--use-split-cross-attention",
        action="store_true",
        help="Use the split cross attention optimization. Ignored when xformers is used.",
    )
    attn_group.add_argument(
        "--use-quad-cross-attention",
        action="store_true",
        help="Use the sub-quadratic cross attention optimization . Ignored when xformers is used.",
    )
    attn_group.add_argument(
        "--use-pytorch-cross-attention",
        action="store_true",
        help="Use the new pytorch 2.0 cross attention function.",
    )
    attn_group.add_argument("--use-sage-attention", action="store_true", help="Use sage attention.")
    attn_group.add_argument(
        "--use-flash-attention", action="store_true", help="Use FlashAttention."
    )
    attn_group.add_argument(
        "--use-ck-attention", action="store_true", help="Use Comfy Kitchen attention."
    )
    parser.add_argument("--disable-xformers", action="store_true", help="Disable xformers.")
    # Upstream groups this with execution-cache modes, which the embedded
    # pipeline does not expose.
    parser.add_mutually_exclusive_group().add_argument(
        "--high-ram",
        action="store_true",
        help="Allow more aggressive use of RAM and pinned memory, including pagefile use.",
    )
    upcast.add_argument(
        "--force-upcast-attention",
        action="store_true",
        help="Force enable attention upcasting, please report if it fixes black images.",
    )
    upcast.add_argument(
        "--dont-upcast-attention",
        action="store_true",
        help="Disable all upcasting of attention. Should be unnecessary except for debugging.",
    )
    vram_group.add_argument(
        "--gpu-only",
        action="store_true",
        help="Store and run everything (text encoders/CLIP models, etc... on the GPU).",
    )
    vram_group.add_argument(
        "--highvram",
        action="store_true",
        help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory.",
    )
    vram_group.add_argument(
        "--lowvram",
        action="store_true",
        help="Doesn't do anything if dynamic vram is enabled. If dynamic vram isn't being used this option makes the text encoders run on the CPU.",
    )
    vram_group.add_argument("--novram", action="store_true", help="When lowvram isn't enough.")
    vram_group.add_argument(
        "--cpu", action="store_true", help="To use the CPU for everything (slow)."
    )
    parser.add_argument(
        "--reserve-vram",
        type=float,
        default=None,
        help="Set the amount of vram in GB you want to reserve for use by your OS/other software. By default some amount is reserved depending on your OS.",
    )
    parser.add_argument(
        "--vram-headroom",
        type=float,
        default=0,
        help="Set the amount of vram in GB for DynamicVRAM to maintain as extra headroom above default. ComfyUI will try and keep this much VRAM completely free and unused, even counting VRAM from other apps.",
    )
    parser.add_argument(
        "--disable-nvml-pressure",
        action="store_true",
        help="Use CUDA instead of NVML for DynamicVRAM memory pressure.",
    )
    parser.add_argument(
        "--async-offload",
        nargs="?",
        const=2,
        type=int,
        default=None,
        metavar="NUM_STREAMS",
        help="Use async weight offloading. An optional argument controls the amount of offload streams. Default is 2. Enabled by default on Nvidia.",
    )
    parser.add_argument(
        "--disable-async-offload",
        action="store_true",
        help="Disable async weight offloading.",
    )
    parser.add_argument(
        "--disable-dynamic-vram",
        action="store_true",
        help="Disable dynamic VRAM and use estimate based model loading.",
    )
    parser.add_argument(
        "--enable-dynamic-vram",
        action="store_true",
        help="Enable dynamic VRAM on systems where it's not enabled by default.",
    )
    parser.add_argument(
        "--fast-disk",
        action="store_true",
        help="Force disk-backed dynamic loading and offload over unpinned RAM. Can be faster for users with fast NVME disks.",
    )
    parser.add_argument(
        "--disable-fast-disk",
        action="store_true",
        help="Disable disk-backed dynamic loading and offload over unpinned RAM. Overrides --fast-disk.",
    )
    parser.add_argument("--disable-cuda-graphs", action="store_true", help="Disable CUDA graphs.")
    parser.add_argument(
        "--disable-comfy-compiler",
        action="store_true",
        help="Disable the Comfy model compiler, including its CUDA graph subfeature.",
    )
    parser.add_argument(
        "--assert-graph-breaks",
        action="store_true",
        help="Fail on Comfy model compiler graph breaks.",
    )
    parser.add_argument(
        "--force-non-blocking",
        action="store_true",
        help="Force ComfyUI to use non-blocking operations for all applicable tensors. This may improve performance on some non-Nvidia systems but can cause issues with some workflows.",
    )
    parser.add_argument(
        "--disable-smart-memory",
        action="store_true",
        help="Force ComfyUI to agressively offload to regular ram instead of keeping models in vram when it can.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Make pytorch use slower deterministic algorithms when it can. Note that this might not make images deterministic in all cases.",
    )
    parser.add_argument(
        "--fast",
        nargs="*",
        type=PerformanceFeature,
        help="Enable some untested and potentially quality deteriorating optimizations. This is used to test new features so using it might crash your comfyui. --fast with no arguments enables everything. You can pass a list specific optimizations if you only want to enable specific ones. Current valid optimizations: {}".format(
            " ".join(map(lambda c: c.value, PerformanceFeature))
        ),
    )
    parser.add_argument(
        "--disable-pinned-memory",
        action="store_true",
        help="Disable pinned memory use.",
    )
    parser.add_argument(
        "--mmap-torch-files",
        action="store_true",
        help="Use mmap when loading ckpt/pt files.",
    )
    parser.add_argument(
        "--disable-mmap",
        action="store_true",
        help="Don't use mmap when loading safetensors.",
    )
    vram_group.add_argument(
        "--normalvram",
        action="store_true",
        help="Legacy normal-VRAM mode: disables dynamic VRAM.",
    )
