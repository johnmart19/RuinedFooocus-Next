# Installation, runtimes and troubleshooting

Local image generation, editing and chat, using a pinned ComfyUI backend and a
shared local llama.cpp runtime. Supports AMD ROCm, NVIDIA CUDA and CPU, with deprecated
explicit DirectML compatibility where the selected model and runtime are compatible.

## Install

Git startup updates preserve tracked edits and allow unrelated untracked work;
safe checkout refuses collisions instead of overwriting local or ignored files.
Pinned backend checks run once per startup and reuse an already matching checkout.
Both launchers print a status before selecting Python and run it unbuffered.
The updater also prints a flushed status before checking the profile, packages or
Git. Application/backend fetches stop after 10 seconds; failed updates report the
reason and continue with installed source. First-use backend clones announce a
separate 300-second deadline and publish only a completed checkout. Package/model
installation has its own work and compatibility checks. See the
[startup contract](integrations/startup.md) and
[measured stages and limits](integrations/startup-timings.md).
Local model lists are ready before publisher artwork work starts in the background
after the browser opens. **Refresh All Files** scans local files only and preserves
search text, model selection, prompts and active adapters. It does not retry online
artwork lookups; existing background workers deduplicate requests and retain cached
previews when a publisher is unavailable.

RuinedFooocus needs Python 3.11–3.14 and Git. Python 3.10 remains a deprecated compatibility path and is end-of-life. Python 3.13 is suitable for new environments with matching GPU wheels; an existing Python 3.12 environment does not need replacing for this UI stack. The launchers create a
`venv` folder, install the right PyTorch build for your GPU and start the web UI. Package installation can
download several GB. The optional default image model is
[sd_xl_base_1.0_0.9vae.safetensors](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors)
and belongs in `models/checkpoints`. Download it explicitly in **Create → Models → Model → Add image models → Download default image model**,
or add another compatible checkpoint. If you already have it, keep your existing copy. Optional prompt enhancement reuses the selected local Chat model. Retired GPT-2/SuperPrompt weights
remain untouched; no separate enhancement model is installed.

Chat and optional prompt enhancement default to [LFM3 1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF),
using the official Q4_K_M GGUF (about 1.1 GB). In Chat, click **Download** for missing
weights, then **Load**. Selecting or downloading a candidate preserves the active
conversation and model; see [Chat preparation](chat-preparation.md).
Saved model/quantization choices stay selected across refresh and Installed/Download views, including unavailable files.
Chat explains source-audited support for the selected runtime; device execution is a separate check.
Known unsupported architectures are checked again on the worker using GGUF metadata, not just filenames.
Search the Installed or Download model list by name, quantization or filename without replacing the selected model.
Model sources and evidence distinguish published file/bundle sizes, unmeasured memory envelopes, licenses and attributed benchmarks from local validation.
**Chat → Download** includes 25 additional source-reviewed language weights (Qwen 3.5/3.8, Gemma 4, LFM3-VL, MiniCPM-V, Granite, Ministral, Ling and Nemotron) with 19 paired projectors. Downloads are explicit; projectors are components, not Chat choices. Source support does not certify local speed, tool accuracy or image quality.
Settings → Local Chat → Local chat and prompt runtime → Chat sampling exposes publisher-default opt-out and optional numeric overrides; blank means automatic and zero remains explicit.
Automatic projector pairing requires matching published weight and projector content; custom models can use an explicit pair.
Text-only calls do not load a projector. Image input or enabled persistent vision selects the vision load, and temporary image review restores the prior mode.
**Chat → Vision image detail** offers an optional 2048-pixel PNG for image references and review; Standard keeps the 1024-pixel transport.
High detail uses more memory, and the model may resize it further. The image-review API accepts `image_detail: "standard"` or `"high"`.
Original pixels, framing and metadata are preserved in the source; transport uses a separate oriented RGB copy.
Saved model choices stay selected. GPT-OSS, Qwen and custom local GGUFs remain available as advanced options.
Verified Chat weights share publisher sampling across Chat, API calls, prompt enhancement and image review.
Explicit request values override native server arguments, saved sampler overrides and model recommendations.
Custom or altered weights retain their embedded defaults; a matching filename alone does not establish identity.
LFM3 uses the native llama.cpp runtime; its tool calls are handled by the same application tools.
Its defaults favor quick replies and basic tools; advanced users can override the
native reasoning budget with `--reasoning-budget` in the llama.cpp server arguments.

Encoder/VAE choices are under **Create → Models → Model → Image model components → Component choices for this checkpoint**,
beside the component inventory. **Save this component choice** saves it for that
checkpoint in the current profile and applies it on the checkpoint's next load.
Built-in components remain fixed; no restart is required for a saved component choice.

[Gradio 6.32.0](https://github.com/gradio-app/gradio/releases/tag/gradio%406.32.0)
supports Pillow 13 and explicitly retains integer PNG output through 16-bit
clipping. RF selects [Pillow 13.0.0](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html)
on Python 3.11+; Python 3.10 retains its latest compatible Pillow 12.3.0.
The [frontend contract](integrations/gradio-frontend-completion.md) records the
exact guarded native assets and focused CPU checks. Package compatibility and
CPU checks remain separate from rendered browser or device qualification.

### Windows

    git clone https://github.com/johnmart19/RuinedFooocus-ROCm.git RuinedFooocus
    cd RuinedFooocus
    RuinedFooocus.bat

Set `PYTHON` to a python.exe, or activate a virtual environment, to use it instead of `venv`.
The order is explicit `PYTHON`, active environment, local `venv`, then creation
of a new local environment. Arguments and the application's exit code are kept.

### Linux and WSL

    git clone https://github.com/johnmart19/RuinedFooocus-ROCm.git RuinedFooocus
    cd RuinedFooocus
    ./RuinedFooocus.sh

On Ubuntu install `python3-venv` and `git` first. Interpreter selection follows
the same order as Windows. An explicit `PYTHON` is used directly, including when
a local `venv` exists; activate or select the intended environment before launch.
Without a selection, the shell creates a local `venv` with `python3` when needed.
Pass `--listen` (with `--auth user/password`) to open a LAN port.

### GPU backends

The launcher prefers a compatible GPU. Automatic selection falls back to CPU
when acceleration is unavailable and prints the affected feature and reason.
Use `--cpu` to select CPU explicitly.

| GPU                                                                                                       | PyTorch                        | Notes                                                                                    |
| --------------------------------------------------------------------------------------------------------- | ------------------------------ | ---------------------------------------------------------------------------------------- |
| NVIDIA Maxwell–Pascal                                                                                     | CUDA 12.6 / 12.4               |                                                                                          |
| NVIDIA Turing and newer                                                                                   | CUDA 13.4 / 13.2 / 13.0 / 12.8 | Newest stable bundle your driver and Python support. Nightlies require `--cuda-nightly`. |
| AMD RX 5000/6000/7000/9000, RDNA 3.5 APUs (Windows, Linux, WSL), Instinct MI100/MI200/MI300/MI350 (Linux) | ROCm 10.1                      | `--rocm10` reinstalls/repairs it                                                         |
| Intel Arc A/B-series, Core Ultra (Meteor/Arrow/Lunar Lake) iGPUs (Windows, Linux, WSL)                    | XPU 2.15.0                     | Official `whl/xpu` wheels; Chat can use the SYCL llama.cpp build, Python 3.11+           |
| Other AMD/Intel GPUs on Windows                                                                           | CPU by default                 | Explicit legacy `--directml` remains available on Python 3.10–3.12                       |
| No supported GPU                                                                                          | CPU                            | or `--cpu`                                                                               |

Use `--legacy` for an explicit NVIDIA CUDA 12.6/12.4 compatibility profile (including GTX 1050/1060).
CPU, XPU and modern CUDA 13.2/13.4 use Torch 2.15 / torchvision 0.29.2 on Python 3.11+.
Python 3.10 retains CPU/CUDA 13.2 Torch 2.14.1; CUDA 12.6/13.0 retain their latest published 2.14.1 pair and CUDA 12.8 retains 2.11.0.
The [PyTorch 2.15 release](https://pytorch.org/blog/pytorch-2-15-release-blog/) and official wheel indexes were checked on 2026-11-13.
CUDA 13.4 is a stable wheel channel; its experimental Rubin coverage still requires separate hardware acceptance.
Legacy mode validates all selected GPUs, the driver and wheel tags before installation. Offline/frozen launches require an
already installed compatible legacy bundle. Small VRAM still requires model-appropriate resolution and offloading; this
profile does not make every large model fit 2–3 GB. AMD uses its separately qualified vendor bundle.
**Settings → Image runtime → Image compute → Active image runtime** separates the active device/backend from compiled HIP/CUDA and wheel-SDK versions, and shows initialized Dynamic VRAM plus requested compute options. It does not run hardware probes or certify model quality/stability.

**Sampling preview at launch** shows the policy this process started with.
Change **Sampling preview**, save, and restart to apply a profile preference.
`--preview-method` takes priority. Auto uses Fast RGB for models that support
it. Approximate VAE needs a matching preview decoder and can fall back to
Fast RGB when the model supports it.

If Dynamic VRAM reports no visible devices, check GPU selection and visibility
settings. If AIMDO rejects visible device indices, check the preceding controller
messages. ComfyUI keeps using the selected GPU with legacy memory management.
Dynamic VRAM was verified on a Windows RX 7900 XTX with ROCm wheels 10.1.0,
PyTorch 2.14.0 and `comfy-aimdo` 0.6.0.
AMD target selection requires device extras in the pinned Torch/vision bundle; installer eligibility is not hardware certification. Check AMD's [released OS/GPU/driver matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html), especially for WSL and older Radeon hardware. Experimental build-only targets remain outside automatic selection.
A bounded Windows RX 7900 XTX run completed 75 automatic-setting image cases. All passed runtime, prompt transport and geometry checks. Five failed semantic quality review. Selected native comparisons reproduced SDXL/Janku object-count errors and a Qwen edit-preservation failure. The bounded Pascal Windows/WSL qualification below does not qualify other NVIDIA devices, bare-metal Linux or other Windows ML devices. Automatic Windows ML selection for background removal remains disabled pending model qualification.

### Bounded GTX 1060 3GB qualification — 2026-11-10

One PC used driver 582.78, Windows 11 / Python 3.12.10 and Ubuntu 26.04.1
WSL2 / Python 3.14.4, with separate environments and PyTorch 2.14.1+cu126.
Both normal launchers selected CUDA automatically without CPU, legacy, backend,
offline or dependency-skip forcing. Native ComfyUI was
`fca60fe70e5b84c5c6b058263653d15cb3401e44`; llama.cpp was b12110
(Windows CUDA 12.4, WSL CUDA 12.8). Installation/operator evidence is separate
from these actual model executions.

| Case                                          | Windows                                      | Ubuntu WSL2                      |
| --------------------------------------------- | -------------------------------------------- | -------------------------------- |
| Janku Model Defaults, recommended 1024×1024   | Completed, 109s sampling                     | Completed, 164s sampling         |
| LFM3 compact arithmetic reply                 | Passed on CUDA                               | Passed on CUDA                   |
| GPT-OSS exact `RESULT=12` reply, 2048 context | Passed, mixed CPU/CUDA                       | Passed, 8/25 layers on CUDA      |
| Native BiRefNet, 512×512 RGBA mask            | Passed on CUDA                               | Passed on CUDA                   |
| AnimeSharp 2×, 512→1024 with alpha retained   | Passed on CUDA                               | Passed on CUDA                   |
| Depth Anything 3 Small, 512×512 map           | Passed on CUDA                               | Passed on CUDA                   |
| Florence-2-base caption                       | Passed, native image-device path             | Passed, native image-device path |
| RT-DETR v4 FP32, annotated 512×512 detections | Passed on CUDA                               | Passed on CUDA                   |
| Janku image-to-image, 768×512, denoise 0.25   | Passed, framing/prompt/PNG metadata retained | Not tested                       |

The automatic Janku PNGs record 25 steps, CFG 3, Euler, normal scheduler and
ClipSkip 2, built-in encoders/VAE and no LoRA/component overrides. Online creator
card fields supply 25/3/Euler; exact-version gallery inference supplies ClipSkip 2;
the Illustrious family supplies scheduler/size. The separately tested offline
cached-example recipe infers 30/3/Euler ancestral/ClipSkip 2 when higher-priority
creator fields are unavailable. It is not a conflicting creator recommendation.
The images execute correctly but weak anime styling and an extra vessel in the
WSL case remain semantic limitations.

Automatic text-to-image evidence used `f8c3371`; image-to-image and utility
evidence used `93a0263` (unchanged native compute paths). The latter fixes background identity discovery leaving
the selected family stale: an actual UI check updated SDXL→Illustrious without
reload while preserving the prompt and manual size. Its 135-feature prerequisite
replay and 27 focused tests passed. Background CPU/OOM recovery contracts passed
25 isolated CPU tests on each OS; actual native BiRefNet completed on GPU after
allocator trimming, so the CPU recovery branch was not physically exercised.
Depth auxiliary-head and TorchScript deprecation warnings were present.

Curated payload identities (SHA-256; utility copies matched across OSes):

| Payload                                                     | SHA-256 / immutable source                                                                                                                                                        |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| JANKU v7.777.safetensors, publisher v7.77 / version 2786084 | `88177d224ce97f60cf3e908f87902f913ed4562bd79d237f84238a4354601efb`                                                                                                                |
| LFM3-1.7B-Q4_K_M.gguf                                       | `3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4`                                                                                                                |
| gpt-oss-20b-MXFP4.gguf                                      | `65d06d31a3977d553cb3af137b5c26b5f1e9297a6aaa29ae7caa98788cde53ab`                                                                                                                |
| birefnet.safetensors                                        | `9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154`                                                                                                                |
| 2x-AnimeSharpV4_Fast_RCAN_PU.safetensors                    | `b641c9eb10b43f26538177aa8f0fef8b9fc2a153afd1431d0a062a84c49ce6d0`                                                                                                                |
| depth_anything_3_small.safetensors                          | `9c0a53d157c5b315d4c82908a43d93eeaa3a9ac3de9d15c2b4242ef46b5508ad`                                                                                                                |
| rt_detr_v4-x-hgnet_fp32.safetensors                         | `1ba4a7794c2f8b176327c887ba38fbe0f977049cbef149960288aa923d0239f3` ([Comfy-Org revision](https://huggingface.co/Comfy-Org/RT-DETR/tree/83a30ba5a7843e3947622fcca04085cea84e2932)) |
| florence-community/Florence-2-base                          | Hub revision `00921df66db728a9ceb750f5eca43e5c203a2051`                                                                                                                           |

This is a finite execution qualification, not all-feature or extended stability
certification. SAM point segmentation lacked a local supported SAM3/SAM31 asset;
no large/gated segmentation weights were acquired. Full Canny/depth/recolour/
sketch ControlLoRA generation, custom ControlNet/T2I adapters, reference adapters,
LoRA/embeddings, inpainting and cropped-face restoration remain unqualified;
the corresponding conditioning/adapter assets or a suitable face fixture were
absent. A depth-map pass does not qualify depth-conditioned diffusion. Native
instruction editing is not a Janku capability. No video/training/driver work was
included. GPT's existing partial was recovered with a verified range helper;
ordinary interrupted-download resume with the installed Hub SDK remains a known
unfixed limitation, distinct from successful model execution.

Set `TORCH_PLATFORM` (for example `cu128`, `rocm10.1`, `cpu`) to override detection, or choose the image, Chat (`llama.cpp`) and background-removal backends in **Settings → Hardware & backends** (see [Settings](settings.md)). Chat models run on the
native llama.cpp server. Auto tries cached compatible vendor GPU builds first,
then Vulkan independently of the image runtime. Select explicit ROCm,
CUDA, SYCL or CPU under **Settings → Hardware & backends** where supported.
When no compatible GPU is available, Auto logs the reasons and uses CPU. If a
runtime fails during execution, the error stays visible for diagnosis.
The alternative XLlamacpp runtime is installed only when selected; choose its wheel (CPU, Vulkan, CUDA 12.8 /
13.2, or ROCm 7.2 on Linux/WSL) under **Settings → Hardware & backends → Chat runtime xllamacpp**. No xllamacpp
ROCm 10 wheel exists, and the ROCm 7.2 wheel crashes in the same process as ROCm 10 PyTorch, so it is offered only
without ROCm 10 PyTorch; with ROCm 10 use native llama.cpp, whose official ROCm 10.0 build runs on Windows, Linux and
WSL. On WSL the Vulkan builds need Mesa's Dozen (`dzn`) Vulkan driver to see the GPU. Automatic
prefers Vulkan on Windows/Linux independently of the installed PyTorch build.
Offline xllamacpp loads require a matching version, backend source, interpreter
identity and intact recorded payload. Explicit CPU may reuse a tracked GPU wheel
with every applicable offload disabled. Native llama.cpp binaries and SDKs use a
separate cache and do not replace PyTorch. New native caches record publisher
archive digests and extracted-file hashes. A legacy cache can be adopted against
an independently verified offline manifest with `adopt_verified_cache`; an
untracked binary remains usable by explicitly setting `LLAMA_SERVER` to its
path, without claiming verified publisher provenance or deleting the old cache.
An initialized xllamacpp device query runs before model-server creation; a query
failure stays visible, while an Auto result with no GPU explicitly disables all
offload. Offline Auto can retain a verified cached CUDA/ROCm wheel when its real
GPU is usable. This device check does not qualify a model workload.
To switch PyTorch later, choose the image backend and click **Install / switch on next start** in **Settings → Hardware & backends**, or start with `TORCH_PLATFORM` set or with `--rocm10`, `--cuda-nightly`, `--cpu` or
`--directml` (deprecated). WSL host adapters appear as inventory information;
their presence does not qualify a Linux guest runtime. Use a guest-specific
recipe, matching host driver and explicit guest visibility evidence. Earlier
RX 7900 XTX WSL execution evidence does not certify other devices or bundles.
After installing XPU it checks that PyTorch sees the Intel GPU and
explains which driver is missing if not. Existing explicit compatible environments remain usable. New installs use native CUDA/qualified ROCm or CPU instead of silently downgrading to Torch 2.4. ONNX DirectML is also deprecated and available only by explicit selection; Auto never chooses it. Neither path constrains the modern runtime bundles.
Create an empty `reinstalltorch` file in the RuinedFooocus folder to reinstall it on the next start
(or `reinstall` to reinstall all packages),
or an empty `freezetorch` file to stop the launcher from changing PyTorch (for example after installing it
yourself). Deleting the `venv` folder gives a completely fresh install. The launchers replace the
old portable setup helpers and Conda environment file; reinstall/freeze marker files are created locally.

A source ZIP uses the same launchers and still requires Python and Git; it does not bundle a runtime or models.
Set `RF_DATA_DIR` to keep settings, model paths, outputs and runtime caches in a separate writable folder.
Reinstall/freeze markers can go there too; existing markers in the application folder remain supported.
Git updates run only in a source checkout and skip local edits; upgrade a source ZIP by replacing application
files while keeping user data. Bundled Windows/Linux CUDA/ROCm portable releases are a future build target.

Use `--skip-dependency-install` to launch with packages you manage yourself. Source updates remain enabled;
use `--offline` to also disable startup updates and downloads. Explicit reinstall/repair requests remain
pending while dependency installation is disabled. The old package-skip spelling remains a hidden alias.

If you see **"MetadataIncompleteBuffer"**, a model file is incomplete or damaged; download it again. **Create → Models → Model → Checkpoint tools** can check a checkpoint for damage.

### Optional image engine controls

**Settings → Hardware & backends → Optional image engine** saves Automatic,
Native ComfyUI, optional stable-diffusion.cpp Vulkan, or its separate CPU build.
Automatic retains a compatible native GPU first, then considers qualified
optional Vulkan before automatic CPU fallback. Explicit global or image CPU
choices prevent Vulkan execution. Unsupported optional workloads remain on the
native route in Automatic; an explicit optional request reports its refusal.

**Install optional runtime** prepares the pinned payload in the bounded I/O
queue. **Verify optional runtime device** is a separate action admitted through
the shared compute worker. The exact `VulkanN` backend name is not a physical
adapter identity. Installation, enumeration and CPU help checks do not grant
model/operation acceptance. No acceptance receipt is created by these controls.
An offline Install can reuse an existing immutable bundle but cannot acquire a
missing one. **Cancel runtime action** cancels this browser's action. Status
refresh reads cached/process facts without loading, hashing or probing a device.
See the [optional engine contract](integrations/sdcpp.md) for supported workloads,
platform limits and remaining physical-device qualification.

## Runtime behavior notes

### Combined package planning

`packaging/package_plan.py` exposes immutable `PackageLayer` and `PackagePlan`
records. Bootstrap reads requirements from the exact pinned ComfyUI Git revision
and combines them with the reviewed Torch recipe and RF/API overlay. The
[overlay classification](../pip/modules.layers.json) identifies shared bounds,
RF additions and legacy exceptions; it does not replace upstream requirements.
AMD's source-only Python SDK selector is built from pinned source before the
wheel-only resolution. Its provenance and the resulting wheel are recorded.
The resolver report supplies exact URLs, SHA256 values, wheel tags,
Requires-Python and transitive dependencies. `plan_packages` applies the shared
capability contract and rejects combined conflicts, wrong vendor sources, ABI
or host/guest mismatches and unqualified legacy core API exceptions.

AMD's `rocm` selector is built from a SHA256-pinned source archive, retained by
content hash and included in the combined package plan. Native Torch and SDK
packages still use publisher wheels. AMD's `rocm/pytorch/whl-next` download
redirect is accepted alongside its `rocm/whl-next` index. Each detected GFX
target selects its own Torch and torchvision device extras.

`execute_plan` consumes that digest and interpreter/environment identity under
a cross-process lock. It downloads the entire hash-locked closure before any
main-environment mutation and installs those local files without consulting a
mutable index. Native imports are checked before acquisition and again before
application. A staging failure leaves the environment intact; pip installation
itself is in-place and cannot promise rollback after a partial failure. Restart
and repair are then required. This is not portable-environment qualification.
Bootstrap tooling needed to run this resolver is a separate prerequisite phase.
An absent AMD detection helper uses its own hash-locked prerequisite environment;
it does not install AMD-only tooling into every RF environment. Offline startup
cannot acquire that helper. Unchanged installed Torch companions and wheel-owned
vendor SDK packages are pinned while resolving RF additions; this preservation
is package acquisition, not a new GPU execution qualification.

PyTorch package plans accept the official index's `download-r2.pytorch.org`
artifact host for the same selected `/whl/` channel. Other hosts and channels
remain rejected; every wheel still requires its recorded SHA256 identity.

Optional native ROCm library reinstalls publish a new environment and active
pointer only after dependency and required-library checks. They retain the prior
SDK directory, so a running native server's DLLs are not overwritten. A failed
stage leaves the previous pointer intact.

Windows acquisition evidence is read from CIM display-driver and OS metadata;
no compute runs in that inspection. The audited ROCm 10.1 Radeon/Ryzen route uses
Windows 11 25H2 and its listed driver, mapped to driver-store 32.0.31041.5005 by
[AMD's release notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-ROCM-10-01.html).
See the [ROCm matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).
Newer drivers on the same 32.x driver-store branch are admitted for installation,
with native runtime validation still required afterward. Other major branches
remain unknown. WSL guest visibility can use an existing `rocminfo` when OpenCL's
`clinfo` is absent. The XPU 2.14 route checks its
listed Arc/Core Ultra families, Windows 11 and driver 32.0.101.8801 or later from
[Intel's versioned prerequisites](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-14.html).
These sources were checked 2026-11-09. Linux/WSL require separate released
OS/driver and guest-visibility evidence; a Windows host record is insufficient.
Native Linux metadata uses physical KFD/DRM PCI identities, the released
distro/kernel rows and loaded-versus-installed module source identity. AMD's
vendor-driver branches and documented APU inbox-driver route remain distinct;
Intel additionally needs its matching userspace compute runtime. Unknown rows
remain unknown rather than borrowing a neighboring release. In WSL, `clinfo`
is a separate explicit acquisition-time device query, never a passive Settings
refresh; ambiguous identical adapters remain unbound. ROCm 10.1 WSL is a
[technical preview](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
requiring explicit selection. Source/CPU fixtures do not certify these platforms.

Before guest ROCm exists, WSL can obtain GFX identities from the Windows driver's
`clinfo.exe`. Host architecture and `/dev/dxg` permit acquiring compatible wheels;
guest execution must then pass validation with the installed runtime. Debian 13
WSL is an application preview available through explicit `--rocm10` selection.
Native Linux retains its separate OS, kernel and driver requirements. On
apt-based systems, the shell launcher can install missing Python, venv and
`libgomp1` prerequisites with administrator access. Offline launches require
these packages to be present already.

### Shared capability decisions

`packaging/runtime_support.py` provides pure `RuntimeTarget`, `RuntimeArtifact`
and `RuntimeCandidate` records plus `capability_decision` and `select_capability`.
Backend family/engine, exact platform and versioned recipe identity remain
separate: for example, saved `cuda` can resolve to platform `cu126` and a pinned
legacy recipe. Catalogues retain the family while installation receives the
exact platform. An abstract CUDA/ROCm family alone cannot select a wheel route.
Callers supply the actual interpreter ABI/wheel tags, native or WSL environment,
versioned recipe and exact artifact metadata, dependency closure, adapter/driver
support and current runtime evidence. Planning performs no imports of Torch,
device enumeration, package installation or model loading. Active helpers such
as `inspect_installed_cuda` are separate explicit validation operations.

A decision distinguishes unknown or unavailable support, an installable route,
installed but unverified evidence, usable runtime, workload qualification and
active execution. Wheel availability, publisher-validated Python versions and
native plugin payload availability are separate facts. Missing transitive
artifacts, incompatible wheel tags and conflicting pins block eligibility;
dependency markers referencing missing target metadata remain unresolved until
that metadata is supplied, rather than borrowing host values or skipping pins;
unverified evidence never becomes usable. Native executable recipes do not
inherit Python wheel restrictions. MIGraphX plugin, legacy MIGraphX, Windows ML,
experimental HIP and WSL recipes must retain separate platform/dependency
contracts; consult the [ONNX integration guide](integrations/onnx-providers.md).

Passive hardware inventory preserves multiple identical cards and separates
host from guest records. Product PCI IDs are not physical runtime identities.
An ordinal fallback is explicitly unstable and requires runtime rebinding;
stable bus or Windows instance identities are retained when available. Host
presence does not prove WSL guest availability. `gpu_inventory` also distinguishes
an unreadable scan from an empty inventory. Passive reads do not run
`nvidia-smi`, load a model or perform kernel checks. GPU and NPU records cannot
substitute for one another; core `torch_npu` refers to Huawei Ascend.

Auto selection evaluates every qualified GPU route before CPU, and a requested
GPU failure stays visible. Image-only CPU fallback never disables independent
Vulkan Chat or ONNX. `cpu_selection_scope` records origin and scope before
bootstrap mutates `args.cpu`: explicit `--cpu` and the saved global switch apply
globally; saved image CPU and `TORCH_PLATFORM=cpu` apply to the image runtime.
CLI/environment choices retain precedence over saved preferences. No selection
rewrites saved model, quantization or unavailable backend choices.

`support_matrix.support_rows(decisions=...)` renders these exact decisions and
reasons. Older vendor-only callers display **Candidate** or **Installed
(unverified)** rather than claiming execution support. Capability-aware choice
catalogues accept `decisions=`. Installer/bootstrap, model-specific qualification
and Settings composition must supply their own exact recipe/evidence inputs;
the pure API alone does not certify those consumers or physical hardware.

Focused mocked checks:

```sh
python tools/check_cpu.py test_hardware_inventory.py test_runtime_support.py test_backend_preferences.py test_gpu_support_matrix.py
```

These cover ABI/dependency failures, stale evidence, mixed and duplicate adapters,
host/guest boundaries, CPU scope and GPU/NPU separation. Linux RDNA3.5 APU memory
and V620 SR-IOV applicability fixtures reject RX 7900 XTX/gfx1100; they do not
apply system tuning. No device execution or real-model qualification is implied.

- Automatic CUDA allocation follows the pinned ComfyUI policy; explicit allocator environment settings and CLI choices take precedence. WSL ROCm retains its expandable-allocation safeguard.
- AMD wide-head attention retains the gfx11 math-kernel safeguard used by the native backend; other recognized AMD architectures keep native SDPA selection. The installed wheel SDK supplies PyTorch libraries independently of an older system HIP SDK; optional Chat and ONNX runtimes keep their own ABI requirements.
- Background removal retains the native rembg model and output dimensions. The qualified rembg 2.0.87 Bria session binds its fixed 1024-pixel preprocessing dimensions before ONNX initialization; other models/releases keep native shape handling. This is not WinML GPU qualification.
- Ordinary background cutouts use native straight alpha to avoid darkening soft edges twice. Explicit cutout, matting and foreground-decontamination options keep their native behavior; existing transparency is retained unless removal is forced.
- Image-to-image preserves source proportions by default. Explicit API output sizes use uniform fitting and neutral padding; cropping requires `image_resize_mode: "crop"`. Create's Image framing control also offers **Fit selected size (pad)**. Inpaint masks follow the same transform, with unpainted padding.
- Native image edits can repaint fine details and lighting. For small refinements, choose **Image tools → Image to image** and a low denoise value. Exact object counts and lettering remain model-dependent; matching a prompt in metadata does not establish visual accuracy. Standalone API upscaling retains uploaded PNG text, EXIF and ICC metadata, using ComfyUI's native alpha handling.
- **Settings → Image runtime → Image compute → Advanced native compute** exposes precision, attention/upcasting and native compiler/graph opt-outs. Auto follows ComfyUI; CLI choices take priority. Save and restart to apply. CPU ignores saved FP16/BF16 precision overrides.
- **Settings → Hardware & backends → Background removal (ONNX Runtime)** keeps vendor providers in separate environments, preserving the app's CPU runtime, original model and native rembg preprocessing. Status distinguishes initialization with assignment unverified from first-inference executed EP node events and CPU partitions. ORT constructor/run retries are disabled; explicit RF CPU recovery records whether initialization or inference failed and its reason. Existing integrated MIGraphX and saved DirectML choices remain available. The new native-Linux MIGraphX plugin route requires exact CPython 3.12, GFX, driver, wheel hashes, `migraphx-libs` and complete dependency closure; it stays outside Auto pending real Bria qualification. WSL preview and incompatible HIP/Python/GFX routes retain separate blockers.
- **Settings → Hardware & backends → Background removal** distinguishes requested choices from the last shared runtime status. Windows ML choices come from the last explicit local catalogue inspection: only certified, bridge-compatible candidates are offered, while unavailable saved intent stays labelled. **Refresh local Windows ML catalogue** reads the installed isolated publisher bridge without acquisition; it does not call provider readiness or device APIs. Install catalogue/provider support explicitly if that bridge is missing. Provider acquisition and device refresh use shared worker admission and browser-scoped cancellation. Global/CLI or background CPU authority prevents accelerated install/discovery; Automatic device actions never infer an NPU from an AMD/Intel vendor label. Each acquired provider record remains individually validated. AMDGPU acquisition is blocked because its required Windows ML 2.5.96 Python/App SDK/ORT closure is unavailable; the audited publisher projection remains 2.3 (2026-11-09). New Windows ML routes and compiled-model cache reuse remain outside Auto pending real-model qualification. See the [ONNX integration contract](integrations/onnx-providers.md); catalogue availability, device discovery and canaries do not qualify hardware quality, speed or stability.

On the Windows RX 7900 XTX, real Bria removal with explicit DirectML was verified
alongside a CPU image backend, preserving dimensions, RGB and ICC data. Windows ML
MIGraphX acquisition and device discovery succeeded, but Bria initialization timed
out after 180 seconds and recovered on CPU; its GPU inference remains unqualified.
WSL native ROCm Chat with LFM3 also passed. NVIDIA, Intel and native-Linux
hardware require separate qualification.
