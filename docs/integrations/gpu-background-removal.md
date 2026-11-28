# GPU background removal

Create → Image tools → RemBG offers native BiRefNet by default when no background
model was saved. Existing explicit local models and the empty Hardware/ONNX
selection are preserved. BiRefNet follows ComfyUI's image device, including CUDA,
ROCm and explicit CPU mode. CUDA execution is not hardware-qualified by the AMD
results below.

The 444 MB Comfy-Org weight is pinned by revision, size and SHA-256 in
`resources/model_catalogue/background_removal.json`. The image preparation lane
downloads it on first use, before compute. Offline mode refuses a missing file;
startup and model browsing never download it. Local alternatives still require
the native architecture contract. No encoder, VAE or diffusion checkpoint is needed.

The model registry annotates that same `birefnet.safetensors` as an Image tools
resource. The installed file matched the pinned publisher's 444,473,596 bytes
and SHA-256. The header records FP16 floating tensors, and the publisher declares
MIT for the repack. The adjacent Lucida file is a separate artifact.
[Pinned publisher card](https://huggingface.co/Comfy-Org/BiRefNet/blob/884af709e9b795d89dafbf97f29c8d6a2386f5c6/README.md)

`native_background.py` reuses pinned ComfyUI's loader, model patcher and
`RemoveBackground` node. It retains source dimensions, RGB and alpha semantics.
Transparent inputs stay unchanged unless reprocessing is selected. Metadata
records the model and initialized device without exposing its local folder.
An out-of-memory failure releases the native model and warns before CPU ONNX
recovery. Invalid weights and other inference errors are not hidden by that retry.
Recovery sessions retain a CPU-specific cache identity, so a later ONNX Auto
request re-evaluates its provider instead of silently reusing the CPU session.

## WSL MIGraphX discovery

On this RX 7900 XTX host, HIP reports one AMD GPU, while WSL's PCI display entry
has Microsoft's vendor ID `0x1414`. The registered ORT 1.29 / MIGraphX 2.18 ROCm
10.1 plugin exposes only CPU. The upstream
[MIGraphX factory](https://github.com/onnxruntime/onnxruntime-ep-amdgpu/blob/58afd43bc4c2ddc7582712e1d75cbfd4c3bb03fc/src/migraphx/mgx_factory.cc)
counts HIP devices but then requires an AMD GPU in ORT's separate platform
inventory. This explains why healthy HIP execution does not produce an EP device.
ORT's [CUDA discovery fix](https://github.com/microsoft/onnxruntime/pull/32517)
handles the analogous WSL gap using runtime-discovered descriptors; the audited
MIGraphX factory has no equivalent fallback.

Do not spoof PCI/vendor IDs, remove the WSL acquisition gate or count CPU Bria
as MIGraphX GPU qualification. Native BiRefNet supplies the working GPU route
without replacing the main CPU ONNX environment or rebuilding a vendor provider.
The original Bria ONNX and explicit provider settings remain available.

## Acceptance

On 2026-11-10, the real app with RX 7900 XTX / ROCm 10.1 completed matched
1024×1024 removal in 2.11 seconds on Debian Python 3.13 and 1.65 seconds on Ubuntu
Python 3.14. Native repeated inference measured 0.71 seconds warm. The earlier
CPU Bria route took about 23 seconds; these are different models, not a claim of
numerical equivalence. A real 928×1664 portrait also retained exact dimensions
and RGB. Outputs were visually reviewed, alpha covered 0–255 and metadata
reported `cuda:0`. This is bounded acceptance, not universal mask-quality proof.

Windows Python 3.12 on the same GPU also completed removal in the actual app
with the default BiRefNet selection at 1024×1024 and 928×1664. Both preserved
dimensions and source RGB, with alpha spanning 0–255 and initialized device
`cuda:0`. An already-transparent input passed through with identical RGBA pixels.
Explicit CPU-only native BiRefNet completed in 125 seconds under a GPU-initialization
guard. A separately injected native OOM exercised real Bria CPU recovery in
14.4 seconds, emitted the fallback warning and cleared stale GPU provenance.
That is failure-path acceptance, not a naturally occurring OOM or stability test.

Focused checks:

```sh
python tools/check_cpu.py test_native_background.py test_image_preparation.py test_image_rembg_options.py test_onnx_runtime.py test_api_image_edits.py test_localization.py
python tools/check_cpu.py test_native_background.py test_onnx_runtime.py test_rembg_runtime.py
```

These cover saved choices, missing offline assets, captured downloads, source
identity, transparency, cancellation, OOM recovery and invalid-weight errors.
The maintained [native ComfyUI workflow](https://docs.comfy.org/tutorials/utility/remove-background-birefnet)
is the reference for model compatibility.
