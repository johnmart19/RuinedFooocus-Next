# WSL model execution

Qualification on 2026-11-10 used an RX 7900 XTX, Ubuntu/Python 3.14.4 and
Debian/Python 3.13.5. Application runtime source: `d85edd5`; the fixture-only
`62d44ac` does not change those runtime results. ComfyUI was pinned to
`fca60fe70e5b84c5c6b058263653d15cb3401e44`, with PyTorch
`2.14.0+rocm10.1.0`, Transformers `5.22.2`, rembg `2.0.87` and CPU ORT `1.31.1`.
These results certify the listed cases, not every catalogue model or GPU.

| Workload                 | Models                                                                                                               | Result on both distributions                                                                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Text Chat                | LFM3 1.7B Q4_K_M, GPT-OSS 20B MXFP4                                                                                  | Real WebUI replies with native llama.cpp ROCm; prior source-matched acceptance retained                                                             |
| Visual Chat              | SmolVLM 256M Q8_0 and matching projector                                                                             | Real visual replies with native llama.cpp ROCm; Ubuntu also checked through the WebUI                                                               |
| Context retrieval        | Pinned all-MiniLM-L6-v2                                                                                              | Real GPU indexing and semantic retrieval through the app's txtai options                                                                            |
| Captioning               | Florence-2-base, BrainBLIP, CLIP ViT-L-14/openai                                                                     | Real ROCm captions through shipped adapters; output semantics remain model-dependent                                                                |
| Upscaling                | Twelve installed RCAN, AniScale, Futsuu, DAT, ESRGAN, HAT, OmniSR, RealESRGAN, SRFormer, SwiftSR and SwinIR variants | Actual native GPU inference; correct 2×/4×/8× dimensions, finite pixels and alpha retained                                                          |
| Depth                    | Depth Anything 3 small/base                                                                                          | Actual native GPU inference; source dimensions retained                                                                                             |
| Detection / segmentation | RT-DETR v4, SAM 3.1 multiplex                                                                                        | Actual GPU detection and point cutouts; Debian also checked through Image tools                                                                     |
| Background removal       | Original Bria RMBG 2.0 ONNX                                                                                          | Actual CPU inference with the GPU-unavailable warning; dimensions and alpha retained, transparent inputs unchanged unless reprocessing is requested |
| Image generation         | Janku v7.77, Janima 2.9B                                                                                             | Prior real automatic-settings WebUI generations on both distributions; Janima acquired its required encoder and VAE                                 |

The utility checks used unchanged source artwork and separate derived inputs.
Upscaler comparisons, depth maps, detections and cutouts were visually reviewed.
Captioning and small visual models can omit details or describe colors incorrectly;
successful transport and GPU execution do not establish universal semantic accuracy.
These cases do not certify every ControlNet, LoRA, editing workflow or absent model.

## Provider limits

- Native llama.cpp ROCm is the qualified accelerated Chat route here. WSL Vulkan
  has no usable GPU ICD on this host. The old xllamacpp ROCm wheel cannot safely
  share the image runtime's newer HIP stack. Keep the reported fallback reason.
- Windows ML and DirectML are Windows runtimes, not WSL providers.
- An isolated Debian wheel probe resolved ORT `1.29.0`, MIGraphX
  `2.18.0+rocm10.1.0` and its EP `1.0.0+rocm10.1.0`. Registration exposed no AMD
  device; the shipped registration adapter rejected it. No Bria GPU run was
  possible. Keep the WSL acquisition gate and CPU recovery; do not replace the
  application's CPU ORT or enable automatic MIGraphX on this evidence.
- [AMD's ONNX installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/onnxruntime.html)
  describes native Linux requirements; wheel resolution alone does not qualify WSL.
  NVIDIA, Intel, NPU and native-Linux hardware results are outside this acceptance.

## Focused validation

Run with each checkout's own interpreter:

```sh
venv/bin/python tools/check_cpu.py test_onnx_runtime.py test_caption_runtime.py test_native_vision.py test_image_depth.py test_upscale_transparency.py test_chat_vision_modes.py
```

All 46 checks passed on Windows and both WSL distributions. The independently
replayed fixture feature passed 17 checks with its declared prerequisites.
The CPU checks cover contracts and recovery; they do not substitute for the real
model runs above. Keep models on the guest Linux filesystem for execution, and
retain user settings separately from disposable validation profiles.
