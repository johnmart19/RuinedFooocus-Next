# Image and optional Chat dependencies

RF installs Torch and TorchVision for image execution. TorchAudio is no longer
part of RF's wheel selection, runtime admission or startup requirement. The
pinned ComfyUI `989d93d0` requirements and image callers do not require it. Its
audio DSP uses native code without importing TorchAudio. Existing user packages
are not uninstalled by this change.

The bootstrap caller declares the same Torch/TorchVision closure to runtime
capability admission. Its caller regression resolves a mocked fresh image plan
through the real capability policy and retains package, Python ABI and machine
compatibility rejection without running installers or device probes:

```sh
python tools/check_cpu.py test_bootstrap_package_contract.py test_package_plan.py test_runtime_support.py
```

The RF overlay no longer requests `imageio-ffmpeg`: RF has no remaining caller,
and ImageIO declares it only for unselected FFmpeg extras. ImageIO stays because
the installed scikit-image image-processing closure requires it. PyAV stays both
as a pinned ComfyUI requirement and for RF's AVIF/EXR still-image readers. Removing
audio/video features does not remove still-image codecs.

Package plans still preserve Torch/TorchVision ABI, selected device wheels,
vendor runtime dependencies and interpreter eligibility. Optional Chat runtimes
keep their independent environments. No package installation, removal, device
probe or inference is needed to check these plan contracts:

```sh
python tools/check_cpu.py test_installer_gpu.py test_installer_rocm.py test_runtime_bundles.py test_package_plan.py test_dependency_requirements.py test_download_link_audit.py
```

The overlay classification also includes the existing `pypdf` text attachment
parser; it remains part of optional local Chat's document ingestion closure.
The compact Chat attachment projection checks the declared `pypdf` version from
`document_parsers.json` against installed package metadata without launching the
parser child. A missing or mismatched package removes PDF from advertised picker
formats; the parser still rejects unsupported PDF extraction safely. DOCX uses
the existing bounded standard-library XML/ZIP parser and needs no new package.

## Current package branches

The November 13, 2026 requirement audit uses published PyPI release metadata and
the exact pinned ComfyUI requirements, rather than replacing the native base.
Python 3.11 and later select Diffusers 0.43, Transformers through 5.23, ONNX Runtime
1.32, FastAPI 0.162.7 and filelock 4.2.10. Python 3.10 retains Diffusers 0.41,
Transformers through 5.20.2 and ONNX Runtime 1.23.2, and selects FastAPI 0.162.5,
python-multipart 0.0.32 and filelock 4.1. The later filelock releases require
Python 3.11; an unconditional 4.2 pin cannot resolve a Python 3.10 environment.

ONNX Runtime 1.24.3 still advertises Python 3.10 in metadata but publishes no
compatible Windows CPython 3.10 wheel. The overlay retains the last such wheel
instead of interpreting the metadata floor as artifact availability. DirectML's
Torch 2.4 producer retains Transformers 5.13 and its compatible tokenizers; the
normal overlay's newer upper bounds do not replace that explicit recipe.

Gradio 6.32 permits Pillow 13, whose Python floor is 3.11; Python 3.10 retains
Pillow 12.3. Gradio's MCP extra still caps Pydantic at 2.12.5, so newer Pydantic
2.14 releases are outside this declared closure. The PDF requirement and parser
manifest move together to pypdf 7.1, with its existing bounded extraction API and
without selecting PDF image, OCR or cryptography extras.

Dependency-only updates preserve exact installed Torch/TorchVision and AMD
SDK/device wheel versions while resolving the combined native, application,
bootstrap and API layers. Package metadata and CPU checks do not establish GPU
or model-output qualification.
