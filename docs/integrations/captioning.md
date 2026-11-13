# Image captioning

Image-upload captioning is owned by
`ruinedfooocus/features/image_processing/interrogate.py`. Florence and BrainBLIP
use Transformers; CLIP interrogation uses clip-interrogator and its low-VRAM
configuration. Calls enter the shared image compute lane so model placement and
generation do not race. `caption_runtime.py` reserves native device memory,
releases models to CPU afterward and retries GPU out-of-memory failures on CPU.
DirectML captioning uses CPU because operator coverage is not qualified.

CLIP's precomputed artist, flavor and other text-vector tables belong in
`cache/clip-interrogator`, under the configured app cache root. They are not text
encoder weights and must not be written into `models/text_encoders`. Actual model
weights continue to use their native Hugging Face cache. Existing files in an old
model folder are not automatically moved or deleted by a source update.

Run `python tools/check_cpu.py test_caption_runtime.py` with the checkout's
interpreter for device routing, cache location, release and recovery checks.
Real Florence, BrainBLIP and CLIP caption inference was also verified on Ubuntu
Python 3.14 and Debian Python 3.13 with ROCm 10.1 on RX 7900 XTX. Captions remain
model predictions and can omit details; runtime success does not certify accuracy.
