# Imported request intent and utility isolation

The submission capture and worker execution prefix resolve imported model/control
intent before sampling defaults. `generation.request_metadata.metadata_intent`
uses the existing bounded metadata fields and control replay reader, then restores
the imported checkpoint identity. The effective operation decides whether any
generation model, prompt preparation or sampling defaults are needed.

For image generation, named defaults use that checkpoint. The normal metadata
restorer runs afterward, so recorded steps, CFG and other explicit values win.
An already captured sampling request is not defaulted again. A utility capture
retargeted to sampling still requires an explicit fresh submission.

For removal/upscale utilities, `restore_utility_metadata` replays inert saved
fields without model-family/guidance lookup or prompt-model preparation. Native
detection/segmentation/depth requests retain their separate sanitized worker data
and ignore source generation metadata. API and tool-call JSON prompts remain
literal. No model download, tensor read, user-profile change or image generation
is performed by these helpers.

Explicit utility controls take precedence over free `#!`, `search:` and logo
commands; explicit Chat/API/tool routing retains its existing authority. The
worker never hashes the selected diffusion checkpoint for a utility. It retains
the utility initializer with its own selected background/upscaler identifier,
then uses the supplied image dimensions, one utility step and empty generation
conditioning. The utility's own prepared model loading remains in its pipeline.
Hidden generation steps, aspect ratio, styles, prompts and LoRAs cannot redirect
that operation.

The worker's earlier ordering could apply the currently selected checkpoint's
defaults before reading a different checkpoint from imported PNG/JSON metadata.
It could also resolve a generation checkpoint while replaying utility metadata.
The caller suite covers both capture and execution ordering, metadata override
precedence, utility model/prompt independence, literal API/tool prompts and strict
utility-to-sampling recapture.

```sh
python tools/check_cpu.py test_utility_sampling_capture.py test_image_preparation.py test_chat_boundaries.py test_resource_lifecycle.py
```
