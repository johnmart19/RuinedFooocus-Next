# RuinedFooocus

Local image generation, editing and optional Chat, using a pinned ComfyUI backend
and a shared local llama.cpp runtime. The main destinations are **Create**, **Chat**,
**Library** and **Settings**. Create keeps the image above its prompt and the model
controls beside Generation.

## Start

Install Git and a supported Python interpreter. Python 3.11–3.14 is supported when
the selected runtime provides matching wheels; Python 3.13 is a suitable starting
point. Existing compatible Python 3.12 environments can stay in place. Python 3.10
is a deprecated compatibility path.

```sh
git clone https://github.com/johnmart19/RuinedFooocus-ROCm.git RuinedFooocus
cd RuinedFooocus
```

On Windows, run `RuinedFooocus.bat`. On Linux/WSL, run `./RuinedFooocus.sh`.
Both launchers print a status immediately, then use explicit `PYTHON`, an active
virtual environment or the checkout's `venv`, in that order. They create `venv`
when no environment has been selected. Git update checks have a 10-second deadline;
see [startup behavior and measured limits](docs/integrations/startup.md).
On apt-based systems, the shell launcher can install missing Python, venv and
OpenMP runtime packages with administrator access. Offline launches require those prerequisites
to be present already. An explicit `PYTHON` selection is kept.
The launchers install the selected platform's dependencies when installation is
enabled. The first installation can download several GB.

In **Create → Models**, select an installed checkpoint, download the optional
default image model, or inspect its required components. Downloading is explicit.
Choose a model, enter a prompt below the image and select **Generate**. Images and
their metadata appear in **Library**. LoRAs and checkpoint tools are beside the
model browser; image editing and prompt tools stay inside Create.

Chat and optional prompt enhancement share the selected local model. The default
LFM3 1.7B Q4_K_M GGUF downloads on first use; advanced or custom local models can be
selected instead. Ordinary random prompts and wildcards do not require an LLM.

## Guides

- [Documentation index](docs/README.md)
- [Workflows and UI controls](docs/user-guide.md)
- [Installation, runtimes and troubleshooting](docs/runtime-and-troubleshooting.md)
- [Settings, profiles and launch options](docs/settings.md)
- [Prompt datasets and custom wildcards](docs/prompt-data.md)
- [UI translations](docs/localization.md)
- [Development and validation](docs/development.md)
- [Code and backend ownership](docs/architecture.md)
- [Legacy migrations](docs/legacy-migration.md)
- [Update log](update_log.md)

## Support limits

Native CUDA, qualified ROCm and CPU are the primary image runtimes. Legacy NVIDIA
has a separate compatibility profile. ONNX Windows ML tools have their own
provider/device qualification and do not replace ComfyUI's image backend. See
the runtime guide for exact compatibility and qualification limits, including
the pinned Gradio/Pillow constraint.

A source ZIP uses the same launchers and still needs Python and Git. Certified
bundled Windows/Linux CUDA/ROCm portable releases remain a future build target.
Use `RF_DATA_DIR` for a separate writable data directory. Upgrades must preserve
weights, outputs, settings and custom presets. Offline/package-managed startup
is documented in the runtime guide.

## Credits and support

Built on [Fooocus](https://github.com/lllyasviel/Fooocus),
[RuinedFooocus](https://github.com/runew0lf/RuinedFooocus),
[ComfyUI](https://github.com/comfyanonymous/ComfyUI),
[AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui),
[MoonRide](https://github.com/MoonRide303/) and
[One Button Prompt](https://github.com/AIrjen/OneButtonPrompt).
Join the [Discord support server](https://discord.gg/CvpAFya9Rr).
