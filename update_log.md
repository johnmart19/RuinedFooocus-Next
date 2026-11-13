### Unreleased (development)

- Use six desktop sidebar destinations: Create, Editor, Gallery, Models, Chat and Settings. Keep the shared image draft, group optional upscaling and advanced Create controls, and separate Installed models from Download.
- Preview available themes immediately and Save the profile choice for later launches; keep Settings profile actions at the page end with a desktop bottom bar, expose the host folder dialog beside in-page folder browsing, and use derived Gallery thumbnails while retaining originals for image actions.
- Bound the derived Gallery thumbnail cache, add its Maintenance size/clear controls, and keep active copies safe from cleanup. Preserve image orientation and original files.
- Keep Gallery image actions tied to the latest page and selection when requests overlap.

- Keep image-only CPU selection separate from global CPU through bootstrap, avoid reloading Chat after sharing the selected model, refresh backend install states automatically, and show certified Windows ML alternatives on failure. Real DirectML/Bria and WSL ROCm Chat checks passed; Windows ML MIGraphX/Bria still times out and recovers on CPU.
- Show a **Supported on this PC** table in Hardware & backends and choose backends for ComfyUI, llama.cpp, xllamacpp, ONNX Runtime and Windows ML, filtered to the detected GPU and OS; install vendor ONNX Runtime builds (CUDA, OpenVINO, MIGraphX) and the selected xllamacpp wheel from Settings; detect AMD and Intel GPUs on WSL through the Windows host; keep an installed Intel XPU build instead of reinstalling it on every start; fix saving the XPU image backend.
- WSL with AMD: keep image generation (ROCm 10.1) and llama.cpp (ROCm 10.0) on the GPU, enable DXG detection for older ROCm runtimes, and stop offering the xllamacpp ROCm 7.2 wheel beside ROCm 10 PyTorch, where it crashes.
- Support Intel Arc and Core Ultra GPUs: pinned PyTorch XPU wheels selectable in Hardware & backends, recognised XPU installs, the official SYCL llama.cpp build for Chat, and no startup failure on laptops with an AMD GPU plus a Core Ultra iGPU.
- Translate every tab name, card list and status line when the UI language changes, including Settings tabs opened later.
- Add a **Settings → Hardware & backends** tab showing detected GPUs and active runtimes, with hardware-filtered backend choices for image generation, Chat (`llama.cpp`) and background removal, a reversible **Use CPU for everything** switch, and explicit runtime install/switch actions that respect CLI and environment overrides.
- Accept short LoRA names in prompts (`<lora:realismenhancer>`, optional weight, any case or unique part of the file name) and in the API; apply a LoRA selected as a card and typed in the prompt only once.
- Send negative embeddings (lazyneg, lazyhand, EasyNegative, ...) to the negative prompt on card activation; pick up new embedding files and trigger words without a restart; keep double-click activation from being dropped.
- Make the same prompt, seed and LoRA give the same image regardless of the previous job by settling LoRA-patched text encoders under DynamicVRAM.
- Style Generate/Settings Save as primary and Stop/remove as stop actions, explain LoRA and embedding names in the browser help, and fix the LoRA list crashing with an empty search.
- Keep Library actions on screen on phones with a labelled Slideshow button, give accordion headers and image toolbars touch-sized targets, and stop truncating prompt tool choices in narrow layouts.

- Keep Random Prompt and configurable One Button separate, preview changes before applying, restore indexed nine-candidate variations and preserve explicit prompt generation from replacement switches.
- Cache immutable prompt datasets and artist indexes, preserve custom override ordering and seeded choices, and remove the duplicate token dictionary without changing canonical data.
- Save settings and folder profiles with recoverable paired-file transactions; expose startup preferences, native compute provenance and distinct Save as / Use next launch actions.
- Add client-local UI language selection, validated English fallback and five draft core translations without changing prompts or model identifiers; prevent native toolbar translation markers leaking into accessible labels.
- Discover textual-inversion embeddings in configurable folders, including existing models/embedding files; use native ComfyUI loading and avoid inheriting the preceding LoRA's weight.
- Deduplicate overlapping Library folders; add maintained user, settings, prompt-data, localization and developer guides.

- Separate UI panels, launcher activation and legacy retirement in feature history; retain packaged prompt/wildcard data and restore Canny threshold regression coverage
- Recognize direct launch.py instances during port-conflict handling; correct stale tool imports and prompt-variation hints

- Honor offline/install-skip settings in optional ONNX installation and native CUDA builds; reject incomplete performance commands without saving partial presets
- Include bundled templates, language data and images with their consuming features; validate resource prerequisites and isolate CPU feature tests

- Pin Pillow 12.3.0 with Gradio 6.31.0; document Pillow 13's pending security fixes and Gradio's upper bound, retaining compatible Python and GPU environments

- Keep original uploaded files in Create's main preview and gallery instead of silently converting transparency, palettes, grayscale or 16-bit pixels to RGB

- Use neutral borders for explicit-size fitting to avoid stretched-edge streaks, preserving source framing and unpainted inpaint padding
- Preserve standalone upscale PNG/EXIF/ICC metadata; retain native ComfyUI alpha processing
- Honor CPU/GPU/NPU during explicit Windows ML provider acquisition, preserve old explicit provider choices and show initialization failures during CPU recovery; keep vendor automatic selection and compiled-model cache reuse disabled
- Record bounded AMD image/native-comparison results separately from semantic quality and driver stability; defer heavy additional qualification while power is limited

- Honor explicit image output sizes with uniform fitting/padding instead of implicit cropping; add the same framing option to the UI and keep source/mask geometry aligned

- Add explicit isolated Windows ML catalogue acquisition and CPU/GPU/NPU device selection for AMD, Intel, NVIDIA and Qualcomm providers; retain CPU recovery, exact device identities and offline policy, with automatic vendor selection/cache reuse disabled until qualified
- Deprecate DirectML and remove ONNX DirectML from automatic fallback selection; retain explicit legacy choices without downgrading modern runtimes
- Expose restart-bound native precision/attention/compiler controls with Auto defaults and CLI precedence; reuse ComfyUI task cleanup so CPU tasks never probe CUDA

- Make the 25 source-reviewed language weights and their 19 exact vision projectors selectable in Chat; preserve previous download identities, selected models and explicit-download behavior

- Stop native CPU Chat requests from running accelerator device enumeration; retain CPU offload flags and GPU-request validation

- Reuse one checksum-aware payload-total helper for Chat and conditional image components; distinguish published/local sizes, preserve unknown totals and embedded-component behavior, and replace the eight-column inventory/880px minimum with a compact sidebar view and collapsed source details

- Explain the active image backend, compiled HIP/CUDA component, ROCm wheel SDK and initialized/requested compute state in Runtime settings without extra device probes

- Show refreshable Chat model/conditional-projector file presence and unique missing payload totals, keep unknown or mismatched sizes unresolved, and fit evidence columns inside the sidebar

- Align AMD installer targets with published bundle extras and current platform matrices: add gfx1153 APUs and Linux MI100/MI200/MI350, retain experimental/platform exclusions, and remove the obsolete RDNA 4 roadmap warning

- Limit the wide-head ROCm attention workaround to gfx11 and unknown architectures, keyed by the tensor device; retain native SDPA on other recognized AMD GPUs

- Preserve RGB at soft background-removal edges using native straight alpha; retain explicit matting/cutout options and the transparent-input guard

- Bind the qualified rembg Bria preprocessing dimensions consistently for native, CPU-recovery and isolated ONNX sessions, without changing the selected model or uploaded image dimensions

- Reuse the pinned ComfyUI automatic CUDA allocator policy before Torch initialization, including its legacy-device blacklist and autotune handling; preserve explicit allocator settings and WSL ROCm protection

- Stop fresh automatic torch-directml selection from downgrading the main environment; retain explicit legacy selection and declared-compatible installed packages, without changing ONNX DirectML fallback

- Expose verified Chat-default opt-out and blank-preserving sampling overrides in Runtime settings, shared by Chat, image review and prompt enhancement; retain saved values when the form reloads

- Add selection-preserving Chat model filters and attributed model/bundle evidence; retain exact quantization when dismissing the model menu or switching filtered library views

- Add bounded opt-in 2048-pixel PNG vision transport beside Chat reference/review controls and in the review API; preserve Standard defaults, source pixels/metadata and proportions, share the review encoder, and fix in-memory EXIF orientation handling

- Load Chat vision projectors only for explicit image or persistent-vision requests, include load mode and file identity in reuse checks, snapshot settings during loading, verify automatic projector pairs, and restore the previous mode after temporary review

- Share content-verified Chat sampling and task-specific reasoning across UI/API/completion/rewrite/enhancement routes, retain explicit zero/false overrides and native tool templates, and remove filename-only LFM3 defaults

- Preserve exact Chat model/quantization selections across refresh and library views, include vision language weights without projectors, explain source-qualified runtime/task support, and check actual GGUF architecture before loading; automatic projector pairing requires verified content identity and reuses the shared hash cache

- Centralize runtime bundle versions, update modern CUDA 13.2 to publisher-compatible Torch 2.15 / vision 0.29.2 / audio 2.11.1, and add explicit `--legacy` NVIDIA selection with offline safeguards; preserve the matched AMD bundle

- Repair the Qwen3 4B encoder binary URL and pin the historical Heretic-v2 weight/projector pair while preserving local filenames and installed copies

- Add lazy source-bound model annotations outside the download catalogues, with distinct language weights/components, proposed artifacts and attributed benchmark protocols; metadata cannot replace downloads or certify GPU execution

- Restore responsive Library thumbnail grids with complete images and bounded desktop action panels; use supported Gradio gallery column/row values in Library, saved setups and LoRAs. Let slider values and reset buttons grow with enlarged text

- Refresh compatible Python/runtime and formatting dependencies while preserving the AMD Torch bundle and older-platform markers; remove unused DiskCache. Retain Gradio-compatible Pillow, MCP and Pydantic versions

- Preserve dirty native backend checkouts and use safe checkout for pinned updates, including concurrent-edit conflicts

- Move advanced encoder/VAE overrides from Settings to the Models component inventory with their own profile save action; retain independent automatic sampling-shift reset in Settings

- Decouple settings storage from app/UI imports, isolate mutable profile defaults, preserve saved values and use neutral styles for fresh generation profiles; retire the unused advanced-mode default while retaining legacy LoRA migration

- Move prompt variation into Prompt tools with separate preview and apply actions; retain Words/One Button choices without automatic generation, image-count changes or unrelated CLIP-subword mutation

- Package official wildcards and the example generation preset under resources, retain custom/legacy precedence, cache wildcard choices by file identity and correct the GFPGAN restoration folder guidance

- Use official LFM3 1.7B Q4_K_M as the small shared Chat/prompt default with first-use download, overridable sampling/reasoning defaults and retained advanced/saved model choices; structured enhancement additions keep original text, strip prefilled reasoning and reject altered quoted labels. Use supported-language examples instead of Ukrainian-specific regression fixtures

- Clearly label creative rewrite presets and keep the rendering-detail preset faithful to literal input instead of teaching it to invent actions or settings

- Preserve distinct local GGUF identities across configured folders, reject ambiguous filenames, exclude projector files and recommend GPT-OSS for unconfigured Chat profiles without changing saved model selections

- Replace GPT-2 Flufferizer and standalone T5 SuperPrompt enhancement with the selected local Chat model; migrate legacy selections once per request, keep enhancement off by default, preserve downloaded weights, and remove obsolete engine assets/tests while retaining UTF-8 and prompt-preservation coverage

- Update native llama.cpp to b12110 and its optional isolated CUDA build toolchain; validate migrated CUDA labels, use native reasoning templates, honor offline installation policy, and install XLlamacpp only when selected with upgrade-safe backend markers outside package metadata

- Refresh pinned ComfyUI and GGUF image backends with matching comfy-kitchen support; retain required CPU/DirectML and bounded-prefill compatibility, and expose actionable warnings instead of suppressing every user warning

- Share wildcard discovery between autocomplete and expansion; accept scalar or list roots and resolve relative user paths against `RF_DATA_DIR`, with deterministic user overrides

- Add `--skip-dependency-install` consistently to both launch paths, retain the hidden old alias, preserve pending repairs and allow supported Linux XPU selection

- Add Library prompt/full-setup reuse and original-image edit/inpaint/upscale/background-removal handoffs; isolate page selections per session, validate cached model/LoRA identity, clear stale references, preserve denoise and restore preset-hidden controls

- Group styles under non-selectable categories; keep filtered selections, apply styles on Generate, separate prompt processors, and add preview/bake/undo with original prompt and style provenance in PNG metadata

- Move Models beside Generation inside Create; use cached publisher names and versions while keeping file identities, and reject stale gallery selections

- Keep category headings out of style selections, preserve prompts with empty templates, and load custom style overrides without rewriting user files

- Put image model selection directly above Saved settings (PNG) in Generation; explain reusable PNG recipes and name their selectable setting groups clearly

- Image-first Create layout: preview above the prompt/Generate controls, image model selection at the top of the right sidebar, and stacked controls on narrow screens

- Source ZIP builder records exact source/backend revisions and excludes user data and weights; archive launches skip Git updates, source updates preserve local work, and runtime caches/reinstall requests follow the writable data directory
- Fresh installs show missing checkpoints clearly and provide an explicit default-model download; refreshing files updates the Create model selector without losing its current selection
- Include the moved One Button presets/style dataset and preserve bundled PNG preset discovery with separate user data, without overwriting custom files

- Pinned developer formatting for owned Python, Markdown, JSON, CSS, HTML and JavaScript; executable Python and JSON values checked for preservation, with model/tokenizer/data/vendor files excluded

- Task-based Create, Chat, Library, Models and Settings navigation; model/prompt controls first, grouped preferences, browser-specific text sizing, explicit generation-default saving and opt-in startup weight loading
- Prompt rewrite and enhancement share the serial local chat worker; rewrite preview/apply/undo protects user edits, and Chat exposes its per-message image-tool permission
- Organize source by feature and backend under `ruinedfooocus`; separate bundled assets/defaults from user content, with legacy assistant/rewrite preset overrides preserved and explicit writable-data paths
- Installers for NVIDIA CUDA, AMD ROCm 10.1 (Windows, Linux and WSL), DirectML and CPU, with validated PyTorch bundles and `RuinedFooocus.bat`/`RuinedFooocus.sh` launchers
- Removed obsolete upstream portable setup helpers, their test/wrapper, the embedded-Python builder and Conda environment file; PyTorch switch/reinstall/freeze instructions are inline in the readme
- Local chat models run on native llama.cpp with a Vulkan, ROCm, CUDA or CPU backend; chat bots gained vision, image generation with an optional reference and a browser Python runner
- Rewritten chat bots that work offline, and new prompt-rewrite presets for local chat models
- Prompt enhancement uses the Chat bots model and keeps your prompt; Erniehancer is replaced by the "Local chat model" style
- One Button Prompt formats follow the checkpoint: SD/SDXL tags, Illustrious/NoobAI/Pony tags, Anima and natural language
- Renamed the bundled Weird Hats preset, migrating its old saved name without replacing custom preset edits
- Preserve distinct custom presets when both old and new names exist, and tolerate missing bundled anime defaults
- Native image edit in Image tools, chat and the API on edit-capable checkpoints, with image-to-image as the fallback
- Checkpoint tools: inspect, check weights and create FP16 copies; clear messages when a model cannot load
- Keep checkpoint conversion destinations protected from overwrite on no-link filesystems and remove partial copies after write failures
- Removed unused merge-recipe handling, legacy YAML checkpoint configs, style-model/merge-cache placeholders and merge artwork; native ComfyUI loading and checkpoint tools remain
- OpenAI-compatible API (`--api`) and a restricted guest page (`--guest`)
- Responsive UI for phones, laptops and 4K screens, and modernised HTML pages
- Many fixes for ComfyUI integration, image, video and prompt handling

### 26.10.31

- [Fibo 2.0](https://huggingface.co/briaai/Fibo-2.0-dev) support
- [Flux3](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy) support
- [MOSS Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) support

### 26.09.02

- [LTX 2.5](https://huggingface.co/Lightricks/LTX-2.5) support
- [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) support
- Updated torch version for NVIDIA gpus
- Bugfixes

### 26.07.26

- [Mage Flow/Mage Flow Edit](https://huggingface.co/microsoft/Mage-Flow) support
- Esc to toggle "edit mode"
- Button in Image browser to copy metadata to Prompt.
- GGUF support for Krea 2 [models](https://huggingface.co/molbal/krea2-gguf)
- New settings for wildcards, preset behaviour and metadata comment.

### 26.06.27

- [Krea 2](https://huggingface.co/Comfy-Org/Krea-2) support

### 26.06.21

- Limited [Ideogram4](https://huggingface.co/ideogram-ai/ideogram-4-fp8) support
- [Boogu-Image](https://huggingface.co/Boogu/Boogu-Image-0.1-Base) support

### 26.05.30

- [PixelDiT](https://huggingface.co/Comfy-Org/PixelDiT/tree/main/diffusion_models) support
- Support for manual [torch installation](https://github.com/runew0lf/RuinedFooocus/wiki/Installation#torchruntime-and-torch-version)

### 26.05.12

- [HiDreamO1](https://huggingface.co/Comfy-Org/HiDream-O1-Image) support

### 26.04.29

- [LTXV 2.3](https://huggingface.co/Lightricks/LTX-2.3) support
- Code cleanup

### 26.04.15

- [Ernie Image](https://huggingface.co/unsloth/ERNIE-Image-GGUF) support ([Turbo](https://huggingface.co/unsloth/ERNIE-Image-Turbo-GGUF))
- Erniehancer

### 26.04.13

- Fixes and help files

### 26.04.02

- Gradio 6.10.0 & Torchruntime 2.8.0
- Fixes for Flux2 models and chatbots

### 26.03.16

- [Anima](https://huggingface.co/circlestone-labs/Anima) support
- Gradio 6.9.0
- Bumped ComfyUI and GGUF version
- Bugfixes

### 26.01.13

- Gradio 6.3
- Bugfixes

### 25.12.20

- NewBie Image support

### 25.11.29

- Z-Image Turbo support
- Changed to ComfuUI-GGUF

### 25.11.26

- Flux.2 Dev support
- RF Linux
- Fixes

### 25.10.14

- AuraFlow support
- CosmoPredict2 support
- QwenImage support
- HiDream support
- SD1.5 (gguf/unet) support
- PixArt (gguf/unet) support

### 25.10.07

- Updated xllamacpp, gradio and transformers
- Warnings if inbox has already downloaded models
- Support character cards for the chatbots
- Enable "context shift" to allow long llama-input
- Small bugfixes

### 25.08.31

- Support WAN 2.2 5b
- Switch from NexaAI to XLlamaCPP

### 25.07.28

- Bump gradio version
- Workaround for Nexa modules
- Links to civitaiarchive for missing models
- Fixes

### 25.07.13

- Chatbots can now have access to a image generation tool

### 25.06.27

- Flux.1 Kontext support
- Fix Florence
- Clear gradio cache at start
- Locking seeds
- Remove diffusers pipeline
- Smarter model loader

### 25.06.01

- Removed MergeManager
- Preset images
- First steps to enable internationalization

### 25.05.14

- LTXV support
- Cleanup and fixes
- --mcp to start MCP server
- Hashbang-commands to simplify some tasks
- Shift attention, specify ranges of weights
- Use last image as input if generating multiple images with img2img

### 25.04.27

- Settings tab
- Add preview for WAN and Hunyuan Video

### 25.04.11

- Multiple Fixes
- Added torchutils to add support for all gfx cards/systems
- Added models/inbox for models to get sorted
- Code cleanup

### 25.03.17

- Improve model loading
- Initial support for WAN 2.1 t2v and i2v models

### 25.03.09

- Initial support for Hunyuan Videos models

### 25.03.05

- Support different pip requirements files
- Support cpu-only
- Support some of the ComfyUI args about vram usage

### 25.02.23

- Chatbots can have embeddings and read markdown from github

### 25.02.21

- Support Lumina2 safetensors
- Bump ComfyUI and GGUF versions

### 25.02.12

- Bugfixes

### 25.02.06

- Fix for browser high CPU usage
- Code cleanup

### 25.02.02

- Fix for StabilityMatrix

### 25.01.30

- Borrow llama_cpp_python from oobaboga
- Update ImageBrowser DB during generation
- List for "archive_folders" in settings
- Chatbots

### 25.01.25

- Image Browser
- Inpainting works again

### 2.0.1

- Post-upgrade bugfixes
- llama and image generation API

### 2.0.0

- RF2!

### 1.58.0

- Support SD1
- Fixes when loading gguf-models

### 1.57.0

- Added BrainBlip interrogator
- StabilityMatrix compatibility
- Added Llama 3.2 button

### 1.56.0

- Support for Flux GGUF unet models
- RemBG
- Img2STL
- Florence as default Interrogator

### 1.55.0

- Pixart and Flux oh my!

### 1.53.0

- MergeMaker

### 1.52.0

- Merge-recipes for checkpoints

### 1.51.0

- Bugfixes
- Now works with the SD3 LARGE model.
- OBP Update!!!

### 1.50.0

- SD3 BBY

### 1.40.1

- Support animated thumbnails

### 1.40.0

- New LoRA selection
- Updated OBP

### 1.39.0

- New model selection with model previews!

### 1.38.1

- Minor Fixes

### 1.38.0

- Added HyperPrompt to styles / OBP
- Added handy hints!

### 1.37.1

- Fixed missing clip skip in some performances

### 1.37.0

- Updated Comfy Version
- Tidied Code
- Added CLipSkip

### 1.36.0

- Added search pipeline, Type "search:" in prompt to get todays images

### 1.35.0

- Added LayerDiffuse to create transparent images with an optional background image `Powerup Tab`
- Artify Update: NMake sure you check the style selection for a TON of new styles to try
- OBP Update

### 1.34.0

- Added llyasviel's prompt tokenizer model for improving prompts, This can be used by adding the `Flufferizer` style

### 1.33.1

- Moved upscale to its own pipeline
- Various Bug Fixes

### 1.33.0

- Updated to latest comfy version
- Now supports Playground 2.5 model!

### 1.32.0

- Various Bugfixes
- Adds the ability to upgrade to a newer torch and xformers, just create a blank `reinstall` file in the RF directory

### 1.31.0

- Added presets / custom preset for OBP
- Various Bugfixes

### 1.30.0

- The OBP Update
- added "the tokinator" mode
- Fintetunes on autonegative prompt: -- Landscape removed
- Added list of artist shorthands (like van gogh)
- Added secret sauce mode (mixing OBP + SDXL styles). Happens more on lower insanity levels
- Added more consistency on lower insanity levels
- EVO with OPB variant works better:
- Small prompts now work
- (almost) always changes something
- Fix for line ends creeping into EVO + OPB Variant
- Added loads of new styles into the lists
- Added loads of more fictional characters into the lists (finally, harley quinn is in :p)

### 1.29.0

- Automatic Negative prompt, save yourself the heartache and hassle of writing negative prompts!

### 1.28.2

- Fixed some dodgy regex to support wildcards with `-` in the name

### 1.28.1

- Added better error descriptions for the civit api

### 1.28.0

- Automatically grabs lora keywords from civit on startup

### 1.27.0

- Advance prompt editing - See [Wiki](https://github.com/runew0lf/RuinedFooocus/wiki/Features#advanced-prompt-editing)
- Moved Evolve to the `Powerup` tab to be cleaner
- Bufixes

### 1.26.1

- Changed so loras with triggerwords textfiles dont automatically get added, instead displays trigger words to add manually

### 1.26.0

- Fixed issue with controlnet caching
- Custom resolutions can now be saved directly, just select custom from the resolution dropdown (Lavorther)
- re-enabled comfy messages

### 1.25.5

- Make Custom option for Performance Mode copy last selected mode's setings (Lavorther)

### 1.25.4

- Fixed an issue with wildcards and strengths getting stuck in Catastrophic backtracking

### 1.25.3

- Changed a lot of os.path to pathlibs
- Worked on backend code for future extensibility
- Added support for --auth

### 1.25.2

- Fixed an os.path issue by replaceing with pathlib

### 1.25.1

- Now displays 🗒️ at the end of the loraname if it has a keyword .txt file

### 1.25.0

- Prompt Bracketing Now works ie `[cat|dog]`
- Updated comfy version

### 1.24.2

- More Tidying and tweaks
- Autoselect LCM lora if Lcm is selected from the quality or sampler dropdowns

### 1.24.1

- Code Tidying and tweaks

### 1.24.0

- Added Evolve which generates variations on your generated picture.
- Fixed wild imports
- Added instant One Button Prompt
- Added keybind of `ctrl+shift` to toggle `hurt me plenty`` mode
- Code Cleanup
- Fixed Issue with dropdown box's being case sensitive when sorting

### 1.23.2

- Adds token count (thanks Lavorther)
- Fixed lora keywords only working the first time the lora is loaded

### 1.23.1

- Use OneButtonPrompt with infinite mode, simply put a tick in `BYPASS SAFETY PROTOCOLS`

### 1.23.0

- Inpainting Update: Adds inpainting to the powerup tab

### 1.22.0

- Now comes with a built in clip interrogator, just drag your image onto the main image to generate the prompt

### 1.21.0

- Π Time - Click the small Π symbol to get a fullscreen image of your last generation great for slideshows
- updated comfy version, NEW SAMPLER TIME

### 1.20.0

- Added wildcard helper, just start typing __ and your wildcards will display
- Added slideshow

### 1.19.5

- Fixed old bug of switching models when the stop button is pressed (old code from OG-F)

### 1.19.4

- Old experimental lcm-pipeline removed
- Generate forever if Image Number is set to 0
- Updated comfy version to latest
- Nested wildcards now supported

### 1.19.3

- Random styles now correctly applying to each image

### 1.19.2

- Gradio Rollback to v3 until v4 is fixed

### 1.19.1

- WildCard Fixes
- Automatcially downloads LCM Models
- Now checks subdirectories for models

### 1.19.0

- Gradio update to V4

### 1.18.0

- New Random Style Selection
- Adding one button prompt overrides in wildcards now
- Added wildcards official
- Other stuff i'll have to get arljen to explain

### 1.17.2

- Limit seeds to 32 bits
- Sort model dropdowns
- Use caches better

### 1.17.1

- removed groop and faceswap as it was causing dependency issues on some systems

### 1.17.0

- Changed minimum cfg scale to 0
- Updated to latest comfy and diffusers (Now supports LCM Loras)
- You NEEED to set the custom settings to use lcm and sgm_sampler, steps of 4 and REALLY low config (between 0 and 1.5)

### 1.16.0

- Facewapping
- Groop

### 1.15.1

- Updated Comfy Version

### 1.15.0

- Different pipelines supporting lcm and sdxl/ssd
- Let async_worker handle model preload
- Lots of small fixes
- fixed metadata bug when stopped

### 1.14.1

- Fixed small issue with metadata not updating

### 1.14.0

- Added Metadata Viewer for Gallery items (Viewable in `Info` Tab)
- Refresh Files now also reloads your `styles.csv` file

### 1.13.0

- Automatically download 4xUltrasharp Upscaler
- Added the ability to upscale images wth upscaler of your choosing
- Changed Powerup Settings so if there is a missing key from defaults it adds it to your custome settings.

### 1.12.1

- Refactored backend code to allow for future pipeline changes

### 1.12.0

- Automatically read triggerwords from <lora_filename>.txt

### 1.11.0

- Updated Comfy Version
- Added support for [SSD-1B Models](https://huggingface.co/segmind/SSD-1B)

### 1.9.0

- Removed ref redundant code.

### 1.8.2

- Update Comfy version and fix changes :D

### 1.8.1

- Improved image2image and allowed settings to be changed when "custom" is selected form the PowerUp Tab.

### 1.8.0

- Added the basics for image 2 image
- Renamed Controlnet to PowerUp
- Now uses `powerup.json` as default

### 1.7.2

- Wildcards can now use subdirectories
- Fixed issue where if you placed 2 placeholders with the same name, you got the same results, a new one is now chosen
- Updated status to show model loading / vae decoding

### 1.7.1

- Update to one button prompt (provided by [Alrjen](https://github.com/AIrjen/OneButtonPrompt))

### 1.7.0

- Custom Controlnet Modes
- minor bugfixes
- moved the controlnet tab to its own ui file.

### 1.6.1

- Added sketch controlnet!

### 1.6.0

- Updated gradio version
- Added recolour controlnet!

### 1.5.2

- Restored gallery preview on all images
- renamed more variables to make sense
- bugfixes

### 1.5.1

- Added all the settings/customization to their own `settings` folder **NOTE:** you will need to move / copy your settings into the new directory
- Bugfix where clicking stop mid-generation stopped working
- code cleanup and some renaming
- inference model speed up
- now only shows gallery when needed

### 1.5.0

- removed metadata toggle, it will now always save metadata
- save your own custom performances
- tidied ui
- fix crash when failing to load loras
- hide all but one lora dropdown showing "None"

### 1.4.2

- change fooocusversion.py to version.py for easier updating
- Moved controlnet to its own tab for easier updates
- updated gradio version
- minor wording changes

### 1.4.1

- `paths.json` will now be updated if there are any missing defaults paths

### 1.4.0

- Now supports controlnet

### 1.3.0

- Updated onebutton prompt so you can now add multiple random prompts by clicking the `Add To Prompt` button

### 1.2.2

- Update comfy version - Lora weights are now calculated on the gpu so should apply faster

### 1.2.1

- Bug fixes and backend updates
- changed `resolutions.csv` to `resolutions.json`
- updated readme

### 1.2.0

- Prompt now splits correctly using `---`
- added the ability to change styles in the prompt by using <style:stylename>

### 1.1.7

- Added init image

### 1.1.6

- Fixed issue with wildcards if file not found.

### 1.1.5

- Fixed sorting on subfolders, so directories are displayed first

### 1.1.4

- Allowed main image window to recieve drag and drop
- Added a gallery preview underneath that will activate image window.

### 1.1.3

- Added support for subdirectories with models/loras so you can get all organised!

### 1.1.2

- showed imported image in gallery
- moved `---` split into prompt generation
- correctly updates progressbar
- fixed importing of width / height

### 1.1.1

- In the json prompt, setting a seed of `-1` allows you to generate a random seed

### 1.1.0

- Render different subjects so you can process a whole list of prompts. Seperate each prompt by placing `---` on a new line

### 1.0.0

- New Beginnings. The official start of the updates!
