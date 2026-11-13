# Legacy source retirement

The application runs from `ruinedfooocus/`. The retired root folders were source
and bundled defaults, not a request to delete local models or user overrides.
History separates panel implementation, launcher activation and subsequent
retirement. Installer commits remain separate. Large deletion counts in the data
retirement commits mostly represent duplicate CSV/text datasets.

| Former source                                                                                   | Maintained owner or disposition                                                                                                                                                      |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `webui.py`, `modules/ui/`                                                                       | `ui/app.py` and each feature's `ui*.py` under `ruinedfooocus/`                                                                                                                       |
| `argparser.py`, `shared.py`, worker/API modules                                                 | `app/arguments.py`, `app/state.py`, `app/async_worker.py`, `app/api*.py`                                                                                                             |
| Image and model pipeline modules                                                                | `features/generation/`, `features/models/`, `backends/comfy/`; native ComfyUI loaders, samplers and nodes remain the execution baseline                                              |
| Canny threshold helper                                                                          | `features/image_editing/canny_utils.py`; its original three regression cases are retained                                                                                            |
| Gallery, image tools and metadata modules                                                       | `features/gallery/`, `features/image_editing/`, `features/image_processing/`                                                                                                         |
| Chat, search and template pipelines                                                             | `features/chat/`, `features/automation/`; shared local Chat runtime                                                                                                                  |
| Browser Python runner (`features/chat/{chat_code,browser_python}.py`, `ui/assets/chat_python*`) | Intentionally retired with its UI controls and asset routes; ordinary Chat code blocks stay readable. Existing caches and chat history remain untouched                              |
| Last image page                                                                                 | Intentionally retired from the UI; `features/gallery/ui.py` retains the Library slideshow and result actions. Existing outputs and Library records remain untouched                  |
| `random_prompt/`                                                                                | `features/prompting/one_button/` (code, presets and CSV datasets); one canonical token CSV serves both logical names                                                                 |
| `wildcards_official/`                                                                           | `resources/wildcards/`; all 32 source files retain identical contents                                                                                                                |
| `llamas/`, `chatbots/`                                                                          | `resources/prompt_presets/`, `resources/assistants/`; local user override directories remain supported                                                                               |
| `language/`, `hints/`, `html/`, default settings/presets                                        | Packaged `resources/`, `ui/assets/`, and maintained settings services                                                                                                                |
| `prompt_expansion/`, `superprompter/`                                                           | Retired separate GPT-2/SuperPrompt enhancement; explicit enhancement uses the selected local Chat model                                                                              |
| Token-based prompt evolution                                                                    | Opt-in cached single-stream tokenizer mode; unloaded/multi-encoder tokenizers explicitly unsupported. Words/One Button, indexed parents and nine-candidate previews remain available |
| `.merge` model recipes and merge preview                                                        | Retired custom loader; supported checkpoints use native loading; checkpoint conversion writes separate copies                                                                        |
| Legacy face-swap marker                                                                         | Replaced with GFPGAN face-restoration guidance; unsupported face-swap requests remain explicitly rejected                                                                            |
| `models/configs/*.yaml`                                                                         | Unused static templates; native ComfyUI detects checkpoint configurations                                                                                                            |
| `help/rf-setup.*`, embedded-Python helper, `environment.yaml`                                   | Replaced by maintained platform installers and root launchers; source archives use `packaging/distribution.py`                                                                       |

Paths in the maintained-owner column are relative to `ruinedfooocus/` unless
explicitly qualified. Source coverage and CPU contracts do not establish GPU
stability, generated image quality, or video execution. Native adapters retain
app-specific cancellation, cache identity, metadata and memory-recovery contracts;
similarity to ComfyUI alone is not evidence that an adapter is redundant.

## Current UI and saved data

The browser Python runner and Last image page are intentional retirements, as
are the elaborate Chat model metadata-filter panel, duplicate moved-setting
notices and raw compute-status JSON. Chat retains installed/custom model search
and quantization selection. Library metadata search, slideshow, metadata
inspection and result reuse remain available. Chat still displays ordinary code
blocks; it no longer exposes executable Python controls or their asset routes.

Saved assistants are edited in Chat. Existing assistant directories, legacy
`chatbots` entries and imported character cards remain readable; saving a bundled
assistant creates a user copy. See [saved assistants](chat-assistants.md).
[Component choices](model-folders.md) remain scoped to the selected checkpoint;
missing or unverified saved alternatives stay visible instead of silently
becoming Automatic.

[Themes](themes.md) and [ordered model folders](paths.md) are saved with the
profile and apply after restart. Selecting a folder does not move models or
outputs. Optional image and background runtime actions live in Hardware &
backends; installation, device verification and actual workload acceptance are
separate. The [optional image contract](integrations/sdcpp.md) preserves native
routing for unsupported requests and requires matching acceptance before
optional execution. These UI changes do not authorize cache or user-data cleanup.
