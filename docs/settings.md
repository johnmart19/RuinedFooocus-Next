# Settings and profiles

Settings keeps its search, navigation, fields and save actions together at a
readable width on large displays, while using the available width on phones.
Search lists matching fields and actions. Choose a result to open its category
and collapsed groups, scroll to it and focus the exact control. Search and
navigation keep the edited form; they do not save it or run an install/maintenance
action. Cache, hash, Library index and size searches include Maintenance.

Create and Chat contain controls for the current task. Settings stores persistent
preferences. **Create → Save current defaults** saves the current model, prompt,
styles, LoRAs, sampling and size; it does not save uploads or a generated image.
Clear an active PNG preset before saving those defaults.

## Where to look

| Section               | Controls and scope                                                                                                                                       |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Appearance & language | Browser text size/language, profile language default, local metadata, theme and UI limits                                                                |
| Profiles & storage    | Startup browser/port/offline/package policy, model/embedding/output folders and profile actions                                                          |
| Hardware & backends   | Detected GPUs, a **Supported on this PC** table, **Use CPU for everything**, and backends for ComfyUI, llama.cpp, xllamacpp, ONNX Runtime and Windows ML |
| Image runtime         | Visible active backend/device/build, collapsed runtime diagnostics, preload and native memory/precision/attention controls                               |
| Local Chat            | Shared runtime choice, context, sampling, vision, prompt-rewrite preferences and collapsed Chat model diagnostics                                        |
| Generation defaults   | Advanced startup defaults and PNG preset reuse policy; current generation stays in Create                                                                |
| Connections           | Live local API status, model IDs, copyable requests, and API/MCP/listen startup preferences                                                              |
| Maintenance           | Cache sizes and targeted derived-cache cleanup; explicit package/Torch reinstall requests                                                                |

Installed checkpoints, components and LoRAs/embeddings are under **Models → Installed**.
Available models and transfer status are under **Models → Download**.

Native advanced choices default to **Auto**. Unsupported choices are explained;
selecting a label does not prove the installed hardware supports that operation.
Background removal's ONNX provider/device identity is separate from PyTorch's
image device. Refreshing/discovering providers is explicit.

**Local Chat → Chat model diagnostics** shows the model and quantization selected
in this browser. It separates the last Chat session's load and sampling messages
from the currently initialized runtime evidence. Expand it to inspect the
evidence; use **Refresh file presence** or **Refresh sampling
recommendations** when needed. Opening Settings does not load a Chat model.

Choosing **Image backend → CPU** affects ComfyUI after a restart and keeps Chat
and background-removal choices independent. **Use CPU for everything** or the
explicit `--cpu` launch option forces all three components onto CPU. Neither
choice requires replacing an installed CUDA or ROCm PyTorch build to run on CPU.

**Hardware & backends** lists only the options that suit the detected GPUs and
the operating system (Windows, WSL or Linux). **Supported on this PC** shows every
option of each component with its state: _Installed_, _Available_ (supported here
but not installed yet) or _Not available_ with the reason, for example
"Needs an NVIDIA GPU" or "Native Linux only". Installed states come from package
metadata and saved install records; viewing the table never loads a model or
probes a device.

Install actions refresh this table automatically. Windows ML acquisition failures
list the providers offered by the local catalogue and their certification state;
an installed provider still needs qualification with the selected model.

| Component            | Choices (shown when the hardware supports them)                                 | Install                                    |
| -------------------- | ------------------------------------------------------------------------------- | ------------------------------------------ |
| ComfyUI (PyTorch)    | CUDA (NVIDIA), ROCm (AMD), XPU (Intel), CPU; DirectML only for a legacy install | **Install / switch on next start**         |
| llama.cpp            | CPU, Vulkan, CUDA, ROCm (AMD), SYCL (Intel)                                     | **Install now**                            |
| xllamacpp            | CPU, Vulkan, CUDA 12.8, CUDA 13.2, ROCm 7.2 (Linux and WSL)                     | **Install xllamacpp now**                  |
| ONNX Runtime         | CPU, CUDA (NVIDIA), MIGraphX (AMD, native Linux), OpenVINO (Intel)              | **Install / verify selected ONNX runtime** |
| Windows ML (Windows) | AMD GPU/VitisAI/MIGraphX, Intel OpenVINO, NVIDIA RTX, Qualcomm QNN              | **Install / verify Windows ML provider**   |

On WSL the GPUs are read from the Windows host, because WSL itself shows only a
virtual display adapter.

## Save and select a profile

**Save** writes the current profile. **Save as profile** writes a named copy of
the edited form. **Use next launch** selects an already saved profile. An explicit
`--settings <name>` on a launcher remains authoritative. The save status identifies
the running profile separately from the saved one; saving does not replace a
running worker's model or restart the application.

The profile summary separates **Running profile**, **Save target** and
**Next launch**, with an **Unsaved changes** indicator. Profile status and actions
follow the Settings categories in normal page flow. Scroll to the end of Settings
to save; the action bar does not cover the form. Each browser keeps its own Save target: another
browser's **Save as profile** does not redirect this browser's **Save**. A successful
save records the submitted form, so edits made while saving remain dirty. A failed
save leaves the form and dirty indicator intact. Draft comparison stays in browser
memory and is not copied to browser storage. Reloading the page loads saved values.
The **Profile name** field is the proposed **Save as profile** or **Use next launch**
name; typing it alone does not change **Save target**.
Settings keeps a server-held base for that session and merges the latest saved pair
from its own profile. Unknown settings and path keys survive; an unrelated profile
saved by another browser is not used as its source.

Set **Default listen port** to 0 for automatic selection. Explicit `--port` remains
authoritative; 0 is saved as unset, never as port zero.

The writable root defaults to the clone directory. `RF_DATA_DIR` selects another
root; the default profile is `settings/settings.json`, named profiles are
`settings/<name>/settings.json`, and each profile has its own `paths.json`.
`settings/startup.json` records the optional next-launch selection. Legacy
relative model folders keep their old resolution rules. Unknown custom settings
are retained, and UTF-8 files with or without a byte-order mark are accepted.

Named sizes saved in Create use `image_sizes.json` or `video_sizes.json` beside
the current profile's `settings.json`, with separate entries for each model family.
They are user choices, outside the bundled recommendation resources. Legacy
`settings/resolutions.json` sizes remain readable; an existing profile's numeric
size and `resolution_auto` intent survive refresh. Choosing Custom or a size
disables automatic following. **Use model recommended size** explicitly restores
it. Fresh profiles follow applicable publisher dimensions, then the documented
family recipe; unknown models show an identified generic fallback. Size recipes
and benchmark/hardware qualification are separate facts.

The [model folder guide](model-folders.md) lists canonical names and legacy
aliases. Diffusion Model Folders extends image-model discovery beyond Checkpoint
Folders; Text Encoder Folders keeps the `path_clip` profile key. New profiles use
`models/text_encoders`, while existing `models/clip` values keep their priority.

In **Profiles & storage → Paths**, **Select folder** opens an in-page browser for
the server's filesystem. Use **Open path**, **Parent folder** or **Open subfolder**,
then **Use folder** to update the draft. Multiple-folder controls also offer
**Add folder**, **Remove folder**, **Move up** and **Move down** to set discovery
priority. **Advanced path** allows direct entry. **Open host dialog** opens a
desktop chooser when a local owner session and the host desktop support it;
otherwise use the in-page browser. A remote browser chooses folders on the app's
host, and WSL paths use the Linux server namespace. Folder browsing requires
owner access. Selecting a folder does not move files; Save and restart apply
the changed paths.

Settings and paths are committed as a recoverable pair: an undo journal is written
before either file changes. Failed or process-interrupted saves restore the exact prior
files before the next read, subject to successful recovery I/O. This is not a
guarantee against storage corruption or sudden power loss. A successful save publishes the new in-memory values
and notification only after disk commit. Do not delete a pending recovery journal.
Concurrent application processes writing the same profile are unsupported; use
separate profiles for separate servers.

## When a choice applies

- Text size, the browser language selection and an available theme apply in that browser immediately.
  Text size's Reset to default value restores 100% immediately and remembers it in that browser.
  Save stores the selected theme in the profile for later launches. See [Themes](themes.md).
- The local metadata preference applies after saving, while explicit offline
  startup still wins.
- Startup defaults, paths, image device and native compute options need a
  restart. Saving does not mutate an initialized backend.
- Local Chat runtime/model changes apply on the next appropriate model load;
  active conversations are not silently replaced by browsing settings.
- Repair, package replacement and cache cleanup are explicit maintenance actions,
  never a hidden consequence of changing a normal UI checkbox.

Explicit CLI choices override saved values; documented environment overrides such
as `TORCH_PLATFORM`, `LLAMA_SERVER` and `RF_OFFLINE` retain their authority (precedence:
explicit CLI flags > `TORCH_PLATFORM` / `LLAMA_SERVER` > saved profile > automatic
selection). **Use CPU for everything** forces CPU for image generation after a
restart and for Chat and background removal on their next load, while keeping your
individual component backend selections intact so turning it off restores them.
Switching to a different GPU PyTorch build (`cuda` or `rocm`) only installs on an
online restart after clicking **Install / switch on next start**. Native **Auto** follows
the pinned ComfyUI policy. The Image runtime section shows the active image
backend, device and PyTorch build above its editable preferences. **Runtime
diagnostics** holds the remaining build details and requested policies separately;
saved form changes do not rewrite these initialized-state facts.
Active runtime information distinguishes requested
profile values, CLI overrides and initialized state. Effective CPU operation
suppresses saved FP16/BF16 overrides; explicit precision flags remain explicit.
Zero and false are real values, not missing preferences. A blank optional numeric
override means automatic only where the control says so.

Native Chat **Auto** prefers usable cached vendor GPU builds, then Vulkan on
Windows and Linux. Its choice is independent of the image PyTorch build. WSL needs a
guest Vulkan driver. Explicit Chat CPU selection disables weight, KV, operation
and projector offload even when a tracked GPU wheel is already cached. Changing
an imported xllamacpp backend requires a restart. Offline loads validate the
selected backend and interpreter identity; package presence alone is insufficient.
When Auto finds no usable GPU it warns in the terminal and uses CPU. xllamacpp
checks its actual devices before loading a model, and backend replacement after
import requires a restart.

Package installation resolves the pinned ComfyUI base, the matching Torch
companions and RF additions together before replacing packages. An incompatible
Python ABI, missing wheel, unsupported driver or dependency conflict stops the
change. Existing reinstall requests remain pending after a failed plan. Package
acquisition/staging failures preserve the current environment; a failure during
in-place pip application requires restart and repair rather than a promised
automatic rollback.

Chat context and output limits can be cleared to use the verified model defaults.
Saving and reopening the page preserves blanks, explicit `0` context and `-1`
output limits; new requests use the saved values. Existing loaded contexts are
not resized while a request is running.

**Local Chat → Local chat and prompt runtime → Enable MoE cache CPU share** sits
beside Chat GPU layers. It starts enabled; **MoE cache share options** contains
the numeric fraction from 0 to 1, initially 0.5. This default is RF policy.
The status separates the form, saved profile and initialized Chat runtime.
Save to keep changes, then reload Chat; **Refresh MoE status** reads existing
runtime state without loading a model or probing a device. Explicit native
arguments or environment values take precedence over the RF preference.
The pinned b12110 runtime lacks this option, so the requested preference remains
visible and inactive. xllamacpp also keeps it inactive. See the
[MoE cache contract](integrations/llama-moe-cache.md) for supported conditions.

## Reasoning preferences

In **Local Chat → Reasoning preferences**, save separate Chat and prompt-rewrite
defaults. Load a model in Chat first: available computation and effort choices
come from its initialized runtime and verified embedded template. **Show
reasoning** in Chat changes display only. Unsupported saved preferences remain
visible but inactive; changing models does not silently erase them.

**Keep saved rewrite behavior** retains earlier profiles' model-specific policy
and **Saved rewrite effort**. Selecting Auto explicitly adopts the initialized
template's default. Saving another setting does not migrate the legacy policy.
Chat's **Use saved preference** follows those defaults on the next request.
Choosing **Auto** explicitly uses the initialized template default instead.
Conversation overrides stay local to that browser and do not rewrite the saved
profile or another conversation.
See [assistants and reasoning](chat-assistants.md) for exact support limits.

## Local API workspace

**Settings → Connections** shows whether the owner API is mounted for the
current launch, its local base URL, and whether requests require a bearer key.
It lists the public Chat and image model IDs and shows the model selected by
each default alias alongside any initialized runtime model. An initialized
image model can be offloaded. Use **Refresh API status** after changing model
selection or loading a model.

Open the API reference or OpenAPI schema for endpoint details. The copyable
Python examples use the standard library and cover Chat streaming, image
generation and masked editing. Substitute a listed model ID if the default
alias is unavailable. Bearer examples show `YOUR_API_KEY` for local replacement.
API and MCP startup choices remain under **Advanced local connections**.

## Cache storage

Open **Settings → Maintenance → Cache storage** and choose **Refresh cache sizes**.
The panel identifies the configured model/Library cache and, when different, the
uploads/runtime cache. Sizes are bounded snapshots taken on request.

**Clear selected cache** can reset local model hash records, the Library index,
Gallery thumbnails or startup environment verification. Hashes regenerate from local files. After
clearing the Library index, use **Refresh images** in Library; original images
and saved metadata remain in their files. Index files can retain allocated disk
space until SQLite reuses it. Startup verification runs under the usual startup
and offline policy; clearing its marker does not install packages.

**Library thumbnails** regenerate when browsing Gallery. The app prunes recognized
thumbnail copies toward 256 MiB and 2,048 files, with cleanup deferred while a copy
is in use. Clearing this category preserves original images and metadata.

Active or queued entries are skipped. Legacy hash records, artwork, uploaded
inputs, Chat sources, resumable downloads, Inbox state, installed runtimes,
models, outputs and profiles are retained. Other tools and external framework
caches keep their own maintenance actions. See the
[cache ownership guide](integrations/cache-contract.md) for exact categories.

## Offline and access

Offline startup is read before startup updates/downloads. **Use installed
packages** skips dependency installation while allowing online source updates;
`--skip-dependency-install` is the equivalent CLI flag. `--offline` additionally
disables network startup work. Existing repair requests stay pending when installs
are disabled.

Binding beyond localhost still requires UI authentication and API credentials
when the API is enabled. Set credentials using `RF_UI_AUTH` / `RF_API_KEY` or their
CLI options; ordinary profiles have no password/key fields. Restricted guest mode
and public-tunnel restrictions remain unchanged. Saving a connection preference
does not open a port until restart.

See [runtime support](runtime-and-troubleshooting.md) for platform-specific flags
and [development](development.md) for private-profile validation.
