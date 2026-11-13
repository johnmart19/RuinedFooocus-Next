# User guide

The main destinations are **Create**, **Editor**, **Gallery**, **Models**, **Chat** and **Settings**.
They appear in the left sidebar on desktop and wrap above the page on narrow screens.
Create starts new images. Editor contains image editing, masks, background removal, composition and
upscaling. They share one image preview, prompt, model selection and generation controls; changing
destinations keeps the draft and image history. Create's **Generate** starts a new image without Editor's source
or mask. Returning to Editor restores its selected image action and source.
The preview stays above the prompt and Generate button, with the concise image model picker at the top of
**Generation** in the right sidebar. Its **Models** button opens the full manager without copying the selection.
Models contains checkpoint browsing, LoRAs, model tools and the download catalogue. Checkpoints show cached
publisher names and versions; details retain filenames. The selected checkpoint card follows the Generation picker.
Generation, Prompt tools and Details remain available beside the shared image; Editor also exposes **More tools**
for existing full image-tool controls. **Generation → Size and upscale → Upscale** contains optional upscaling.
**Generation → More options** groups saved PNG settings, native output and sampling, layout/attention,
Layered and saved-tensor controls; model-specific sections appear when applicable. The changelog is in Details.
When the working area is narrow or text is enlarged, the image controls follow the image and prompt
and scroll with the page.
Settings groups appearance/language, profiles/storage, hardware/backends, image runtime, Local Chat, generation defaults, connections and maintenance. Text size, language and theme can apply immediately in the browser; Save keeps the theme for later launches. Startup preferences need a restart. See [Settings](settings.md).
Optional hints open with the **?** beside a control. Click, tap or focus it and
press Enter to show the hint; activate it again to close. Hints stay in the page
flow so their links remain usable without covering another control.

## Image models

- **Models → Installed** browses local checkpoints and LoRAs/embeddings; **Models → Download** opens the shared download catalogue.
  Filter by category, workflow or quantization; select a card for its cached preview,
  source/license, companion status and resource information. Downloads have progress
  and cancellation and do not activate a model. Use an installed checkpoint explicitly
  in Create, or keep selecting local models there. **Browse model catalogue** in Create
  opens this same view. See [model folders and acquisition](model-folders.md).
- SD 1.5, SDXL (including Pony, Illustrious and NoobAI), SD3, Flux, Flux 2 and Flux 3, Qwen-Image, HiDream and
  HiDream O1, Lumina 2, Z-Image, Anima, Krea 2, MOSS Image, Fibo 2, Mage Flow, Ernie Image, NewBie Image,
  PixArt, PixelDiT, AuraFlow, Cosmos Predict 2, Boogu and Ideogram 4 (limited). See the [update log](../update_log.md).
- Safetensors and GGUF checkpoints, including UNet-only files. **Models → Installed → Model → Image model components** lists
  and downloads the text encoders and VAE a checkpoint needs. The compact status view separates local/published sizes and reports unique required missing bytes; embedded components add no download. Paths and URLs are under **Component locations and sources**. Unknown/custom sizes keep totals unresolved; refresh does not download.
  ComfyUI detects checkpoint configuration directly. An explicitly selected local YAML recipe can override prediction and CLIP-layer settings for ordinary native Stable Diffusion; filenames do not attach recipes automatically. Merge-recipe files are not used.
- **Models → Installed → Model → Checkpoint tools** reports a checkpoint's size and tensor types, checks its weights for
  damage (NaN/Inf), and can create a smaller FP16 copy while keeping the original.
  Existing destination files are never overwritten, including on drives without hard-link support.
- If a checkpoint cannot load, the error names the cause: a missing file, a missing text encoder or VAE, a
  damaged or incomplete download, or a file that is not an image checkpoint.
- Drop a finished model file into `models/inbox` while RuinedFooocus is running.
  The sorter checks it in the background and moves it to the configured model
  folder once it can identify its category. See [Inbox sorting](model-folders.md#inbox-sorting).
- Custom model and output folders in `settings/paths.json` (on Windows use `/` or `\\` instead of `\`).

## Prompts and styles

- **Create → Generation → Styles** groups visual styles by category. Select several to apply them in
  order when generating, or add `<style:name>` directly. Search keeps selections in other groups.
  Preview shows style text without running enhancement; Bake into prompt applies it once and clears
  those selections, with Undo that protects later edits. Optional prompt processors are in **Prompt tools**.
  Styles come from `settings/styles.csv` and the bundled catalog.
  Category headings are never styles. Custom rows override bundled styles by name; loading styles
  does not rewrite your file, and empty templates keep your prompt intact.
- Wildcards: type `__name__` to insert a random line from your `wildcards/name.txt`. For a bundled example, use `__words__` from the [packaged word list](../ruinedfooocus/resources/wildcards/words.txt). Official choices ship in `ruinedfooocus/resources/wildcards/`; custom files take precedence.
- **Prompt variations** preserves the original and an indexed parent, with 1–9 reviewed candidates. Preview does not generate or change image count. Apply/Undo protect later edits. **Generate reviewed candidates** explicitly queues one image per candidate and disables automatic typed-prompt replacement. Words and One Button work without image weights. Tokens is opt-in and requires the selected model's compatible tokenizer already cached in RAM; unloaded, changed and multi-encoder tokenizers are unsupported. It protects quoted text, numbers and prompt syntax, but arbitrary token changes can still produce poor wording.
  Preview/Apply/Undo check the current browser draft again when a response arrives.
  Typing while a result is returning keeps your newer text. A delayed Random Prompt
  or One Button suggestion remains available as a pending result; applying it is an
  explicit replacement or append action. Undo refuses to replace a newer manual edit,
  including an edit that restores the same visible text.
- LoRAs from **Models → Installed → LoRAs & embeddings** or with `<lora:name>` / `<lora:name:0.8>` in the prompt. The name is the
  file name without folder or extension, in any case, or any part of it that matches only one installed LoRA
  (`<lora:realismenhancer>`); an ambiguous name lists the matching LoRAs. A LoRA selected as a card and typed in the
  prompt is applied once, with the prompt's weight. Trigger words are read from a `.txt`
  file next to the LoRA or downloaded from Civitai; enable **Append LoRA trigger words** to include them.
- Put `---` on its own line to queue several prompts. Automatic negative prompt is available.
- **Image tools → Scheduled LoRA** adds one local LoRA with manual or interpolated
  strength keyframes. It starts Off and applies after the selected/prompt LoRAs.
  Model and CLIP strengths are separate; CLIP keyframes are optional. See
  [scheduled LoRA behavior](integrations/native-lora-hooks.md) for intervals,
  compatible targets and explicit full-setup replay.
- **Prompt tools** has [One Button Prompt](https://github.com/AIrjen/OneButtonPrompt) for random prompts. Its
  **Generate prompt** action puts text in your draft; the main **Generate** starts the image separately.
  Choose a preset and optionally enter a subject. Custom options and optional enhancement stay collapsed.
  Prompt format follows the checkpoint: tags for SD/SDXL, anime tags for Illustrious/NoobAI/Pony, tags with a
  description for Anima, and natural language for Flux, Qwen and similar models.
  The bundled **Weird Hats** preset keeps existing saved selections and custom edits under its old spelling.
  If both spellings already contain custom presets, each saved selection keeps its own values.
- Old saved style selections stay inactive until explicitly selected or saved with **Save current defaults**.
  PNG metadata records the original prompt and applied styles alongside the final rendered prompt, so
  replay does not apply styles twice.
- **Prompt tools → Rewrite with chat model → Preview rewrite** and prompt enhancement use the selected local chat model. Rewrite instructions are bundled
  under `ruinedfooocus/resources/prompt_presets`; add your own in `prompt_presets`. Legacy `llamas` entries still work.

## Embeddings

Open **Models → Installed → LoRAs & embeddings** for one preview-card list. Search by publisher name or filename; the optional All/LoRA/Embedding filter only narrows the list. LoRAs and embeddings can be active together.

Double-click a card or drag it into the selected area to activate it. Select a LoRA card to change its weight or remove it. Each embedding has its own **Embedding target**: **Automatic** recognizes negative embeddings such as `lazyneg`, `lazyhand`, `EasyNegative` and `badhandv4`; other embeddings default to Positive. Choose a different target and press **Activate** to move an unchanged reference inserted by this session. Other resources and prompt text remain intact. Manually typed, edited or replayed references stay under your control; edit those directly in the prompt.

You can also type an installed embedding's name, for example `lazypos, 1girl` and `lazyneg, lazyhand` in the negative prompt. Readable triggers go at the start of the chosen prompt. The native CLIP encoding path resolves installed, unambiguous triggers to ComfyUI references internally; explicit `embedding:name` syntax remains supported. Selected cards show active LoRA weights and Positive/Negative references. **Remove** undoes only an unchanged insertion owned by this session.

Publisher artwork and names come from local sidecars or the metadata cache. Exact `<filename.ext>.preview.png` sidecars take priority. Missing artwork uses a local category image labelled **Default preview**, including image-model and download cards. Failed or rate-limited publisher requests retain usable cached artwork; Refresh respects the publisher cooldown. Full filenames remain in resource details and searchable; internal identities distinguish resources without displaying technical IDs or long filenames. Metadata lookup runs in the existing background model scan and respects offline/local-metadata preferences.

Canonical embedding files belong in `models/embeddings`. Legacy folders remain readable when configured, but new installations do not create duplicate alias directories. Configure **Settings → Profiles & storage → Paths → Embedding Folders**, save and restart for custom roots.

For the author's Lazy examples with compatible Illustrious/Noob/SDXL models, add `lazypos` to the positive prompt and `lazyneg` to the negative prompt (`embedding:lazypos` works the same). Pony examples use `lazyup`/`lazydn`; select only installed resources. These are prompt references, not substitutes for model compatibility or a guarantee of image quality.

Textual inversion requires a compatible text encoder and matching embedding dimensions. Native SD 1.x, SD 2.x
and SDXL tokenizers support matching embeddings; availability in the browser does not certify compatibility
with every checkpoint or text encoder. Adding a reference edits prompt text only; Generate remains a separate action.

## Image tools

The full tool panels are in **Editor → More tools**. The main **Image action** selector
handles the common editing steps; the same source, mask and generation controls serve both.

Use **Latest generated** to open a completed image, or **Choose from Gallery** and
select its tile. An empty Latest selection keeps the image already open. The
**Image tools** shortcuts open the existing face-landmark, pose and depth controls.
With automatic size and Preserve framing, prompt edits and inpainting keep the
source dimensions; an explicit size, Pad or Crop remains your choice.

SAM segmentation keeps the photo open. Choose returned instances under **Objects**
and use **Use result as selection mask** to attach their combined mask for
inpainting. Cutouts remain separate results and cannot become selection masks.
**Restore cropped face** processes an already cropped face; it does not detect
and replace faces in a full photo.

Choose **Adjust image** for **Transform**, **Mask** or **Color**, then choose its
operation and press **Run image action**. Transform resizes, crops, pads, rotates
or flips. Mask extracts or edits an exact-frame selection. Color blends a
reference, transfers its colors or makes an RGB8 palette. These tools need no
model or prompt. Precision originals use PNG16 or EXR output. After a Mask
result, **Use result as selection mask** restores its original image and attaches
the selection for inpainting; this handoff requires an 8-bit sRGB image and mask.
See [adjustments and precision](integrations/native-image-adjustments.md).

- **Transform → Crop to selection** crops to the exact nonzero bounds of an
  opaque grayscale selection in the source frame, without resizing. An empty
  selection is rejected.
- **Color → Text overlay** adds text to an 8-bit sRGB source, including opacity
  over transparent areas. **Morphology** applies erode, dilate and related RGB
  operations while retaining source alpha. **SDR to HDR** expands SDR brightness;
  choose EXR 16-bit or 32-bit float and the **HDR** output color space in its
  contextual output controls. These controls mirror the existing output settings;
  browsing operations does not change the saved choices. See
  [color operations and precision](integrations/native-image-color-tools.md).

Ordinary Stable Diffusion checkpoints expose **Native layout and attention**.
Both selectors start Off. Expand the section to list local GLIGEN/hypernetwork
files, then use ordered text boxes on a fresh canvas or signed hypernetwork
strength with matching native attention dimensions. See the
[layout and attention guide](integrations/native-layout-attention.md) for the
JSON example, configured folders, compatibility checks and exact replay.

- **WeVisOne detection** draws objects named one per line. **WeVisOne matting**
  accepts a subject description and returns a cutout, grayscale matte or overlay.
  Add one still source, explicitly select a WeVisOne image encoder, and press
  Generate. These tools preserve source size and require no diffusion checkpoint
  or generation prompt. See [detection and matting](integrations/native-wevisone.md).

Compatible Anima, Qwen Image, Z-Image and Flux checkpoints expose **Native model
patch** for local LLLite, DiffSynth, Fun and USO weights. A ControlNet tool also
offers **Native ControlNet recipe** for ordinary Union modes or Qwen AliMama
inpaint/canvas extension. Each named recipe uses its own images and mask.
See [native model patches](integrations/native-model-patches.md) for compatible
files, dedicated inputs, source denoise and Library/API behavior.

- **RGB curves** edits normalized master/red/green/blue point tables. **GLSL image
  effect** selects Invert RGB, BT.709 grayscale or Posterize with an amount slider.
  These explicit source operations preserve size and alpha and accept at most
  4 megapixels and 4096 pixels per side. CPU mode works without GL. Requested
  acceleration falls back to CPU with a terminal warning when native EGL/GLES
  is unavailable. GL/platform equivalence and rendered visual acceptance remain
  pending. See [curves and effects](integrations/native-image-curves-effects.md).

- Choose **Blur image**, **Sharpen image**, **Adjust brightness** or **Adjust
  contrast**, add a source and press Generate to apply that filter. The source
  size and alpha are preserved; no model or prompt is needed. Radius must be
  smaller than both image dimensions. These operations stay off until selected.
  See [native image filters](integrations/native-image-filters.md).

- **Histogram analysis** opens a source-only analysis panel. Press **Analyze
  histogram** to see RGB, luminance and individual channel counts; **Cancel
  analysis** or changing the source discards pending results. This uses stored
  RGB values and counts transparent pixels. It needs no model or prompt and does
  not generate an image. See [native histograms](integrations/native-image-histogram.md).

- **Make image grid** and **Compose images** use ordered image uploads without a
  generation model. Grid fits each whole image into a cell and uses an opaque
  black background. Composition takes a base then an overlay, fits the overlay
  inside its rectangle and supports a white-selects mask or native Porter–Duff
  alpha modes. Select a thumbnail to move it earlier or later. Generate runs once.
  Library's **Start image grid** or **Start image composition** seeds the selected
  original; add further inputs in Editor. Saved defaults and full setup restore
  controls, and require fresh image uploads. See [framing, alpha and limits](integrations/native-image-composition.md).
- **Generation → Size and upscale → Output target** lists the selected model's recommended sizes and
  **Custom...**. Custom opens integer **Target width** and **Target height**, plus **1080p**, **1440p**
  and **4K** shortcuts. **Use model recommended size** restores automatic following. A deliberate size
  survives model refreshes and model switches; **Size details** identifies the recommendation source.
  **Save size** stores a named size for the current profile and image model family.
  Existing profiles retain their saved size; **Save current defaults** remembers the follow setting too.
  New PNGs retain final target intent; older PNGs retain their recorded generation size. For image to image and native editing, the
  source-frame note shows how **Image framing** uses the selected pixel budget; denoise and framing stay
  under your control.
- Image to image with a Denoise slider, and **Native image edit** on checkpoints that can edit images
  (Qwen-Image, Mage Flow, Fibo 2, HiDream O1, MOSS with an edit format, Flux Kontext). Adding an image picks the
  right one automatically; on other checkpoints it falls back to image to image.
- With Qwen Image 2.1, load the source in **Main** and turn on **Localized edit**.
  Paint the region red and describe the change. **Paint guide** shows the region to Qwen;
  **Mask** also preserves the source outside the brush stroke. See
  [localized edits and native background removal](integrations/native-special-image-workflows.md#qwen-image-21-localized-edits-and-background-removal).
- Identified Qwen Image Layered checkpoints expose **Qwen Layered** with an optional RGBA source
  and **Output layers**. **Image count** still controls independent generations. Use PNG or EXR,
  keep automatic upscaling off and describe the overall scene. See the
  [Layered workflow](integrations/native-qwen-layered.md) for output order and replay.
- **ControlNet / T2I adapter** uses a supplied guide; Canny, Depth, Sketch and Recolour keep their
  existing processors and strength/start/end controls. Select a compatible local file under the tool.
  **Automatic** retains the stock SDXL control presets. Other supported native families require an
  explicit local model. Files with unknown compatibility are labelled and require deliberate selection;
  incompatible files are excluded. Refresh lists newly installed files. Saved tool presets and Library
  metadata retain the selected file; changed or missing files need a new selection. Inpainting and
  background removal remain separate tools.
- For Lotus depth, choose **Depth → Estimate from photo → Depth model → Lotus**,
  then select an installed Lotus checkpoint and compatible SD VAE. The fixed native
  recipe estimates a white-near depth map; input dimensions must be divisible by eight.
  **Use supplied depth map** skips estimation. DA3 remains available from the same selector.
- **Lotus depth** and **MoGe depth** create standalone relative depth maps.
  MoGe also offers **Surface normals** with OpenGL or DirectX convention for
  perspective images, and **360° panorama** depth for equirectangular sources.
  Panorama supports depth only. **Face landmarks** offers **Landmark overlay**
  or **Face region mask**. Select **All face regions** or **Custom regions** to
  build a mask; **Use result as selection mask** restores the original image and
  its alpha with the retained soft selection. This handoff requires an 8-bit sRGB
  source and mask. **Whole-body pose** draws
  an SDPose map for one person, and **Marigold albedo** estimates surface color
  without lighting. Select the installed components shown for the chosen tool.
  These tools preserve the source image size and alpha and use their own model,
  independently of the generation checkpoint, prompt and sampler. Marigold needs
  its dedicated albedo LoRA, VAE and saved conditioning with a compatible Qwen
  Image Edit model. See [still geometry utilities](integrations/native-still-geometry-utilities.md)
  for component folders, API fields, replay and qualification limits.
- Rembg can use an installed native BiRefNet model selected under **Background model**. Leave it on **Hardware background removal (ONNX)** for the existing default. Native models use the image device; install them locally in your configured `background_removal` folder. Existing alpha is kept unless **Reprocess transparent images** is enabled.
- Upscaling is in **Generation → Size and upscale → Upscale**. Choose **2×** or **4×** to filter models by their
  actual native factor; **All native factors** exposes compatible specialist factors. Selecting a model
  can acquire its registered file on first use. The target is the final file size: 1920 × 1080 at 2×
  generates 960 × 540 content, removes only native alignment padding, then enlarges it to Full HD.
  Odd targets use uniform fitting and padding to retain every edge. **Effective stages** shows the
  extra geometry. This is separate from latent/diffusion upscale workflows.
  **Input image only** skips generation and preserves source size × factor while recommended-size
  following is enabled. Choose a target manually to uniformly fit the upload with padding; alpha and
  the original upload remain intact. Masked edits and source-preserving generation keep their own geometry.
  **Saved settings (PNG)** is at the bottom of Generation; its active-preset notice stays near the model.
- **Detect objects** uses an installed RT-DETR v4 checkpoint to draw object boxes.
  **Segment by points** uses SAM 3 or 3.1: click the input image to add points to
  keep or exclude, then choose a transparent cutout or black-and-white mask.
  **Undo last point** and **Clear points** affect only the markers. The separate
  marker preview leaves the input pixels intact; replacing the input clears its
  points. Advanced coordinates are source-image pixels, with at most 256 points
  of each kind. These tools use the selected local utility model, independently
  of the generation model, prompt and sampler. They do not download weights.
  **Refresh All Files** keeps the selected utility and its contextual controls.
- A recognized native SD4x diffusion checkpoint exposes **Image tools → Diffusion 4× upscale**.
  Add its source there; the output-size note shows four times the source dimensions. Resolution and framing
  controls stay read-only for this operation. Transparent sources keep their alpha, resized uniformly over
  generated RGB. This specialized model workflow is separate from the ordinary pixel upscaler.
  For its official full `.ckpt`, first use **Models → Installed → Model → Checkpoint tools → Inspect**; its workflow controls refresh automatically.
  Inspect verifies the published source checksum; it does not scan tensor integrity. Its encoder and VAE
  appear as built in, so separate component downloads are unnecessary.
- Drop an image on the main image with an empty prompt to reuse its saved prompt text, or
  caption it with the configured interrogator if it has no usable prompt metadata. An existing
  prompt stays unchanged. Type `clip:`, `florence:` or `brainblip:` before uploading to request
  that captioner explicitly.

## Generations and metadata

- **Image tools → Image prompt adapter** adds a separate reference image: StyleAdapter for SD1,
  Redux for Flux, or PhotoMaker for SDXL. Choose installed local adapter/vision files; no download
  starts here. PhotoMaker needs a prompt such as `portrait of person photomaker`. This reference
  does not select image-to-image or change your canvas. Library replay asks for a new reference.
  See the [native adapter guide](integrations/native-reference-adapters.md) for supported files.
- **Image tools** shows contextual options for native special workflows. Stable Cascade uses your
  selected Stage C model and a local bundled Stage B checkpoint; blank Stage B sampling fields use
  its verified official recipe when available, otherwise enter the values explicitly. Hunyuan image
  refiner and pixel diffusion use **Native image edit** and a source image. Pixel diffusion also needs
  the source latent format and a compatible local input VAE. These controls do not download models.
- For ordinary native Stable Diffusion checkpoints, **Image tools → Local checkpoint recipe** selects
  an optional YAML file from your configured checkpoint folders. **Current Clip Skip** remains the
  default; **Selected recipe** explicitly delegates layer selection to the file. Editing Clip Skip or
  choosing a PNG preset restores Current Clip Skip. Choose None, then select a deliberately edited
  recipe again to accept its new contents; replay reports changed files rather than silently using them.

- **Generation → More options → Saved settings (PNG)** reuses a RuinedFooocus PNG's model/LoRAs, sampling and/or size for each
  generation while keeping your current prompt. Clear the preset to return to your controls. Choose the
  groups in **Settings → Generation defaults → Saved settings (PNG)**; **Save current defaults** instead remembers
  your choices for the next launch.
- Full generation settings are embedded in each PNG. Uploading one on the main image fills an
  empty prompt with its rendered prompt, falling back to its original prompt when needed; it
  does not change the selected model or other controls. Plain-text PNG parameters remain text.
  Use **Saved settings (PNG)** or Gallery's **Reuse full setup** to restore settings deliberately.
  **Details** shows the metadata; pasting its JSON into the prompt box also reuses the saved setup.
- `Ctrl+Enter` generates, `Esc` toggles edit mode, **Cancel** stops, and Image number `0` generates forever.
  A seed of `-1` is random.
- **Gallery** searches earlier generations and keeps every output accessible. The grid uses cached WebP
  thumbnails for large still images; metadata and editing resolve the selected original. The compact pager
  shows current/total pages and disables unavailable directions. Select an image to work in its separate
  **Gallery draft**. **Use prompt** and **Reuse full setup** fill that local draft; missing or mismatched
  resources are reported. If a saved setup no longer matches the selected image, reuse its setup or choose
  **Use current sampling settings** to clear the saved setup while keeping local text/source.
  **Edit image**, **Inpaint**, **Upscale** and **Remove background** select local
  actions. Change the local prompt/model/tool settings and press **Run** explicitly. Its result and original
  downloads remain in Gallery, and its **Stop** affects only that job. Browsing another image prepares a
  separate next draft without cancelling the accepted job. **Open result in Editor** opens its completed
  image history. **Open in Editor**, **Start image grid**, **Start image composition** and **Continue in Create**
  remain explicit navigation choices. Create keeps its current draft while you work locally in Gallery.
  The separate **Slideshow** fits complete images within the screen. Thumbnail size and cleanup are under
  [Settings → Maintenance → Cache storage](settings.md#cache-storage).
- **Editor → Image workspace** offers **Open current image**, **Latest generated** and **Choose from Gallery**.
  Current adopts the displayed still. Latest imports the accepted final generation/edit image with its selected
  batch identity; maps, masks, helpers and previews do not replace that shortcut. Those outputs remain accessible
  in Gallery and their explicit guide actions. Choose from Gallery opens a bounded picker inside Editor, using
  the original file rather than thumbnail pixels. Changed or missing selections must be chosen again.
  **Gallery → Open in Editor** restores the selected original, chosen rendered/original prompt and available
  saved model through the draft guard. Auto sizing uses the original dimensions; manual targets remain yours.
  **Undo**, **Redo** and **Reopen image history** move through completed results without regeneration.
  Source/history adoption defers while an accepted job owns the Editor terminal: finish it or use **Stop**,
  then retry. Typing changes the next draft without cancelling the captured job. Failed operations retain the
  current image/history and draft. Precision originals and alpha stay unchanged; tools that require an 8-bit
  sRGB working copy report that conversion.
- To replace a background, open the foreground, choose **Remove background**, then explicitly select
  **Select background** with a background file or **Generate background** with your prompt. Choose
  **Compose foreground and background**, then **Edit selected image** and **Upscale selected image**
  as needed. Each Run performs only the selected step. Background removal and composition keep their
  source canvas; an edit-size change requires explicit Pad or Crop. Upscaling uses both output dimensions.
  **Mask and composition inputs** accepts an exact-frame grayscale mask, union/instance selection and
  explicit inversion; the existing painted inpaint editor also works. A new source/result needs a new mask.
  **Preview image readiness** reports captured resources without downloading or generating.
- Custom generation-complete sounds are currently unavailable.

## Chat

- The conversation and model controls share an adjustable divider when there is room.
  At narrower widths or larger text sizes, the model controls move below the conversation.
  **Choose / load model** opens and focuses the model library in either layout.
- Choose an assistant; **New**, **Edit**, **Delete**, **Refresh** and **Use at startup** remain available.
  Expand **Model library** for Installed/Download, search, Model, exact Quantization, **Load**, **Import GGUF**,
  **Download**, **Cancel download** and **Unload**. Opening the library revalidates external files; ordinary
  search reuses inventory and keeps your selected model when its row is filtered out. **Load**, **Loading…**
  and **Loaded** follow the exact selected file and captured
  initialized runtime, with visible acquisition/probe/load progress, **Cancel loading** and retry. Downloading
  a candidate never activates it or replaces the conversation. Offline operation uses available local resources.
- Local GGUF chat models on llama.cpp, using Vulkan, ROCm, CUDA or CPU (**Settings → Hardware & backends**).
  Explicit CPU requests skip the device-enumeration preflight.
- **Settings → Local Chat → Chat model diagnostics** retains file/source and sampling refresh, last-session
  state and distinct current initialized identity, requested/effective configuration and observed placement.
  Unknown placement stays unavailable. File-presence refresh checks configured paths and sizes without loading
  or downloading weights. Text and automatic vision bundles keep separate missing payload totals; size mismatches
  stay unresolved, and custom/disabled projectors do not inherit automatic bundle totals.
- Use **New** or **Edit** beside the Assistant picker, enter a name and system prompt, then **Save**.
  Saving preserves your current conversation, message draft and model; select the saved assistant to use it.
  Upload an optional assistant image; otherwise the General chat image is used. **Delete** asks for confirmation.
  General chat can be renamed and keeps its protected identity and empty default system prompt.
  Existing `assistants` folders, legacy `chatbots` entries and
  SillyTavern character cards remain available; bundled assistants live under `ruinedfooocus/resources/assistants`.
  Select **General chat** for a conversation without a character persona. See [saved assistants](chat-assistants.md)
  for Save as, startup selection and conflict handling.
- **Image generation**, **Rewrite image prompts** and **Update Create prompt** start Off and are independent.
  Generation permits one image action for the message and resets after submission; **Check image readiness**
  previews resources. Optional **Image prompt to preserve** is an explicit supplied visual prompt; rewriting Off
  preserves it exactly. Empty means a new visual request, without importing Chat prose or an unrelated Create draft.
  Authorized Create updates use the draft guard with pending **Apply result**, **Dismiss result** and safe **Undo**;
  cancellation, changed authority and newer drafts cannot be overwritten. New conversation/assistant clears the
  supplied prompt and source. Code blocks remain readable Chat content.
- **Attach files** displays accepted extensions (`.txt`, `.md`, `.csv`, `.json`, `.pdf`, `.docx` and `.rag` when
  the configured parser supports them), up to four files of 256 KiB each, and `.rag` alone. Ready files keep
  Remove/export. Image references appear for an enabled image action or model vision, with upload only for
  **Uploaded image**; image formats follow the installed decoder's input support.
- Mandatory reasoning models show only supported Effort choices; their inapplicable computation selector is
  hidden. **Show reasoning** is display-only and never disables native reasoning.
- **Settings → Local Chat → Local chat and prompt runtime** has **Enable MoE cache CPU share**
  enabled by default with a saved share of **0.5**. Save and reload Chat to apply changes.
  The selected native executable must support the option; it does not move experts to RAM or choose a cache size.
  Full GPU loads keep the preference dormant. Passing a supported option does not prove CPU cache-miss execution;
  see the [MoE cache contract](integrations/llama-moe-cache.md).

## Settings and access

- Defaults for the UI in `settings/settings.json`, custom resolutions in `settings/resolutions.json`, custom
  performance presets, Gradio themes and [UI translations](localization.md).
- **Create → Save current defaults** stores the selected model, prompt, styles, LoRAs, size, seed and sampling
  choices for the next launch. It never stores uploads or a previous generated image. Clear an active generation
  preset before saving. Prompt rewrite offers Preview, Apply and Undo, protecting later edits to your prompt.
- **Settings → Hardware & backends** lists detected GPUs, selects the compute backend for image generation, Chat (`llama.cpp`) and background removal, and provides a single **Use CPU for everything** switch.
- **Auto** follows the native device, memory and precision policy. Explicit CPU
  choices remain authoritative. Image adjustments, Face and MoGe report why an
  operation uses CPU; supported accelerator failures can retry once with fresh
  CPU data or models. Face and MoGe recovery requires native CPU offloading;
  explicit GPU-only mode can prevent it. Disable GPU-only and use Auto, or restart
  with `--cpu`, then submit again. Recovery does not save a new backend choice. See
  [image utility devices and CPU recovery](integrations/native-image-utility-device-policy.md).
  Captioning and Chat text embeddings explain their existing DirectML CPU route
  separately in [auxiliary fallback reasons](integrations/auxiliary-device-fallback-reasons.md).
- Model weights load on demand. **Settings → Image runtime → Image compute → Preload selected image model** optionally
  loads the selected image model during startup.
- `--listen` opens the UI on your network and requires `--auth user/password` (or `RF_UI_AUTH`). `--guest`
  starts a restricted, generation-only page for other people. `--nobrowser` skips opening a browser.
- `--api` enables an OpenAI-compatible API; set `--api-key` (or `RF_API_KEY`), which is required when listening
  beyond localhost. `--mcp` starts an MCP server.
- Reinstall, repair and PyTorch freeze markers are described in [runtimes and troubleshooting](runtime-and-troubleshooting.md).

## Preview without generating

1. **Random Prompt:** open Create → Prompt tools → Random Prompt and click **Generate prompt**, with an optional subject. Its quick defaults do not reuse configurable One Button state or call Chat. Prompt options contains seed and append mode. Undo restores your previous draft.
2. **One Button Prompt:** choose a preset, optionally enter a subject and click **Generate prompt**. Choose Custom for advanced controls; Format and enhancement contains optional model-aware format and local Chat enhancement. If you edited the draft while a result was being built, it appears as Pending prompt with Apply result or Dismiss result. The main **Generate** remains the separate image action. Applying a generated draft turns off automatic prompt replacement so the image uses that draft.
3. **Styles:** choose a category and real styles (headings cannot be selected), Preview, then Bake into prompt if desired. Inspect the result before Generate.
4. **Variations:** put up to nine parent prompts in the prompt box separated by `---`, choose Use current prompts as parents, select a parent and Preview variations. Select an indexed candidate; Apply changes text only.
5. **Gallery:** select a result and use Original/Rendered prompt or Reuse full setup in its local draft; Create stays unchanged. Choose a local action without running inference during navigation-only checks. Open in Editor explicitly and check original dimensions/history. In Editor, check Latest generated and Choose from Gallery, including a technical output that must not replace Latest. Page/search/refresh invalidate Gallery selections; browsing never means Stop. PNG metadata cannot reconstruct absent uploads or masks.
6. **Editor → More tools:** choose Image to image, input a reference and set denoise/framing. Preserve source proportions; explicit padded sizing fits uniformly. The tool is configured now; Generate executes it later.

Random Prompt, One Button, Random Style, seed randomness, wildcards, prompt variations and optional LLM enhancement are distinct operations. See [prompt data](prompt-data.md) for their data ownership. No preview action is a claim of final image quality.
