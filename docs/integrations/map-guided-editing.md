# Editing with owned depth, normal and face maps

Editor treats these maps as editing guides. Creating a technical map does not
establish the quality of a finished image.

After an owned Face region-mask result, **Use result as selection mask** restores
the source and selects Inpaint. The existing exact-frame mask handoff retains the
original pixels and alpha. The prompt remains editable; Run is a separate action.

For an owned perspective MoGe depth/normal or Lotus depth result, explicitly
select a compatible installed ControlNet under **More tools**. Use **Use map for
editing** to restore the source and attach the retained map separately. The
action opens the existing control panel. It preserves the current prompt,
negative prompt, target dimensions and manual control strength/start/end values.
**Edit denoise** is a transient control, initially 0.35; lower values retain more
of the source. Nothing samples or downloads when the guide is displayed.

With automatic resolution and preserve framing, a supported source keeps its
dimensions instead of the model's recommended square. Oversized sources fit the
model's working bound with uniform downscaling and padding, leaving the original
unchanged. For an intentional output-size override, disable automatic resolution
and choose fit/pad framing; the original and guide use the same geometry.

The initial admitted consumers are the exact source-verified SDXL rank128 depth
Control-LoRA and SDXL Union base6 artifacts. A normal guide requires Union's
normal mode and a source-qualified MoGe camera convention. The compatibility
receipt declares scalar channel inversion for that consumer; the backend applies
it once to a fresh guide copy before common framing. Supplied maps are not
estimated again, spatially flipped or copied over the source. Panorama and
unverified source/control combinations fail explicitly.

Both original and map must be owned, exact-frame 8-bit sRGB artifacts. The opaque
server-side handoff captures their separate identities, signatures and convention.
At Run, `EditorWorkspace._capture_guided_request` checks the current handles,
resolves the existing effective PNG checkpoint and freezes a `GuideRequest`
before worker/ticket/plan admission. Component and native admission revalidate
that request. Shared native geometry frames source and guide consistently; output
alpha comes from the original image.

Choosing another Editor action clears the guide's extra-image role without
changing image history. It cannot become a composition foreground. Create also
omits the retained guide and its transient consumer model/mode from its copied
request, while retaining the Editor widgets. A stale, foreign, changed or replaced
source/map/control is refused before source restoration or native admission.
The handoff and denoise do not enter saved startup defaults.

UI ownership is `gallery/ui_guided_results.py` over the existing workspace;
`image_editing/guide_handoffs.py` owns source/map capture and
`guide_control_compat.py` owns the finite consumer contract. The UI's compatibility
preflight runs only on the explicit CTA, never during rendering or file browsing.
The automatic guide projection declares only its five guide controls and nine
source widgets needed to detach a guide and uses `show_progress_on=[]`.
Native ImageEditor snapshot getters
can export canvas blobs and upload them, so hidden view events must avoid retaining
the editor as a progress target. Explicit map/selection actions retain
their complete control outputs and own status target. This avoids broadcasting projection loading
state through unrelated controls and navigation containers in Gradio 6.31,
including when progress is hidden; owner/version fences and source restoration
remain unchanged. `test_guided_result_ui.py` checks the registered output
contract and real Gradio postprocessing as well as attach/detach behavior.
The guide status resolves its existing label and message template for the
request's locale before combining them. Gradio 6.31 translation markers are for
static properties; their `.format()` method does not interpolate placeholders.
The required utility-model picker explains that a model must be selected. A
blank picker remains unselected even when only one compatible file is installed.
`test_guided_result_ui.py` uses installed Gradio callbacks and real Editor
transactions; source-level, CPU and actual model/quality evidence remain distinct.
