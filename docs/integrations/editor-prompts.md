# Editor prompt conditioning

Editor **Edit selected image** captures the current owned image, selected model,
exact instruction, dimensions and denoise before execution. The existing
`image_components.edit_control` selects a native image-edit route when supported,
otherwise ordinary image-to-image. This does not change the selected model.

Native editors receive the exact instruction directly. Ordinary image models,
including SDXL/Janku and Anima, receive the frozen source generation's positive
description followed by the exact instruction. This gives image-to-image useful
scene context without an LLM, captioning pass, download or additional inference.
It does not turn those models into native instruction editors: their ability to
make a requested change still depends on the model and denoise.

`features/image_editing/edit_prompt.py` changes rendered positive conditioning
only for an explicit captured `prompt-edit` workflow using `img2img`, or owned
SD/SDXL/Flux masked inpainting identified by the bound image plan. Qwen Image 2.1
local masked editing still receives its literal instruction. Unknown inpaint
families do not borrow source context. Requested
`prompt`, `raw_prompt` and the operation ticket retain the exact current text.
PNG `image_edit_prompt` metadata records the base source positive, current
instruction, up to three recent prior instructions and the actual route. Repeated
edits reuse that base instead of nesting previously combined prompts. Prior
instructions have a combined 4,096-character budget; whole oldest entries are
dropped. Older changes remain in the current pixels and owned history.

Source context longer than 16,384 characters is omitted with a metadata reason;
the current instruction is never truncated. Missing or empty source descriptions
leave the instruction alone, without inventing a caption. The source recipe stays
in its existing lineage. Its negative prompt, style choices, LoRA selections,
model settings and source handles are not imported or reapplied. Styles and random/enhancement processing remain
off by default for Editor instructions. A later new-image request or native edit
does not use the ordinary image-to-image context descriptor.

Inherited explicit LoRA, style and embedding references and the legacy `<facerestore>` marker remain recorded
as literal source/history text, but are removed from inherited conditioning so
they cannot silently activate extra models. Current instruction directives retain
their existing explicit behavior.
The shared `prompt_resources.RESOURCE` grammar is unchanged from prompt rewrite;
ordinary words and bare embedding trigger words keep their normal meaning.

## Explicit image description

Prompt tools **Describe image** uses the existing queued caption service and the
image description engine selected in Settings. It follows Run's active upload
versus displayed-image choice, freezes a working pixel copy, and validates an
original file again before publishing. Precision preview markers resolve their
actual original. The startup placeholder is not a caption source.

The description replaces the prompt directly and **Undo description** restores
the previous draft while it remains current. The existing browser draft fence
also watches only the two candidate image controls; active-source changes revoke
that caption's image context even when display URLs stay identical. A newer
typed prompt or source wins over a late caption. Uploads never trigger caption
inference, and captioning does not change owned Editor history.

Focused CPU checks:

```sh
python tools/check_cpu.py test_editor_prompt_routing.py test_editor_workspace_actions.py test_library_editing_workflow.py test_image_workflow_handoffs.py test_image_workflow_modes.py test_prompt_processing.py test_metadata_guidance_replay.py test_image_request_isolation.py
python tools/check_cpu.py test_prompt_caption.py test_prompt_draft_fence.py
```

These checks cover request, prompt and metadata contracts with owned tiny images.
Actual model instruction quality, GPU generation and browser behavior require
their own acceptance evidence.
