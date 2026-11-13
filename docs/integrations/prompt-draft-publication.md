# Prompt draft publication

Gradio 6.31.0 owns the components, events and existing compute admission.
`features/prompting/ui_draft_fence.py` adds no worker, model loader or queue.
It retains each tool's server validation and routes its output through one hidden
JSON envelope. Only the following browser-only success callback owns the prompt,
undo record and related checkbox/count updates.

The browser captures the current textarea text and revision before validation.
Input events advance the revision even if a user later restores identical text.
Visible context fields also receive input/change revisions. Before Gradio paints
a validated result, the synchronous client callback checks the live text, revision,
context and textarea-wide request serial. Starting a different prompt tool also
supersedes an older action on that same draft. A superseded response cannot publish; an intervening
edit leaves the current controls unchanged. Random Prompt and One Button retain a
pending suggestion that can be explicitly applied or dismissed. A failed validation
does not trigger the client success callback.

Undo records that the client updates use hidden JSON components, since a
browser-only callback cannot mutate Gradio's server-only `State`. The record carries
the applied browser revision. An edit away and back still invalidates Undo, even
when the final text matches the applied text. State stays inside this browser's
component/DOM memory; the fence does not write localStorage, files or telemetry.

Existing model/reference validation remains with the tool. Chat rewrite and
official Qwen enhancement remain separate engines. Qwen's server validation still
checks checkpoint, operation and reference identity before applying a preview.
Style bake consumes the existing style selection once; Variations retains explicit
reviewed image admission. That admission counter travels with the fenced prompt
and count update, so rejection cannot start an image request.

Focused source/client checks:

```sh
python tools/check_cpu.py test_prompt_draft_fence.py test_prompt_workflows.py test_style_workflow.py test_qwen_image21_enhancement.py test_prompt_variations.py test_llama_rewrite.py
```

These CPU checks use the production JavaScript in a deterministic Node harness and
real Gradio event construction. They separate server validation from the final
client publication and cover late edits, identical-text restoration, context edits,
out-of-order callbacks, two browser contexts and stale undo. A private no-model
Gradio fixture is used for browser integration; its receipt does not qualify model
execution, visual output or whole-app responsiveness. Independent W11 acceptance
still checks the full application and actual viewport/text-size matrix.
