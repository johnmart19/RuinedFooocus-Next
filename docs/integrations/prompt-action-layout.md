# Prompt and Generate action layout

Create sizes the action column from its label instead of a fixed share of the
prompt row. The prompt retains usable editing space; an action that cannot fit
beside it wraps below. On phones the action owns the full row. Labels retain the
normal theme font and can wrap inside the available width, including long
localized labels, without clipping or creating document overflow.

The source contract is `ui/assets/ui.css` on the app-owned `prompt-row` and
`prompt-actions` IDs. Guarded CPU checks cover intrinsic sizing, the Gradio
minimum-width override, bounded long labels and the phone fallback:

```sh
python tools/check_cpu.py test_prompt_action_layout.py test_html_templates.py
```

The tablet `Generat` / `e` split was observed on exact source `f3bcdcda`.
The prompt markup and shared theme CSS are unchanged by that source's card-fill
commit; a visual baseline comparison is separate. Root's browser acceptance after
integration must verify actual label layout, long translations, five viewports
and document overflow. CPU CSS contracts do not substitute for that rendering.

## Shared action and filter rows

Use `elem_classes=["rf-actions"]` on native Gradio action Rows. Their buttons share
the theme, normal text size, wrapping labels, consistent gaps and a minimum height
of `max(44px, 2.75rem)`. Add `rf-actions-compact` for content-width utility actions;
use Gradio's native `variant="primary"` for the main action. Existing Chat, prompt,
Gallery and one-button action selectors retain the same shared rules. The mixed
Settings profile row keeps its bottom-aligned field and content-width actions.

Use `rf-filters` for a Row of related filters. Each filter starts with a readable
12rem cell, bounded to the available width, and wraps below its neighbors. Gradio
6.31.0 groups adjacent input components in a generated Form; the stylesheet sizes
both the Form and its children. Sidebar tab cells also bound their minimum width
to their container. At phone widths, Gallery pagination keeps Previous, the page
input and Next together and places the total on its own row.

```sh
python tools/check_cpu.py test_responsive_ui_layout.py test_prompt_action_layout.py test_compact_model_profile_controls.py test_html_templates.py
```

These checks verify static CSS contracts and native component/layout configuration
without launching a server. They do not measure rendering, document overflow,
actual browser zoom, mobile behavior or visual quality. Those acceptance checks
remain blocked while the in-app browser provider is unavailable.
