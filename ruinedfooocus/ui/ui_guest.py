from ruinedfooocus.core.translation_manager import catalogue, locale_change_js, translate_ui

"""Generation-only LAN interface, with media scoped to the browser's login."""

import html
from pathlib import Path
import queue
import secrets
import time

import gradio as gr
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from starlette.middleware import Middleware

import ruinedfooocus.app.state as shared
import version
from ruinedfooocus.integrations.web_access import PrivateResponses, login_token, validate_web_access
from ruinedfooocus.ui import html as app_html
from ruinedfooocus.ui.gradio_frontend import GradioCompletionGuard, normalize_hidden_progress


class GuestMedia:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.entries = {}

    def add(self, path, owner):
        path = Path(path).resolve()
        if not owner or not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Invalid guest result.")
        token = secrets.token_urlsafe(32)
        self.entries[token] = (path, owner, time.monotonic())
        while len(self.entries) > 256:
            self.entries.pop(next(iter(self.entries)))
        return token

    def get(self, token, owner):
        entry = self.entries.get(token)
        if (
            not entry
            or not owner
            or not secrets.compare_digest(entry[1], owner)
            or time.monotonic() - entry[2] > 86400
            or not entry[0].is_file()
        ):
            raise HTTPException(404, "Result unavailable for this login.")
        return entry[0]


def launch_guest(args):
    validate_web_access(args)
    from ruinedfooocus.ui.themes import resolve_theme, theme_id

    saved_theme = shared.settings.default_settings.get("theme")
    theme, theme_warning = resolve_theme(saved_theme)
    if theme_warning:
        print(theme_warning)
    # Change only this process's output destination, never the owner's saved paths.
    output = (
        Path(shared.path_manager.model_paths["temp_outputs_path"])
        / "guests"
        / secrets.token_hex(16)
    )
    output.mkdir(parents=True, exist_ok=True)
    shared.path_manager.model_paths["temp_outputs_path"] = output
    shared.path_manager.model_paths["temp_preview_path"] = output / "preview.jpg"
    import ruinedfooocus.app.api_runtime as backend
    from ruinedfooocus.integrations.openai_api import ImageRequest

    media = GuestMedia(output)
    models = {key: info for key, info in backend.models().items() if info["kind"] == "image"}
    selected = next(
        (
            key
            for key, info in models.items()
            if info["name"] == shared.settings.default_settings.get("base_model")
        ),
        next(iter(models), None),
    )

    def options(model):
        info = models.get(model)
        if info is None:
            return [], []
        presets = shared.performance_settings.choices_for_model(info["family"], info["name"])
        # This compact view has no custom sampler controls. Prefer a matching
        # family recipe instead of whichever preset happens to be listed first.
        presets = [
            name for name in presets if name != shared.performance_settings.CUSTOM_PERFORMANCE
        ]
        preferred = (
            info["family"]
            if info["family"] in presets
            else shared.settings.default_settings.get("performance")
        )
        if info["kind"] == "image" and preferred in presets:
            presets.remove(preferred)
            presets.insert(0, preferred)
        sizes = shared.resolution_settings.choices_for_model(info["family"])
        sizes = [size for size in sizes if size in shared.resolution_settings.aspect_ratios]
        return presets, sizes

    def change_model(model):
        presets, sizes = options(model)
        return (
            gr.update(choices=presets, value=presets[0] if presets else None),
            gr.update(choices=sizes, value=sizes[0] if sizes else None),
        )

    def generate(prompt, negative, model, preset, size, seed, request: gr.Request):
        owner = login_token(request.request)
        if not owner:
            raise gr.Error("Please log in again.")
        presets, sizes = options(model)
        if preset not in presets or size not in sizes:
            raise gr.Error("Select an available model, performance preset and resolution.")
        width, height = shared.resolution_settings.aspect_ratios[size]
        body = dict(
            prompt=prompt,
            negative_prompt=negative,
            model=model,
            performance=preset,
            size=f"{width}x{height}",
            seed=seed,
            styles=[],
            loras=[],
            n=1,
        )
        try:
            payload = ImageRequest(**body).model_dump(exclude_none=True)
            job = backend.images(payload, models[model])
        except (ValueError, RuntimeError):
            raise gr.Error(
                "Cannot start this request. Check the model and generation settings."
            ) from None
        try:
            yield "", "Generating your result…"
            while True:
                try:
                    kind, value = job.events.get(timeout=1)
                except queue.Empty:
                    continue
                if kind in ("error", "invalid_request"):
                    raise gr.Error("Generation failed. Ask the host to check the application log.")
                if kind != "result":
                    continue
                rendered = []
                for path in value:
                    token = media.add(path, owner)
                    url = html.escape(f"guest-media/{token}", quote=True)
                    rendered.append(
                        f'<img src="{url}" alt="Your generated result" style="max-width:100%;max-height:65vh">'
                    )
                    rendered.append(f'<p><a href="{url}" download>Download your result</a></p>')
                yield "".join(rendered), "Complete. Results are saved on the host."
                return
        finally:
            job.cancelled.set()

    presets, sizes = options(selected)
    with gr.Blocks(
        title=f"RuinedFooocus {version.version} — Guest", analytics_enabled=False
    ) as app:
        gr.Markdown(
            translate_ui(
                "# RuinedFooocus — Guest\nGenerate images. Existing outputs and owner controls are unavailable here."
            )
        )
        with gr.Row():
            with gr.Column(scale=2):
                result = gr.HTML()
                status = gr.Markdown()
                prompt = gr.Textbox(label=translate_ui("Prompt"), lines=4)
                negative = gr.Textbox(label=translate_ui("Negative prompt"), lines=2)
                run = gr.Button(
                    translate_ui("Generate"), variant="primary", interactive=bool(models)
                )
            with gr.Column():
                model = gr.Dropdown(
                    [(info["name"], key) for key, info in models.items()],
                    value=selected,
                    label=translate_ui("Model"),
                )
                preset = gr.Dropdown(
                    presets,
                    value=presets[0] if presets else None,
                    label=translate_ui("Performance"),
                )
                size = gr.Dropdown(
                    sizes, value=sizes[0] if sizes else None, label=translate_ui("Resolution")
                )
                seed = gr.Number(
                    value=-1,
                    minimum=-1,
                    maximum=4294967295,
                    precision=0,
                    label=translate_ui("Seed (-1: random)"),
                )
        model.change(change_model, model, [preset, size], api_visibility="private")
        run.click(
            generate,
            [prompt, negative, model, preset, size, seed],
            [result, status],
            concurrency_limit=1,
            api_visibility="private",
        )
        app.load(
            fn=None,
            inputs=[],
            outputs=[],
            js=locale_change_js(
                args.language or shared.settings.default_settings.get("ui_language", "auto"),
                initialize=True,
            ),
            queue=False,
            api_visibility="private",
        )
    app.queue(max_size=8)
    normalize_hidden_progress(app)
    server, _, _ = app.launch(
        server_name=args.listen,
        server_port=args.port,
        auth=args.auth.split("/", 1),
        share=False,
        inbrowser=not args.nobrowser,
        prevent_thread_lock=True,
        pwa=False,
        allowed_paths=[],
        app_kwargs={
            "middleware": [
                Middleware(PrivateResponses, guest=True),
                Middleware(GradioCompletionGuard),
            ]
        },
        enable_monitoring=False,
        i18n=catalogue().native(),
        theme=theme,
        head=app_html.startup_head(
            app.app_id, themed=not theme_warning and bool(theme_id(saved_theme)), sidebar=False
        ),
        js=app_html.scripts,
    )

    @server.get("/guest-media/{token}", include_in_schema=False)
    def guest_media(token: str, request: Request):
        return FileResponse(media.get(token, login_token(request)))

    try:
        while True:
            time.sleep(100)
    finally:
        app.close()
