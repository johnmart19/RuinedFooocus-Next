"""Read-only status and client examples for the mounted owner API."""

from pathlib import Path
from pprint import pformat
from urllib.parse import urlsplit

import gradio as gr

from ruinedfooocus.core.translation_manager import localize, resolve_locale, translate_ui as t


def api_base_url(local_url):
    try:
        url = urlsplit(local_url)
        host, port = url.hostname, url.port
    except ValueError:
        return ""
    if url.scheme not in ("http", "https") or not host:
        return ""
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    if ":" in host:
        host = f"[{host}]"
    address = host if port is None else f"{host}:{port}"
    return f"{url.scheme}://{address}{url.path.rstrip('/')}/v1"


def request_example(base_url, route, payload, requires_key, *, mask=False):
    lines = ["import json", "from urllib.request import Request, urlopen"]
    if mask:
        lines += ["from base64 import b64encode", "from pathlib import Path"]
    headers = {"Content-Type": "application/json"}
    if requires_key:
        headers["Authorization"] = "Bearer YOUR_API_KEY"
    lines += ["", f"payload = {pformat(payload, sort_dicts=False, width=68)}"]
    if mask:
        lines += [
            'payload["input_image"] = b64encode(Path("source.png").read_bytes()).decode()',
            'payload["mask"] = b64encode(Path("mask.png").read_bytes()).decode()',
        ]
    lines += [
        f"headers = {pformat(headers, sort_dicts=False, width=68)}",
        f"request = Request({base_url + route!r}, json.dumps(payload).encode(), headers)",
        "with urlopen(request) as response:",
    ]
    if payload.get("stream"):
        lines += ["    for line in response:", "        print(line.decode().rstrip())"]
    else:
        lines.append("    print(json.load(response))")
    return "\n".join(lines)


def _initialized_model_path(kind, pipeline):
    if kind == "chat":
        paths = getattr(pipeline, "loaded_paths", ())
        if getattr(pipeline, "llm", None) is not None and paths:
            return paths[0]
    elif getattr(pipeline, "xl_base", None) is not None and getattr(pipeline, "xl_base_hash", None):
        signature = getattr(pipeline, "base_file_signature", None)
        if signature:
            return signature[0]
    return None


def _published_model_id(path, kind, catalogue):
    try:
        path = Path(path).resolve()
    except (OSError, RuntimeError):
        return None
    for model_id, row in catalogue.items():
        if row["kind"] == kind:
            try:
                if Path(row["path"]).resolve() == path:
                    return model_id
            except (OSError, RuntimeError):
                continue
    return None


def model_status(backend, service_pipeline):
    """Project the existing inventory and worker state to public model IDs only."""
    catalogue = backend.models()
    results = {}
    for kind in ("chat", "image"):
        available = tuple(sorted(model for model, row in catalogue.items() if row["kind"] == kind))
        default = None
        if available:
            try:
                default, _ = backend.resolve_model("default", kind)
            except ValueError:
                pass
        loaded_path = _initialized_model_path(kind, service_pipeline(kind))
        active = _published_model_id(loaded_path, kind, catalogue) if loaded_path else None
        results[kind] = (available, default, active, bool(loaded_path))
    return results


def _refresh(locale):
    from ruinedfooocus.app import state as shared

    requires_key = getattr(
        getattr(shared.server_app, "state", None), "rf_openai_api_requires_key", None
    )
    if requires_key is None:
        return (
            localize(
                "API disabled for this launch. Use Advanced local connections and restart to enable it.",
                locale,
            ),
            gr.update(visible=False),
            "",
            "",
            [],
            "",
            "",
            "",
            "",
        )

    from ruinedfooocus.app import api_runtime
    from ruinedfooocus.app.resource_lifecycle import service_pipeline

    status = localize(
        "API enabled. Bearer key required."
        if requires_key
        else "API enabled. No bearer key required.",
        locale,
    )
    base_url = api_base_url(shared.local_url)
    rows = []
    model_ids = []
    try:
        models = model_status(api_runtime, service_pipeline)
        for kind, title in (("chat", "Chat"), ("image", "Images")):
            available, default, active, initialized = models[kind]
            model_ids.extend(available)
            rows.append(
                [
                    localize(title, locale),
                    str(len(available)),
                    default or localize("Default unavailable", locale),
                    active
                    or localize(
                        "Outside API inventory" if initialized else "No initialized model", locale
                    ),
                ]
            )
    except (OSError, ValueError, RuntimeError):
        status += " " + localize("Model inventory unavailable. Refresh to try again.", locale)

    links = " · ".join(
        f"[{localize(label, locale)}]({url})"
        for label, url in (
            ("API reference", "/v1/docs"),
            ("OpenAPI schema", "/v1/openapi.json"),
            ("Image tools schema", "/tools/openapi.json"),
        )
    )
    examples = (
        (
            request_example(
                base_url,
                "/chat/completions",
                {
                    "model": "chat/default",
                    "messages": [{"role": "user", "content": "Explain why leaves change color."}],
                    "stream": True,
                },
                requires_key,
            ),
            request_example(
                base_url,
                "/images/generations",
                {
                    "model": "image/default",
                    "prompt": "A watercolor of a mountain lake at sunrise",
                    "response_format": "url",
                },
                requires_key,
            ),
            request_example(
                base_url,
                "/images/generations",
                {
                    "model": "image/default",
                    "prompt": "Add clouds within the editable area",
                    "inpaint_denoise": 0.65,
                    "response_format": "url",
                },
                requires_key,
                mask=True,
            ),
        )
        if base_url
        else ("", "", "")
    )
    return status, gr.update(visible=True), base_url, links, rows, "\n".join(model_ids), *examples


def create_api_workspace(tab):
    with gr.Group(elem_id="local-api-workspace"):
        status = gr.Markdown(t("Checking API status…"))
        refresh = gr.Button(t("Refresh API status"), size="sm")
        with gr.Column(visible=False, min_width=0) as enabled:
            base_url = gr.Textbox(label=t("API base URL"), interactive=False, buttons=["copy"])
            links = gr.Markdown()
            models = gr.Dataframe(
                headers=[
                    "Model role",
                    "Available",
                    "Default API model",
                    "Initialized runtime model",
                ],
                column_count=4,
                interactive=False,
                wrap=True,
            )
            with gr.Accordion(t("Available model IDs"), open=False):
                model_ids = gr.Textbox(
                    label=t("Chat and image model IDs"),
                    interactive=False,
                    lines=5,
                    max_lines=8,
                    buttons=["copy"],
                )
            with gr.Accordion(t("Request examples"), open=False):
                gr.Markdown(
                    t(
                        "Use a model ID from the API inventory if a default is unavailable. An initialized image model may be offloaded."
                    )
                )
                chat = gr.Code(
                    label=t("Chat stream"),
                    language="python",
                    interactive=False,
                    wrap_lines=True,
                    buttons=["copy"],
                )
                images = gr.Code(
                    label=t("Generate an image"),
                    language="python",
                    interactive=False,
                    wrap_lines=True,
                    buttons=["copy"],
                )
                gr.Markdown(
                    t(
                        "Masked editing needs an SDXL or Flux Inpaint model and a same-size PNG mask with alpha. Transparent pixels are editable."
                    )
                )
                masked = gr.Code(
                    label=t("Edit with a mask"),
                    language="python",
                    interactive=False,
                    wrap_lines=True,
                    buttons=["copy"],
                )
            from ruinedfooocus.ui.api_activity import create_api_activity_controls

            create_api_activity_controls(tab)

    outputs = [status, enabled, base_url, links, models, model_ids, chat, images, masked]

    def show(request: gr.Request):
        return _refresh(resolve_locale(request=request))

    def show_when_selected(request: gr.Request, event: gr.SelectData):
        return show(request) if event.selected else tuple(gr.skip() for _ in outputs)

    options = dict(outputs=outputs, queue=False, api_visibility="private", show_progress="hidden")
    tab.select(show_when_selected, **options)
    refresh.click(show, **options)
