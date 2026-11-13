"""Private owner Chat attachment controls and transient source display."""

from html import escape

import gradio as gr
from gradio.context import LocalContext

from ruinedfooocus.app.resource_lifecycle import current_owner
from ruinedfooocus.core.translation_manager import (
    localize,
    localized_output,
    resolve_locale,
    translate_ui,
)
from ruinedfooocus.features.chat.attachments import (
    AttachmentSessions,
    MAX_FILES,
    MAX_FILE_BYTES,
    MAX_MANIFEST_BYTES,
    attachment_capabilities,
)
from ruinedfooocus.integrations.web_access import authenticated_owner


_PICKER_JS = """
const conversation = element.closest("#chat-conversation");
if (!conversation.dataset.attachmentKey) {
    conversation.dataset.attachmentKey = Array.from(crypto.getRandomValues(new Uint8Array(16)),
        value => value.toString(16).padStart(2, "0")).join("");
}
const status = () => element.querySelector("[role=status]");
element.addEventListener("click", event => {
    if (event.target.matches("button")) element.querySelector("input").click();
});
element.addEventListener("change", async event => {
    const input = event.target;
    if (input.type !== "file") return;
    const files = Array.from(input.files);
    input.value = "";
    if (!files.length) return;
    const list = conversation.querySelector("#chat-attachment-list [data-revision]");
    const revision = Number(list?.dataset.revision || 0);
    const manifest = files.some(file => file.name.toLowerCase().endsWith(".rag"));
    if (manifest && files.length !== 1) {
        status().textContent = props.manifest_alone;
        return;
    }
    const limit = manifest ? props.max_manifest_bytes : props.max_file_bytes;
    if (files.length > props.max_files || files.some(file => file.size > limit)) {
        status().textContent = manifest ? props.manifest_too_large : props.limit_message;
        return;
    }
    const button = element.querySelector("button");
    button.disabled = true;
    status().textContent = `${props.reading} ${files.map(file => file.name).join(", ")}`;
    try {
        const read = file => new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onerror = reject;
            reader.onabort = reject;
            reader.onload = () => resolve({name: file.name, data: reader.result.split(",", 2)[1]});
            reader.readAsDataURL(file);
        });
        trigger("upload", {key: conversation.dataset.attachmentKey, revision,
            files: await Promise.all(files.map(read))});
    } catch (_) {
        status().textContent = props.read_error;
    } finally {
        button.disabled = false;
    }
});
"""

_LIST_JS = """
const downloadButton = () => element.querySelector("button[data-export]");
const status = () => element.querySelector("[role=status]");
element.addEventListener("click", async event => {
    const button = downloadButton();
    if (event.target !== button) return;
    button.disabled = true;
    status().textContent = "";
    const revision = props.value.revision;
    try {
        const {manifest, error} = await server.export_manifest({
            key: element.closest("#chat-conversation").dataset.attachmentKey, revision
        });
        if (revision !== props.value.revision) return;
        if (error) {
            status().textContent = error;
            return;
        }
        const blob = new Blob([manifest], {type: "application/json;charset=utf-8"});
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "chat-attachments.rag";
        document.body.append(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (_) {
        status().textContent = props.export_error;
    } finally {
        button.disabled = false;
    }
});
function render() {
    const list = element.querySelector("ul");
    list.replaceChildren();
    list.dataset.revision = props.value.revision;
    for (const file of props.value.files) {
        const row = document.createElement("li");
        const text = document.createElement("span");
        text.textContent = `${file.id} · ${file.name} · ${props.ready}`;
        const remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = props.remove;
        remove.setAttribute("aria-label", `${props.remove} ${file.id} ${file.name}`);
        remove.addEventListener("click", () => trigger("click", {
            key: element.closest("#chat-conversation").dataset.attachmentKey,
            revision: props.value.revision, id: file.id
        }));
        row.append(text, remove);
        list.append(row);
    }
    downloadButton().hidden = !props.value.files.length;
    status().textContent = props.value.status;
    element.closest("#chat-conversation")?.querySelector("#chat-attachment-picker [role=status]")
        ?.replaceChildren();
}
watch(["value", "remove", "ready", "export_label"], render);
render();
"""

_BUTTON_STYLE = """
button { padding: .35rem .75rem; border-radius: var(--radius-md); background: var(--button-secondary-background-fill);
    color: var(--button-secondary-text-color); border: 1px solid var(--border-color-primary); cursor: pointer; }
button:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; }
button:disabled { opacity: .6; cursor: wait; }
[role=status] { overflow-wrap: anywhere; }
"""


def picker_labels(locale):
    capability = attachment_capabilities()
    return {
        "attach": localize("Attach files", locale),
        "formats": capability["label"],
        "help": localize("Select up to four files of 256 KiB each.", locale)
        + " "
        + localize(capability["notes"], locale),
        "reading": localize("Reading files…", locale),
        "limit_message": localize("Select up to four files of 256 KiB each.", locale),
        "manifest_alone": localize("Select one .rag manifest by itself.", locale),
        "manifest_too_large": localize("The .rag manifest is too large.", locale),
        "read_error": localize(
            "Could not read the selected files. Try selecting them again.", locale
        ),
    }


def file_labels(locale):
    return {
        "remove": localize("Remove", locale),
        "ready": localize("Ready", locale),
        "export_label": localize("Export attachments (.rag)", locale),
        "export_error": localize("Could not export the attachments. Try again.", locale),
    }


class AttachmentControls:
    # Insert the tab's capability into the first placeholder of normal Gradio events.
    capture_key_js = """(...values) => [
        document.querySelector("#chat-conversation")?.dataset.attachmentKey || "",
        ...values.slice(1)
    ]"""

    def __init__(self):
        self.sessions = AttachmentSessions()

        async def export_manifest(data, request: gr.Request):
            try:
                # The component-server route injects the actual FastAPI request
                # and has already parsed the same JSON envelope to route it.
                envelope = await request.json()
                session_hash = envelope.get("session_hash")
                if (
                    not isinstance(data, dict)
                    or not isinstance(session_hash, str)
                    or not session_hash
                ):
                    raise PermissionError("The attachment conversation is no longer available.")
                prefix = authenticated_owner(request)
                if prefix is None:
                    if request.app.auth is not None:
                        raise PermissionError("Sign in before exporting attachments.")
                    prefix = ("ui", "", "local")
                session = self.sessions.get(prefix + (session_hash,), data.get("key"), create=False)
                return {"manifest": session.export_manifest(data.get("revision"))}
            except (ValueError, PermissionError) as error:
                return {"error": localize(str(error), resolve_locale(request=request))}

        accepted_files = attachment_capabilities()["accept"]
        self.picker = gr.HTML(
            "",
            html_template=(
                '<button type="button">{{attach}}</button>'
                f'<input type="file" accept="{accepted_files}" multiple hidden>'
                '<small class="attachment-formats" title="{{help}}">{{formats}}</small>'
                '<span role="status"></span>'
            ),
            js_on_load=_PICKER_JS,
            css_template=_BUTTON_STYLE,
            elem_id="chat-attachment-picker",
            max_files=MAX_FILES,
            max_file_bytes=MAX_FILE_BYTES,
            max_manifest_bytes=MAX_MANIFEST_BYTES,
            **picker_labels("en"),
        )
        self.files = gr.HTML(
            self.empty(),
            elem_id="chat-attachment-list",
            html_template=(
                '<ul data-revision="0"></ul><p role="status"></p>'
                '<button type="button" data-export hidden>{{export_label}}</button>'
            ),
            js_on_load=_LIST_JS,
            css_template=_BUTTON_STYLE + "li { display: flex; align-items: center; gap: .5rem; }"
            "li span { overflow-wrap: anywhere; } ul, p { margin: 0; }",
            server_functions=[export_manifest],
            **file_labels("en"),
        )
        localized_output(
            lambda locale: (gr.update(**picker_labels(locale)), gr.update(**file_labels(locale))),
            [],
            [self.picker, self.files],
        )
        self.sources = gr.HTML("", elem_id="chat-attachment-sources")
        self.stop_reply = gr.Button(translate_ui("Stop reply"), visible=False, size="sm")
        for event, operation, key_options in (
            (self.picker.upload, self.attach, {}),
            (self.files.click, self.remove, {}),
            (self.stop_reply.click, self.stop, {"inputs": self.picker, "js": self.capture_key_js}),
        ):
            event(
                operation,
                outputs=[self.files, self.sources],
                queue=False,
                show_progress="hidden",
                api_visibility="private",
                **key_options,
            )

    @staticmethod
    def empty(status=""):
        return {"revision": 0, "files": [], "status": status}

    def session(self, key):
        return self.sessions.get(current_owner(), key)

    def attach(self, evt: gr.EventData):
        data = evt._data if isinstance(evt._data, dict) else {}
        session = None
        try:
            session = self.session(data.get("key"))
            return session.attach(data.get("revision"), data.get("files")), ""
        except (ValueError, PermissionError) as error:
            status = localize(str(error), attachment_locale())
            return (session.view(status) if session else self.empty(status)), gr.skip()

    def remove(self, evt: gr.EventData):
        data = evt._data if isinstance(evt._data, dict) else {}
        session = None
        try:
            session = self.session(data.get("key"))
            return session.remove(data.get("revision"), data.get("id")), ""
        except (ValueError, PermissionError) as error:
            status = localize(str(error), attachment_locale())
            return (session.view(status) if session else self.empty(status)), gr.skip()

    def stop(self, key):
        self.session(key).stop()
        return gr.skip(), ""

    def clear(self, key):
        if not key:
            return self.empty(), ""
        return self.session(key).clear(), ""


def attachment_locale():
    return resolve_locale(request=LocalContext.request.get(None))


def source_summary(product):
    locale = attachment_locale()
    references = product["references"]
    if not references:
        status = (
            localize("Relevant attachment excerpts did not fit the current model context.", locale)
            if product["relevant"]
            else localize("No relevant attachment excerpts were found for this message.", locale)
        )
        return f"<p role='status'>{escape(status)}</p>"
    rows = []
    for ref in references:
        lines = str(ref["start"])
        if ref["end"] != ref["start"]:
            lines += f"–{ref['end']}"
        page = (
            f"{escape(localize('Page', locale))} {int(ref['page'])} · "
            if ref.get("page") is not None
            else ""
        )
        rows.append(
            f"<li>{escape(ref['id'])} · {escape(ref['name'])} · "
            f"{page}{escape(localize('Lines', locale))} {lines}</li>"
        )
    label = escape(localize("Retrieved attachment sources", locale))
    return f"<p>{label}</p><ul>{''.join(rows)}</ul>"
