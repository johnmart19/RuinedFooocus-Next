from ruinedfooocus.core.paths import asset_path
import gradio as gr
from pathlib import Path
from urllib.parse import quote
import ruinedfooocus.app.state as shared
from ruinedfooocus.app.state import (
    state,
    settings,
)
import ruinedfooocus.features.automation.search_pipeline as search_pipeline
from fastapi.responses import Response
from io import BytesIO
from PIL import Image, ImageOps


def image_file_url(filename):
    """Build a Gradio URL without Windows separators or browser dot segments."""
    path = Path(filename).resolve().as_posix()
    return shared.local_url.rstrip("/") + "/gradio_api/file=" + quote(path, safe="/:")


# "secret" pi slideshow
def get_last_image() -> str:
    """
    Internal use for the secret pi slideshow
    """
    global state
    if "last_image" in state:
        return Path(state["last_image"]).resolve().as_posix()
    else:
        return Path(str(asset_path("logo.png"))).resolve().as_posix()


def add_api():

    import ruinedfooocus.app.api_runtime as api_runtime

    def api_llama(system: str, user: str) -> str:
        """
        Ask the local chat model selected in RuinedFooocus for a reply.

        Args:
            system: Instructions for the model, such as its role or answer style.
            user: The message to answer.
        Returns:
            The model's reply text.
        """
        model, selected = api_runtime.configured_chat_model()
        result = api_runtime.wait_for_result(
            api_runtime.chat(
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": settings.default_settings.get("llm_hp_max_tokens", 256),
                },
                selected,
            )
        )
        return result["choices"][0]["message"]["content"] or user

    gr.api(api_llama, api_name="llama")

    def api_generate_image(prompt: str) -> dict:
        """
        Generate one image with the image model and settings selected in RuinedFooocus.

        Args:
            prompt: Description of the image to generate.
        Returns:
            The generated image file.
        """
        files = api_runtime.wait_for_result(api_runtime.legacy_images(prompt))
        return {"path": str(files[0]), "meta": {"_type": "gradio.FileData"}}

    gr.api(api_generate_image, api_name="generate_image")

    # Search
    def api_search(text: str) -> list[str]:
        """
        Search the RuinedFooocus image library by prompt and metadata text.

        Args:
            text: Words to match. Start with "max:N" to return up to N images (default 10).
        Returns:
            URLs of the matching images.
        """
        prompt = f"search: max:10 {text}"
        files = search_pipeline.search(prompt)
        result = []
        for file in files:
            result.append(image_file_url(file))
        return result

    gr.api(api_search, api_name="search")

    def api_version() -> str:
        """
        Look up the running RuinedFooocus version.

        Returns:
            The version string.
        """
        from version import version

        return version

    gr.api(api_version, api_name="api_version")

    # "secret" pi slideshow: callable by name, but not listed for API docs or MCP
    # because it returns a server-side path.
    gr.api(get_last_image, api_name="last_image", api_visibility="undocumented")


def add_fastapi():
    from fastapi import Request
    from ruinedfooocus.integrations.web_access import require_ui_login

    async def last_image(request: Request, w: int = 0, h: int = 0, format: str = "PNG"):
        require_ui_login(request)
        filename = get_last_image()
        from ruinedfooocus.core.cache_contract import configured_cache_root
        from ruinedfooocus.features.image_processing.native_image_formats import display_image

        filename = display_image(filename, configured_cache_root(shared.path_manager))
        img = Image.open(filename)
        ow, oh = img.size
        if w > 0 or h > 0:
            if w == 0:
                w = int(ow * h / oh)
            if h == 0:
                h = int(oh * w / ow)
            img = ImageOps.fit(
                img, (w, h), method=Image.Resampling.LANCZOS, bleed=0.0, centering=(0.5, 0.5)
            )
        if format == "JPEG":
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format=format)
        return Response(
            content=buf.getvalue(),
            media_type=f"image/{format.lower()}",
        )

    shared.server_app.add_api_route("/last_image", last_image, methods=["GET"])
