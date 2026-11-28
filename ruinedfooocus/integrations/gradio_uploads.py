"""Uploads that arrive twice or stop halfway, without errors or files left behind.

Gradio stores an upload in a folder named after the SHA-256 of its bytes. The
image editor sends the same picture several times at once (for example a blank
layer as both `layer_0.png` and `image.png`). On Windows a file cannot be
renamed over an existing one, so the second arrival falls back to copying over
a file another request has open, fails with WinError 32 and leaves its
temporary file in the upload folder. As the folder name is the content hash, a
file already stored under the same name has the same bytes, and the arrival
can simply be dropped.

A browser that cancels an upload ends the request mid-stream. Gradio then
raises `ClientDisconnect`, which the server logs as an application error, and
keeps the partial temporary file. Nobody is left to answer, so the request ends
quietly and the partial file is removed.
"""

from pathlib import Path
import shutil

_INSTALLED = False


def _same_stored_file(source, destination):
    """True when the destination already holds this upload.

    The destination folder is named after the upload's SHA-256, so a stored
    file of the same size under it is the same content.
    """
    try:
        return destination.is_file() and destination.stat().st_size == source.stat().st_size
    except OSError:
        return False


def _discard(path):
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        # Still open elsewhere; the cache folder is disposable and cleared later.
        pass


def move_to_cache(files, destinations):
    """Gradio's `move_uploaded_files_to_cache`, tolerant of identical uploads."""
    for file, dest in zip(files, destinations, strict=False):
        source, destination = Path(file), Path(dest)
        if _same_stored_file(source, destination):
            _discard(source)
            continue
        try:
            shutil.move(source, destination)
        except OSError:
            # Another request stored the same bytes while this one was moving.
            if not _same_stored_file(source, destination):
                raise
            _discard(source)


def _discard_partial_uploads(parser):
    for file in getattr(parser, "_files_to_close_on_error", ()):
        try:
            file.close()
        except OSError:
            pass
        _discard(file.name)


def install():
    """Patch Gradio's upload handling once; later calls do nothing."""
    global _INSTALLED
    if _INSTALLED:
        return
    import gradio.route_utils as route_utils
    import gradio.routes as routes
    from starlette.requests import ClientDisconnect

    route_utils.move_uploaded_files_to_cache = move_to_cache
    # `routes` imported the function by name before this patch.
    routes.move_uploaded_files_to_cache = move_to_cache
    parser_class = route_utils.GradioMultiPartParser
    original_parse = parser_class.parse

    async def parse(self):
        try:
            return await original_parse(self)
        except ClientDisconnect:
            _discard_partial_uploads(self)
            raise

    parse.__wrapped__ = original_parse
    parser_class.parse = parse
    _INSTALLED = True


class UploadGuard:
    """ASGI middleware: a request whose browser went away ends without an error.

    Constructing it installs the upload fixes above, so every launcher that
    lists this middleware gets both.
    """

    def __init__(self, app):
        install()
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        from starlette.requests import ClientDisconnect

        try:
            await self.app(scope, receive, send)
        except ClientDisconnect:
            # The browser cancelled the request; there is no one to answer.
            return
