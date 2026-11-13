"""Narrow compatibility for the pinned native Gradio frontend; no vendor writes."""

import hashlib
from dataclasses import dataclass
from importlib.metadata import version
from importlib.resources import files
import logging
import threading


@dataclass(frozen=True)
class FrontendContract:
    asset: str
    original_sha256: str
    patched_sha256: str
    before: bytes
    after: bytes


GRADIO_VERSION = "6.32.0"
CONTRACTS = {
    "6.31.0": FrontendContract(
        "Blocks-D1GWYZdr.js",
        "c870d6727be0b3c596cb996e4a7cae42a333ff65bfd17e122c0a560231bc89d1",
        "e00fa530ea3f35046f04669a8c9404dea2a9420a633e2216e723be5b1c27c41c",
        b"if(s.stage===`complete`){n=`closed`,a.forEach",
        b"if(s.stage===`complete`){r.connection_type===`stream`&&(n=`closed`),a.forEach",
    ),
    "6.32.0": FrontendContract(
        "Blocks-WDOEXx6W.js",
        "4b0f4aebda6d990941d1bea997feaa3fbc23f3068334494ff20cf1f6fd37a2cc",
        "8f421d2ad9dbd2df54782338601265da5c2d6ea6f6b53e95dac6bebf2d70d0e8",
        b"if(o.stage===`complete`){e=`closed`,a.forEach",
        b"if(o.stage===`complete`){r.connection_type===`stream`&&(e=`closed`),a.forEach",
    ),
}
_warning_lock = threading.Lock()
_warned = False
_log = logging.getLogger(__name__)


def _patch_blocks(source, contract=None):
    contract = CONTRACTS[GRADIO_VERSION] if contract is None else contract
    if hashlib.sha256(source).hexdigest() != contract.original_sha256:
        raise ValueError("the native Blocks asset hash differs")
    if source.count(contract.before) != 1:
        raise ValueError("the native completion branch does not match exactly once")
    patched = source.replace(contract.before, contract.after, 1)
    if hashlib.sha256(patched).hexdigest() != contract.patched_sha256:
        raise ValueError("the corrected Blocks asset hash differs")
    return patched


def _warn_native(reason):
    global _warned
    with _warning_lock:
        if _warned:
            return
        _warned = True
    _log.warning(
        "Gradio completion compatibility was not applied: %s. "
        "The native frontend is unchanged; review or remove this adapter for the installed Gradio.",
        reason,
    )


class GradioCompletionGuard:
    """Serve one verified native asset through launch's public ASGI middleware hook."""

    def __init__(self, app):
        self.app = app
        self.asset = None
        self.contract = None
        try:
            installed = version("gradio")
            contract = CONTRACTS.get(installed)
            if contract is None:
                raise ValueError(f"unsupported Gradio {installed}")
            source = (
                files("gradio").joinpath("templates/frontend/assets", contract.asset).read_bytes()
            )
            self.asset = _patch_blocks(source, contract)
            self.contract = contract
        except (OSError, ValueError, ModuleNotFoundError) as error:
            _warn_native(error)

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "").removeprefix(scope.get("root_path", ""))
        if (
            self.asset is not None
            and scope["type"] == "http"
            and scope.get("method") in ("GET", "HEAD")
            and path == "/assets/" + self.contract.asset
        ):
            from starlette.responses import Response

            # Both launchers already use PrivateResponses. Keep the asset private
            # here too, and return corrected bytes even with an old native ETag.
            response = Response(
                self.asset if scope["method"] == "GET" else b"",
                media_type="text/javascript",
                headers={
                    "Cache-Control": "private, no-store",
                    # Native outer middleware may encode GET as gzip/Brotli.
                    "ETag": 'W/"' + self.contract.patched_sha256 + '"',
                    "Vary": "Accept-Encoding",
                },
            )
            if scope["method"] == "HEAD":
                # The encoded GET length is determined by native compression.
                del response.headers["Content-Length"]
            return await response(scope, receive, send)
        return await self.app(scope, receive, send)


def normalize_hidden_progress(root):
    """Resolve only hidden callbacks' implicit output scope before native launch."""
    changed = 0
    for function in root.fns.values():
        if function.show_progress == "hidden" and function.show_progress_on is None:
            function.show_progress_on = []
            changed += 1
    if changed:
        root.config = root.get_config_file()
    return changed
