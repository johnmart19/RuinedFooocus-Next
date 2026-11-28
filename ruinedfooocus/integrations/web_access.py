"""Access controls shared by the owner UI and restricted guest server."""

import os
import hashlib
from ipaddress import ip_address


def validate_web_access(args):
    args.auth = args.auth or os.environ.get("RF_UI_AUTH")
    if args.auth:
        username, separator, password = args.auth.partition("/")
        if not separator or not username.strip() or not password.strip():
            raise SystemExit(
                "Set RF_UI_AUTH or --auth to username/password with both fields nonempty."
            )
    host = args.listen or os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    try:
        local = ip_address(host.strip("[]")).is_loopback
    except ValueError:
        local = host.lower() == "localhost"
    if (not local or args.guest) and not args.auth:
        raise SystemExit(
            "LAN/guest access requires a UI login. Set RF_UI_AUTH=username/password before launching."
        )
    if args.share:
        raise SystemExit(
            "Public Gradio tunnels are disabled. Use authenticated --listen access on your trusted LAN or VPN."
        )
    if args.guest and (args.api or args.mcp):
        raise SystemExit(
            "Guest mode does not expose the owner API or MCP. Start without --api and --mcp."
        )


def login_token(request):
    app = request.app
    token = request.cookies.get(f"access-token-{app.cookie_id}") or request.cookies.get(
        f"access-token-unsecure-{app.cookie_id}"
    )
    return token if token and app.tokens.get(token) is not None else None


def require_ui_login(request):
    from fastapi import HTTPException

    if request.app.auth is not None and not login_token(request):
        raise HTTPException(401, "UI login required.")


def authenticated_owner(request):
    """Opaque job ownership derived only from a currently valid server login."""
    app = getattr(request, "scope", {}).get("app") if request is not None else None
    if app is None or not hasattr(app, "cookie_id") or not hasattr(app, "tokens"):
        return None
    token = login_token(request)
    if token is None:
        return None
    return ("ui-login", str(app.tokens[token]), hashlib.sha256(token.encode()).hexdigest())


class PrivateResponses:
    """Do not cache private responses; guest media uses its own checked route."""

    def __init__(self, app, guest=False):
        self.app = app
        self.guest = guest

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        from starlette.requests import Request
        from ruinedfooocus.app.resource_lifecycle import (
            _REQUEST_OWNER,
            registry,
            latest_ui_requests,
        )

        owner = authenticated_owner(Request(scope))
        if self.guest and path.startswith(
            ("/gradio_api/file", "/gradio_api/proxy", "/gradio_api/upload", "/gradio_api/stream/")
        ):
            from starlette.responses import Response

            return await Response(status_code=404, headers={"Cache-Control": "no-store"})(
                scope, receive, send
            )

        owner_context = _REQUEST_OWNER.set(owner)

        async def private_send(message):
            if message["type"] == "http.response.start":
                from starlette.datastructures import MutableHeaders

                headers = MutableHeaders(scope=message)
                headers["Cache-Control"] = "private, no-store"
                headers["Referrer-Policy"] = "no-referrer"
                headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
                if (
                    owner is not None
                    and path.rstrip("/").endswith("/logout")
                    and message.get("status", 500) < 400
                ):
                    registry.cancel_owner(owner)
                    latest_ui_requests.cancel_owner(owner)
            await send(message)

        try:
            await self.app(scope, receive, private_send)
        finally:
            _REQUEST_OWNER.reset(owner_context)
