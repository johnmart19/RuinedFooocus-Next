"""Strict model-output boundary; no resource preparation or compute imports.

The app supplies authority, captured settings and Preview/Run intent. Model
output supplies one instruction and, where applicable, the selected opaque ID.
"""

from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import re
from threading import RLock
from types import MappingProxyType
from uuid import uuid4

MAX_ARGUMENT_BYTES = 32768
MAX_PROMPT_CHARS = 12000
EDIT_OPERATIONS = ("edit", "inpaint", "outpaint")
PROCESS_OPERATIONS = ("upscale", "remove_background", "restore")
_OPAQUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_SOURCE_FIELDS = frozenset(
    (
        "input_image",
        "reference_images",
        "mask_image",
        "mask",
        "image_reference",
        "source_id",
        "mask_id",
        "session_id",
        "editor_session",
        "editor_revision",
        "controlnet",
        "cn_selection",
        "cn_type",
    )
)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return deepcopy(value)


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return deepcopy(value)


class ImageToolValidationError(ValueError):
    def __init__(self, message, code="invalid_tool_call"):
        super().__init__(message)
        self.code = code


def _utf8_length(value):
    try:
        return len(value.encode("utf-8"))
    except UnicodeError as error:
        raise ImageToolValidationError("Image tool text must be valid UTF-8.") from error


@dataclass(frozen=True)
class ChatToolPermissions:
    """Immutable choices plus a server-side, per-browser revocation fence."""

    owner: tuple
    turn_id: str
    enable_image_generation: bool
    rewrite_image_prompt: bool
    update_create_prompt: bool
    prompt_target: str
    original_prompt: str | None
    session: object

    def __deepcopy__(self, memo):
        return self

    def check(self):
        self.session.check(self)


class ChatPermissionSession:
    """A newer turn/revocation invalidates only that authenticated browser owner."""

    def __init__(self):
        self._lock = RLock()
        self._turns = {}

    def __deepcopy__(self, memo):
        # Server-owned identities survive queue snapshots; owners remain isolated.
        return self

    def capture(
        self,
        owner,
        *,
        enable_image_generation=False,
        rewrite_image_prompt=False,
        update_create_prompt=False,
        prompt_target="new",
        original_prompt=None,
    ):
        if not isinstance(owner, tuple) or not owner:
            raise ValueError("Chat permissions require an authenticated app owner.")
        choices = (enable_image_generation, rewrite_image_prompt, update_create_prompt)
        if any(type(value) is not bool for value in choices):
            raise ValueError("Chat permissions must be explicit booleans.")
        if prompt_target not in ("new", "supplied"):
            raise ValueError("Choose a new or explicitly supplied image prompt.")
        if prompt_target == "supplied":
            if (
                not isinstance(original_prompt, str)
                or not original_prompt.strip()
                or len(original_prompt) > MAX_PROMPT_CHARS
                or "\x00" in original_prompt
            ):
                raise ValueError("Supply a bounded nonempty image prompt to preserve.")
            _utf8_length(original_prompt)
        elif original_prompt is not None:
            raise ValueError("A new image request cannot inherit an old image prompt.")
        result = ChatToolPermissions(
            owner, uuid4().hex, *choices, prompt_target, original_prompt, self
        )
        with self._lock:
            self._turns[owner] = result
        return result

    def check(self, permissions):
        with self._lock:
            if self._turns.get(permissions.owner) is not permissions:
                raise ImageToolValidationError(
                    "This Chat turn was superseded or its permissions were revoked.",
                    "revoked_permissions",
                )

    def revoke(self, owner):
        with self._lock:
            self._turns.pop(owner, None)


@dataclass(frozen=True)
class ImageToolContext:
    """Server-created authority, captured once for one Chat turn.

    Source ownership and revision are checked again by the shared image service.
    No public/model DTO may construct this object from supplied owner/settings.
    """

    owner: tuple
    intent: str
    settings: Mapping
    session_id: str | None = None
    source_id: str | None = None
    mask_id: str | None = None
    revision: int | None = None
    edit_operations: tuple = ()
    process_operations: tuple = ()
    inspect_available: bool = False
    grounded: bool = False
    permissions: ChatToolPermissions | None = None

    def __post_init__(self):
        if not isinstance(self.owner, tuple) or not self.owner:
            raise ValueError("Image tools require an authenticated app owner.")
        if self.intent not in ("text", "preview", "run"):
            raise ValueError("Choose text, preview or run in the app.")
        if not isinstance(self.settings, Mapping):
            raise ValueError("Captured image settings must be an object.")
        if self.permissions is not None:
            if (
                not isinstance(self.permissions, ChatToolPermissions)
                or self.permissions.owner != self.owner
            ):
                raise ValueError("Chat permissions must belong to this app owner.")
            self.permissions.check()
        for value in (self.session_id, self.source_id, self.mask_id):
            if value is not None and (
                not isinstance(value, str) or not _OPAQUE_ID.fullmatch(value)
            ):
                raise ValueError("Image references must be opaque app IDs.")
        selected = self.source_id is not None
        if (selected and (self.session_id is None or self.revision is None)) or (
            not selected and (self.session_id is not None or self.revision is not None)
        ):
            raise ValueError("A selected source needs its session and revision.")
        if self.revision is not None and (type(self.revision) is not int or self.revision < 0):
            raise ValueError("Invalid selected image revision.")
        if self.mask_id is not None and not selected:
            raise ValueError("Select an image before selecting its mask.")
        edits, processes = tuple(self.edit_operations), tuple(self.process_operations)
        if set(edits) - set(EDIT_OPERATIONS) or set(processes) - set(PROCESS_OPERATIONS):
            raise ValueError("Unsupported selected image operation.")
        object.__setattr__(self, "settings", _freeze(self.settings))
        object.__setattr__(self, "edit_operations", edits)
        object.__setattr__(self, "process_operations", processes)

    def image_settings(self):
        """Private mutable copy for existing RF request adapters."""
        return _thaw(self.settings)


@dataclass(frozen=True)
class ImageToolRequest:
    call_id: str
    name: str
    arguments: Mapping
    context: ImageToolContext
    runtime_binding: str | None = None

    @property
    def owner(self):
        return self.context.owner

    @property
    def intent(self):
        return self.context.intent

    @property
    def settings(self):
        return self.context.settings

    @property
    def session_id(self):
        return self.context.session_id

    @property
    def source_id(self):
        return self.context.source_id

    @property
    def mask_id(self):
        return self.arguments.get("mask_id")

    @property
    def revision(self):
        return self.context.revision

    @property
    def mutates(self):
        return self.name not in ("inspect_image", "propose_create_prompt")


@dataclass(frozen=True)
class ImageToolContinuation:
    """Worker resumes the same job after leaving Chat compute and preparing."""

    request: ImageToolRequest
    pipeline: object
    resume: Callable[[Mapping], str]
    original_model: str
    original_vision: bool


def _schema(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


_PROMPT = {"type": "string", "minLength": 1, "maxLength": MAX_PROMPT_CHARS}


def image_tool_schemas(context, capability):
    """Only the initialized template and this selected context expose tools."""
    if (
        not isinstance(context, ImageToolContext)
        or context.grounded
        or not isinstance(capability, Mapping)
        or capability.get("available") is not True
        or not capability.get("binding")
    ):
        return []
    policy = context.permissions
    if policy is not None:
        try:
            policy.check()
        except ImageToolValidationError:
            return []
    from ruinedfooocus.features.chat.chat_tools import IMAGE_TOOL

    tools = []
    if policy is not None and policy.update_create_prompt:
        tools.append(
            _schema(
                "propose_create_prompt",
                "Propose one bounded visual prompt for Create without running image generation. Use only for a current request to write or update a prompt. The app owns publication and preserves later draft edits.",
                {"prompt": dict(_PROMPT)},
                ["prompt"],
            )
        )
    if context.intent == "text" or (policy is not None and not policy.enable_image_generation):
        return tools
    tools.append(deepcopy(IMAGE_TOOL))
    if context.source_id is None:
        return tools
    source = {"type": "string", "enum": [context.source_id]}
    edits = tuple(
        value for value in context.edit_operations if value != "inpaint" or context.mask_id
    )
    if edits:
        properties = {
            "prompt": dict(_PROMPT),
            "source_id": source,
            "operation": {"type": "string", "enum": list(edits)},
        }
        if context.mask_id:
            properties["mask_id"] = {"type": ["string", "null"], "enum": [context.mask_id, None]}
        tools.append(
            _schema(
                "edit_image",
                "Edit the currently selected owned image, preserving captured settings and unchanged details. Supply the mask ID for inpaint and null for other edits.",
                properties,
                list(properties),
            )
        )
    if context.process_operations:
        tools.append(
            _schema(
                "process_image",
                "Apply one selected image utility without diffusion or a rewritten prompt.",
                {
                    "source_id": source,
                    "operation": {"type": "string", "enum": list(context.process_operations)},
                },
                ["source_id", "operation"],
            )
        )
    if context.inspect_available:
        tools.append(
            _schema(
                "inspect_image",
                "Ask the selected initialized compatible vision reviewer about actual pixels. Metadata is not vision.",
                {
                    "source_id": source,
                    "question": dict(_PROMPT),
                },
                ["source_id", "question"],
            )
        )
    return tools


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ImageToolValidationError("Duplicate image tool argument.")
        result[key] = value
    return result


def validate_image_tool_call(call, context, capability, *, finish_reason="tool_calls"):
    """Reject incompatible/malformed calls before preparation or dispatch."""
    if finish_reason not in ("tool_calls", "stop"):
        raise ImageToolValidationError("The image tool call did not finish completely.")
    if not isinstance(call, Mapping) or set(call) - {"id", "type", "function"}:
        raise ImageToolValidationError("Expected one complete image tool call.")
    call_id, function = call.get("id"), call.get("function")
    if not isinstance(call_id, str) or not _OPAQUE_ID.fullmatch(call_id):
        raise ImageToolValidationError("The image tool call needs its original call ID.")
    if call.get("type", "function") != "function" or not isinstance(function, Mapping):
        raise ImageToolValidationError("Expected an image function call.")
    if set(function) != {"name", "arguments"}:
        raise ImageToolValidationError("Expected an image tool name and arguments.")
    schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in image_tool_schemas(context, capability)
    }
    name, raw = function["name"], function["arguments"]
    if not isinstance(name, str) or name not in schemas:
        raise ImageToolValidationError("This image tool is unavailable for the selected context.")
    if not isinstance(raw, str) or _utf8_length(raw) > MAX_ARGUMENT_BYTES:
        raise ImageToolValidationError("Image tool arguments must be bounded JSON text.")
    try:
        arguments = json.loads(raw, object_pairs_hook=_object_pairs)
    except (ValueError, TypeError, RecursionError) as error:
        raise ImageToolValidationError("Malformed image tool arguments.") from error
    schema = schemas[name]
    if (
        not isinstance(arguments, dict)
        or set(arguments) - set(schema["properties"])
        or set(schema["required"]) - set(arguments)
    ):
        raise ImageToolValidationError("Unexpected or missing image tool fields.")
    for key, value in arguments.items():
        field = schema["properties"][key]
        nullable_mask = key == "mask_id" and value is None and None in field.get("enum", ())
        if (not isinstance(value, str) and not nullable_mask) or (
            "enum" in field and value not in field["enum"]
        ):
            raise ImageToolValidationError("Invalid selected image reference or operation.")
        if key in ("prompt", "question") and (
            not value.strip() or len(value) > MAX_PROMPT_CHARS or "\x00" in value
        ):
            raise ImageToolValidationError("The image tool needs a bounded nonempty instruction.")
        if key in ("prompt", "question"):
            _utf8_length(value)
    if (
        name == "edit_image"
        and arguments["operation"] == "inpaint"
        and arguments.get("mask_id") != context.mask_id
    ):
        raise ImageToolValidationError("Inpaint requires the currently selected owned mask.")
    if (
        name == "edit_image"
        and arguments["operation"] != "inpaint"
        and arguments.get("mask_id") is not None
    ):
        raise ImageToolValidationError("A mask is only valid for the selected inpaint operation.")
    if name == "generate_image":
        values = {
            key: _thaw(value)
            for key, value in context.settings.items()
            if key not in _SOURCE_FIELDS
        }
        context = replace(
            context, settings=values, session_id=None, source_id=None, mask_id=None, revision=None
        )
    if (
        name in ("generate_image", "edit_image", "propose_create_prompt")
        and context.permissions is not None
    ):
        policy = context.permissions
        if policy.prompt_target == "supplied" and not policy.rewrite_image_prompt:
            arguments["prompt"] = policy.original_prompt
    return ImageToolRequest(call_id, name, _freeze(arguments), context, capability.get("binding"))


def check_tool_permissions(request):
    """Recheck before preparation, execution, retry and final publication."""
    if not isinstance(request, ImageToolRequest):
        raise ImageToolValidationError("A validated app image request is required.")
    policy = request.context.permissions
    if policy is not None:
        policy.check()
        allowed = (
            policy.update_create_prompt
            if request.name == "propose_create_prompt"
            else policy.enable_image_generation
        )
        if not allowed:
            raise ImageToolValidationError("This action is not permitted for this Chat turn.")
        if (
            request.name in ("generate_image", "edit_image", "propose_create_prompt")
            and policy.prompt_target == "supplied"
            and not policy.rewrite_image_prompt
            and request.arguments.get("prompt") != policy.original_prompt
        ):
            raise ImageToolValidationError(
                "The supplied image prompt cannot be rewritten for this turn."
            )
    if request.name != "propose_create_prompt" and request.intent == "text":
        raise ImageToolValidationError("Image generation is Off for this Chat turn.")


def prompt_tool_result(request):
    """A real prompt-only tool has no image preparation, compute or side effects."""
    check_tool_permissions(request)
    if request.name != "propose_create_prompt" or request.context.permissions is None:
        raise ImageToolValidationError("An authorized Create prompt proposal is required.")
    return {
        "status": "completed",
        "call_id": request.call_id,
        "name": request.name,
        "prompt_proposal": create_prompt_proposal(request),
    }


def create_prompt_proposal(request):
    """Optional draft candidate from the same validated action, never a second tool."""
    check_tool_permissions(request)
    policy = request.context.permissions
    if (
        policy is None
        or not policy.update_create_prompt
        or request.name not in ("generate_image", "edit_image", "propose_create_prompt")
    ):
        return None
    return {"prompt": request.arguments["prompt"], "turn_id": policy.turn_id}


class ImageToolCallBuffer:
    """Collect exactly one streamed call, preserving its original call ID."""

    def __init__(self):
        self.call = {"id": None, "type": "function", "function": {"name": None, "arguments": ""}}
        self.seen = False
        self.error = None

    def add(self, calls):
        if self.error:
            return
        try:
            if not isinstance(calls, list) or len(calls) != 1:
                raise ImageToolValidationError("Only one image tool call is allowed per turn.")
            call = calls[0]
            if (
                not isinstance(call, Mapping)
                or set(call) - {"index", "id", "type", "function"}
                or type(call.get("index", 0)) is not int
                or call.get("index", 0) != 0
            ):
                raise ImageToolValidationError(
                    "Parallel or malformed image tool calls are unavailable."
                )
            self.seen = True
            if "id" in call:
                if self.call["id"] is not None and call["id"] != self.call["id"]:
                    raise ImageToolValidationError(
                        "The image tool call ID changed during streaming."
                    )
                self.call["id"] = call["id"]
            if call.get("type", "function") != "function":
                raise ImageToolValidationError("Expected an image function call.")
            function = call.get("function", {})
            if not isinstance(function, Mapping) or set(function) - {"name", "arguments"}:
                raise ImageToolValidationError("Malformed image tool function.")
            if "name" in function:
                if (
                    self.call["function"]["name"] is not None
                    and function["name"] != self.call["function"]["name"]
                ):
                    raise ImageToolValidationError("The image tool name changed during streaming.")
                self.call["function"]["name"] = function["name"]
            fragment = function.get("arguments", "")
            if not isinstance(fragment, str):
                raise ImageToolValidationError("Image tool arguments must be JSON text.")
            self.call["function"]["arguments"] += fragment
            if _utf8_length(self.call["function"]["arguments"]) > MAX_ARGUMENT_BYTES:
                raise ImageToolValidationError("Image tool arguments exceed the limit.")
        except ImageToolValidationError as error:
            self.error = error

    def finish(self, context, capability, finish_reason):
        if self.error:
            raise self.error
        return validate_image_tool_call(self.call, context, capability, finish_reason=finish_reason)


def image_tool_error(call_id, name, error, *, cancelled=False):
    return {
        "call_id": call_id,
        "name": name,
        "status": "cancelled" if cancelled else "error",
        "error": {
            "code": "cancelled" if cancelled else getattr(error, "code", "image_tool_failed"),
            "message": str(error),
        },
    }


def dispatch_image_tool(request, service, *, check_cancelled=None):
    """One typed terminal result; shared services own resource/history fences."""
    if not isinstance(request, ImageToolRequest):
        raise ImageToolValidationError("A validated app image request is required.")
    try:
        if check_cancelled:
            check_cancelled()
        check_tool_permissions(request)
        result = service(request)
        if not isinstance(result, Mapping) or result.get("status") not in (
            "preview",
            "completed",
            "cancelled",
            "error",
        ):
            raise RuntimeError("The image service returned no terminal result.")
        if result.get("call_id", request.call_id) != request.call_id:
            raise RuntimeError("The image service returned a mismatched call ID.")
        if check_cancelled:
            check_cancelled()
        check_tool_permissions(request)
        return {**result, "call_id": request.call_id, "name": request.name}
    except CancelledError as error:
        return image_tool_error(request.call_id, request.name, error, cancelled=True)
    except Exception as error:
        return image_tool_error(request.call_id, request.name, error)


def _public_metadata(value, key="", depth=0):
    if depth > 24:
        return None
    if isinstance(value, Mapping):
        return {
            name: _public_metadata(item, name, depth + 1)
            for name, item in value.items()
            if isinstance(name, str)
            and name not in ("path", "paths", "owner", "settings", "image_settings", "local_file")
            and not name.endswith(("_path", "_paths"))
        }
    if isinstance(value, (tuple, list)):
        return [_public_metadata(item, key, depth + 1) for item in value]
    if (
        isinstance(value, str)
        and "prompt" not in key
        and (re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith(("/", "\\\\")))
    ):
        return "[local path omitted]"
    return value


def public_image_tool_result(result):
    """Return public DTOs; local files and app authority stay internal."""
    return {
        key: _public_metadata(result[key])
        if key in ("editor", "execution", "readiness")
        else result[key]
        for key in (
            "call_id",
            "name",
            "status",
            "editor",
            "execution",
            "readiness",
            "text",
            "error",
            "prompt_proposal",
        )
        if key in result
    }
