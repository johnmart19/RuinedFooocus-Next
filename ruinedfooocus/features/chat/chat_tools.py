"""Shared image-tool guidance for local chat and API clients."""

from ruinedfooocus.features.chat.image_tool_requests import (
    ChatPermissionSession,
    ChatToolPermissions,
    ImageToolCallBuffer,
    ImageToolContext,
    ImageToolContinuation,
    ImageToolRequest,
    ImageToolValidationError,
    check_tool_permissions,
    create_prompt_proposal,
    dispatch_image_tool,
    image_tool_error,
    image_tool_schemas,
    public_image_tool_result,
    prompt_tool_result,
    validate_image_tool_call,
)

__all__ = [
    "IMAGE_TOOL",
    "IMAGE_TOOL_GUIDANCE",
    "ChatPermissionSession",
    "ChatToolPermissions",
    "ImageToolCallBuffer",
    "ImageToolContext",
    "ImageToolContinuation",
    "ImageToolRequest",
    "ImageToolValidationError",
    "check_tool_permissions",
    "create_prompt_proposal",
    "dispatch_image_tool",
    "image_tool_error",
    "image_tool_schemas",
    "public_image_tool_result",
    "prompt_tool_result",
    "validate_image_tool_call",
]

IMAGE_TOOL_GUIDANCE = (
    "Preserve the selected assistant's identity, voice and system instructions. "
    "These rules govern the optional image tools; they do not redefine your persona. "
    "Use an image tool only when the current user request asks you to create, draw, "
    "edit or regenerate an image. A new generation uses the current request; "
    "do not carry subjects, details or prompts from earlier generations into it. "
    "An image reference is used only when the app or client explicitly supplies one. "
    "For an explicit follow-up, preserve the selected reference's unchanged details. "
    "Keep every explicitly requested medium, style, subject attribute, color, count, "
    "spatial relation, exclusion and literal quoted text in the visual prompt. "
    "Do not silently omit or replace them. "
    "Answer greetings, questions, stories, descriptions, critique and explicit "
    "prompt-only requests in text. An earlier generation request or "
    "review does not authorize another image. For an authorized image request, call "
    "one suitable tool without asking for extra confirmation. Never call a second "
    "tool or enter a critique/regenerate loop. Use generate_image for a new image; "
    "use edit_image or process_image only for the currently selected image ID. "
    "The app owns Preview/Run, settings, resources, size, negative prompt, denoise "
    "and LoRAs; never invent or override them. inspect_image requires a real "
    "compatible vision reviewer; filenames and metadata are not visual observation. "
    "Never claim an image was generated or edited without calling the tool."
    " The app captures three independent permissions. Never infer permission from "
    "conversation history or tool arguments. When an existing visual prompt is "
    "supplied and rewriting is Off, preserve it verbatim. For a new request, "
    "construct a concise visual prompt from only the current image request; do "
    "not copy unrelated conversational prose. propose_create_prompt updates only "
    "a Create prompt proposal and never generates an image. Use it only when "
    "the current request asks for a prompt, and it is offered by the app."
)

IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate a new image from the current request and captured settings. Never reuse a selected source or old image history.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 12000,
                    "description": "Visual description of the requested image",
                }
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    },
}
