"""Real Chat request/stream bodies over inert image services; no model runtime."""

from concurrent.futures import CancelledError
import json
import unittest
from unittest.mock import Mock

from ruinedfooocus.features.chat.chat_tools import (
    ChatPermissionSession,
    ImageToolCallBuffer,
    ImageToolContext,
    ImageToolValidationError,
    dispatch_image_tool,
    image_tool_schemas,
    public_image_tool_result,
    validate_image_tool_call,
)
from ruinedfooocus.features.chat.model_profiles import initialized_image_tools, ProfiledChatClient
from ruinedfooocus.features.models.model_registry import load_registry

CAPABILITY = {"available": True, "binding": "fixture-template-runtime-resources"}


def context(**overrides):
    values = {
        "owner": ("test", "owner", "tab"),
        "intent": "run",
        "settings": {
            "negative": "exact exclusion",
            "width": 768,
            "height": 512,
            "denoise": 0.25,
            "loras": [["detail.safetensors", 0.6]],
            "alpha": "preserve",
            "_effective_image_plan": {"fixture": "captured"},
        },
        **overrides,
    }
    values.setdefault(
        "permissions",
        ChatPermissionSession().capture(
            values["owner"], enable_image_generation=values["intent"] != "text"
        ),
    )
    return ImageToolContext(**values)


def selected(**overrides):
    return context(
        **{
            "session_id": "session_id",
            "source_id": "image_id",
            "mask_id": "mask_id",
            "revision": 3,
            "edit_operations": ("edit", "inpaint", "outpaint"),
            "process_operations": ("upscale", "remove_background", "restore"),
            **overrides,
        }
    )


def call(name="generate_image", arguments=None, call_id="call_fixture"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                {"prompt": "Current exact prompt"} if arguments is None else arguments
            ),
        },
    }


class StrictToolTests(unittest.TestCase):
    def test_context_filtered_tools_and_all_fields_are_required(self):
        self.assertEqual(image_tool_schemas(context(intent="text"), CAPABILITY), [])
        self.assertEqual(image_tool_schemas(context(grounded=True), CAPABILITY), [])
        self.assertEqual(image_tool_schemas(context(), {}), [])
        self.assertEqual(
            [item["function"]["name"] for item in image_tool_schemas(context(), CAPABILITY)],
            ["generate_image"],
        )
        tools = image_tool_schemas(selected(inspect_available=True), CAPABILITY)
        self.assertEqual(
            [item["function"]["name"] for item in tools],
            ["generate_image", "edit_image", "process_image", "inspect_image"],
        )
        for item in tools:
            function = item["function"]
            self.assertTrue(function["strict"])
            self.assertFalse(function["parameters"]["additionalProperties"])
            self.assertEqual(
                set(function["parameters"]["required"]), set(function["parameters"]["properties"])
            )

    def test_settings_are_captured_and_model_cannot_override_them(self):
        values = {"negative": "keep", "loras": [["detail", 0.5]], "width": 768}
        captured = context(settings=values)
        values["negative"] = "changed"
        values["loras"][0][1] = 1
        request = validate_image_tool_call(call(), captured, CAPABILITY)
        self.assertEqual(request.context.image_settings()["loras"], [["detail", 0.5]])
        self.assertEqual(request.settings["negative"], "keep")
        with self.assertRaises(TypeError):
            request.settings["width"] = 1
        for field in (
            "negative",
            "width",
            "denoise",
            "loras",
            "intent",
            "owner",
            "settings",
            "graph",
            "url",
            "path",
            "install",
        ):
            with self.subTest(field=field), self.assertRaises(ImageToolValidationError):
                validate_image_tool_call(
                    call(arguments={"prompt": "exact", field: "forbidden"}), captured, CAPABILITY
                )

    def test_new_generation_never_inherits_selected_source(self):
        values = {
            **context().image_settings(),
            "input_image": "old",
            "mask_image": "mask",
            "controlnet": {"old": True},
        }
        request = validate_image_tool_call(call(), selected(settings=values), CAPABILITY)
        self.assertIsNone(request.source_id)
        self.assertIsNone(request.session_id)
        self.assertIsNone(request.revision)
        self.assertIsNone(request.context.mask_id)
        self.assertNotIn("input_image", request.settings)
        self.assertNotIn("controlnet", request.settings)
        for key in (
            "negative",
            "width",
            "height",
            "denoise",
            "loras",
            "alpha",
            "_effective_image_plan",
        ):
            self.assertEqual(request.context.image_settings()[key], context().image_settings()[key])

    def test_edit_and_mask_are_bound_to_the_selected_source(self):
        good = {
            "prompt": "change only background",
            "source_id": "image_id",
            "operation": "inpaint",
            "mask_id": "mask_id",
        }
        request = validate_image_tool_call(call("edit_image", good), selected(), CAPABILITY)
        self.assertEqual(
            (request.session_id, request.source_id, request.mask_id, request.revision),
            ("session_id", "image_id", "mask_id", 3),
        )
        self.assertEqual(request.owner, ("test", "owner", "tab"))
        for changes in (
            {"source_id": "other_image"},
            {"source_id": "C:/private.png"},
            {"mask_id": None},
            {"operation": "edit"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ImageToolValidationError):
                validate_image_tool_call(
                    call("edit_image", {**good, **changes}), selected(), CAPABILITY
                )
        with self.assertRaises(ImageToolValidationError):
            validate_image_tool_call(call("edit_image", good), context(), CAPABILITY)

    def test_process_uses_one_selected_operation_and_inspection_is_explicit(self):
        request = validate_image_tool_call(
            call("process_image", {"source_id": "image_id", "operation": "upscale"}),
            selected(),
            CAPABILITY,
        )
        self.assertTrue(request.mutates)
        inspection = call(
            "inspect_image", {"source_id": "image_id", "question": "What pixels are visible?"}
        )
        with self.assertRaises(ImageToolValidationError):
            validate_image_tool_call(inspection, selected(), CAPABILITY)
        self.assertFalse(
            validate_image_tool_call(
                inspection, selected(inspect_available=True), CAPABILITY
            ).mutates
        )

    def test_bad_json_duplicates_shapes_and_truncation_have_no_service_call(self):
        service = Mock()
        for raw in (
            '{"prompt":',
            '{"prompt":"a","prompt":"b"}',
            "[]",
            "null",
            '{"prompt":42}',
            '{"prompt":""}',
            '{"prompt":"x","tool":"shell"}',
        ):
            invalid = call()
            invalid["function"]["arguments"] = raw
            with self.subTest(raw=raw), self.assertRaises(ImageToolValidationError):
                dispatch_image_tool(
                    validate_image_tool_call(invalid, context(), CAPABILITY), service
                )
        for finish in (None, "length", "content_filter"):
            with self.subTest(finish=finish), self.assertRaises(ImageToolValidationError):
                validate_image_tool_call(call(), context(), CAPABILITY, finish_reason=finish)
        service.assert_not_called()

    def test_paths_urls_and_partial_app_contexts_are_not_handles(self):
        for reference in (
            "C:/image.png",
            "https://server/image",
            "../image",
            "file:///image",
            "a\\b",
        ):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                selected(source_id=reference)
        for values in (
            {"source_id": "image"},
            {"session_id": "session"},
            {"revision": 1},
            {"owner": "client-owner"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                context(**values)

    def test_invalid_unicode_is_a_typed_rejection_before_dispatch(self):
        for invalid in (call(arguments={"prompt": "\ud800"}),):
            with self.assertRaises(ImageToolValidationError):
                validate_image_tool_call(invalid, context(), CAPABILITY)
        buffer = ImageToolCallBuffer()
        buffer.add([{"function": {"arguments": "\ud800"}}])
        with self.assertRaises(ImageToolValidationError):
            buffer.finish(context(), CAPABILITY, "tool_calls")

    def test_streamed_arguments_keep_the_original_call_id(self):
        buffer = ImageToolCallBuffer()
        buffer.add(
            [
                {
                    "index": 0,
                    "id": "original_id",
                    "type": "function",
                    "function": {"name": "generate_image", "arguments": '{"prompt":"a '},
                }
            ]
        )
        buffer.add([{"index": 0, "function": {"arguments": 'fox"}'}}])
        request = buffer.finish(context(), CAPABILITY, "tool_calls")
        self.assertEqual(request.call_id, "original_id")
        self.assertEqual(request.arguments["prompt"], "a fox")

    def test_parallel_changed_ids_unknown_fields_and_oversized_streams_fail_closed(self):
        for chunks in (
            [[{"index": 0}, {"index": 1}]],
            [[{"index": 1}]],
            [[{"id": "first"}], [{"id": "second"}]],
            [[{"unknown": "field"}]],
            [[{"function": {"arguments": "x" * 32769}}]],
        ):
            buffer = ImageToolCallBuffer()
            for chunk in chunks:
                buffer.add(chunk)
            with (
                self.subTest(chunks=repr(chunks)[:80]),
                self.assertRaises(ImageToolValidationError),
            ):
                buffer.finish(context(), CAPABILITY, "tool_calls")

    def test_dispatch_cancellation_call_id_and_restoration_errors_are_typed(self):
        request = validate_image_tool_call(call(), context(), CAPABILITY)
        service = Mock(return_value={"status": "completed", "images": ["internal.png"]})
        terminal = dispatch_image_tool(
            request, service, check_cancelled=Mock(side_effect=CancelledError("stopped"))
        )
        self.assertEqual(terminal["status"], "cancelled")
        service.assert_not_called()
        for result in ({"status": "pending"}, {"status": "completed", "call_id": "other"}):
            self.assertEqual(dispatch_image_tool(request, lambda _: result)["status"], "error")
        failed = dispatch_image_tool(request, Mock(side_effect=RuntimeError("restoration failed")))
        self.assertEqual(failed["call_id"], request.call_id)
        self.assertEqual(failed["status"], "error")

    def test_public_results_exclude_internal_images_and_settings(self):
        result = public_image_tool_result(
            {
                "status": "completed",
                "call_id": "id",
                "images": ["C:/private/image.png"],
                "owner": ("other",),
                "settings": {"secret": "local"},
            }
        )
        self.assertEqual(result, {"status": "completed", "call_id": "id"})

    def test_public_metadata_preserves_prompts_and_redacts_nested_paths(self):
        result = public_image_tool_result(
            {
                "status": "completed",
                "editor": {
                    "source_generation": {
                        "requested_prompt": "C:/literal text in the image",
                        "path": "C:/private/input.png",
                        "effective_plan": {
                            "checkpoint": {"path": "D:/models/weight.safetensors", "sha256": "abc"}
                        },
                    }
                },
                "execution": {"model": "D:/models/weight", "operation": "upscale"},
            }
        )
        self.assertEqual(
            result["editor"]["source_generation"]["requested_prompt"],
            "C:/literal text in the image",
        )
        self.assertNotIn("path", result["editor"]["source_generation"])
        self.assertEqual(result["execution"]["model"], "[local path omitted]")


class InitializedToolBindingTests(unittest.TestCase):
    def artifact(self):
        return load_registry().artifact("existing:llm/Qwen3-1.7B-Q8_0.gguf")

    def test_runtime_template_and_resource_identity_all_bind_transport(self):
        template = "{% if tools %}{{ tools }}{% endif %}{{ messages[0].tool_calls }}"
        kwargs = {"initialized": True, "resources": [("model", 42, 1)]}
        bound = initialized_image_tools(self.artifact(), {}, template, **kwargs)
        self.assertTrue(bound["available"])
        self.assertFalse(bound["behavior_qualified"])
        self.assertFalse(bound["vision"])
        for values, content, options in (
            ({}, template, {**kwargs, "initialized": False}),
            ({}, template, {**kwargs, "resources": ()}),
            ({}, "{{ messages }}", kwargs),
            ({}, "{% broken %}", kwargs),
            ({"llama_server_args": "--chat-template-file custom.jinja"}, template, kwargs),
            ({"llm_runtime": "unqualified"}, template, kwargs),
        ):
            self.assertFalse(
                initialized_image_tools(self.artifact(), values, content, **options)["available"]
            )
        for content, resources in (
            (template + "changed", kwargs["resources"]),
            (template, [("model", 43, 2)]),
        ):
            self.assertNotEqual(
                bound["binding"],
                initialized_image_tools(
                    self.artifact(), {}, content, initialized=True, resources=resources
                )["binding"],
            )

    def test_close_removes_initialized_advertisement(self):
        client = ProfiledChatClient(
            Mock(),
            self.artifact(),
            {},
            initialized=True,
            template="{{ tools }}{{ messages[0].tool_calls }}",
            resources=[("model", 1, 1)],
        )
        self.assertTrue(client.image_tool_capability["available"])
        client.close()
        self.assertFalse(client.image_tool_capability["available"])


if __name__ == "__main__":
    unittest.main()
