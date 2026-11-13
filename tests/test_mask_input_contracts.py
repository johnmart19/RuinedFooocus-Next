"""Bind only a known fixed preprocessor; preserve native options and model choice."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ruinedfooocus.backends.onnx.model_contracts import create_mask_session


class MaskInputContractTests(unittest.TestCase):
    def factory(self, default="bria-rmbg"):
        calls = []

        def create(model_name=default, *args, sess_opts=None, **kwargs):
            calls.append((model_name, args, sess_opts, kwargs))
            return "native session"

        return SimpleNamespace(new_session=create), calls

    def test_default_fixed_preprocessor_binds_shape_and_keeps_provider_options(self):
        rembg, calls = self.factory()
        options = SimpleNamespace(add_free_dimension_override_by_name=Mock())
        providers = [("DmlExecutionProvider", {"device_id": "1"}), "CPUExecutionProvider"]
        with (
            patch.dict("sys.modules", {"rembg": rembg}),
            patch("ruinedfooocus.backends.onnx.model_contracts.version", return_value="2.0.87"),
        ):
            self.assertEqual(
                create_mask_session(
                    sess_opts=options, providers=providers, _factory=rembg.new_session
                ),
                "native session",
            )
        self.assertEqual(
            options.add_free_dimension_override_by_name.call_args_list,
            [unittest.mock.call("height", 1024), unittest.mock.call("width", 1024)],
        )
        self.assertIs(calls[0][2], options)
        self.assertIs(calls[0][3]["providers"], providers)
        self.assertEqual(calls[0][0], "bria-rmbg")

    def test_unknown_versions_and_different_models_keep_native_dimensions(self):
        for default, selected, release in (
            ("bria-rmbg", "bria-rmbg", "2.0.88"),
            ("u2net", None, "2.0.87"),
            ("bria-rmbg", "u2net", "2.0.87"),
        ):
            rembg, calls = self.factory(default)
            options = SimpleNamespace(add_free_dimension_override_by_name=Mock())
            with (
                patch.dict("sys.modules", {"rembg": rembg}),
                patch("ruinedfooocus.backends.onnx.model_contracts.version", return_value=release),
            ):
                create_mask_session(
                    *([selected] if selected else []),
                    sess_opts=options,
                    providers=["CPUExecutionProvider"],
                )
            options.add_free_dimension_override_by_name.assert_not_called()
            self.assertEqual(calls[0][0], selected or default)

    def test_options_are_created_only_for_the_qualified_contract(self):
        rembg, calls = self.factory()
        options = SimpleNamespace(add_free_dimension_override_by_name=Mock())
        ort = SimpleNamespace(SessionOptions=Mock(return_value=options))
        with (
            patch.dict("sys.modules", {"rembg": rembg, "onnxruntime": ort}),
            patch("ruinedfooocus.backends.onnx.model_contracts.version", return_value="2.0.87"),
        ):
            create_mask_session(providers=["CPUExecutionProvider"])
        ort.SessionOptions.assert_called_once_with()
        self.assertIs(calls[0][2], options)


if __name__ == "__main__":
    unittest.main()
