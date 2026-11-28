from ruinedfooocus.core.paths import asset_path, resource_path
from ruinedfooocus.core.paths import asset_path

"""Exercise actual Windows path normalization without importing application services."""
import ast
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote, unquote
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(name, **extra):
    tree = ast.parse((ROOT / "ruinedfooocus/app/api.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(
        asset_path=asset_path,
        resource_path=resource_path,
        Path=Path,
        quote=quote,
        shared=SimpleNamespace(local_url="http://127.0.0.1:7860/"),
        **extra,
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "ruinedfooocus/app/api.py", "exec"), scope
    )
    return scope[name]


class ImageURLTests(unittest.TestCase):
    def test_search_url_resolves_dot_segments_and_quotes_filename(self):
        path = ROOT / "modules" / ".." / "outputs" / "image #1 ?.png"
        url = function("image_file_url")(str(path))
        self.assertNotIn("\\", url)
        self.assertNotIn("/../", url)
        self.assertIn("image%20%231%20%3F.png", url)
        self.assertEqual(unquote(url.split("file=", 1)[1]), path.resolve().as_posix())

    def test_last_image_returns_canonical_path_for_display_and_file_access(self):
        path = ROOT / "modules" / ".." / "outputs" / "image.png"
        get = function("get_last_image", state={"last_image": str(path)})
        self.assertEqual(get(), path.resolve().as_posix())
        self.assertEqual(
            function("get_last_image", state={})(), (ROOT / str(asset_path("logo.png"))).as_posix()
        )


if __name__ == "__main__":
    unittest.main()
