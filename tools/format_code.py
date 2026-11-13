"""Format/check tracked owned source; never traverse models, user data or vendors."""

import argparse
import ast
import json
from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMAND_LINE_UNITS = 24_000


def canonical_html(text):
    """Keep the repository's HTML5 declaration/void style after Prettier.

    Use HTML tokens, so script strings, comments and SVG self-closing tags are
    not altered by a global text replacement.
    """
    edits = []
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    class Tokens(HTMLParser):
        def start(self):
            line, column = self.getpos()
            return offsets[line - 1] + column

        def handle_decl(self, decl):
            if decl.lower() == "doctype html":
                edits.append((self.start(), self.start() + len(decl) + 3, "<!DOCTYPE html>"))

        def handle_startendtag(self, tag, attrs):
            if tag in {
                "area",
                "base",
                "br",
                "col",
                "embed",
                "hr",
                "img",
                "input",
                "link",
                "meta",
                "param",
                "source",
                "track",
                "wbr",
            }:
                raw = self.get_starttag_text()
                edits.append((self.start(), self.start() + len(raw), raw[:-2].rstrip() + ">"))

    Tokens().feed(text)
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    return text


def tracked_files():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True)
    return [Path(name) for name in result.stdout.decode("utf-8").split("\0") if name]


def python_ast(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    # Formatters may normalize documentation indentation; executable literals
    # (including system prompts and code templates) must remain identical.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def command_batches(prefix, files, *, max_units=COMMAND_LINE_UNITS):
    """Keep native quoting and UTF-16 argv length below a conservative Windows budget."""

    def fits(command):
        quoted = subprocess.list2cmdline(command)
        return len(quoted.encode("utf-16-le", errors="surrogatepass")) // 2 + 1 <= max_units

    prefix = list(prefix)
    command = prefix.copy()
    for path in files:
        candidate = [*command, path]
        if fits(candidate):
            command = candidate
            continue
        if len(command) > len(prefix):
            yield command
        command = [*prefix, path]
        if not fits(command):
            raise RuntimeError(
                f"Formatter path exceeds the {max_units}-unit command budget: {path}"
            )
    if len(command) > len(prefix):
        yield command


def run(check=False):
    local_ruff = Path(sys.executable).with_name("ruff.exe" if sys.platform == "win32" else "ruff")
    ruff = str(local_ruff) if local_ruff.is_file() else shutil.which("ruff")
    node = shutil.which("node")
    prettier = ROOT / "tools/format/node_modules/prettier/bin/prettier.cjs"
    if not ruff or not node or not prettier.is_file():
        raise RuntimeError(
            "Install developer tools: pip install -r tools/format/requirements.txt; npm ci --prefix tools/format"
        )
    files = tracked_files()
    python = [
        str(path)
        for path in files
        if path.suffix == ".py" and not path.as_posix().startswith("ruinedfooocus/backends/vendor/")
    ]
    web = [
        str(path)
        for path in files
        if path.suffix in (".js", ".css", ".json", ".md", ".yaml", ".yml")
        and not path.as_posix().startswith("ruinedfooocus/backends/vendor/")
    ]
    before = {} if check else {path: python_ast(path) for path in python}
    json_before = (
        {}
        if check
        else {
            path: json.loads((ROOT / path).read_text(encoding="utf-8"))
            for path in web
            if Path(path).suffix == ".json"
        }
    )
    commands = [
        ([ruff, "format", *(["--check"] if check else [])], python),
        ([ruff, "check"], python),
        ([node, str(prettier), "--check" if check else "--write"], web),
    ]
    failed = False
    for prefix, paths in commands:
        for command in command_batches(prefix, paths):
            failed |= subprocess.run(command, cwd=ROOT).returncode != 0
    for path in files:
        if path.suffix != ".html":
            continue
        result = subprocess.run(
            [node, str(prettier), str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        formatted = canonical_html(result.stdout)
        original = (ROOT / path).read_text(encoding="utf-8")
        if check:
            if original != formatted:
                print(f"HTML needs formatting: {path}")
                failed = True
        elif original != formatted:
            (ROOT / path).write_text(formatted, encoding="utf-8", newline="\n")
    if not check:
        changed = [path for path, tree in before.items() if python_ast(path) != tree]
        if changed:
            raise RuntimeError("Executable Python changed during formatting: " + ", ".join(changed))
        print(f"Executable AST preserved for {len(before)} owned Python files.")
        for path, value in json_before.items():
            if json.loads((ROOT / path).read_text(encoding="utf-8")) != value:
                raise RuntimeError(f"JSON values changed during formatting: {path}")
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    try:
        return run(parser.parse_args().check)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
