"""Small Gradio API examples; --help imports no client or application runtime.

Use the existing app environment. The framework owns JSON, authentication and
queue transport. Returned media stays on the server (download_files=False).
"""

import argparse
from concurrent.futures import TimeoutError
import os
import sys
from urllib.parse import quote, urlsplit


class ExampleError(ValueError):
    """A safe, actionable example error without credentials or server internals."""


def base_url(value):
    parts = urlsplit(value)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise argparse.ArgumentTypeError("Use an HTTP(S) app URL without credentials or query.")
    return value.rstrip("/")


def positive_timeout(value):
    number = float(value)
    if not 0 < number <= 86400:
        raise argparse.ArgumentTypeError("Timeout must be greater than zero and at most 86400.")
    return number


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="Print the server's absolute search URLs")
    search.add_argument("text", nargs="?", default="")
    search.add_argument("--max-results", type=int, default=25)
    llama = commands.add_parser("llama", help="Request a reply from the configured Chat model")
    llama.add_argument("prompt", nargs="?", default="Please tell me a joke about generative AI.")
    llama.add_argument("--system", default="You are a witty AI. Please help the user.")
    image = commands.add_parser("image", help="Generate one image using the current app settings")
    image.add_argument("prompt", nargs="?", default="A small robot tending a garden")
    for command in (search, llama, image):
        command.add_argument("--base-url", type=base_url, default="http://localhost:7860")
        command.add_argument("--timeout", type=positive_timeout, default=300)
    return root


def login(environ):
    username = environ.get("RF_GRADIO_USERNAME")
    password = environ.get("RF_GRADIO_PASSWORD")
    if bool(username) != bool(password):
        raise ExampleError("Set both RF_GRADIO_USERNAME and RF_GRADIO_PASSWORD for UI login.")
    return (username, password) if username else None


def absolute_url(value):
    return (
        isinstance(value, str)
        and urlsplit(value).scheme in {"http", "https"}
        and bool(urlsplit(value).netloc)
    )


def output_lines(command, result, url):
    if command == "search":
        if not isinstance(result, list) or not all(absolute_url(item) for item in result):
            raise ExampleError("Search did not return a list of absolute image URLs.")
        return result
    if command == "llama":
        if not isinstance(result, str):
            raise ExampleError("Chat did not return text.")
        return [result]
    if isinstance(result, dict):
        if absolute_url(result.get("url")):
            return [result["url"]]
        path = result.get("path")
        if isinstance(path, str) and path:
            # gr.api returns FileData. Keep its server path, never resolve it on
            # the caller's OS or download it into Gradio's default local cache.
            return [url + "/gradio_api/file=" + quote(path.replace("\\", "/"), safe="/:")]
    raise ExampleError("Image generation did not return Gradio FileData.")


def run(options, *, client_factory=None, environ=None):
    auth = login(os.environ if environ is None else environ)
    if client_factory is None:
        from gradio_client import Client

        client_factory = Client
    client = client_factory(
        options.base_url,
        auth=auth,
        verbose=False,
        analytics_enabled=False,
        download_files=False,
    )
    try:
        if options.command == "search":
            values = {"text": f"max:{options.max_results} {options.text}"}
        elif options.command == "llama":
            values = {"system": options.system, "user": options.prompt}
        else:
            values = {"prompt": options.prompt}
        endpoint = {"search": "/search", "llama": "/llama", "image": "/generate_image"}[
            options.command
        ]
        job = client.submit(api_name=endpoint, **values)
        try:
            result = job.result(timeout=options.timeout)
        except (TimeoutError, KeyboardInterrupt) as error:
            job.cancel()
            raise ExampleError(
                "Wait interrupted; Gradio cancellation requested. Running server work may continue."
            ) from error
        return output_lines(options.command, result, options.base_url)
    finally:
        client.close()


def main(argv=None):
    cli = parser()
    options = cli.parse_args(argv)
    if options.command == "search" and options.max_results < 1:
        cli.error("--max-results must be positive")
    try:
        for line in run(options):
            print(line)
    except ExampleError as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception as error:
        print(
            f"API request failed ({type(error).__name__}); check the app URL and UI login.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
