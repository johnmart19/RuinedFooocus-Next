"""Search the running app and print its already absolute image URLs."""

import sys

if __package__:
    from .api_examples import main
else:
    from api_examples import main


if __name__ == "__main__":
    raise SystemExit(main(["search", *sys.argv[1:]]))
