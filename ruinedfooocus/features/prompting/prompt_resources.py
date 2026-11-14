"""Existing explicit resource-token grammar shared by prompt text consumers."""

import re


RESOURCE = re.compile(
    r"<(?:lora|style|embedding):[^<>]+>|"
    r"\(embedding:[\w./\\-]+:[+-]?(?:\d+(?:\.\d*)?|\.\d+)\)|"
    r"\bembedding:[\w./\\-]+",
    re.I,
)
