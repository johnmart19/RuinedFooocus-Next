"""Bind static image LoRA intent to existing local files before queueing."""

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path, PurePosixPath
import re

from ruinedfooocus.core.preparation import PreparationRequest
from ruinedfooocus.features.models.lora_names import (
    capture_gguf_lora,
    is_gguf_lora_name,
    merge_loras,
)
from ruinedfooocus.features.models.model_identity import file_signature

KEY = "_captured_image_loras"
PROMPTS_KEY = "_captured_image_lora_prompts"


def _key(name):
    return str(name).replace("\\", "/").casefold()


def _relative_name(name):
    text = str(name).replace("\\", "/")
    relative = PurePosixPath(text)
    if not text or relative.is_absolute() or ":" in text or ".." in relative.parts:
        raise ValueError(
            "Choose an image LoRA by its relative filename in a configured LoRA folder."
        )
    return text


def selected_loras(rows):
    """Read the existing card transport without changing names or explicit weights."""
    selected = []
    for row in rows or ():
        if isinstance(row, dict):
            selected.append(dict(row))
            continue
        thumbnail, caption = row
        weight, name = caption.split(" - ", 1)
        checksum = (
            thumbnail
            if isinstance(thumbnail, str) and re.fullmatch(r"[0-9a-fA-F]{64}", thumbnail)
            else None
        )
        if name != "None":
            selected.append(dict(name=name, weight=float(weight), hash=checksum))
    return selected


def _request_groups(data, models, loras=None):
    from ruinedfooocus.features.models.lora_names import parse_loras

    prompts = data.get("prompt", "")
    prompts = prompts if isinstance(prompts, list) else [prompts]
    if any(not isinstance(prompt, str) for prompt in prompts):
        raise ValueError("Image prompts must be text or a list of text prompts.")
    if loras is not None:
        return [(prompt, loras) for prompt in prompts]
    parsed = [parse_loras(prompt, data.get("negative", ""))[0] for prompt in prompts]
    installed = models.get_names("loras") if any(parsed) else ()
    selected = selected_loras(data.get("loras"))
    return [
        (prompt, merge_loras(selected, typed, installed)) for prompt, typed in zip(prompts, parsed)
    ]


def _alternatives(groups):
    alternatives = {}
    for _, rows in groups:
        for row in rows:
            alternatives.setdefault((_key(row["name"]), row["weight"]), row)
    return list(alternatives.values())


def request_loras(data, models):
    """Use the same literal inline-tag and card merge rules as the worker."""
    return _alternatives(_request_groups(data, models))


def _active(loras):
    result = []
    for row in sorted(loras, key=lambda item: item["name"].lower()):
        name, weight = row.get("name", "None"), row.get("weight", 0)
        if name == "None":
            continue
        _relative_name(name)
        if type(weight) not in (int, float) or not math.isfinite(weight):
            raise ValueError("Image LoRA strength must be a finite number.")
        if is_gguf_lora_name(name) and not -100 <= weight <= 100:
            raise ValueError("GGUF LoRA strength must be a number from -100 to 100.")
        if weight:
            result.append(row)
    return result


@dataclass(frozen=True)
class CapturedLora:
    name: str
    weight: float
    source: Path
    signature: tuple | None
    gguf: bool
    request: PreparationRequest | None = None
    download_key: str | None = None

    @property
    def path(self):
        return Path(self.signature[0]) if self.signature is not None else self.source

    def verify(self):
        if self.signature is None:
            if (
                not isinstance(self.request, PreparationRequest)
                or not self.download_key
                or self.source != self.request.destination
            ):
                raise ValueError("A missing image LoRA needs its captured catalogue request.")
            return
        try:
            unchanged = (
                tuple(file_signature(self.source)) == self.signature
                and tuple(file_signature(self.path)) == self.signature
            )
        except OSError:
            unchanged = False
        if not unchanged:
            raise ValueError(
                "A captured image LoRA changed or was removed. Submit the request again."
            )

    def metadata(self):
        return dict(
            name=self.name,
            weight=self.weight,
            source=str(self.source),
            signature=list(self.signature) if self.signature is not None else None,
            format="gguf" if self.gguf else "native",
            download_key=self.download_key,
            request=(
                dict(
                    url=self.request.url,
                    destination=str(self.request.destination),
                    expected_size=self.request.expected_size,
                    sha256=self.request.sha256,
                )
                if self.request is not None
                else None
            ),
        )


@dataclass(frozen=True)
class CapturedPromptLoras:
    prompt_sha256: str
    pairs: tuple

    def metadata(self):
        return dict(
            prompt_sha256=self.prompt_sha256,
            loras=[dict(name=name, weight=weight) for name, weight in self.pairs],
        )


def _prompt_key(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def captured_prompts(data):
    value = (data or {}).get(PROMPTS_KEY)
    if value is not None and (
        not isinstance(value, tuple)
        or any(
            not isinstance(item, CapturedPromptLoras)
            or not isinstance(item.prompt_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", item.prompt_sha256) is None
            or not isinstance(item.pairs, tuple)
            or any(
                not isinstance(pair, tuple)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or type(pair[1]) not in (int, float)
                or not math.isfinite(pair[1])
                for pair in item.pairs
            )
            for item in value
        )
    ):
        raise ValueError("Capture typed original prompt LoRA selections before loading.")
    return value


def sources_for_prompt(prompt, union, prompts):
    """Project only the original scalar child's captured selection; never resolve names."""
    captured({KEY: union})
    table = captured_prompts({PROMPTS_KEY: prompts})
    matching = [
        item
        for item in table or ()
        if isinstance(prompt, str) and item.prompt_sha256 == _prompt_key(prompt)
    ]
    if len(matching) != 1:
        raise ValueError("The original image prompt was not captured. Submit the request again.")
    allowed = {(_key(item.name), item.weight): item for item in union}
    keys = [(_key(name), weight) for name, weight in matching[0].pairs]
    if any(key not in allowed for key in keys):
        raise ValueError("The original prompt LoRA selection changed after capture. Submit again.")
    return tuple(allowed[key] for key in keys)


def captured(data):
    value = (data or {}).get(KEY)
    if value is not None:
        if not isinstance(value, tuple) or any(
            not isinstance(item, CapturedLora) for item in value
        ):
            raise ValueError("Capture typed image LoRA sources before loading.")
        for item in value:
            item.verify()
    return value


def for_loras(loras, items, *, require_prepared=True):
    """Match runtime prompt output to the frozen selection before any cache hit."""
    captured({KEY: items})
    rows = _active(loras)
    keys = [(_key(row["name"]), row["weight"]) for row in rows]
    if keys != [(_key(item.name), item.weight) for item in items]:
        raise ValueError("Image LoRA selection changed after capture. Submit the request again.")
    if require_prepared and any(item.signature is None for item in items):
        raise ValueError(
            "Prepare the captured image LoRA before loading. Submit the request again."
        )
    return items


def _pending(row, roots, manager):
    name = row["name"]
    catalog = getattr(manager, "DOWNLOADABLE_FILES", {})
    keys = (
        [name]
        if catalog.get(name, {}).get("path") == "path_loras"
        else [
            key
            for key, entry in catalog.items()
            if entry.get("path") == "path_loras" and _key(entry.get("filename", "")) == _key(name)
        ]
    )
    if not keys:
        raise FileNotFoundError(
            f"Could not find installed image LoRA: {name}. Choose an installed or catalogue LoRA."
        )
    requests = [manager.preparation_request(key) for key in keys]
    request, key = requests[0], keys[0]
    if any(candidate != request for candidate in requests[1:]):
        raise ValueError(
            "The missing LoRA name matches several catalogue sources. Choose an exact entry."
        )
    if not any(request.destination.is_relative_to(root) for root in roots):
        raise ValueError("The image LoRA destination is outside its captured LoRA folders.")
    if is_gguf_lora_name(request.destination) != is_gguf_lora_name(name):
        raise ValueError("Select a GGUF LoRA by its explicit relative .gguf filename.")
    item = CapturedLora(
        name, row["weight"], request.destination, None, is_gguf_lora_name(name), request, key
    )
    item.verify()
    return item


def capture_request(data, models, *, loras=None, manager=None):
    """Resolve installed names once; never scan, acquire or hash model weights."""
    result = dict(data)
    groups = _request_groups(result, models, loras)
    rows = _active(_alternatives(groups))
    prompt_rows = {}
    for prompt, selected in groups:
        key = _prompt_key(prompt)
        prompt_rows[key] = CapturedPromptLoras(
            key, tuple((row["name"], row["weight"]) for row in _active(selected))
        )
    prompts = tuple(prompt_rows.values())
    previous = captured(result)
    if previous is not None:
        for_loras(rows, previous, require_prepared=False)
        if captured_prompts(result) != prompts:
            raise ValueError(
                "The original prompt LoRA selections changed after capture. Submit again."
            )
        return result
    result[PROMPTS_KEY] = prompts
    if not rows:
        result[KEY] = ()
        return result
    roots = tuple(Path(root).resolve() for root in models.model_dirs["loras"])
    items, by_name = [], {}
    for row in rows:
        name = row["name"]
        if _key(name) in by_name:
            items.append(replace(by_name[_key(name)], name=name, weight=row["weight"]))
            continue
        gguf = is_gguf_lora_name(name)
        if gguf:
            try:
                source, signature = capture_gguf_lora(name, roots)
            except FileNotFoundError:
                item = _pending(row, roots, manager)
                items.append(item)
                by_name[_key(name)] = item
                continue
        else:
            source = models.get_file("loras", _relative_name(name))
            if source is None and row.get("hash") is not None:
                source = models.get_file_from_hash("loras", row["hash"])
            if source is None:
                item = _pending(row, roots, manager)
                items.append(item)
                by_name[_key(name)] = item
                continue
            if is_gguf_lora_name(source):
                raise ValueError("Select a GGUF LoRA by its explicit relative .gguf filename.")
            signature = file_signature(source)
        if not any(Path(signature[0]).is_relative_to(root) for root in roots):
            raise ValueError("The image LoRA resolves outside its configured LoRA folder.")
        item = CapturedLora(name, row["weight"], Path(source).absolute(), tuple(signature), gguf)
        item.verify()
        items.append(item)
        by_name[_key(name)] = item
    result[KEY] = tuple(items)
    return result


def bind_prepared(original_data, prepared_data):
    """Admit newly acquired files only through the original typed preparation receipt."""
    from ruinedfooocus.features.generation.image_preparation import KEY as ASSETS_KEY
    from ruinedfooocus.features.generation.image_preparation import ImageArtifact, ImageAssets
    from ruinedfooocus.features.models.gguf_adapters import inspect_gguf_lora

    original = captured(original_data)
    if captured(prepared_data) != original:
        raise ValueError("The captured image LoRA request changed during preparation.")
    if captured_prompts(prepared_data) != captured_prompts(original_data):
        raise ValueError("The original prompt LoRA selections changed during preparation.")
    result = []
    for item in original or ():
        if item.signature is not None:
            result.append(item)
            continue
        matches = []
        for data in (original_data, prepared_data):
            assets = data.get(ASSETS_KEY)
            rows = assets.artifacts if isinstance(assets, ImageAssets) else ()
            matching = [
                row
                for row in rows
                if isinstance(row, ImageArtifact)
                and row.role == "loras"
                and row.name == "lora:" + item.name
                and row.request == item.request
            ]
            if len(matching) != 1:
                raise ValueError("The image LoRA is missing its captured preparation receipt.")
            matches.append(matching[0])
        before, after = matches
        if before.signature is not None or after.signature is None:
            raise ValueError(
                "The image LoRA preparation receipt does not match its missing source."
            )
        if (
            after.signature[0] != str(item.source)
            or item.request.expected_size is not None
            and after.signature[1] != item.request.expected_size
        ):
            raise ValueError(
                "The prepared image LoRA does not match its captured destination or size."
            )
        bound = replace(item, signature=tuple(after.signature))
        bound.verify()
        if bound.gguf and inspect_gguf_lora(bound.path).signature != bound.signature:
            raise ValueError("The prepared GGUF LoRA changed before capture.")
        result.append(bound)
    return tuple(result)
