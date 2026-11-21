"""Native ComfyUI orientation choices at the selected model's pixel budget."""

from functools import lru_cache
import logging
import math

_STANDARD_RATIOS = (
    ("1:1", (1, 1)),
    ("2:3", (2, 3)),
    ("3:2", (3, 2)),
    ("3:4", (3, 4)),
    ("4:3", (4, 3)),
    ("9:16", (9, 16)),
    ("16:9", (16, 9)),
    ("21:9", (21, 9)),
)
_KNOWN_ASPECT_LABELS = (
    ("1:1", 1 / 1),
    ("4:5", 4 / 5),
    ("5:4", 5 / 4),
    ("3:4", 3 / 4),
    ("4:3", 4 / 3),
    ("2:3", 2 / 3),
    ("3:2", 3 / 2),
    ("10:16", 10 / 16),
    ("16:10", 16 / 10),
    ("9:16", 9 / 16),
    ("16:9", 16 / 9),
    ("9:21", 9 / 21),
    ("21:9", 21 / 9),
)


def aspect_ratio_label(width, height):
    """Return a concise human-readable aspect ratio label (e.g. '16:9', '2:3', '1:1')."""
    if not width or not height or width <= 0 or height <= 0:
        return ""
    actual = width / height
    best_label, best_ratio = min(_KNOWN_ASPECT_LABELS, key=lambda item: abs(actual - item[1]) / item[1])
    if abs(actual - best_ratio) / best_ratio <= 0.045:
        return best_label
    divisor = math.gcd(int(width), int(height))
    rw, rh = int(width) // divisor, int(height) // divisor
    if rw <= 21 and rh <= 21:
        return f"{rw}:{rh}"
    return f"{actual:.2f}:1" if actual >= 1 else f"1:{(1 / actual):.2f}"


def aspect_presets(width, height, multiple=None):
    """Reuse the pinned node; fall back to identical math when comfy_extras is not loaded yet."""
    try:
        return _aspect_presets(width, height, multiple)
    except ImportError:
        logging.getLogger(__name__).debug(
            "Native resolution selector is not initialized; using equivalent ratio grid"
        )
        return _fallback_aspect_presets(width, height, multiple or 8)


@lru_cache(maxsize=128)
def _fallback_aspect_presets(width, height, multiple):
    multiple = max(2, int(multiple or 8))
    total_pixels = width * height
    sizes = []
    for _, (w_ratio, h_ratio) in _STANDARD_RATIOS:
        scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
        w = round(w_ratio * scale / multiple) * multiple
        h = round(h_ratio * scale / multiple) * multiple
        size = (w, h)
        if size not in sizes and all(256 <= value <= 4096 for value in size):
            sizes.append(size)
    return tuple(sizes)


@lru_cache(maxsize=128)
def _aspect_presets(width, height, multiple):
    from ruinedfooocus.features.models.native_snapshot import load_native_snapshot
    from comfy_extras.nodes_resolution import ASPECT_RATIOS, ResolutionSelector

    schema = load_native_snapshot().schemas["ResolutionSelector"]
    if multiple is None:
        multiple = next(
            row["options"]["default"] for row in schema["inputs"] if row["name"] == "multiple"
        )
    megapixels = width * height / (1024 * 1024)
    sizes = []
    for ratio in ASPECT_RATIOS:
        size = tuple(ResolutionSelector.execute(ratio, megapixels, multiple).result)
        if size not in sizes and all(256 <= value <= 4096 for value in size):
            sizes.append(size)
    return tuple(sizes)


def canvas_multiple(recipe, family=None):
    """Respect native latent scaling and the family's empty-latent node grids."""
    from collections.abc import Mapping
    from math import lcm
    from ruinedfooocus.features.models.native_snapshot import (
        load_native_snapshot,
        get_image_family_facts,
    )

    snapshot = load_native_snapshot()
    contract = recipe.get("applicability", {}).get("component_contract", {})
    native_family = contract.get("native_family") or recipe.get("family") or family
    facts = get_image_family_facts(native_family)
    if facts is None:
        from ruinedfooocus.features.models.image_components import ALIASES

        facts = get_image_family_facts(ALIASES.get(native_family, native_family))
    if facts is None:
        return None
    steps = []
    nodes = set()
    for config in facts.configurations:
        latent = config.get("latent", {})
        scale = latent.get("spacial_downscale_ratio")
        if type(scale) is int and scale > 0:
            steps.append(scale)
        nodes.add(f"Empty{latent.get('class')}LatentImage")
    for example in facts.examples:
        workflow = example if isinstance(example, Mapping) else snapshot.recipes.get(example, {})
        nodes.update(workflow.get("node_ids", ()))
    for name in nodes:
        if name.startswith("Empty") and name.endswith("LatentImage"):
            schema = snapshot.schemas.get(name, {})
            steps.extend(
                row["options"]["step"]
                for row in schema.get("inputs", ())
                if row["name"] in ("width", "height")
                and type(row["options"].get("step")) is int
                and row["options"]["step"] > 0
            )
    return lcm(*steps) if steps else None
