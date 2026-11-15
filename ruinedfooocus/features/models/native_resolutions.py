"""Native ComfyUI orientation choices at the selected model's pixel budget."""

from functools import lru_cache
import logging


def aspect_presets(width, height, multiple=None):
    """Reuse the pinned node; no sampling, downloads or device initialization."""
    try:
        return _aspect_presets(width, height, multiple)
    except ImportError:
        logging.getLogger(__name__).debug("Native resolution selector is not initialized")
        return ()


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
