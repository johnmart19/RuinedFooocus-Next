from ruinedfooocus.core.paths import data_path, resource_path
import json
from pathlib import Path
from threading import RLock
from ruinedfooocus.core.config_io import save_json


class ResolutionSettings:
    DEFAULT_RESOLUTIONS_FILE = resource_path("defaults", "resolutions.default")
    RESOLUTIONS_FILE = data_path("settings", "resolutions.json")
    CUSTOM_RESOLUTION = "Custom..."

    def __init__(self):
        self.profile = None
        self._custom_lock = RLock()
        self.load_resolutions()

    def set_profile(self, profile=None):
        from ruinedfooocus.core.profile_store import profile_name

        with self._custom_lock:
            self.profile = profile_name(profile)
            return self.load_resolutions()

    def _custom_file(self):
        from ruinedfooocus.core.profile_store import profile_folder

        return profile_folder(self.profile) / "image_sizes.json"

    def _custom_entries(self):
        path = self._custom_file()
        if not path.is_file():
            return []
        if path.stat().st_size > 256 * 1024:
            raise ValueError(
                "Saved image sizes exceed the supported file size. Your file was preserved."
            )
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or not isinstance(value.get("sizes"), list)
        ):
            raise ValueError("Unsupported saved image sizes. Your existing file was preserved.")
        from ruinedfooocus.features.generation.size_plan import dimension

        for entry in value["sizes"]:
            if not isinstance(entry, dict) or any(
                not isinstance(entry.get(key), str) or not entry[key].strip()
                for key in ("name", "family")
            ):
                raise ValueError("Invalid saved image size. Your existing file was preserved.")
            dimension(entry.get("width"))
            dimension(entry.get("height"))
        return value["sizes"]

    def save_custom(self, name, width, height, family):
        from ruinedfooocus.features.generation.size_plan import dimension
        from ruinedfooocus.features.models.image_scope import require_image_family

        require_image_family(family)

        width, height = dimension(width), dimension(height)
        name = str(name).strip()
        if not name or len(name) > 80 or name in self.builtin_names:
            raise ValueError("Choose a name of 1–80 characters that differs from built-in sizes.")
        with self._custom_lock:
            entries = [
                item
                for item in self._custom_entries()
                if (item["name"], item["family"]) != (name, family)
            ]
            entries.append(dict(name=name, family=family, width=width, height=height))
            save_json(self._custom_file(), dict(schema_version=1, sizes=entries))
            self.load_resolutions()
        return f"{width}x{height} ({name} · {family})"

    def load_resolutions(self):
        self.base_ratios = {}

        data = {}
        if self.RESOLUTIONS_FILE.is_file():
            with self.RESOLUTIONS_FILE.open() as f:
                data = json.load(f)
        self._saved_resolutions = dict(data)
        # Ignore obsolete recipes while preserving the original saved entries.
        self._obsolete_resolutions = {
            key: value
            for key, value in data.items()
            if value.get("video_models") and not value.get("image_models")
        }
        data = {key: value for key, value in data.items() if key not in self._obsolete_resolutions}

        # Append new defaults without replacing the user's saved resolutions.
        with self.DEFAULT_RESOLUTIONS_FILE.open() as f:
            for ratio, res in json.load(f).items():
                data.setdefault(ratio, res)

        self.profile_custom = {}
        self.custom_errors = []
        try:
            entries = self._custom_entries()
        except (OSError, ValueError) as error:
            self.custom_errors.append(str(error))
            entries = []
        from ruinedfooocus.features.models.image_scope import retired_family

        for entry in entries:
            if retired_family(entry["family"]):
                continue
            name = f"{entry['name']} · {entry['family']}"
            data[name] = dict(
                width=entry["width"], height=entry["height"], image_models=[entry["family"]]
            )
            self.profile_custom[name] = (entry["name"], entry["family"])
        self.image_models = {name: res.get("image_models", []) for name, res in data.items()}
        with self.DEFAULT_RESOLUTIONS_FILE.open() as f:
            self.builtin_names = set(json.load(f))

        # Square, landscape, portrait; then ratio and pixel area within each group.
        def order(item):
            name, res = item
            width, height = res["width"], res["height"]
            return (
                0 if width == height else 1 if width > height else 2,
                max(width, height) / min(width, height),
                width * height,
                name.casefold(),
            )

        for ratio, res in sorted(data.items(), key=order):
            self.base_ratios[ratio] = (res["width"], res["height"])

        self.aspect_ratios = {f"{v[0]}x{v[1]} ({k})": v for k, v in self.base_ratios.items()}
        self.aspect_ratios.update(
            {f"{width}x{height}": (width, height) for width, height in self.base_ratios.values()}
        )

        return self.base_ratios

    def save_resolutions(self, res_options):
        formatted_options = dict(self._obsolete_resolutions)
        for k in res_options:
            if k in self.builtin_names or k in self.profile_custom:
                continue
            from ruinedfooocus.features.generation.size_plan import dimension

            formatted_options[k] = dict(self._saved_resolutions.get(k, {}))
            formatted_options[k].update(
                width=dimension(res_options[k][0]), height=dimension(res_options[k][1])
            )
            if self.image_models.get(k):
                formatted_options[k]["image_models"] = self.image_models[k]

        save_json(self.RESOLUTIONS_FILE, formatted_options)

        return self.load_resolutions()

    def choices_for_model(self, family, checkpoint=None, performance=None):
        from ruinedfooocus.features.models.image_scope import require_image_family

        require_image_family(family)
        choices = [
            f"{width}x{height} ({name})"
            for name, (width, height) in self.base_ratios.items()
            if (
                (not self.image_models[name] or family in self.image_models[name])
                and self.is_custom(f"{width}x{height} ({name})")
            )
        ]
        recommended = self.recommended_sizes(family, checkpoint, performance)
        for width, height in recommended:
            self.aspect_ratios[f"{width}x{height}"] = (width, height)
            if not any(self.aspect_ratios[value] == (width, height) for value in choices):
                label = f"{width}x{height} ({family} recommended)"
                self.aspect_ratios[label] = (width, height)
                choices.append(label)
        if not choices and not recommended:
            label = "1024x1024 (Generic fallback)"
            self.aspect_ratios[label] = (1024, 1024)
            self.aspect_ratios["1024x1024"] = (1024, 1024)
            choices.append(label)
        choices.sort(key=lambda value: self.aspect_ratios[value] not in recommended)
        return choices + [self.CUSTOM_RESOLUTION]

    def recommendation(self, family, checkpoint=None, performance=None):
        """Read bounded, local recommendation data without fetching or sampling."""
        import re
        from ruinedfooocus.features.models.model_recommendations import FAMILY_PRESETS

        if performance is not None and checkpoint:
            size, source = performance.model_recommendations.cached_resolution(checkpoint)
            if size:
                return [size], source
            metadata = performance.model_recommendations.metadata(checkpoint)
            config = metadata.get("generationConfig", {})
            if (
                source is None
                and isinstance(config, dict)
                and not metadata.get("_rf_stale_native_family")
                and self._current_metadata(checkpoint, metadata)
            ):
                size = config.get("width"), config.get("height")
                if all(
                    type(value) is int and 256 <= value <= 4096 and value % 2 == 0 for value in size
                ):
                    return [size], "Cached publisher generation settings"
            preset = performance.family_preset(family, checkpoint)
        else:
            preset = FAMILY_PRESETS.get(family, family)
        from ruinedfooocus.features.models.model_recommendations import family_recipe

        model_recipe = {}
        details = getattr(
            getattr(performance, "model_recommendations", None), "family_details", None
        )
        if checkpoint and callable(details):
            model_recipe = details(checkpoint, family=family)
        if not isinstance(model_recipe, dict) or not model_recipe:
            model_recipe = family_recipe(preset)
        if model_recipe:
            sizes = [tuple(size) for size in model_recipe.get("recommended_sizes", ())]
            default = model_recipe.get("default_size")
            if default:
                default = tuple(default)
                sizes = [default] + [size for size in sizes if size != default]
            source = model_recipe.get("provenance", {}).get("label", f"Local {preset} recipe")
            return sizes, source
        example_reader = getattr(performance, "native_example", None)
        if checkpoint and callable(example_reader):
            example = example_reader(family, checkpoint)
            size = example.get("default_size") if example else None
            if (
                isinstance(size, (list, tuple))
                and len(size) == 2
                and all(
                    type(value) is int and 256 <= value <= 4096 and value % 2 == 0 for value in size
                )
            ):
                return [
                    tuple(size)
                ], "Native workflow example canvas; no publisher size recommendation"
        from ruinedfooocus.features.models.resource_defaults import performance_defaults

        recipes = performance_defaults()
        recipe = recipes.get(preset, {})
        sizes = [
            (int(w), int(h))
            for w, h in re.findall(r"(\d+)x(\d+)", recipe.get("recommended_resolutions", ""))
        ]
        return sizes, f"Local {preset} recipe" if sizes else ""

    @staticmethod
    def _current_metadata(checkpoint, metadata):
        from ruinedfooocus.features.models.model_identity import file_signature
        import sys

        try:
            selected = Path(checkpoint)
            if not selected.is_absolute():
                models = getattr(sys.modules.get("ruinedfooocus.app.state"), "models", None)
                selected = (
                    models.get_file("checkpoints", checkpoint) if models is not None else None
                )
            return selected is not None and file_signature(selected) == metadata.get(
                "_rf_file_signature"
            )
        except (OSError, ValueError, TypeError):
            return False

    def recommended_sizes(self, family, checkpoint=None, performance=None):
        return self.recommendation(family, checkpoint, performance)[0]

    def model_selection(
        self, family, current, automatic, checkpoint=None, performance=None, *, force=False
    ):
        """Follow model defaults until a deliberate size or replay takes ownership."""
        choices = self.choices_for_model(family, checkpoint, performance)
        # Library setup replay restores dimensions through Custom, independently
        # of the previous session's automatic-size checkbox.
        automatic = automatic is True and (force or current != self.CUSTOM_RESOLUTION)
        sizes = self.recommended_sizes(family, checkpoint, performance)
        if automatic:
            current = (
                next(value for value in choices if self.aspect_ratios.get(value) == sizes[0])
                if sizes
                else choices[0]
            )
        elif current not in choices:
            if current in self.aspect_ratios:
                choices.insert(-1, current)
            else:
                import re

                numeric = re.fullmatch(r"(\d+)x(\d+)(?: \(.+\))?", str(current))
                if numeric and min(map(int, numeric.groups())) > 0:
                    self.aspect_ratios[current] = tuple(map(int, numeric.groups()))
                    choices.insert(-1, current)
                elif current == self.CUSTOM_RESOLUTION:
                    pass
                else:
                    current = choices[0]
        return choices, current, automatic

    def recommendation_text(
        self,
        family,
        locale=None,
        *,
        checkpoint=None,
        performance=None,
        selection=None,
        width=None,
        height=None,
        automatic=None,
        source_size=None,
        framing="preserve",
    ):
        from ruinedfooocus.core.translation_manager import localize

        sizes, source = self.recommendation(family, checkpoint, performance)
        text = (
            localize(
                "Recommended: {sizes} ({source}).",
                locale,
                sizes=", ".join(f"{w} × {h}" for w, h in sizes),
                source=source,
            )
            if sizes
            else localize("No recommended size published in the local family recipe.", locale)
        )
        if self.custom_errors:
            text += "  \n" + " ".join(self.custom_errors)
        details = getattr(
            getattr(performance, "model_recommendations", None), "family_details", None
        )
        recipe = details(checkpoint, family=family) if checkpoint and callable(details) else {}
        working = recipe.get("working_recipe", {}) if isinstance(recipe, dict) else {}
        if working.get("default_size"):
            width_working, height_working = working["default_size"]
            text += "  \n" + localize(
                "Working recipe: {width} × {height} ({source}).",
                locale,
                width=width_working,
                height=height_working,
                source=working.get("provenance", {}).get("label", "local working recipe"),
            )
        effective = self.aspect_ratios.get(selection, (width, height)) if selection else None
        if effective and all(value is not None for value in effective):
            text += "  \n" + localize(
                "Selected size: {width} × {height}.",
                locale,
                width=effective[0],
                height=effective[1],
            )
            if automatic is False or selection == self.CUSTOM_RESOLUTION:
                text += " " + localize("Your size override is preserved.", locale)
            if source_size:
                from ruinedfooocus.features.image_editing.image_geometry import reference_plan

                frame = reference_plan(
                    source_size, effective, framing, multiple=32 if family == "HiDreamO1" else 16
                )
                text += "  \n" + localize(
                    "Planned source frame: {width} × {height} ({framing}). Native canvas alignment is applied separately.",
                    locale,
                    width=frame["width"],
                    height=frame["height"],
                    framing=framing,
                )
                if framing == "preserve":
                    text += " " + localize(
                        "The selected size controls the working pixel budget; source proportions are preserved.",
                        locale,
                    )
        return text

    def is_custom(self, selection):
        return any(
            f"{w}x{h} ({name})" == selection and name not in self.builtin_names
            for name, (w, h) in self.base_ratios.items()
        )

    def remove_custom(self, selection):
        name = next(
            (k for k, v in self.base_ratios.items() if f"{v[0]}x{v[1]} ({k})" == selection), None
        )
        if name is None or name in self.builtin_names:
            raise ValueError("Only saved custom resolutions can be removed.")
        if name in self.profile_custom:
            user_name, family = self.profile_custom[name]
            with self._custom_lock:
                entries = [
                    item
                    for item in self._custom_entries()
                    if (item["name"], item["family"]) != (user_name, family)
                ]
                save_json(self._custom_file(), dict(schema_version=1, sizes=entries))
                self.load_resolutions()
            return
        options = dict(self.base_ratios)
        del options[name]
        self.save_resolutions(options)

    def selection_for_model(self, family, current):
        choices = self.choices_for_model(family)
        return current if current in choices else choices[0]

    def get_aspect_ratios(self, name):
        return self.aspect_ratios[name]
