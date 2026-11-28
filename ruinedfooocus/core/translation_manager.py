"""Validated UI catalogues and Gradio native frontend translation metadata."""

from functools import lru_cache
import html
import json
from pathlib import Path
from string import Formatter
from ruinedfooocus.core.paths import data_path, resource_path


LOCALE_COOKIE = "rf_ui_locale"


def native_key(key):
    """Keep app messages separate from Gradio's own common.* messages."""
    return f"ruinedfooocus.{key}"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate translation ID: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)


def placeholders(text):
    return {field for _, field, _, _ in Formatter().parse(text) if field is not None}


def _placeholder_signature(text):
    # Conversions and format specifications are part of a message's contract.
    return {
        (field, spec, conversion)
        for _, field, spec, conversion in Formatter().parse(text)
        if field is not None
    }


def validate_catalogue(data, english=None):
    if not isinstance(data, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in data.items()
    ):
        raise ValueError("A translation catalogue must map string IDs to strings")
    for key, value in data.items():
        fields = _placeholder_signature(value)
        if english is not None:
            if key not in english:
                raise ValueError(f"Obsolete translation ID: {key}")
            if fields != _placeholder_signature(english[key]):
                raise ValueError(f"Translation placeholders differ: {key}")
    return data


class TranslationManager:
    def __init__(self, root=None, legacy_root=None):
        self.root = Path(root) if root is not None else resource_path("translations")
        self.language = "en"
        self.errors = []
        self.translations = {"en": {}}
        self.aliases = {}
        self.manifest = {"locales": {"en": {"name": "English", "review": "source"}}}
        try:
            self.translations["en"] = validate_catalogue(read_json(self.root / "en.json"))
        except (OSError, ValueError, TypeError) as error:
            self.errors.append(f"en: {error}")
        try:
            self.manifest = read_json(self.root / "manifest.json")
            if not isinstance(self.manifest, dict) or not isinstance(
                self.manifest.get("locales"), dict
            ):
                raise ValueError("Locale manifest must contain a locales object")
            if "en" not in self.manifest["locales"] or any(
                not isinstance(code, str)
                or not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or not item["name"].strip()
                or item.get("review") not in ("source", "draft", "reviewed")
                for code, item in self.manifest["locales"].items()
            ):
                raise ValueError(
                    "Locale manifest must declare English, locale names and review status"
                )
        except (OSError, ValueError, TypeError) as error:
            self.errors.append(str(error))
            self.manifest = {"locales": {"en": {"name": "English", "review": "source"}}}
        try:
            aliases = validate_catalogue(read_json(self.root / "legacy-keys.json"))
            if any(key not in self.translations["en"] for key in aliases.values()):
                raise ValueError("Legacy aliases must refer to English message IDs")
            self.aliases = aliases
        except (OSError, ValueError, TypeError) as error:
            self.errors.append(f"Legacy aliases: {error}")
        for locale in self.manifest.get("locales", {}):
            if locale == "en":
                continue
            try:
                self.translations[locale] = validate_catalogue(
                    read_json(self.root / f"{locale}.json"), self.translations["en"]
                )
            except (OSError, ValueError, TypeError) as error:
                self.errors.append(f"{locale}: {error}")
        # Existing custom legacy catalogues are read-only; no packaged duplicate.
        legacy_roots = (
            (Path(legacy_root),)
            if legacy_root is not None
            else (self.root.parent / "language", data_path("language"))
        )
        for legacy_directory in dict.fromkeys(legacy_roots):
            for path in sorted(legacy_directory.glob("*.json")):
                try:
                    custom = validate_catalogue(read_json(path))
                    mapped = {self.aliases.get(k, k): v for k, v in custom.items()}
                    if len(mapped) != len(custom):
                        raise ValueError("Duplicate legacy translation ID after alias migration")
                    validate_catalogue(mapped, self.translations["en"])
                    self.translations.setdefault(path.stem, {}).update(mapped)
                except (OSError, ValueError, TypeError) as error:
                    self.errors.append(f"Legacy {path.name}: {error}")

    def set_language(self, language):
        """Compatibility server default; never call this from a client callback."""
        self.language = language or "en"

    def translate(self, text, mapping=None, *, locale=None, escape=False):
        key = self.aliases.get(text, text)
        english = self.translations["en"].get(key, text)
        result = self.translations.get(locale or self.language, {}).get(key, english)
        if mapping is not None:
            values = {
                k: html.escape(str(v)) if escape and not isinstance(v, (int, float)) else v
                for k, v in mapping.items()
            }
            try:
                result = result.format_map(values)
            except (KeyError, ValueError, TypeError):
                try:
                    result = english.format_map(values)
                except (KeyError, ValueError, TypeError):
                    result = english
        return result

    def native(self):
        import gradio as gr

        english = self.translations["en"]
        return gr.I18n(
            **{
                locale: {
                    # Gradio 6.31's status live region requests this absent native key.
                    "common.complete": "Complete",
                    **{native_key(key): text for key, text in (english | values).items()},
                }
                for locale, values in self.translations.items()
            }
        )


@lru_cache(maxsize=1)
def catalogue():
    return TranslationManager()


def translate_ui(text):
    """Mark registered static properties; unknown literals retain English safely."""
    manager = catalogue()
    key = manager.aliases.get(text, text)
    if key not in manager.translations["en"]:
        return text
    from gradio.i18n import I18nData

    return I18nData(native_key(key))


def resolve_locale(value=None, request=None):
    """Return the catalogue locale a browser uses for server-built text.

    An explicit choice wins; "auto" or no choice uses the language cookie set by
    the picker (not for "auto"), then the browser's Accept-Language, then English.
    """
    known = {code.casefold(): code for code in catalogue().translations}
    candidates = [] if value in (None, "", "auto") else [value]
    if request is not None:
        if value != "auto":
            candidates.append((getattr(request, "cookies", None) or {}).get(LOCALE_COOKIE))
        header = (getattr(request, "headers", None) or {}).get("accept-language", "")
        for part in header.split(","):
            code = part.split(";")[0].strip()
            candidates += [code, code.split("-")[0]]
    for code in candidates:
        if code and (match := known.get(str(code).replace("_", "-").casefold())):
            return match
    return "en"


def localize(text, locale=None, /, **values):
    """Translate server-built text for one browser; English stays the source.

    Use literal text so tools/check_translations.py registers it.
    """
    return catalogue().translate(text, values or None, locale=locale or "en")


_LOCALIZED_OUTPUTS = []


def localized_output(fn, inputs, outputs):
    """Re-render server-built text after a language change or page load.

    ``fn(locale, *inputs)`` returns values for ``outputs``.
    """
    _LOCALIZED_OUTPUTS.append((fn, list(inputs), list(outputs)))


def localized_outputs():
    return list(_LOCALIZED_OUTPUTS)


def locale_choices():
    # Review status stays in the manifest and docs; the picker shows language and region only.
    return [(translate_ui("Auto (browser language)"), "auto")] + [
        (item["name"], code) for code, item in catalogue().manifest["locales"].items()
    ]


def normalize_locale(value):
    """Legacy spelling migration; UI locales have no model-language restrictions."""
    value = {
        "pt_br": "pt-BR",
        "pt-br": "pt-BR",
        "zh-cn": "zh-CN",
        "zh_cn": "zh-CN",
        "zh-tw": "zh-TW",
        "zh_tw": "zh-TW",
    }.get(value, value)
    return value if value == "auto" or value in catalogue().translations else "en"


def locale_change_js(default="auto", *, initialize=False):
    """Use the installed native reactive store, without rebuilding components."""
    import gradio

    assets = Path(gradio.__file__).parent / "templates" / "frontend" / "assets"
    cores = [
        p for p in assets.glob("core-*.js") if " as changeLocale," in p.read_text(encoding="utf-8")
    ]
    if len(cores) != 1:
        raise RuntimeError("Installed Gradio has no unambiguous native changeLocale export")
    script = (
        resource_path("translations", "locale.js")
        .read_text(encoding="utf-8")
        .strip()
        .removesuffix(";")
    )
    return (
        script.replace("__CORE_ASSET__", json.dumps(cores[0].name))
        .replace("__DEFAULT_LOCALE__", json.dumps(normalize_locale(default)))
        .replace("__INITIALIZE__", "true" if initialize else "false")
    )
