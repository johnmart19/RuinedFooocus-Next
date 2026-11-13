"""Bundled and user assistants; model selection is independent of persona."""

import base64
import json
import hashlib
import io
import re
import threading
import uuid
from pathlib import Path

from PIL import Image, ImageOps

from ruinedfooocus.core.translation_manager import translate_ui
from ruinedfooocus.core.paths import APP_ROOT, asset_path, content_roots, data_path, resource_path
from ruinedfooocus.core.config_io import save_bytes, save_json

_SAVE_LOCK = threading.RLock()
_USER_ID = re.compile(r"user:([a-z0-9][a-z0-9_-]{0,63})\Z")
_AVATAR_FILE = re.compile(r"avatar-[0-9a-f]{64}\.png\Z")
_KEEP_AVATAR = object()


def _saved_path(identity):
    if identity != "__normal__":
        return user_assistant_path(identity)
    root = data_path("assistants").resolve()
    path = root / ".general" / "info.json"
    if path.resolve().parent.parent != root:
        raise ValueError("Assistant path is outside the saved catalogue.")
    return path


def _avatar(path, info):
    name = info.get("avatar_file")
    if isinstance(name, str) and _AVATAR_FILE.fullmatch(name) and (path / name).is_file():
        return path / name
    if "avatar_file" not in info and (path / "avatar.png").is_file():
        return path / "avatar.png"
    return asset_path("chat_normal.png")


def _general_info(raw=_KEEP_AVATAR):
    info = {"name": translate_ui("General chat"), "system": ""}
    path = _saved_path("__normal__")
    try:
        if raw is _KEEP_AVATAR:
            raw = path.read_bytes()
        saved = json.loads(raw) if raw is not None else {}
        if isinstance(saved.get("name"), str) and saved["name"].strip():
            info["name"] = saved["name"]
        if isinstance(saved.get("system"), str):
            info["system"] = saved["system"]
        info["avatar_file"] = saved.get("avatar_file")
    except (OSError, ValueError, AttributeError):
        pass
    return dict(info, avatar=_avatar(path.parent, info), embed="[]", chatstart=[])


def user_assistant_path(identity):
    """Only stable catalogue IDs can name writable assistant files."""
    match = _USER_ID.fullmatch(identity) if isinstance(identity, str) else None
    if not match:
        raise ValueError("Invalid saved assistant ID.")
    root = data_path("assistants").resolve()
    path = root / match[1] / "info.json"
    if path.resolve().parent.parent != root:
        raise ValueError("Assistant path is outside the saved catalogue.")
    return path


def assistant_revision(identity):
    path = _saved_path(identity)
    if identity == "__normal__" and not path.exists():
        return "builtin-general-v1"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assistant_edit_snapshot(selected):
    """Read the draft and its conflict token from the same bytes."""
    selected = assistant_identity(selected)
    if selected == "__normal__":
        try:
            raw = _saved_path(selected).read_bytes()
        except FileNotFoundError:
            raw = None
        revision = hashlib.sha256(raw).hexdigest() if raw is not None else "builtin-general-v1"
        return _general_info(raw), selected, revision
    if selected.startswith("user:"):
        raw = user_assistant_path(selected).read_bytes()
        return json.loads(raw), selected, hashlib.sha256(raw).hexdigest()
    return assistant_info(selected), None, None


def save_assistant(name, system, identity=None, revision=None, *, guest=False, avatar=_KEEP_AVATAR):
    """Create a distinct entry or compare-and-save an explicitly opened draft.

    Display-name collisions are allowed; an existing ID needs its exact revision.
    Atomic replacement keeps the previous definition intact if publication fails.
    """
    if guest:
        raise PermissionError("Guest sessions cannot save assistants.")
    if not isinstance(name, str) or not name.strip() or len(name) > 200:
        raise ValueError("Enter an assistant name of at most 200 characters.")
    if not isinstance(system, str) or len(system) > 100000:
        raise ValueError("The system prompt must contain at most 100000 characters.")
    with _SAVE_LOCK:
        creating = identity is None
        identity = identity or "user:assistant-" + uuid.uuid4().hex
        path = _saved_path(identity)
        if creating:
            if path.parent.exists():
                raise ValueError("Assistant ID already exists. Save again to create a new ID.")
            info = {}
        else:
            if (identity != "__normal__" and not path.is_file()) or assistant_revision(
                identity
            ) != revision:
                raise ValueError(
                    "This assistant changed or was removed. Reopen Edit before saving."
                )
            info = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if not isinstance(info, dict):
            raise ValueError("Assistant definition is not a JSON object.")
        info.update(name=name.strip(), system=system)
        image_path = None
        created_image = False
        if avatar is not _KEEP_AVATAR:
            info["avatar_file"] = None
            if avatar is not None:
                if not isinstance(avatar, Image.Image):
                    raise ValueError("Upload an assistant image first.")
                image = ImageOps.exif_transpose(avatar)
                image = image.convert("RGBA" if image.has_transparency_data else "RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                raw = buffer.getvalue()
                info["avatar_file"] = "avatar-" + hashlib.sha256(raw).hexdigest() + ".png"
                image_path = path.parent / info["avatar_file"]
                created_image = not image_path.exists()
                if created_image:
                    save_bytes(image_path, raw)
        try:
            save_json(path, info)
        except BaseException:
            if created_image:
                image_path.unlink(missing_ok=True)
            raise
        return identity, assistant_revision(identity)


def delete_assistant(identity, revision=None, *, guest=False, before_delete=None):
    """Delete an owned definition or hide a bundled entry; never delete shared images."""
    if guest:
        raise PermissionError("Guest sessions cannot delete assistants.")
    if not isinstance(identity, str) or not identity:
        raise ValueError("Select an available assistant first.")
    identity = assistant_identity(identity)
    if identity == "__normal__":
        raise ValueError("General chat cannot be deleted.")
    with _SAVE_LOCK:
        if identity not in {value for _, value in assistant_choices()}:
            raise ValueError("This assistant is no longer available.")
        if identity.startswith("user:"):
            path = user_assistant_path(identity)
            if assistant_revision(identity) != revision:
                raise ValueError("This assistant changed. Reopen Delete before confirming.")
            if before_delete is not None:
                before_delete()
            path.unlink()
            for image in path.parent.iterdir():
                if image.is_file() and _AVATAR_FILE.fullmatch(image.name):
                    try:
                        image.unlink()
                    except OSError as error:
                        print(f"Cannot remove unused assistant image {image}: {error}")
            if not any(path.parent.iterdir()):
                try:
                    path.parent.rmdir()
                except OSError:
                    pass
        else:
            if before_delete is not None:
                before_delete()
            hidden = _hidden_assistants()
            save_json(data_path("assistants", ".hidden.json"), sorted(hidden | {identity}))


def _hidden_assistants():
    try:
        values = json.loads(data_path("assistants", ".hidden.json").read_text(encoding="utf-8"))
        return (
            {item for item in values if isinstance(item, str)}
            if isinstance(values, list)
            else set()
        )
    except (OSError, ValueError):
        return set()


def character_card(path):
    with Image.open(path) as image:
        encoded = image.info.get("chara")
    if not encoded:
        return None
    try:
        card = json.loads(base64.b64decode(encoded))
    except (ValueError, RecursionError) as error:
        raise ValueError("Unsupported character card") from error
    data = card.get("data", card) if isinstance(card, dict) else None
    if not isinstance(data, dict) or not all(
        isinstance(data.get(key), str) for key in ("name", "first_mes", "personality", "scenario")
    ):
        raise ValueError("Unsupported character card")
    return {
        "name": data["name"],
        "greeting": data["first_mes"],
        "avatar": path,
        "system": f"Your name is {data['name']}.\nYou are: {data['personality']}\nScenario: {data['scenario']}",
        "embed": [["text", f"Summary: {data.get('description', data.get('summary', ''))}"]],
    }


def assistant_choices():
    choices = {}
    hidden = _hidden_assistants()
    for folder in content_roots("assistants", "chatbots"):
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            try:
                if path.name == "info.json":
                    if path.resolve() == _saved_path("__normal__").resolve():
                        continue
                    info = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(info, dict) or not isinstance(info.get("name"), str):
                        raise ValueError("Assistant definition needs a name.")
                    identity = str(path.parent)
                    if path.parent.parent.resolve() == data_path("assistants").resolve():
                        candidate = "user:" + path.parent.name
                        if _USER_ID.fullmatch(candidate):
                            identity = candidate
                    choices[path.parent.relative_to(folder).as_posix()] = (
                        info["name"],
                        identity,
                    )
                elif path.suffix.lower() == ".png" and not path.with_name("info.json").exists():
                    info = character_card(path)
                    if info:
                        choices[path.relative_to(folder).as_posix()] = (info["name"], str(path))
            except (OSError, ValueError, KeyError) as error:
                print(f"Cannot read assistant {path}: {error}")
    from ruinedfooocus.core.translation_manager import translate_ui

    return [(_general_info()["name"], "__normal__")] + sorted(
        (item for item in choices.values() if item[1] not in hidden),
        key=lambda item: item[0].casefold(),
    )


def assistant_identity(selected):
    """Resolve legacy absolute user selections without accepting arbitrary files."""
    choices = {value for _, value in assistant_choices()}
    if selected in choices:
        return selected
    if isinstance(selected, str) and Path(selected).is_absolute():
        for value in choices:
            if (
                value.startswith("user:")
                and user_assistant_path(value).parent == Path(selected).resolve()
            ):
                return value
    return selected or "__normal__"


def assistant_info(selected):
    selected = assistant_identity(selected)
    if selected == "__normal__":
        return _general_info()
    try:
        if (selected or "").startswith("user:"):
            path = user_assistant_path(selected).parent
        else:
            # Legacy absolute selections still work, but only for known catalogue entries.
            known = {value for _, value in assistant_choices()}
            if selected not in known:
                raise ValueError("Assistant is not in the catalogue")
            path = Path(selected)
        if path.is_dir():
            info = json.loads((path / "info.json").read_text(encoding="utf-8"))
            if not isinstance(info, dict):
                raise ValueError("Assistant definition is not a JSON object.")
            info["avatar"] = _avatar(path, info)
        else:
            info = character_card(path)
        if not info:
            raise ValueError("No character definition")
        info["embed"] = json.dumps(info.get("embed", []))
        info.setdefault("system", "")
        greeting = info.get("greeting")
        info["chatstart"] = [{"role": "assistant", "content": greeting}] if greeting else []
        return info
    except (OSError, ValueError, KeyError) as error:
        print(f"Cannot load assistant {selected}: {error}")
        return {
            "name": "Unavailable assistant",
            "avatar": asset_path("error.png"),
            "system": "",
            "embed": "[]",
            "chatstart": [],
        }


def context_file(value):
    """Resolve local context without allowing files outside application/data roots."""
    candidate = Path(value)
    roots = (data_path(), APP_ROOT)
    candidates = (
        [base / candidate for base in roots] if not candidate.is_absolute() else [candidate]
    )
    if not candidate.is_absolute() and candidate.parts and candidate.parts[0] == "chatbots":
        candidates.append(resource_path("assistants", *candidate.parts[1:]))
    for path in candidates:
        path = path.resolve()
        if any(base.resolve() in path.parents for base in roots) and path.is_file():
            return path
    raise ValueError(f"Chat context file not found inside the application/data folder: {value}")
