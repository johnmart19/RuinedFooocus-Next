"""Shared request profiles for native and embedded llama.cpp clients."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from functools import lru_cache
import math
from pathlib import Path
import shlex

from ruinedfooocus.features.models.model_registry import load_registry, RegistryError
from ruinedfooocus.core.paths import resource_path

COMPACT_CANDIDATE = "LFM2.5-1.2B-Instruct-Q8_0.gguf"
COMPACT_SHA256 = "f6b981dcb86917fa463f78a362320bd5e2dc45445df147287eedb85e5a30d26a"
COMPACT_TEMPLATE_SHA256 = "ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6"
COMPACT_TEMPLATE = "lfm25-instruct-publisher.jinja"
QWEN3_CANDIDATE = "Qwen3-1.7B-Q8_0.gguf"
QWEN3_SHA256 = "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a"


@dataclass(frozen=True)
class ChatLoadProfile:
    settings: dict
    native_defaults: tuple = ()
    candidate: bool = False
    template_path: Path | None = None
    template: str | None = None
    template_sha256: str | None = None
    batch: int | None = None
    microbatch: int | None = None

    @property
    def identity(self):
        return (self.template_sha256, self.batch, self.microbatch)


def chat_load_profile(artifact, values):
    """Resolve load defaults only from an already content-verified annotation.

    Callers obtain artifact through verified_local_artifact; catalogue membership
    or a same-named missing/custom file must never supply this argument. Template
    correction is independent of the publisher-sampling preference. Saved values
    (including zero/None) and explicit native CLI remain authoritative.
    """
    settings = deepcopy(values)
    native = (
        tuple(native_load_defaults(artifact)) if values.get("llm_use_model_defaults", True) else ()
    )
    if (
        artifact is not None
        and artifact.get("id") == "existing:llm/" + QWEN3_CANDIDATE
        and artifact.get("profile_id") == "language:qwen3_1_7b"
        and artifact.get("artifact", {}).get("bytes") == 1834426016
        and artifact.get("artifact", {}).get("sha256") == QWEN3_SHA256
    ):
        settings.setdefault("llm_n_ctx", 2048)
        settings.setdefault("llm_n_predict", 256)
        # Use the verified artifact's own publisher template. The initialized
        # client binds reasoning controls to those actual template bytes.
        return ChatLoadProfile(
            settings,
            native + ("--batch-size", "256", "--ubatch-size", "128"),
            candidate=True,
            batch=256,
            microbatch=128,
        )
    if not (
        artifact is not None
        and artifact.get("id") == "existing:llm/" + COMPACT_CANDIDATE
        and artifact.get("profile_id") == "language:lfm25_1_2b"
        and artifact.get("artifact", {}).get("bytes") == 1246253888
        and artifact.get("artifact", {}).get("sha256") == COMPACT_SHA256
    ):
        return ChatLoadProfile(settings, native)
    settings.setdefault("llm_n_ctx", 2048)
    settings.setdefault("llm_n_predict", 256)
    native += ("--batch-size", "256", "--ubatch-size", "128")
    flags = {
        token.partition("=")[0] for token in shlex.split(settings.get("llama_server_args", ""))
    }
    custom_template = settings.get("llm_runtime", "llama.cpp") == "llama.cpp" and any(
        flag in ("--chat-template", "--chat-template-file", "--no-jinja") for flag in flags
    )
    if custom_template:
        return ChatLoadProfile(settings, native, candidate=True, batch=256, microbatch=128)
    path = resource_path("chat_templates", COMPACT_TEMPLATE)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != COMPACT_TEMPLATE_SHA256:
        raise ValueError(
            "The bundled LFM2.5 publisher template has changed; restore the shipped resource."
        )
    return ChatLoadProfile(
        settings,
        native + ("--chat-template-file", str(path)),
        candidate=True,
        template_path=path,
        template=payload.decode("utf-8"),
        template_sha256=COMPACT_TEMPLATE_SHA256,
        batch=256,
        microbatch=128,
    )


def configure_embedded_profile(params, profile):
    """Apply the same verified load plan to the installed embedded binding."""
    if profile.candidate:
        params.n_batch = profile.batch
        params.n_ubatch = profile.microbatch
        if profile.template is not None:
            params.chat_template = profile.template


SAMPLERS = {
    "temperature": (0, None),
    "top_p": (0, 1),
    "top_k": (0, None),
    "min_p": (0, 1),
    "typical_p": (0, 1),
    "repeat_penalty": (0, None),
    "presence_penalty": (None, None),
    "frequency_penalty": (None, None),
}
CLI_SAMPLERS = {"--temp": "temperature", **{"--" + key.replace("_", "-"): key for key in SAMPLERS}}


def validate_sampling(values):
    if not isinstance(values, Mapping):
        raise ValueError("Chat sampling overrides must be an object.")
    result = {}
    for name, value in values.items():
        if name not in SAMPLERS:
            raise ValueError(f"Unsupported Chat sampling option: {name}")
        low, high = SAMPLERS[name]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (low is not None and value < low)
            or (high is not None and value > high)
            or (name == "top_k" and type(value) is not int)
        ):
            raise ValueError(f"Invalid Chat sampling value: {name}")
        result[name] = value
    return result


def cli_sampling(arguments):
    tokens = shlex.split(arguments or "")
    result = {}
    for index, token in enumerate(tokens):
        flag, separator, value = token.partition("=")
        name = CLI_SAMPLERS.get(flag)
        if name:
            value = value if separator else tokens[index + 1] if index + 1 < len(tokens) else ""
            try:
                result[name] = int(value) if name == "top_k" else float(value)
            except ValueError:
                raise ValueError(f"{flag} requires a numeric value.") from None
    return validate_sampling(result)


def profile_for(artifact):
    try:
        return (
            (load_registry().profile(artifact["profile_id"]) or {}) if artifact is not None else {}
        )
    except RegistryError:
        return {}


def native_load_defaults(artifact):
    # The working LFM3 template needs thinking enabled plus a bounded native
    # reasoning budget. Explicit server arguments follow and override this.
    return (
        ["--reasoning-budget", "0"]
        if artifact is not None
        and artifact["profile_id"] in ("language:lfm3_1_7b", "language:atomic_lfm3_1_7b_uncensored")
        else []
    )


def reasoning_kwargs(profile, effort):
    if effort is None:
        return {}
    if not isinstance(effort, str):
        raise ValueError("Choose a listed reasoning effort.")
    thinking = profile.get("capabilities", {}).get("thinking", {})
    control = thinking.get("control") or {}
    if "reasoning_effort" in control:
        allowed = thinking.get("allowed", ("low", "medium", "high"))
        if effort.lower() not in allowed:
            raise ValueError(f"This model supports reasoning effort: {', '.join(allowed)}.")
        return {"reasoning_effort": effort.lower()}
    if "enable_thinking" in control:
        return {
            "enable_thinking": True
            if profile.get("identity", {}).get("gguf_architecture") == "lfm3"
            else effort.lower() != "low"
        }
    return {}


def validate_reasoning_request(values):
    """Keep enabled state separate from the model's optional effort levels."""
    if not isinstance(values, Mapping) or set(values) - {"mode", "effort"}:
        raise ValueError("Reasoning must contain mode and optional effort.")
    mode, effort = values.get("mode", "Auto"), values.get("effort", "Auto")
    if mode not in ("Auto", "Off", "On"):
        raise ValueError("Choose reasoning Auto, Off or On.")
    if effort not in ("Auto", "Low", "Medium", "High"):
        raise ValueError("Choose a listed reasoning effort or Auto.")
    return {"mode": mode, "effort": effort}


def initialized_reasoning(
    artifact, values, template, *, initialized=False, initialized_runtime=None
):
    """Advertise only controls present in this verified, initialized template.

    The loader supplies embedded GGUF template bytes after successful backend
    initialization. A custom native template/global reasoning switch invalidates
    that binding. Parsing a template proves transport support, not model quality.
    """
    result = {
        "modes": ("Auto",),
        "efforts": (),
        "control": None,
        "reason": "Load a verified model to configure reasoning.",
        "template_sha256": None,
    }
    if (
        not initialized
        or artifact is None
        or not artifact.get("artifact", {}).get("sha256")
        or not isinstance(template, str)
        or not template
    ):
        return result
    result["template_sha256"] = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if values.get("llm_runtime", "llama.cpp") == "llama.cpp":
        flags = {
            token.partition("=")[0] for token in shlex.split(values.get("llama_server_args", ""))
        }
        if any(
            flag.startswith(("--chat-template", "--reasoning")) or flag == "--no-jinja"
            for flag in flags
        ):
            return {
                **result,
                "reason": "Custom template or server reasoning arguments control this load. Remove them and reload to use these controls.",
            }
    from ruinedfooocus.features.models.model_capabilities import (
        configured_runtime,
        task_eligibility,
    )

    runtime = initialized_runtime or configured_runtime(values.get("llm_runtime", "llama.cpp"))
    decision = task_eligibility(artifact, runtime)
    if decision.status != "source_supported":
        return {
            **result,
            "reason": "Reasoning controls are not established for this initialized runtime build.",
        }
    profile = profile_for(artifact)
    if profile.get("identity", {}).get("gguf_architecture") == "lfm3":
        return {
            **result,
            "reason": "LFM3 reasoning controls are unavailable while its template and native reasoning budget interaction is unresolved.",
        }
    from jinja2 import Environment, TemplateError, meta

    try:
        variables = meta.find_undeclared_variables(Environment().parse(template))
    except TemplateError:
        return {
            **result,
            "reason": "The initialized template's reasoning controls could not be inspected.",
        }
    thinking = profile.get("capabilities", {}).get("thinking", {})
    control = thinking.get("control") or {}
    if "enable_thinking" in control and "enable_thinking" in variables:
        return {
            **result,
            "modes": ("Auto", "Off", "On"),
            "control": "enable_thinking",
            "reason": "Auto keeps the template default; Off/On sends its thinking flag. No graded effort is established.",
        }
    if "reasoning_effort" in control and "reasoning_effort" in variables:
        efforts = tuple(
            value.title()
            for value in thinking.get("allowed", ())
            if value in ("low", "medium", "high")
        )
        if efforts:
            return {
                **result,
                "modes": ("Auto", "On"),
                "efforts": efforts,
                "control": "reasoning_effort",
                "reason": "This model exposes reasoning effort. Low is not Off; no off switch is established.",
            }
    return {
        **result,
        "reason": "This initialized model/template has no established reasoning control.",
    }


def controlled_reasoning_kwargs(capability, values):
    request = validate_reasoning_request(values)
    mode, effort = request["mode"], request["effort"]
    if mode == "Auto" and effort == "Auto":
        return {}
    if capability["control"] is None:
        raise ValueError(capability["reason"])
    if mode not in capability["modes"]:
        raise ValueError("This model has no established reasoning off switch.")
    if effort != "Auto" and (effort not in capability["efforts"] or mode == "Off"):
        raise ValueError("This model/mode does not support that reasoning effort.")
    if capability["control"] == "enable_thinking":
        return {} if mode == "Auto" else {"enable_thinking": mode == "On"}
    return {} if effort == "Auto" else {"reasoning_effort": effort.lower()}


def initialized_image_tools(
    artifact,
    values,
    template,
    *,
    initialized=False,
    resources=(),
    vision=False,
    initialized_runtime=None,
):
    """Bind tool transport to a loaded runtime, exact template and file identities.

    This is source/template transport evidence, never behavior or image quality
    qualification. Explicit custom templates remain authoritative but unbound.
    """
    result = {
        "available": False,
        "binding": None,
        "template_sha256": None,
        "vision": False,
        "behavior_qualified": False,
        "reason": "Load a source-compatible model with its tool template first.",
    }
    if (
        not initialized
        or artifact is None
        or not artifact.get("artifact", {}).get("sha256")
        or not resources
        or not isinstance(template, str)
        or not template
    ):
        return result
    result["template_sha256"] = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if values.get("llm_runtime", "llama.cpp") == "llama.cpp":
        flags = {
            token.partition("=")[0] for token in shlex.split(values.get("llama_server_args", ""))
        }
        if any(flag.startswith("--chat-template") or flag == "--no-jinja" for flag in flags):
            return {
                **result,
                "reason": "The explicit native template controls this load; image tools are not bound to it.",
            }
    from ruinedfooocus.features.models.model_capabilities import (
        configured_runtime,
        task_eligibility,
    )
    from jinja2 import Environment, TemplateError, meta

    runtime = initialized_runtime or configured_runtime(values.get("llm_runtime", "llama.cpp"))
    if task_eligibility(artifact, runtime).status != "source_supported":
        return {
            **result,
            "reason": "Tool transport is not established for this initialized runtime build.",
        }
    if not profile_for(artifact).get("capabilities", {}).get("tools"):
        return {**result, "reason": "This exact model has no established tool-template contract."}
    try:
        variables = meta.find_undeclared_variables(Environment().parse(template))
    except TemplateError:
        return {**result, "reason": "The initialized tool template could not be inspected."}
    if "tools" not in variables or "tool_calls" not in template:
        return {**result, "reason": "The initialized template does not expose tool calls."}
    identity = {
        "runtime": (runtime.name, runtime.build, runtime.package, runtime.origin),
        "artifact": artifact["artifact"]["sha256"],
        "template": result["template_sha256"],
        "resources": resources,
        "vision": bool(vision),
    }
    return {
        **result,
        "available": True,
        "binding": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        "vision": bool(
            vision
            and len(resources) > 1
            and task_eligibility(artifact, runtime, "image").status == "source_supported"
        ),
        "reason": "Initialized source/template tool transport; model behavior and image quality remain unqualified.",
    }


@lru_cache(maxsize=8)
def _embedded_reasoning_template(path, signature):
    from gguf import GGUFReader

    field = GGUFReader(path).get_field("tokenizer.chat_template")
    template = field.contents() if field else ""
    return template if isinstance(template, str) and len(template) <= 1024 * 1024 else ""


def embedded_reasoning_template(path):
    """Read GGUF metadata only, without importing a tensor/device runtime."""
    from ruinedfooocus.features.models.model_identity import file_signature

    try:
        signature = tuple(file_signature(path))
        template = _embedded_reasoning_template(str(Path(path).resolve()), signature)
        return template if signature == tuple(file_signature(path)) else ""
    except (OSError, ValueError, TypeError, KeyError):
        return ""


def publisher_sampling(profile, template):
    publisher = profile.get("publisher_settings", {})
    config = publisher.get("generation_config", {})
    values = {name: config[name] for name in SAMPLERS if name in config}
    if "repetition_penalty" in config:
        values["repeat_penalty"] = config["repetition_penalty"]
    for recommendation in publisher.get("recommendations", ()):
        task = recommendation.get("task", "").lower()
        if task and not any(word in task for word in ("general", "chat")):
            continue
        if "nonthinking" in task and template.get("enable_thinking", True):
            continue
        if (
            "thinking" in task
            and "nonthinking" not in task
            and not template.get("enable_thinking", True)
        ):
            continue
        values.update({name: recommendation[name] for name in SAMPLERS if name in recommendation})
        break
    return validate_sampling(values)


def compose_sampling(profile, user, cli, data, template):
    """The request and its displayed provenance use one validated precedence path."""
    explicit = validate_sampling({key: value for key, value in data.items() if key in SAMPLERS})
    layers = [("publisher_defaults", publisher_sampling(profile, template))]
    if data.get("_rf_task") == "prompt_enhancement":
        layers.append(("prompt_enhancement", {"temperature": 0.1}))
    layers += [("saved_overrides", user), ("native_cli", cli), ("request_overrides", explicit)]
    effective, origins = {}, {}
    for origin, values in layers:
        effective.update(values)
        origins.update(dict.fromkeys(values, origin))
    return effective, origins


class ProfiledChatClient:
    """One request boundary covers UI, API, streaming, completion and enhancement.

    Only verified artifacts or reviewed variant sources supply sampling defaults.
    Variant sampling qualification does not establish runtime/reasoning support.
    Request values win over
    server CLI, saved user overrides and publisher defaults. Never alter messages,
    native tool definitions, schemas or explicit JSON false/zero values.
    """

    def __init__(
        self,
        client,
        artifact,
        settings,
        *,
        close=None,
        template=None,
        initialized=False,
        sampling_source=None,
        resources=(),
        vision=False,
        initialized_runtime=None,
    ):
        self.client = client
        self.profile = profile_for(artifact) if settings.get("llm_use_model_defaults", True) else {}
        self.use_model_defaults = settings.get("llm_use_model_defaults", True)
        self.sampling_source = deepcopy(sampling_source or {})
        from ruinedfooocus.features.chat.sampling_sources import verified_sampling_profile

        self.sampling_profile = self.profile or (
            verified_sampling_profile(self.sampling_source) if self.use_model_defaults else {}
        )
        self.user = validate_sampling(settings.get("llm_sampling", {}))
        self.cli = (
            cli_sampling(settings.get("llama_server_args", ""))
            if settings.get("llm_runtime", "llama.cpp") == "llama.cpp"
            else {}
        )
        self.reasoning = settings.get("llm_rewrite_reasoning", "Low")
        self.reasoning_capability = initialized_reasoning(
            artifact,
            settings,
            template,
            initialized=initialized,
            initialized_runtime=initialized_runtime,
        )
        self.image_tool_capability = initialized_image_tools(
            artifact,
            settings,
            template,
            initialized=initialized,
            resources=resources,
            vision=vision,
            initialized_runtime=initialized_runtime,
        )
        self.reasoning_settings = deepcopy(
            {
                key: settings[key]
                for key in (
                    "llm_chat_reasoning_mode",
                    "llm_chat_reasoning_effort",
                    "llm_rewrite_reasoning_mode",
                    "llm_rewrite_reasoning_effort",
                )
                if key in settings
            }
        )
        self._close = close

    def sampling_state(self, data=None):
        """Effective explicit fields and their winning origins; never fetch metadata."""
        data = dict(data or {})
        template = self.reasoning_state(data)["template_kwargs"]
        effective, origins = compose_sampling(
            self.sampling_profile, self.user, self.cli, data, template
        )
        publisher_active = "publisher_defaults" in origins.values()
        if any(
            origin in ("saved_overrides", "native_cli", "request_overrides")
            for origin in origins.values()
        ):
            status = "user_overrides"
            message = (
                "User sampling overrides active; remaining fields use publisher defaults."
                if publisher_active
                else "User sampling overrides active; remaining fields use runtime defaults."
            )
        elif "prompt_enhancement" in origins.values():
            status = "task_defaults"
            message = (
                "Prompt enhancement temperature active; remaining fields use publisher defaults."
                if publisher_active
                else "Prompt enhancement temperature active; remaining fields use runtime defaults."
            )
        elif publisher_active:
            status = "publisher_defaults"
            message = (
                "Verified variant sampling baseline active."
                if self.sampling_source.get("status") == "reviewed_variant"
                else "Verified publisher sampling defaults active."
            )
        else:
            status = "runtime_defaults"
            if not self.use_model_defaults:
                message = "Publisher defaults disabled; runtime defaults apply."
            elif self.sampling_source.get("status") == "source_unverified":
                message = "Source metadata found; publisher and variant are unverified. Runtime defaults apply."
            elif self.sampling_source.get("status") == "offline":
                message = "Publisher recommendations unavailable offline; runtime defaults apply."
            else:
                message = "Publisher recommendations not found; runtime defaults apply."
        return dict(
            status=status,
            message=message,
            effective=effective,
            origins=origins,
            publisher_source=self.sampling_profile.get("publisher_settings", {}).get("source_url"),
            source_status=self.sampling_source.get("status", "not_found"),
        )

    def reasoning_state(self, data):
        task = data.get("_rf_task", "chat")
        rewrite = task in ("rewrite", "prompt_enhancement")
        prefix = "llm_rewrite" if rewrite else "llm_chat"
        explicit = "_rf_reasoning" in data
        legacy = (
            rewrite
            and self.reasoning_settings.get(prefix + "_reasoning_mode") is None
            and not explicit
        )
        requested = (
            data.get("_rf_reasoning")
            if explicit
            else {
                "mode": self.reasoning_settings.get(prefix + "_reasoning_mode", "Auto"),
                "effort": self.reasoning_settings.get(prefix + "_reasoning_effort", "Auto"),
            }
        )
        status = "Using the initialized template default."
        if legacy:
            try:
                kwargs = reasoning_kwargs(self.profile, self.reasoning)
                status = "Using the saved legacy rewrite policy; Low is model-specific."
            except ValueError:
                kwargs = {}
                status = "Saved legacy reasoning preference is inactive."
        else:
            try:
                kwargs = controlled_reasoning_kwargs(self.reasoning_capability, requested)
            except ValueError as error:
                if explicit:
                    raise
                # Unknown/unsupported saved preferences cannot break model startup.
                kwargs = {}
                status = "Saved reasoning preference is inactive: " + str(error)
        override = data.get("chat_template_kwargs", {})
        if override is not None:
            if not isinstance(override, Mapping):
                raise ValueError("chat_template_kwargs must be a JSON object.")
            kwargs.update(override)
        if kwargs:
            status = "Template request: " + ", ".join(
                f"{key}={value!r}" for key, value in kwargs.items()
            )
        return {
            "requested": deepcopy(requested),
            "template_kwargs": kwargs,
            "status": status,
            "template_sha256": self.reasoning_capability["template_sha256"],
        }

    def parameters(self, data):
        request = dict(data)
        request.pop("_rf_task", None)
        template = self.reasoning_state(data)["template_kwargs"]
        request.pop("_rf_reasoning", None)
        explicit_template = request.pop("chat_template_kwargs", {})
        explicit_template = {} if explicit_template is None else explicit_template
        thinking = self.profile.get("capabilities", {}).get("thinking", {})
        if "reasoning_effort" in explicit_template and thinking.get("allowed"):
            if explicit_template["reasoning_effort"] not in thinking["allowed"]:
                raise ValueError(
                    f"This model supports reasoning effort: {', '.join(thinking['allowed'])}."
                )
        sampling, _ = compose_sampling(self.sampling_profile, self.user, self.cli, data, template)
        sampling.update({key: value for key, value in request.items() if key not in SAMPLERS})
        if template:
            sampling["chat_template_kwargs"] = template
        return sampling

    def handle_chat_completions(self, data, callback=None):
        return self.client.handle_chat_completions(self.parameters(data), callback)

    def close(self):
        try:
            if self._close is not None:
                self._close()
        finally:
            self.client = None
            self._close = None
            self.image_tool_capability = {
                "available": False,
                "binding": None,
                "vision": False,
                "behavior_qualified": False,
                "reason": "The model was unloaded.",
            }
