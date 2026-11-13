"""Resolve explicit prompt schedules without changing ordinary bracketed text."""

import math
import random
import re


_BRACKETS = re.compile(r"\\.|\[(?P<body>(?:\\.|[^\[\]\\])*)\]")


def _split(text, delimiter):
    parts = []
    start = index = 0
    while index < len(text):
        if text[index] == "\\" and index + 1 < len(text) and text[index + 1] in "\\[]:~^/?|":
            index += 2
            continue
        if text[index] == delimiter:
            parts.append(text[start:index])
            start = index + 1
        index += 1
    return parts + [text[start:]]


def _scheduled_text(body, step, steps):
    def boundary(value):
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(value)
        return int(number * steps) if number < 1 else int(number)

    parts = _split(body, ":")
    if len(parts) > 1:
        try:
            if len(parts) == 4 and parts[2] == "":
                body = parts[0] if boundary(parts[1]) <= step < boundary(parts[3]) else ""
            elif len(parts) == 3:
                body = parts[0] if step < boundary(parts[2]) else parts[1]
            elif len(parts) == 2:
                body = parts[0] if step >= boundary(parts[1]) else ""
            else:
                return None
        except ValueError:
            return None
        # A schedule's selected side may itself alternate: [red|blue:green:0.5].
        selected = _scheduled_text(body, step, steps)
        return body if selected is None else selected

    for operator in "~^/\\?|":
        options = _split(body, operator)
        if len(options) == 1:
            continue
        if operator != "|" and not all(options):
            return None
        if operator == "~":
            index = step // max(round(steps / 10), 2)
        elif operator == "?":
            index = random.randrange(len(options))
        elif operator == "|":
            index = step
        else:
            index = round(step + (steps - step) * step / steps)
            if operator in ("^", "\\") and step > steps / 2:
                options.reverse()
            if operator == "/" and step > steps / 2:
                return options[-1]
            if operator == "\\" and step < steps / 2:
                return options[0]
        return options[index % len(options)]
    return None


def prompt_switch_per_step(prompt, steps):
    """Return scheduled prompts, or [] when native tokenization can use the input.

    Invalid schedules, ordinary brackets and escaped brackets stay literal. Each
    successful substitution removes a bracket pair, so unmatched input terminates.
    """
    if steps <= 0 or "[" not in prompt or "]" not in prompt or prompt.lstrip().startswith("{"):
        return []
    prompts = []
    for step in range(steps):
        text = prompt
        scheduled = False

        def replace(match):
            nonlocal scheduled
            if match["body"] is None:
                return match[0]
            replacement = _scheduled_text(match["body"], step, steps)
            if replacement is None:
                return match[0]
            scheduled = True
            return replacement

        while True:
            changed = _BRACKETS.sub(replace, text)
            if changed == text:
                break
            text = changed
        if not scheduled:
            return []
        prompts.append(text)
    return prompts


def bind_prompt_steps(conditioning, model, steps, sampler, scheduler, denoise, *, sigmas=None):
    """Bind explicit switches to native sigma intervals after img2img denoise is known."""
    if not any("_rf_prompt_step" in values for _, values in conditioning):
        return conditioning
    import torch
    from comfy.samplers import KSampler

    if sigmas is None and sampler == "lcm":
        from comfy_extras.nodes_custom_sampler import BasicScheduler

        sigmas = BasicScheduler.execute(
            model=model,
            scheduler=scheduler,
            steps=steps,
            denoise=1.0 if denoise is None else denoise,
        )[0]
    elif sigmas is None:
        sigmas = KSampler(
            model, steps, device="cpu", sampler=sampler, scheduler=scheduler, denoise=denoise
        ).sigmas
    if len(sigmas) == 0:
        return conditioning  # Zero denoise: the native sampler returns the input latent.
    values = sigmas.tolist()
    if len(values) != steps + 1 or any(a <= b for a, b in zip(values, values[1:])):
        raise ValueError(
            "Prompt switching requires distinct descending sigmas; use fewer steps or another scheduler."
        )
    lower = torch.nextafter(sigmas[1:], torch.full_like(sigmas[1:], float("inf"))).tolist()
    sampling = model.get_model_object("model_sampling")
    result = []
    for tensor, metadata in conditioning:
        metadata = metadata.copy()
        step = metadata.pop("_rf_prompt_step", None)
        if step is not None:
            start = float("inf") if step == 0 else values[step]
            end = 0.0 if step == steps - 1 else lower[step]
            if "clip_start_percent" in metadata:
                start = min(start, sampling.percent_to_sigma(metadata.pop("clip_start_percent")))
            if "clip_end_percent" in metadata:
                end = max(end, sampling.percent_to_sigma(metadata.pop("clip_end_percent")))
            metadata.pop("start_percent", None)
            metadata.pop("end_percent", None)
            metadata.update(timestep_start=start, timestep_end=end)
        result.append([tensor, metadata])
    return result
