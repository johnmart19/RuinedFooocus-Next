"""Audit native image tokenizers on CPU, without weights, downloads or inference.

Run with the app's Python: python tools/validate_prompt_tokenizers.py --output outputs/prompt-tokenizers
Checks native IDs, weights, chunking and templates for short/Unicode/long inputs.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def prompt_cases():
    sentence = (
        "A red fox beside a blue teapot on a wooden table. The scene has soft daylight, "
        "precise detail, balanced composition and natural colors. "
    )
    tail = " The final detail is a violet umbrella marked END_SENTINEL_94721."
    cases = {
        "empty": "",
        "one-character": "f",
        "one-word": "fox",
        "unicode": "\u732b",
        "multilingual": "Київ, Одеса; 日本語の猫; مرحبا; café naïve; 👩🏽‍🚀",
        "whitespace": " ,  red\tfox\nblue cup  , ",
        "literal-brackets": "A sign reads [red, blue] and [label: text].",
        "unmatched-brackets": "A [red fox and a blue cup.",
        "weighted": "A (red fox:1.5) beside (blue cup).",
        "escaped": r"A \(red fox\) beside a cup.",
        "punctuation": "Do not add cats—only 2 dogs! “A:B”, C++ & {text}.",
        "chunk-boundary": "fox " * 76 + "LAST_CUP",
    }
    for size, label in ((16000, "large-16k"), (64000, "ultra-64k"), (128000, "ultra-128k")):
        cases[label] = (sentence * (size // len(sentence) + 1))[:size] + tail
    return cases


def native_tokenizers():
    sys.path.insert(0, str(ROOT / "repositories/ComfyUI"))
    from comfy.cli_args import args

    args.cpu = True
    from comfy.sdxl_clip import SDXLTokenizer
    from comfy.text_encoders.krea2 import Krea2Tokenizer
    from comfy.text_encoders.anima import AnimaTokenizer
    from comfy.text_encoders.moss_image import MOSSImageTokenizer
    from comfy.text_encoders.qwen_image import QwenImageTokenizer
    from comfy.text_encoders.mage_flow import MageFlowTokenizer

    return {
        "SDXL": SDXLTokenizer,
        "Krea2": Krea2Tokenizer,
        "Anima": AnimaTokenizer,
        "MOSS": MOSSImageTokenizer,
        "Qwen edit": QwenImageTokenizer,
        "Mage edit": MageFlowTokenizer,
    }


def streams(tokenizer, family):
    if family == "SDXL":
        return {"l": tokenizer.clip_l, "g": tokenizer.clip_g}
    if family == "Anima":
        return {"qwen3_06b": tokenizer.qwen3_06b, "t5xxl": tokenizer.t5xxl}
    return {tokenizer.clip_name: getattr(tokenizer, tokenizer.clip)}


def reference_segments(tokenizer, family, case, prompt):
    if family in ("SDXL", "Anima"):
        if case == "weighted":
            return [("A ", 1.0), ("red fox", 1.5), (" beside ", 1.0), ("blue cup", 1.1), (".", 1.0)]
        return [(prompt.replace(r"\(", "(").replace(r"\)", ")"), 1.0)]
    if family == "MOSS":
        template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        prompt = prompt.strip()
    else:
        template = tokenizer.llama_template
    return [(template.format(prompt).replace(r"\(", "(").replace(r"\)", ")"), 1.0)]


def check_streams(tokenizer, family, case, prompt, tokens):
    details = {}
    for key, stream in streams(tokenizer, family).items():
        expected = []
        for text, weight in reference_segments(tokenizer, family, case, prompt):
            if family == "Anima" and key == "qwen3_06b":
                weight = 1.0  # Native Anima applies emphasis through its T5 stream only.
            expected.extend(
                (token, weight) for token in stream.tokenizer.encode(text, add_special_tokens=False)
            )
        actual = [
            (token, weight) for chunk in tokens[key] for token, weight, word in chunk if word != 0
        ]
        ids = [token for token, _ in actual]
        details[key] = {
            "ids_and_weights_match": actual == expected,
            "unknown_tokens": ids.count(stream.tokenizer.unk_token_id),
            "vocabulary_size": len(stream.tokenizer),
            "ids_in_range": all(0 <= token < len(stream.tokenizer) for token in ids),
            "decoded_sha256": hashlib.sha256(stream.decode(ids).encode("utf-8")).hexdigest(),
        }
    return details


def check_images(tokenizer, family):
    if family not in ("Qwen edit", "Mage edit", "Krea2"):
        return []
    # These tokenizer paths accept opaque image objects; no pixels/weights are evaluated.
    results = []
    for count in (1,) if family == "Qwen edit" else (1, 2):
        images = [object() for _ in range(count)]
        prompt = "Keep the sign: КИЇВ [A:B]."
        tokens = tokenizer.tokenize_with_weights(prompt, images=images)
        embedded = [
            token["data"]
            for chunks in tokens.values()
            for chunk in chunks
            for token, _ in chunk
            if isinstance(token, dict) and token.get("type") == "image"
        ]
        stream = next(iter(streams(tokenizer, family).values()))
        if family == "Mage edit":
            prefix = "".join(
                f"Image {i + 1}: <|vision_start|><|image_pad|><|vision_end|>" for i in range(count)
            )
            template = tokenizer.llama_template_images.replace("{}", prefix + "{}", 1)
        else:
            template = tokenizer.llama_template_images.replace(
                "<|vision_start|><|image_pad|><|vision_end|>",
                "<|vision_start|><|image_pad|><|vision_end|>" * count,
                1,
            )
        expected = stream.tokenizer.encode(template.format(prompt), add_special_tokens=False)
        actual = [
            151655 if isinstance(token, dict) else token
            for chunks in tokens.values()
            for chunk in chunks
            for token, _ in chunk
        ]
        results.append(
            {
                "family": family,
                "case": f"edit-references-{count}",
                "passed": (
                    len(embedded) == count
                    and all(a is b for a, b in zip(images, embedded))
                    and actual == expected
                ),
            }
        )
    return results


def check_template_slicing(tokenizer, family):
    if family not in ("Krea2", "Qwen edit", "Mage edit"):
        return []
    import torch
    from unittest.mock import patch
    from comfy.sd1_clip import SD1ClipModel
    from comfy.text_encoders.krea2 import Krea2TEModel
    from comfy.text_encoders.qwen_image import QwenImageTEModel
    from comfy.text_encoders.mage_flow import MageFlowTEModel

    model = {"Krea2": Krea2TEModel, "Qwen edit": QwenImageTEModel, "Mage edit": MageFlowTEModel}[
        family
    ]
    results = []
    for text in ("", "fox", "Київ and a red cup END_SENTINEL_94721"):
        tokens = tokenizer.tokenize_with_weights(text)
        sequence = next(iter(tokens.values()))[0]
        # Verify native slicing with positional CPU values instead of encoder weights.
        positions = torch.arange(len(sequence), device="cpu", dtype=torch.float32)
        out = positions.reshape(1, -1, 1)
        if family == "Krea2":
            out = out.unsqueeze(1).expand(-1, 12, -1, -1)
        mask = torch.ones((1, len(sequence)), device="cpu")
        with patch.object(
            SD1ClipModel, "encode_token_weights", return_value=(out, None, {"attention_mask": mask})
        ):
            result = model.encode_token_weights(model.__new__(model), tokens)[0]
        prefix = tokenizer.llama_template.split("{}")[0]
        stream = next(iter(streams(tokenizer, family).values()))
        start = len(stream.tokenizer.encode(prefix, add_special_tokens=False))
        results.append(
            {
                "family": family,
                "case": f"template-slice-{len(text)}",
                "passed": torch.equal(result[0, :, 0], positions[start:]),
            }
        )
    return results


def audit(factory, family, cases):
    try:
        tokenizer = factory()
    except Exception as error:
        return [{"family": family, "error": str(error), "passed": False}]
    results = []
    for case, prompt in cases.items():
        started = time.monotonic()
        row = {"family": family, "case": case, "characters": len(prompt)}
        try:
            tokens = tokenizer.tokenize_with_weights(prompt, return_word_ids=True)
            counts = {key: [len(chunk) for chunk in chunks] for key, chunks in tokens.items()}
            tail_retained = None
            if "END_SENTINEL_94721" in prompt:
                changed = tokenizer.tokenize_with_weights(
                    prompt.replace("END_SENTINEL_94721", "TAIL_REPLACED_68439"),
                    return_word_ids=True,
                )
                tail_retained = tokens != changed
            details = check_streams(tokenizer, family, case, prompt, tokens)
            row.update(
                tokens={key: sum(sizes) for key, sizes in counts.items()},
                chunks={key: len(sizes) for key, sizes in counts.items()},
                tail_retained=tail_retained,
            )
            row["streams"] = details
            row["passed"] = (
                bool(counts)
                and all(sum(sizes) > 0 for sizes in counts.values())
                and tail_retained is not False
                and all(d["ids_and_weights_match"] and d["ids_in_range"] for d in details.values())
            )
        except Exception as error:
            row.update(error=str(error), passed=False)
        row["seconds"] = round(time.monotonic() - started, 3)
        results.append(row)
        print(json.dumps(row), flush=True)
    results.extend(check_images(tokenizer, family))
    results.extend(check_template_slicing(tokenizer, family))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/prompt-length-tokenizers")
    options = parser.parse_args()
    cases = prompt_cases()
    results = []
    for family, factory in native_tokenizers().items():
        results.extend(audit(factory, family, cases))
    report = {
        "comfy_commit": subprocess.check_output(
            ["git", "-C", str(ROOT / "repositories/ComfyUI"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "app_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_files": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "ruinedfooocus/features/generation/image_pipeline.py",
                "ruinedfooocus/features/prompting/prompt_processing.py",
                "ruinedfooocus/features/prompting/prompt_scheduling.py",
                "ruinedfooocus/features/image_editing/shift_attention.py",
                "repositories/ComfyUI/comfy/sd1_clip.py",
                "repositories/ComfyUI/comfy/sdxl_clip.py",
                *(
                    f"repositories/ComfyUI/comfy/text_encoders/{name}.py"
                    for name in (
                        "anima",
                        "krea2",
                        "qwen_image",
                        "qwen3vl",
                        "qwen35",
                        "mage_flow",
                        "moss_image",
                    )
                ),
            )
        },
        "tokenizer_files": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("sd1_tokenizer", "t5_tokenizer", "qwen25_tokenizer", "qwen35_tokenizer")
            for p in (
                ROOT
                / "repositories/ComfyUI/comfy"
                / ("text_encoders/" if folder != "sd1_tokenizer" else "")
                / folder
            ).rglob("*")
            if p.is_file()
        },
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("transformers", "tokenizers", "torch", "sentencepiece")
        },
        "validation": "CPU tokenization only; no text-encoder forward pass or image generation",
        "results": results,
        "passed": all(row["passed"] for row in results),
    }
    options.output.mkdir(parents=True, exist_ok=True)
    (options.output / "native-tokenizers.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (options.output / "prompts.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return not report["passed"]


if __name__ == "__main__":
    raise SystemExit(main())
