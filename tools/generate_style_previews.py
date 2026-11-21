"""Generate 160x160 WebP style preview thumbnails for all bundled styles.

Uses RuinedFooocus's native ComfyUI image pipeline so the selected checkpoint
loads once on GPU and renders consistent, representative thumbnails for every
style in `ruinedfooocus/resources/defaults/styles.default`.
"""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image


DEFAULT_SUBJECT = (
    "a red fox sitting beside a glowing vintage lantern on a cobblestone street "
    "in front of a cozy cottage at golden hour, centered composition"
)

SUBJECT_OVERRIDES = {
    "ads-automotive": "a sleek crimson sports car parked on a coastal mountain road at sunset",
    "ads-corporate": "a modern glass office atrium with clean desks and plants",
    "ads-fashion editorial": "a stylish woman in a tailored trench coat posing on a sunlit city avenue",
    "ads-food photography": "a gourmet artisanal berry tart plated beside a steaming espresso cup on rustic wood",
    "ads-luxury": "an ornate gold mechanical wristwatch resting on dark marble with dramatic rim light",
    "ads-real estate": "a modern architectural villa with warm interior lights and infinity pool overlooking hills",
    "ads-retail": "a colorful artisanal coffee tin and gift box arranged on a clean studio pedestal",
    "ads-advertising": "a vibrant handcrafted botanical perfume bottle on a clean studio pedestal",
    "futuristic-cyberpunk cityscape": "a towering neon-lit metropolis with rain-slicked streets and flying vehicles",
    "game-fighting game": "two agile martial artists facing off in a dramatic courtyard arena",
    "game-streetfighter": "a heroic martial artist in a dynamic power pose in a vibrant harbor street",
    "game-strategy game": "a medieval castle town with tiny armies, roads, forests, and rivers",
    "misc-architectural": "a cantilevered contemporary pavilion beside a reflecting pool and pine trees",
    "misc-character referencesheet": "a young adventurer wearing a scarf and leather satchel",
    "misc-chibi": "a cheerful young traveler holding a glowing lantern with a tiny red fox",
    "misc-funkopop": "a cute traveler with a lantern and a tiny fox",
    "misc-3D-Caricature": "a cheerful explorer with glasses holding a glowing lantern",
    "artstyle-pixar": "a friendly young traveler with a glowing lantern and a curious red fox",
    "sai-texture": "weathered mossy cobblestone bricks with warm autumn leaves",
    "photo-tilt-shift": "a bustling European harbor village with colorful boats and red roofs from above",
    "photo-silhouette": "a lone traveler holding a lantern on a misty ridge against a glowing twilight sky",
    "photo-long exposure": "a rocky coastline waterfall under starry twilight clouds and silky glowing water",
    "photo-neon noir": "a rain-slicked midnight street with glowing magenta and cyan reflections and a lone figure",
    "photo-film noir": "a detective in a trench coat standing under a misty streetlamp on a wet alley",
    "photo-alien": "a surreal bioluminescent alien canyon with glowing crystal spires under twin moons",
}


def subject_for(name: str) -> str:
    bare = name.removeprefix("Style: ").removeprefix("Artify: ")
    return SUBJECT_OVERRIDES.get(bare, DEFAULT_SUBJECT)


def main():
    parser = argparse.ArgumentParser(description="Generate WebP style preview thumbnails.")
    parser.add_argument(
        "--checkpoint",
        default="AnimaMayhemV1-2.9B.safetensors",
        help="Checkpoint filename in models/checkpoints/",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "ruinedfooocus" / "ui" / "assets" / "style_previews"),
        help="Output directory for WebP thumbnails",
    )
    parser.add_argument("--size", type=int, default=768, help="Generation width/height (square)")
    parser.add_argument("--thumb", type=int, default=160, help="Saved WebP width/height (square)")
    parser.add_argument("--steps", type=int, default=18, help="Sampling steps")
    parser.add_argument("--cfg", type=float, default=4.5, help="CFG scale")
    parser.add_argument("--sampler", default="er_sde", help="ComfyUI sampler name")
    parser.add_argument("--scheduler", default="simple", help="ComfyUI scheduler name")
    parser.add_argument("--seed", type=int, default=424242, help="Base seed")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for quick smoke testing")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing thumbnails")
    parser.add_argument("--only", nargs="*", default=None, help="Optional subset of style names/slugs")
    args = parser.parse_args()
    # Keep RuinedFooocus's internal argparse from seeing our script flags.
    sys.argv = [sys.argv[0], "--offline", "--skip-dependency-install"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Boot ComfyUI paths & native image pipeline using the exact installed GPU runtime.
    for extra in (ROOT / "repositories", ROOT / "repositories" / "ComfyUI"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    import ruinedfooocus.backends.comfy.comfy_args  # noqa: F401
    import ruinedfooocus.app.state as shared
    from ruinedfooocus.backends.comfy.model_paths import register_image_model_paths
    from ruinedfooocus.features.generation.image_pipeline import pipeline as ImagePipeline

    register_image_model_paths(shared.path_manager)
    from ruinedfooocus.features.prompting.sdxl_styles import apply_style
    from ruinedfooocus.features.prompting.style_catalog import load_catalog, style_preview_slug

    catalog = load_catalog()
    targets = []
    only_set = set(args.only) if args.only else None
    for name, style in catalog.items():
        slug = style_preview_slug(name)
        if only_set and name not in only_set and slug not in only_set:
            continue
        dest = out_dir / f"{slug}.webp"
        if dest.is_file() and not args.overwrite:
            continue
        targets.append((name, slug, style, dest))

    if args.limit > 0:
        targets = targets[: args.limit]

    print(
        f"Style preview generator: {len(targets)} to render "
        f"(total catalog={len(catalog)}, checkpoint={args.checkpoint}, "
        f"gen={args.size}x{args.size}, thumb={args.thumb}x{args.thumb}, "
        f"steps={args.steps}, cfg={args.cfg}, sampler={args.sampler}/{args.scheduler})",
        flush=True,
    )
    if not targets:
        print("All style previews already exist.", flush=True)
        return

    t0 = time.perf_counter()
    pipeline = ImagePipeline()
    pipeline.load_base_model(args.checkpoint)
    print(f"Loaded {args.checkpoint} in {time.perf_counter() - t0:.1f}s", flush=True)

    done = 0
    failed = []
    start_gen = time.perf_counter()
    for idx, (name, slug, style, dest) in enumerate(targets, start=1):
        base_subject = subject_for(name)
        positive, negative = apply_style(
            name, base_subject, "text, watermark, signature, logo, ugly, blurry", ""
        )
        gen_data = {
            "task_id": f"style-preview-{idx}",
            "positive_prompt": positive,
            "negative_prompt": negative,
            "prompt": positive,
            "negative": negative,
            "base_model_name": args.checkpoint,
            "width": args.size,
            "height": args.size,
            "steps": args.steps,
            "cfg": args.cfg,
            "clip_skip": -1,
            "sampler_name": args.sampler,
            "scheduler": args.scheduler,
            "seed": args.seed,
            "image_number": 1,
            "loras": [],
            "style_selection": [],
            "task_type": "style_preview",
            "preview_method": "none",
            "performance_selection": "Custom",
        }
        t_item = time.perf_counter()
        try:
            images = pipeline.process(gen_data)
            if not images:
                raise RuntimeError("Pipeline returned no images")
            raw = images[0]
            img = raw if isinstance(raw, Image.Image) else Image.fromarray(raw)
            img = img.convert("RGB")
            thumb = img.resize((args.thumb, args.thumb), Image.Resampling.LANCZOS)
            thumb.save(dest, format="WEBP", quality=82, method=6)
            done += 1
            elapsed = time.perf_counter() - start_gen
            avg = elapsed / done
            eta = avg * (len(targets) - done)
            print(
                f"[{idx}/{len(targets)}] {slug}.webp ({dest.stat().st_size} B) "
                f"in {time.perf_counter() - t_item:.1f}s | avg {avg:.1f}s | ETA {eta / 60:.1f}m",
                flush=True,
            )
        except Exception as exc:
            failed.append((name, str(exc)))
            print(f"[{idx}/{len(targets)}] FAILED {name}: {exc}", flush=True)

    total_time = time.perf_counter() - start_gen
    print(
        f"Finished: {done}/{len(targets)} rendered in {total_time:.1f}s, failures={len(failed)}",
        flush=True,
    )
    if failed:
        for name, err in failed[:10]:
            print(f"  FAIL {name}: {err}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
