from ruinedfooocus.core.paths import asset_path
import os
import ruinedfooocus.app.async_worker as worker
from PIL import Image
import ruinedfooocus.app.state as shared
import json


from comfy.samplers import KSampler


class pipeline:
    pipeline_type = ["hashbang"]

    model_hash = ""

    def parse_gen_data(self, gen_data):
        gen_data["original_image_number"] = gen_data["image_number"]
        gen_data["image_number"] = 1
        gen_data["show_preview"] = False
        return gen_data

    def load_base_model(self, name, hash=None):
        # We don't need a model
        self.model_hash = name
        return

    def load_loras(self, loras):
        return

    def process(
        self,
        gen_data=None,
        callback=None,
    ):
        worker.add_result(
            gen_data["task_id"],
            "preview",
            (-1, f"Working ...", str(asset_path("logo.png"))),
        )

        image = Image.open(str(asset_path("logo.png")))

        # Get command
        lines = gen_data["prompt"].splitlines()
        cmd = lines[0][2:].strip()
        data = "\n".join(lines[1:])

        match cmd:
            case "echo":
                # Simple test
                print(data)

            case "help":
                print("TODO...")

            case "add performance":
                if len(lines) < 2:
                    print("ERROR: Add a name: line before performance settings.")
                    return []
                new_perf = dict(
                    shared.performance_settings.default_settings
                )  # Never edit shared defaults.
                new_perf["name"] = ""

                metadata = None
                if (
                    lines[1].startswith("name:")
                    and len(lines) > 2
                    and lines[2].lstrip().startswith("{")
                ):  # Try parsing json
                    new_perf["name"] = lines[1].split(":", 1)[1].strip()
                    try:
                        metadata = json.loads("\n".join(lines[2:]))
                    except json.JSONDecodeError:
                        metadata = None

                    if metadata is not None:
                        try:
                            new_perf["custom_steps"] = metadata["steps"]
                            new_perf["cfg"] = metadata["cfg"]
                            new_perf["sampler_name"] = metadata["sampler_name"]
                            new_perf["scheduler"] = metadata["scheduler"]
                            new_perf["clip_skip"] = metadata["clip_skip"]
                        except (KeyError, TypeError):
                            metadata = None

                if metadata is None:
                    # It wasn't json-data, try parsing as the example below:
                    #
                    # #!add performance
                    # name: WAN
                    # steps: 30
                    # cfg: 4
                    # scheduler: simple
                    # sampler: uni_pc

                    for line in lines[1:]:
                        if not line.strip():
                            continue
                        kv = line.split(":", 1)
                        if len(kv) < 2:
                            print(f"ERROR: Can't find key and value on line: {line}")
                            return []
                        k = kv[0].strip().lower()

                        # Some helpful translations
                        if k in ["steps"]:
                            k = "custom_steps"
                        if k in ["sampler"]:
                            k = "sampler_name"
                        if k in ["clip-skip", "clip skip", "clipskip"]:
                            k = "clip_skip"

                        if k in ["custom_steps", "clip_skip", "cfg"]:
                            try:
                                v = float(kv[1]) if k == "cfg" else int(kv[1])
                            except ValueError:
                                print(f"ERROR: Invalid number for {k}: {kv[1]}")
                                return []
                        elif k in ["sampler_name", "scheduler", "name"]:  # Handle strings
                            v = kv[1].strip()
                            if k == "sampler_name" and v not in KSampler.SAMPLERS:
                                print(f"ERROR: Unknown sampler: {v}")
                                print(f"Known Samplers: {KSampler.SAMPLERS}")
                                return []
                            if k == "scheduler" and v not in KSampler.SCHEDULERS:
                                print(f"ERROR: Unknown scheduler: {v}")
                                print(f"Known Schedulers: {KSampler.SCHEDULERS}")
                                return []
                        else:
                            print(f"ERROR: Unknown key: {k}")
                            return []
                        new_perf[k] = v

                if not new_perf["name"]:
                    print("ERROR: Performance settings need a nonempty name.")
                    return []
                perf_options = shared.performance_settings.load_performance()
                try:
                    opts = {
                        "custom_steps": new_perf["custom_steps"],
                        "cfg": new_perf["cfg"],
                        "sampler_name": new_perf["sampler_name"],
                        "scheduler": new_perf["scheduler"],
                        "clip_skip": new_perf["clip_skip"],
                    }
                    perf_options[new_perf["name"]] = opts

                    shared.performance_settings.save_performance(perf_options)
                    print(f"#!: Saved performance: {new_perf['name']}: {opts}")
                    shared.update_cfg()
                except Exception as e:
                    image = Image.open(str(asset_path("error.png")))
                    print(f"ERROR: {e}")

            case "clean cache":
                gradio_cache = os.environ["GRADIO_TEMP_DIR"]
                import shutil

                try:
                    if gradio_cache.endswith("ruinedfooocus_cache"):
                        shutil.rmtree(gradio_cache, ignore_errors=True)
                        print("Cache cleared. Please reload the browser to avoid errors.")
                    else:
                        print(f"ERROR: {gradio_cache} doesn't look like a RF cache. Not deleting.")
                except FileNotFoundError:
                    pass
                except PermissionError:
                    pass
                except Exception as e:
                    print(f"ERROR: {str(e)}")

            case "branch":
                import pygit2

                repo = pygit2.Repository(".")
                print(f"#!: Switching to branch {data}. Please restart.")
                try:
                    remote_ref = f"origin/{data}"
                    remote_branch = repo.lookup_branch(remote_ref, pygit2.GIT_BRANCH_REMOTE)
                    local_branch = repo.lookup_branch(data)

                    if local_branch is None:
                        commit = repo.get(remote_branch.target)
                        local_branch = repo.create_branch(data, commit)
                        local_branch.upstream = remote_branch

                    repo.set_head(local_branch.name)
                    repo.checkout_head()
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    print(f"ERROR: {e}")

            case "hf_login":
                import huggingface_hub  # requires !pip install huggingface_hub

                print(
                    "INFO: Go to https://huggingface.co/settings/tokens to create an account and get a HF_TOKEN if you need one."
                )
                try:
                    huggingface_hub.login()
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    print(f"ERROR: {e}")

            case _:
                print(f"ERROR: Unknown command #!{cmd}")
                image = Image.open(str(asset_path("error.png")))

        # Return finished image to preview
        if callback is not None:
            callback(gen_data["steps"], 0, 0, gen_data["steps"], image)

        return []
