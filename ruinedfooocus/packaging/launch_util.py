from ruinedfooocus.core.paths import APP_ROOT
import os
import importlib
import importlib.util
import subprocess
import sys
import re
import logging
from pathlib import Path
import importlib.metadata
from tempfile import TemporaryDirectory
import packaging.version
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

logging.getLogger("torch.distributed.nn").setLevel(logging.ERROR)  # sshh...
logging.getLogger("xformers").addFilter(
    lambda record: "A matching Triton is not available" not in record.getMessage()
)

python = sys.executable
default_command_live = os.environ.get("LAUNCH_LIVE_OUTPUT") == "1"
index_url = os.environ.get("INDEX_URL", "")

modules_path = Path(__file__).resolve().parent
script_path = APP_ROOT
dir_repos = "repositories"

re_requirement = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")


def git_clone(url, dir, name, hash=None):
    from ruinedfooocus.packaging.git_transport import (
        CLONE_TIMEOUT,
        FETCH_TIMEOUT,
        bounded_git_operation,
    )

    repo = None
    try:
        import pygit2

        pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)

        if Path(dir).is_dir():
            repo = pygit2.Repository(dir)
        else:
            Path(dir).parent.mkdir(parents=True, exist_ok=True)
            print(f"Cloning {name} (up to {CLONE_TIMEOUT} seconds)...", flush=True)
            # An interrupted acquisition must not become the installed checkout.
            with TemporaryDirectory(prefix="rf-git-clone-", dir=Path(dir).parent) as staging:
                checkout = Path(staging) / "repository"
                bounded_git_operation("clone", checkout, url)
                if Path(dir).exists():
                    raise RuntimeError("The checkout appeared during cloning; preserving it.")
                checkout.rename(dir)
            repo = pygit2.Repository(dir)
            print(f"{name} cloned.", flush=True)

        if repo.status():
            print(f"{name} has local changes; preserving its checkout without updating.")
            return False

        if hash and str(repo.head.target) == hash:
            print(f"{name} already at the pinned revision.")
            return True

        # A pinned commit already present locally needs no publisher round trip.
        commit = repo.get(hash) if hash else None
        if commit is None:
            print(f"Checking {name} source (up to {FETCH_TIMEOUT} seconds)...", flush=True)
            bounded_git_operation("fetch", Path(dir), "origin")
            commit = repo.get(hash)
        # SAFE also protects changes made after the status check (e.g. another chat).
        repo.checkout_tree(commit, strategy=pygit2.GIT_CHECKOUT_SAFE)
        repo.set_head(commit.id)
        print(f"{name} update check complete.")
        return True
    except Exception as e:
        print(f"Git clone failed for {name}: {str(e)}", flush=True)
        return False
    finally:
        if repo is not None:
            repo.free()


def repo_dir(name):
    return str(Path(script_path) / dir_repos / name)


def is_installed(package):
    m = re.match(re_requirement, package)
    if m is None:
        return False

    package = m.group(1).strip()
    version_required = (m.group(2) or "").strip()

    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    try:
        version_installed = re.sub(r"\+.*$", "", importlib.metadata.version(package))
    except Exception:
        return False

    if version_required != "" and packaging.version.parse(
        version_required
    ) != packaging.version.parse(version_installed):
        return False

    return True


def run(
    command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live
) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": "utf8",
        "errors": "ignore",
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return result.stdout or ""


def run_pip(command, desc=None, live=default_command_live):
    index_url_line = (
        f" --index-url {index_url}" if index_url and "--index-url" not in command else ""
    )
    return run(
        f'"{python}" -m pip {command} --prefer-binary{index_url_line}',
        desc=f"Installing {desc}",
        errdesc=f"Couldn't install {desc}",
        live=live,
    )


def reinstall_requirements(requirements_file, constraints=""):
    """Reinstall exact versions after dependency resolution, without replacing Torch."""
    resolved = {}
    for line in Path(requirements_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        requirement = Requirement(line.split(" #", 1)[0])
        if requirement.marker and not requirement.marker.evaluate():
            continue
        installed = importlib.metadata.version(requirement.name)
        if not requirement.specifier.contains(installed, prereleases=True):
            raise RuntimeError(
                f"Resolve {requirements_file} before reinstalling {requirement.name}."
            )
        resolved[requirement.name] = installed
    # --no-deps does not check dependency bounds. Reusing version ranges here
    # could replace DirectML-compatible OpenCV/rembg/Tokenizers with newer builds.
    with TemporaryDirectory(prefix="ruinedfooocus-reinstall-") as directory:
        pinned = Path(directory) / "requirements.txt"
        pinned.write_text(
            "".join(f"{name}=={version}\n" for name, version in resolved.items()), encoding="utf-8"
        )
        run_pip(
            f'install --force-reinstall --no-deps -r "{pinned}"{constraints}', "resolved modules"
        )


def _requirement_met(requirement, checked):
    installed = importlib.metadata.version(requirement.name)
    if not requirement.specifier.contains(installed, prereleases=True):
        return False
    key = (canonicalize_name(requirement.name), frozenset(requirement.extras))
    if key in checked:
        return True
    checked.add(key)
    # Check requested extras as well as the base dependency closure. Merely
    # finding gradio installed does not establish that gradio[mcp] is usable.
    extras = {"", *requirement.extras}
    for raw in importlib.metadata.requires(requirement.name) or []:
        dependency = Requirement(raw)
        if dependency.marker and not any(
            dependency.marker.evaluate({"extra": extra}) for extra in extras
        ):
            continue
        if not _requirement_met(dependency, checked):
            return False
    return True


def requirements_met(requirements_file):
    """Check applicable PEP 508 requirements and their installed extra closure."""
    checked = set()
    with open(requirements_file, "r", encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith(("#", "--")):
                continue
            try:
                requirement = Requirement(line.split(" #", 1)[0])
                if requirement.marker and not requirement.marker.evaluate():
                    continue
                if not _requirement_met(requirement, checked):
                    return False
            except (InvalidRequirement, importlib.metadata.PackageNotFoundError, ValueError):
                return False
    return True
