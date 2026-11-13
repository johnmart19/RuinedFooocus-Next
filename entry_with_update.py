import os
import subprocess
import sys

print("Starting RuinedFooocus Python; checking startup policy...", flush=True)

from pathlib import Path
from ruinedfooocus.core.paths import source_checkout, startup_marker
from ruinedfooocus.packaging.startup_options import skip_dependency_install, offline_startup

root = Path(__file__).resolve().parent
sys.path.append(str(root))
os.chdir(root)

offline = offline_startup()

if not skip_dependency_install():
    from ruinedfooocus.core.dependency_cache import needed, record

    # Bootstrap before importing application helpers, which require packaging.
    reinstall_bootstrap = ["--force-reinstall"] if startup_marker("reinstall").exists() else []
    if needed(root, "bootstrap", force=bool(reinstall_bootstrap)):
        print("Check pre-requirements", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                *reinstall_bootstrap,
                "-r",
                "requirements_versions.txt",
            ],
            check=True,
        )
        record(root, "bootstrap")
    sys._rf_bootstrap_checked = True
    # ruinedfooocus/app/bootstrap.py must not replace pygit2/cffi after the updater loads their DLLs.

if not offline and source_checkout(root):
    bupdated = False
    repo = None
    try:
        import pygit2

        pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)

        repo_path = Path(__file__).resolve().parent
        repo = pygit2.Repository(str(repo_path))
        from ruinedfooocus.packaging.source_updates import tracked_changes

        changes = tracked_changes(repo)
        if changes:
            raise RuntimeError(
                "Tracked source changes found. Keeping them; automatic update skipped: "
                + ", ".join(sorted(changes)[:5])
            )
        if repo.head_is_detached:
            raise RuntimeError("Detached source revision; automatic update skipped.")
        expected_head = (repo.head.name, repo.head.target)
        branch_name = repo.head.shorthand

        remote_name = "origin"
        from ruinedfooocus.packaging.git_transport import FETCH_TIMEOUT, bounded_git_operation

        print(f"Checking application updates (up to {FETCH_TIMEOUT} seconds)...", flush=True)
        bounded_git_operation("fetch", repo_path, remote_name)

        remote_reference = f"refs/remotes/{remote_name}/{branch_name}"
        remote_commit = repo.revparse_single(remote_reference)

        merge_result, _ = repo.merge_analysis(remote_commit.id)

        if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
            print("You have the latest version", flush=True)
        elif merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
            from ruinedfooocus.packaging.source_updates import fast_forward

            fast_forward(repo, remote_commit, expected_head)
            print("Updating Files", flush=True)
            bupdated = True
        elif merge_result & pygit2.GIT_MERGE_ANALYSIS_NORMAL:
            print(
                "Local and remote history diverge; automatic update skipped. Reconcile the branch manually."
            )
    except Exception as e:
        print("Automatic update not applied. Continuing with installed source.", flush=True)
        print(str(e), flush=True)
    finally:
        if repo is not None:
            repo.free()
    if bupdated:
        print("Update succeeded!!")
        # Check the new dependency files in a fresh process, before updater
        # imports can hold native package DLLs open on Windows.
        os.execv(sys.executable, [sys.executable, *sys.argv])
else:
    print(
        "Offline mode. No update."
        if offline
        else "Archive installation: Git updates are disabled. Replace application files to upgrade.",
        flush=True,
    )
print("Preparing application runtime...", flush=True)
from ruinedfooocus.app.bootstrap import *
