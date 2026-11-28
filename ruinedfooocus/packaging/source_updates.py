"""Apply source updates without replacing local work or ignored user files."""


def tracked_changes(repo):
    """Untracked files are protected by SAFE checkout, not treated as edits."""
    import pygit2

    return {
        path: flags for path, flags in repo.status().items() if flags != pygit2.GIT_STATUS_WT_NEW
    }


def fast_forward(repo, target, expected_head):
    import pygit2

    # Hold both refs while checking and checking out; another Git process must
    # not switch branches or advance this branch beneath a running update.
    with repo.transaction() as transaction:
        transaction.lock_ref("HEAD")
        transaction.lock_ref(expected_head[0])
        if tracked_changes(repo) or (repo.head.name, repo.head.target) != expected_head:
            raise RuntimeError(
                "Source changed during the update. Keeping local work; update skipped."
            )
        analysis, _ = repo.merge_analysis(target.id)
        if not analysis & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
            raise RuntimeError("The update is not a fast-forward. Keeping local history.")
        repo.checkout_tree(
            target,
            strategy=pygit2.GIT_CHECKOUT_SAFE | pygit2.GIT_CHECKOUT_DONT_OVERWRITE_IGNORED,
        )
        transaction.set_target(expected_head[0], target.id, message="RuinedFooocus source update")
