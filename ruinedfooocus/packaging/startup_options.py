"""Dependency-install policy shared by both startup entrypoints."""

import os
import sys

LEGACY_SKIP_FLAG = "--iINSTallLEDmYOwNPaCKaGeS"


def saved_profile_path(argv=None):
    """Select this launch's profile without importing application arguments or devices.

    Selection syntax is the same --settings / --settings=value spelling accepted
    by argparse. All actual application arguments are still parsed in one place.
    """
    from ruinedfooocus.core.profile_store import selected_profile, profile_folder

    options = list(sys.argv[1:] if argv is None else argv)
    explicit = None
    for index, option in enumerate(options):
        if option == "--":
            break
        spelling = option.split("=", 1)[0]
        is_settings = spelling.startswith("--se") and "--settings".startswith(spelling)
        if is_settings and "=" in option:
            explicit = option.split("=", 1)[1]
        elif is_settings and index + 1 < len(options):
            explicit = options[index + 1]
    return profile_folder(selected_profile(explicit)) / "settings.json"


def saved_startup_policy(argv=None):
    """Read only startup booleans before updater/dependency work; no feature imports."""
    from ruinedfooocus.core.profile_store import read_profile_data

    values = read_profile_data(saved_profile_path(argv))
    policy = {}
    for name in ("startup_offline", "startup_skip_dependency_install"):
        if name in values:
            if type(values[name]) is not bool:
                raise ValueError(f"{name}: choose enabled or disabled (a boolean).")
            policy[name] = values[name]
    return policy


def _policy_flag(options, flag, minimum):
    # These prefixes are unambiguous in the production parser. Honor argparse's
    # accepted abbreviations before the updater can perform any network work.
    return any(option.startswith(minimum) and flag.startswith(option) for option in options)


def offline_startup(argv=None):
    options = sys.argv[1:] if argv is None else argv
    return (
        os.environ.get("RF_OFFLINE") == "1"
        or _policy_flag(options, "--offline", "--of")
        or saved_startup_policy(options).get("startup_offline", False)
    )


def skip_dependency_install(argv=None):
    options = sys.argv[1:] if argv is None else argv
    return (
        offline_startup(options)
        or _policy_flag(options, "--skip-dependency-install", "--sk")
        or LEGACY_SKIP_FLAG in options
        or saved_startup_policy(options).get("startup_skip_dependency_install", False)
    )
