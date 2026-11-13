# RuinedFooocus development

## Start with current evidence

Confirm physical cwd, Git root/common directory, branch, HEAD and existing edits.
Other chats may share this checkout. Follow the current request; an old plan or
checkpoint grants no new authorization. Preserve unrelated edits, profiles,
outputs and inputs. Model replacements, GPU use, installs and restarts require
scope from the current conversation; do not infer it from this file.

After resume, read global `CONTINUITY.md`, this file and the **current section of
this chat's exact record** in `.codex/tasks/`. Read `.agents/continuity.md` if it
exists. Revalidate the recorded revision and next action.
Keep old evidence by reference instead of repeatedly loading checkpoint history.
For a handoff, use a separate checkpoint for the receiving chat and cite the
originating record. Verify tools, permissions and GPU scope in the current
client and conversation.

## Source owners

| Need                                          | Start here                                                                             |
| --------------------------------------------- | -------------------------------------------------------------------------------------- |
| Installation, device/Python/wheel eligibility | `ruinedfooocus/packaging/`, `app/bootstrap.py`                                         |
| Native runtimes and ComfyUI adaptation        | `ruinedfooocus/backends/{comfy,llama,onnx}/`                                           |
| Worker, APIs, runtime state                   | `ruinedfooocus/app/{async_worker,api_runtime,state}.py`                                |
| Models, registry, components, resource cards  | `ruinedfooocus/features/models/`                                                       |
| Chat, tools, vision, prompt enhancement       | `features/chat/`, `features/prompting/` under `ruinedfooocus/`                         |
| Generation, editing, utilities, Library       | `features/{generation,image_editing,image_processing,gallery}/` under `ruinedfooocus/` |
| Shared UI, Hardware/Settings, assets          | `ruinedfooocus/ui/`; feature panels stay with their feature                            |
| Paths, downloads, settings/profiles           | `ruinedfooocus/core/`; ownership detail in `docs/architecture.md`                      |

Read `docs/development.md` or a specific workflow/runtime guide on demand. Use
scoped `rg` searches. Do not recursively read weights, caches, outputs, venvs,
vendor repositories or prompt datasets to learn the app layout.

## Runtime and validation contracts

- Use the verified project interpreter (`venv/Scripts/python.exe` on this Windows
  checkout), not global Python. WSL uses its own explicit checkout and venv.
  Package presence/version does not prove Python ABI or device compatibility.
- Use the pinned `repositories/ComfyUI` revision recorded by bootstrap. Reuse
  native functionality; do not duplicate samplers/device management speculatively.
- Run focused checks with `python tools/check_cpu.py test_<feature>.py` using
  that interpreter. It supplies isolated data and GPU guards. Add caller tests
  for actual dependencies; broaden only for cross-cutting changes or failures.
- UI checks use the built-in browser and a private profile. CPU launches use
  `--cpu --offline --skip-dependency-install --settings <owned-profile>`.
  Preserve unrelated servers. Actual browser zoom is not viewport resizing.
- Keep source/CPU, UI, actual generation, visual quality, performance and GPU
  stability evidence separate. Record exact source/dependency/artifact identities.
  A provider list or installation canary does not certify real-model execution.
- Shared compute state/cleanup is not thread-safe merely because an event has
  a separate Gradio queue. Review resource ownership before adding concurrency.

## Agent context and completion

Delegate only when requested or instructed, with the user's current model choice.
Send a bounded task, revision, relevant paths, permissions and expected evidence;
do not copy the full chat or ask every agent to rerun the same tests. Assign
non-overlapping edit ownership. Require concise findings with source references.
Each worker updates the guide for its change. Follow `docs/development.md`
and use `docs/integrations/` for native API and platform contracts.

Use Context7 only for a specific API/version gap; reuse its resolved library ID.
TokenEconomizer may provide useful facts, but check its actual status and verify
facts against this checkout. Do not claim savings from enabled settings alone.
Keep navigation here and task state and evidence in the exact checkpoint.

## Tooling reliability

Use a small `apply_patch` Update File for ordinary edits; do not delete and add
the same path in one patch. For large generated evidence, use a short project-
Python or normal Node script to write atomically; return only its path, size,
hash and summary. Do not embed entire upstream responses in tool arguments.
Keep calls bounded and report substantive progress at least every 60 seconds.

Browser/Node REPL is not required for file editing. Diagnose an observed error
before changing dependencies. If a task actually needs bundled document packages,
discover their current paths with `load_workspace_dependencies`; REPL module
configuration is per chat. Use normal Node for packages incompatible with that
REPL. Do not install duplicates or alter security settings to repair a guessed
problem. Formatting uses the repository's existing `tools/format` dependencies.

Before publication: inspect scoped diff, run appropriate formatting/checks,
register cohesive commits under `tools/features/` through
`tools/feature_commits.json`, and validate their declared prerequisite closures.
Keep installers separate. Include `Co-authored-by: Codex <noreply@openai.com>` and
verify remote publication/attribution. Do not force-push without current authority.
Remove only owned disposable fixtures/processes and retain useful evidence.
