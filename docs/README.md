# Documentation

RF is scoped to local image generation/editing and optional local Chat. See the
[product boundary and retained shared consumers](integrations/image-chat-scope.md).

| Task                                                             | Guide                                                          |
| ---------------------------------------------------------------- | -------------------------------------------------------------- |
| Install, choose CUDA/ROCm/CPU, diagnose runtime failures         | [Runtimes and troubleshooting](runtime-and-troubleshooting.md) |
| Create, edit, prompt, Chat and reuse Library results             | [User guide](user-guide.md)                                    |
| Download a Chat candidate while using the active model           | [Chat preparation](chat-preparation.md)                        |
| Create, edit and save an assistant or choose its startup persona | [Chat assistants](chat-assistants.md)                          |
| Save preferences, profiles and startup choices                   | [Settings](settings.md)                                        |
| Choose a built-in or cached custom theme                         | [Themes](themes.md)                                            |
| Select host folders and manage ordered model roots               | [Folder selection](paths.md)                                   |
| Place model weights and retain legacy folder aliases             | [Model folders](model-folders.md)                              |
| Customize One Button lists, presets and wildcards                | [Prompt data](prompt-data.md)                                  |
| Select or contribute UI languages                                | [Localization](localization.md)                                |
| Change code, run focused checks and publish features             | [Development](development.md)                                  |
| Find UI, service, resources and native backend owners            | [Architecture](architecture.md)                                |
| Locate replacements for old source paths                         | [Legacy migration](legacy-migration.md)                        |
| Understand platform APIs, package boundaries and qualification   | [Integration contracts](integrations/README.md)                |
| Audit model, component and runtime download links                | [Download audit](download-audit.md)                            |

Preview and configuration actions do not imply generation. The guides distinguish
saved preferences, the current request and actions that load a model. CPU contracts,
browser checks, image quality and physical GPU qualification are separate evidence.
