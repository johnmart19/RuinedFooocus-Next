# Choosing folders

Open **Settings → Profiles & storage → Paths**. Each path row shows the host
filesystem used by RuinedFooocus. **Select folder** opens a host-folder browser:
enter a path and choose **Open path**, or choose a subfolder and **Open subfolder**.
**Parent folder** goes up one directory. **Use folder** updates the form;
**Cancel** leaves it unchanged.

For an ordered list, choose the row to replace before selecting a folder.
**Add folder** appends another root; **Remove folder**, **Move up** and
**Move down** change the list. Earlier roots keep priority. Selecting a duplicate
directory, including another spelling or a link to the same directory, reports
an error. Advanced manual lists retain existing duplicate entries; consumers
continue their existing deduplication behavior.

Expand **Advanced path** to type or paste paths directly. Lists use one folder
per line. The help text shows the base for relative paths. Manual paths may name
folders that do not yet exist; selecting through the browser requires an
accessible existing directory. Output selection also checks the host's writable
access flag. Actual file writes can still fail later if permissions or storage
change.

Archive roots use the same controls under **Image Browser**. The picker covers
checkpoints, diffusion models, text encoders, LoRAs, embeddings, inbox, geometry
models, output, wildcards and archives.

**Save** writes the draft into the current profile. **Save as profile** writes a
named profile. Restart to activate the new model/runtime folders consistently.
Selecting or rearranging folders does not move or delete files, start downloads,
or change the running worker's derived model folders. If saving fails, the
existing profile and active path state remain intact; the form can be corrected
and saved again.

Library's next refresh reads the saved output and archive paths. The running
worker keeps its derived model/runtime folders until restart, so restarting
after saving remains the consistent way to activate the complete profile.

## Host paths and relative paths

Paths refer to the machine running RuinedFooocus, including when the browser is
on a phone or another computer. Browser directory-upload APIs are not used.

| Field                                | Relative path base                                                                    |
| ------------------------------------ | ------------------------------------------------------------------------------------- |
| Model roots, inbox, geometry, output | `modules/` below the writable data root, preserving the existing legacy path contract |
| Wildcards                            | Writable data root                                                                    |
| Archive roots                        | Server process working directory, preserving Library's existing contract              |

For example, with data root `D:\RuinedFooocus`, a model value
`../models/checkpoints` refers to `D:\RuinedFooocus\models\checkpoints`.
The default wildcard value `wildcards` refers to `D:\RuinedFooocus\wildcards`.
An absolute path avoids working-directory dependence for archives.

Windows uses drive paths such as `D:\Models` or UNC paths. A WSL/Linux server
uses Linux paths such as `/mnt/d/Models`. The picker labels that namespace and
rejects newly edited paths in the wrong syntax; it does not translate Windows
paths into WSL paths or treat client folders as server folders. Existing saved
values are retained when unchanged, so a profile can be inspected or corrected
without blocking unrelated saves.

Symbolic links and Windows junctions are resolved for selection and duplicate
checks. The selected absolute destination is stored. Existing manually entered
spellings remain unchanged until edited. Browsing lists only immediate child
directories, including directory links; it never recursively expands a tree.
Large listings are capped. Type a full path to reach a folder omitted by that
cap.

## Optional native host dialog

On the local host, **Advanced path → Open host dialog** can open a native folder
chooser. Windows uses its existing PowerShell/WinForms dialog in an STA process
with the console hidden; no additional GUI package is installed. Linux uses an
existing Zenity or KDialog when a desktop display is available. Cancellation
keeps the draft. Dialog failures or a two-minute timeout open the host-browser
fallback.

This action requires the actual server-observed client, URL and browser Origin
to be loopback, with no forwarding headers, plus an interactive desktop.
Remote requests, WSL and headless environments use the host browser instead.
Loopback forwarding can resemble a local connection; normal **Select folder**
and **Add folder** always use the host browser, and the desktop dialog requires
its separate explicit action.

Remote browsing requires the configured owner's authenticated Gradio request
identity. A submitted username or client-side field does not authorize it.
Guest mode cannot enumerate host folders. Cross-origin requests are rejected.
The host browser is a full owner operation, not a filesystem sandbox for guests.

## Implementation and validation

`ui/path_picker.py` owns reusable Gradio controls, request checks, shallow
enumeration, native dialog adaptation and path syntax validation. It returns one
editable Textbox per setting; summary controls, browsing state and queue state
are not persisted. `ui/ui_settings.py` registers those fields with the existing
Settings save transaction. The save hook checks edited path syntax without
creating directories or changing existing consumer precedence.

Resolution reuses `core.paths.legacy_model_path` for model paths and matches
`features.prompting.wildcards.wildcard_roots` and Library's `Path(...).resolve()`
rules. Duplicate selection uses filesystem identity (`Path.samefile`) so Windows
long names, 8.3 names, symlinks and junction aliases are handled correctly.

The installed Gradio 6.31.0 `Request` injection and `Blocks.process_api` were
checked locally on 2026-11-09. Draft edits, browsing and order changes use
unqueued events with hidden progress. Only explicit native dialog events use
the bounded `host-folder-dialog` concurrency group; they do not occupy the
normal application queue. Per-session Gradio State tracks add versus replace.

Focused CPU checks use private data and mocked native chooser calls:

```powershell
venv/Scripts/python.exe tools/check_cpu.py test_path_picker.py test_ui_themes.py test_profile_transactions.py test_settings_ui.py test_model_folders.py test_gallery_scan.py test_wildcard_catalog.py
```

Coverage includes real Gradio callback processing, request authorization,
choose/cancel, append/replace/order, Unicode and spaces, duplicate directory
identity, permission failures, wrong namespaces, profile save/reload/failure,
and model/output/wildcard/Library consumers. Windows junction identity is tested
with an owned temporary fixture when symlink privilege is unavailable. Native
dialog commands are mocked; real GUI selection and Linux desktop behavior still
need host acceptance. Actual browser layout, mobile behavior and restart checks
are separate from these CPU contracts.
