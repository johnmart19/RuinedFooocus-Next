"""Pick a local GGUF without uploading or copying its weights."""

import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def select_gguf():
    wsl = os.name != "nt" and (
        "microsoft" in platform.release().lower() or bool(os.environ.get("WSL_DISTRO_NAME"))
    )
    if os.name == "nt" or wsl:
        powershell = shutil.which("powershell.exe")
        if not powershell:
            raise RuntimeError("Windows file picker requires PowerShell and WSL interoperability.")
        script = """
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Select a GGUF model'
$dialog.Filter = 'GGUF models (*.gguf)|*.gguf'
$dialog.CheckFileExists = $true
if ($dialog.ShowDialog() -eq 'OK') { [Console]::Write($dialog.FileName) }
$dialog.Dispose()
"""
        command = [powershell, "-NoProfile", "-STA", "-Command", script]
    elif shutil.which("zenity"):
        command = [
            "zenity",
            "--file-selection",
            "--title=Select a GGUF model",
            "--file-filter=GGUF models | *.gguf",
        ]
    elif shutil.which("kdialog"):
        command = ["kdialog", "--getopenfilename", str(Path.home()), "*.gguf|GGUF models"]
    else:
        command = [
            sys.executable,
            "-c",
            """
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
try:
    print(filedialog.askopenfilename(title='Select a GGUF model',
          filetypes=[('GGUF models', '*.gguf')]), end='')
finally:
    root.destroy()
""",
        ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    selected = result.stdout.strip()
    if result.returncode == 1 and command[0] in ("zenity", "kdialog"):
        return None
    if result.returncode:
        raise RuntimeError(
            "Could not open the file picker. On Linux, a desktop session and "
            "Zenity, KDialog or python3-tk are required. You can also set "
            "Custom GGUF path in Settings and refresh the model list."
        )
    if not selected:
        return None
    if wsl:
        selected = subprocess.check_output(["wslpath", "-u", selected], text=True).strip()
    path = Path(selected)
    if path.suffix.lower() != ".gguf" or path.name.lower().startswith("mmproj"):
        raise ValueError("Select a GGUF language model, not a projector file.")
    with path.open("rb") as file:
        if file.read(4) != b"GGUF":
            raise ValueError("The selected file is not a GGUF model.")
    return str(path.resolve())
