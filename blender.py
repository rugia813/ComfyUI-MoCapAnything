"""Locate Blender installed on the host system."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def _windows_installs():
    import winreg

    installs = []
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall") as root:
                for index in range(winreg.QueryInfoKey(root)[0]):
                    try:
                        with winreg.OpenKey(root, winreg.EnumKey(root, index)) as key:
                            if winreg.QueryValueEx(key, "DisplayName")[0].lower().startswith("blender"):
                                installs.append(Path(winreg.QueryValueEx(key, "InstallLocation")[0]) / "blender.exe")
                    except OSError:
                        pass
        except OSError:
            pass
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        if os.environ.get(variable):
            installs.extend((Path(os.environ[variable]) / "Blender Foundation").glob("Blender*/blender.exe"))
    return installs


def find_blender():
    candidates = [shutil.which("blender")]
    if sys.platform == "win32":
        candidates.extend(_windows_installs())
    elif sys.platform == "darwin":
        candidates.extend((Path.home() / "Applications/Blender.app/Contents/MacOS/Blender",
                           Path("/Applications/Blender.app/Contents/MacOS/Blender")))
    versions = []
    for candidate in filter(None, candidates):
        path = Path(candidate)
        if not path.is_file():
            continue
        try:
            output = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=10,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except (OSError, subprocess.TimeoutExpired):
            continue
        match = re.search(r"Blender (\d+)\.(\d+)", output.stdout)
        if output.returncode == 0 and match:
            version = tuple(map(int, match.groups()))
            if version >= (4, 2):
                versions.append((version, str(path.resolve())))
    if versions:
        return max(versions)[1]
    raise RuntimeError("Blender 4.2+ was not found. Install Blender, then restart ComfyUI. "
                       "For a portable Blender build, add its executable to PATH.")
