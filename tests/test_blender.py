import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import bootstrap
from mocap_extension.blender import find_blender


class BlenderDiscoveryTests(unittest.TestCase):
    def test_installed_windows_blender_without_path(self):
        old, current = Path("old-blender.exe"), Path("current-blender.exe")
        def version(args, **kwargs):
            release = "3.0.0" if args[0] == str(old) else "5.2.1"
            return subprocess.CompletedProcess(args, 0, f"Blender {release}", "")
        with patch("mocap_extension.blender.sys.platform", "win32"), \
             patch("mocap_extension.blender.shutil.which", return_value=None), \
             patch("mocap_extension.blender._windows_installs", return_value=[old, current]), \
             patch("mocap_extension.blender.Path.is_file", return_value=True), \
             patch("mocap_extension.blender.subprocess.run", side_effect=version):
            self.assertEqual(find_blender(), str(current.resolve()))

    def test_path_on_linux(self):
        path = Path("blender-test")
        with patch("mocap_extension.blender.sys.platform", "linux"), \
             patch("mocap_extension.blender.shutil.which", return_value=str(path)), \
             patch("mocap_extension.blender.Path.is_file", return_value=True), \
             patch("mocap_extension.blender.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "Blender 4.2.0", "")):
            self.assertEqual(find_blender(), str(path.resolve()))

    def test_macos_app_without_path(self):
        app = Path("/Applications/Blender.app/Contents/MacOS/Blender")
        with patch("mocap_extension.blender.sys.platform", "darwin"), \
             patch("mocap_extension.blender.shutil.which", return_value=None), \
             patch("mocap_extension.blender.Path.is_file", autospec=True, side_effect=lambda path: path == app), \
             patch("mocap_extension.blender.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "Blender 4.5.0", "")):
            self.assertEqual(find_blender(), str(app.resolve()))

    def test_missing_blender_explains_portable_setup(self):
        with patch("mocap_extension.blender.sys.platform", "linux"), \
             patch("mocap_extension.blender.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "add its executable to PATH"):
                find_blender()


if __name__ == "__main__":
    unittest.main()
