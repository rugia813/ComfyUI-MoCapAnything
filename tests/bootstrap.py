import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT.parents[1]
sys.path.insert(0, str(COMFY))
package = types.ModuleType("mocap_extension")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
