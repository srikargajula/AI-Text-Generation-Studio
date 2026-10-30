import os
import sys
from pathlib import Path

# Add project root to sys.path for pytest discovery
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# On Windows, register DLL directories for TensorFlow
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    for search_dir in [
        Path(sys.prefix) / "Scripts",
        Path(sys.prefix) / "Lib" / "site-packages" / "tensorflow",
    ]:
        if search_dir.exists():
            try:
                os.add_dll_directory(str(search_dir))
            except OSError:
                pass

