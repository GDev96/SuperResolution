import sys
from pathlib import Path
import importlib.util

def setup_paths():
    # Calcola la root: .../SuperResolution/
    SRC_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_DIR.parent
    
    paths_to_add = [
        PROJECT_ROOT / "models" / "BasicSR",
        PROJECT_ROOT / "models" / "HAT",
        PROJECT_ROOT / "models" / "HAT" / "hat",
    ]
    
    for p in paths_to_add:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

setup_paths()

def import_external_archs():
    RRDBNet = None
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError: pass

    HAT = None
    try:
        from hat.archs.hat_arch import HAT
    except ImportError:
        try: from archs.hat_arch import HAT
        except: pass
            
    return RRDBNet, HAT