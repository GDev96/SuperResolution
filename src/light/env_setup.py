import sys
import os
from pathlib import Path

def setup_paths():
    # File corrente: .../SuperResolution/src/light/env_setup.py
    CURRENT_DIR = Path(__file__).resolve().parent
    
    # Risaliamo alla root: light -> src -> SuperResolution
    PROJECT_ROOT = CURRENT_DIR.parent.parent
    
    print(f"\n🔍 CONFIGURAZIONE PATH (LIGHT MODE):")
    print(f"   📂 Root rilevata: {PROJECT_ROOT}")
    
    # 1. Aggiungi Models Root
    models_root = PROJECT_ROOT / "models"
    if str(models_root) not in sys.path:
        sys.path.insert(0, str(models_root))

    # 2. Cerca e aggiungi BasicSR (Logica di ricerca ricorsiva sicura)
    try:
        basicsr_path = list(models_root.rglob("rrdbnet_arch.py"))
        if basicsr_path:
            bsr_root = basicsr_path[0].parent.parent.parent
            if str(bsr_root) not in sys.path:
                sys.path.insert(0, str(bsr_root))
                print(f"   ✅ BasicSR trovato in: {bsr_root}")
    except Exception:
        pass

    # 3. Cerca e aggiungi HAT
    try:
        hat_path = list(models_root.rglob("hat_arch.py"))
        if hat_path:
            hat_root = hat_path[0].parent.parent.parent
            if str(hat_root) not in sys.path:
                sys.path.insert(0, str(hat_root))
                print(f"   ✅ HAT trovato in: {hat_root}")
    except Exception:
        pass

setup_paths()

def import_external_archs():
    RRDBNet = None
    HAT = None

    # --- IMPORT BASICSR ---
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except Exception as e:
        print(f"   ❌ BasicSR Import Error: {e}")

    # --- IMPORT HAT ---
    try:
        from hat.archs.hat_arch import HAT
    except ImportError:
        try:
            import hat.archs.hat_arch as hat_module
            HAT = hat_module.HAT
        except Exception:
            pass
    except Exception:
        pass

    return RRDBNet, HAT