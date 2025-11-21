import sys
import os
from pathlib import Path

def setup_paths():
    # 1. Trova la root del progetto
    # Tentativo A: Relativo a questo file
    SRC_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_DIR.parent
    
    # Tentativo B: Assoluto standard RunPod (Fallback sicuro)
    RUNPOD_ROOT = Path("/root/SuperResolution")
    if RUNPOD_ROOT.exists():
        PROJECT_ROOT = RUNPOD_ROOT

    print(f"\n🔍 CONFIGURAZIONE PATH (Root: {PROJECT_ROOT})")
    
    # 2. Aggiungi Models Root
    models_root = PROJECT_ROOT / "models"
    if str(models_root) not in sys.path:
        sys.path.insert(0, str(models_root))

    # 3. Cerca e aggiungi BasicSR
    # Cerca ricorsivamente perché la struttura potrebbe variare dopo git clone
    basicsr_path = list(models_root.rglob("rrdbnet_arch.py"))
    if basicsr_path:
        # Risaliamo fino alla root che contiene 'basicsr'
        bsr_root = basicsr_path[0].parent.parent.parent
        if str(bsr_root) not in sys.path:
            sys.path.insert(0, str(bsr_root))
            print(f"   ✅ BasicSR collegato: {bsr_root}")
    else:
        print("   ⚠️  BasicSR non trovato in models/!")

    # 4. Cerca e aggiungi HAT
    hat_path = list(models_root.rglob("hat_arch.py"))
    if hat_path:
        # Risaliamo fino alla root che contiene 'hat'
        hat_root = hat_path[0].parent.parent.parent
        if str(hat_root) not in sys.path:
            sys.path.insert(0, str(hat_root))
            print(f"   ✅ HAT collegato: {hat_root}")
    else:
        print("   ⚠️  HAT non trovato in models/!")

setup_paths()

def import_external_archs():
    RRDBNet = None
    HAT = None

    # --- IMPORT BASICSR ---
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError:
        print("   ❌ Errore import BasicSR. Controlla path e installazione.")

    # --- IMPORT HAT ---
    try:
        # Prova import standard
        from hat.archs.hat_arch import HAT
    except ImportError:
        try:
            # Fallback per alcune strutture folder
            import hat.archs.hat_arch as hat_module
            HAT = hat_module.HAT
        except Exception as e:
            print(f"   ⚠️  HAT non caricato: {e}")

    return RRDBNet, HAT