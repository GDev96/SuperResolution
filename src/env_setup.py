import sys
import os
from pathlib import Path
import traceback

def setup_paths():
    # Calcola la root: .../SuperResolution/
    SRC_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_DIR.parent
    
    print(f"\n🔍 CONFIGURAZIONE PATH INTELLIGENTE:")
    
    # 1. Aggiungi Models Root
    models_root = PROJECT_ROOT / "models"
    if str(models_root) not in sys.path:
        sys.path.insert(0, str(models_root))

    # 2. Cerca e aggiungi BasicSR
    basicsr_path = list(models_root.rglob("rrdbnet_arch.py"))
    if basicsr_path:
        # Prendi la cartella padre di 'basicsr' (es: .../models/BasicSR)
        # basicsr_path[0] è .../basicsr/archs/rrdbnet_arch.py
        # .parent -> archs
        # .parent -> basicsr (package)
        # .parent -> ROOT DA AGGIUNGERE
        bsr_root = basicsr_path[0].parent.parent.parent
        if str(bsr_root) not in sys.path:
            sys.path.insert(0, str(bsr_root))
            print(f"   ✅ BasicSR trovato in: {bsr_root}")

    # 3. Cerca e aggiungi HAT (Fix 'archs' not found)
    hat_path = list(models_root.rglob("hat_arch.py"))
    if hat_path:
        # hat_path[0] è .../hat/archs/hat_arch.py
        # Vogliamo aggiungere la cartella che contiene il pacchetto 'hat'
        hat_root = hat_path[0].parent.parent.parent
        if str(hat_root) not in sys.path:
            sys.path.insert(0, str(hat_root))
            print(f"   ✅ HAT trovato in: {hat_root}")
    else:
        print("   ⚠️  File 'hat_arch.py' non trovato in models!")

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
    print("   🔍 Caricamento HAT...", end=" ")
    try:
        # Prova import standard
        from hat.archs.hat_arch import HAT
        print("OK (Standard)")
    except ImportError:
        try:
            # Fix per alcune versioni di repo HAT
            import hat.archs.hat_arch as hat_module
            HAT = hat_module.HAT
            print("OK (Modulo Diretto)")
        except Exception as e:
            print(f"\n   ⚠️  HAT disabilitato: {e}")
            print("      (Assicurati di aver fatto: pip install einops timm)")
    except Exception as e:
        print(f"\n   ⚠️  HAT Error: {e}")

    return RRDBNet, HAT