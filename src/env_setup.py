import sys
import os
from pathlib import Path
import importlib.util
import traceback

def setup_paths():
    # Calcola la root: .../SuperResolution/
    SRC_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_DIR.parent
    
    # Lista di possibili percorsi dove cercare le librerie
    paths_to_check = [
        PROJECT_ROOT / "models" / "BasicSR",
        PROJECT_ROOT / "models" / "HAT",
        PROJECT_ROOT / "models" / "Real-ESRGAN",
    ]
    
    print(f"\n🔍 DEBUG PATHS in env_setup.py:")
    for p in paths_to_check:
        if p.exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
                print(f"   ✅ Aggiunto path: {p.name}")
            else:
                print(f"   ℹ️  Già presente: {p.name}")
        else:
            print(f"   ⚠️  Path non trovato: {p}")

setup_paths()

def import_external_archs():
    RRDBNet = None
    HAT = None

    print("\n🔍 TENTATIVO IMPORT BasicSR...")
    try:
        # Prova l'import standard
        from basicsr.archs.rrdbnet_arch import RRDBNet
        print("   ✅ RRDBNet importato con successo!")
    except Exception as e:
        print(f"   ❌ FALLITO IMPORT BasicSR: {e}")
        print("   📜 Traceback completo (mostra questo errore):")
        traceback.print_exc()
        print("   --------------------------------------------------")

    print("\n🔍 TENTATIVO IMPORT HAT...")
    try:
        from hat.archs.hat_arch import HAT
        print("   ✅ HAT importato con successo!")
    except ImportError:
        try: 
            from archs.hat_arch import HAT
            print("   ✅ HAT importato (percorso alternativo)!")
        except Exception as e:
            print(f"   ⚠️ HAT non trovato (opzionale): {e}")
    except Exception as e:
        print(f"   ⚠️ Errore generico HAT: {e}")
            
    return RRDBNet, HAT