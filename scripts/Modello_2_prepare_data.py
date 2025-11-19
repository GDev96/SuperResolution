import json
import random
import sys
import subprocess
from pathlib import Path
from astropy.io import fits
import numpy as np
from tqdm import tqdm

# ================= CONFIGURAZIONE =================
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent 

# Path Dati Input (Relativo alla root del progetto)
SOURCE_PATH = PROJECT_ROOT / "data" / "M33" / "6_patches_from_cropped" / "paired_patches_folders"

# Path Output JSON (Nella cartella data della ROOT)
OUTPUT_SPLITS = PROJECT_ROOT / "data" / "splits"
# ==================================================

def get_fits_dims(path):
    try:
        with fits.open(path, mode='readonly', memmap=True) as hdul:
            for hdu in hdul:
                if hasattr(hdu, 'shape') and len(hdu.shape) >= 2:
                    return hdu.shape[-2:]
    except Exception:
        pass
    return (0, 0)

def prepare_dataset():
    print(f"🔍 Scansione cartella dati: {SOURCE_PATH}")
    
    if not SOURCE_PATH.exists():
        print(f"❌ Errore: La cartella dati non esiste!")
        print(f"   Ho cercato in: {SOURCE_PATH}")
        return

    pair_folders = sorted(list(SOURCE_PATH.glob("pair_*")))
    print(f"   Trovate {len(pair_folders)} cartelle di coppie.")
    
    valid_pairs = []
    
    print("   Analisi file...")
    for p_dir in tqdm(pair_folders):
        fits_files = list(p_dir.glob("*.fits"))
        if len(fits_files) != 2: fits_files = list(p_dir.glob("*.fits*"))
        if len(fits_files) != 2: continue
        
        f1, f2 = fits_files[0], fits_files[1]
        
        try:
            dims1 = get_fits_dims(f1)
            dims2 = get_fits_dims(f2)
            area1, area2 = dims1[0]*dims1[1], dims2[0]*dims2[1]
            
            if area1 == 0 or area2 == 0: continue

            if area1 < area2:
                ground, hubble = f1, f2
                g_dim, h_dim = dims1, dims2
            else:
                ground, hubble = f2, f1
                g_dim, h_dim = dims2, dims1
                
            valid_pairs.append({
                "patch_id": p_dir.name,
                "ground_path": str(ground),
                "hubble_path": str(hubble),
                "lr_shape": g_dim, "hr_shape": h_dim
            })
        except Exception as e:
            continue

    print(f"\n✅ Coppie valide: {len(valid_pairs)}")
    if not valid_pairs: return

    random.shuffle(valid_pairs)
    n = len(valid_pairs)
    n_tr = int(n * 0.8)
    n_val = int(n * 0.1)
    
    OUTPUT_SPLITS.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SPLITS / "train.json", "w") as f: json.dump(valid_pairs[:n_tr], f, indent=2)
    with open(OUTPUT_SPLITS / "val.json", "w") as f: json.dump(valid_pairs[n_tr:n_tr+n_val], f, indent=2)
    with open(OUTPUT_SPLITS / "test.json", "w") as f: json.dump(valid_pairs[n_tr+n_val:], f, indent=2)
        
    print(f"\n💾 JSON salvati in: {OUTPUT_SPLITS}")

def ask_continue_to_next_step():
    """Chiede se proseguire con il prossimo script."""
    next_script_name = 'Modello_3_verifica.py'
    HERE = Path(__file__).resolve().parent
    next_script_path = HERE / next_script_name

    print("\n" + "="*70)
    print("🎯 STEP 2 COMPLETATO!")
    print("="*70)
    print(f"\n📋 PROSSIMO STEP: Verifica Dataset ({next_script_name})")
    print("   Questo script prova a caricare il dataset per assicurarsi che tutto funzioni.")
    
    while True:
        print("\n" + "─"*70)
        choice = input(f"👉 Vuoi avviare '{next_script_name}' ora? [S/n]: ").strip().lower()
        if choice in ('', 's', 'si', 'y', 'yes'):
            if next_script_path.exists():
                print(f"\n🚀 Avvio {next_script_name}...")
                try:
                    subprocess.run([sys.executable, str(next_script_path)])
                except Exception as e:
                    print(f"❌ Errore durante l'avvio dello script: {e}")
            else:
                print(f"❌ Errore: File non trovato: {next_script_path}")
            return
        elif choice in ('n', 'no'):
            print(f"\n👋 Ok. Puoi eseguire lo script manualmente in seguito: python {next_script_name}")
            return
        else:
            print("❌ Scelta non valida.")

if __name__ == "__main__":
    prepare_dataset()
    ask_continue_to_next_step()