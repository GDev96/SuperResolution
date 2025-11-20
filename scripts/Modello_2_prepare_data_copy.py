"""
MODELLO - STEP 2: PREPARAZIONE DATASET (SMART FILTER VERSION)
Gestisce cartelle con più di 2 file selezionando automaticamente LR (più piccolo) e HR (più grande).
"""

import json
import random
import sys
import subprocess
import shutil
from pathlib import Path
from astropy.io import fits
import numpy as np
from tqdm import tqdm

# ============================================================================
# CONFIGURAZIONE PATH
# ============================================================================
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# DIMENSIONI TARGET
TARGET_HR_SIZE = 512
TARGET_LR_SIZE = 128
SCALE_RATIO = 4.0

print(f"📂 Project Root: {PROJECT_ROOT}")
print(f"📂 Data Dir:     {ROOT_DATA_DIR}")

# ============================================================================
# FUNZIONI UTILI
# ============================================================================
def select_target_directory():
    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    except Exception: return None
    
    if not subdirs: return None
    
    print("\nCartelle target disponibili:")
    for i, d in enumerate(subdirs): print(f"   {i+1}: {d.name}")
    
    while True:
        try:
            c = input(f"👉 Seleziona (1-{len(subdirs)}): ").strip()
            if c == 'q': return None
            idx = int(c) - 1
            if 0 <= idx < len(subdirs): return subdirs[idx]
        except ValueError: pass

def get_fits_dims(path):
    try:
        with fits.open(path, mode='readonly', memmap=True) as hdul:
            # Cerca la prima estensione con dati validi
            for hdu in hdul:
                if hasattr(hdu, 'data') and hdu.data is not None:
                    shape = hdu.data.shape
                    # Gestisce (Channel, H, W) o (H, W)
                    if len(shape) == 3: return shape[-2:]
                    elif len(shape) == 2: return shape
    except Exception: pass
    return (0, 0)

def copy_pair_to_split(pair_info, split_dir, pair_id):
    try:
        pair_folder = split_dir / f"pair_{pair_id:05d}"
        pair_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pair_info['ground_path'], pair_folder / Path(pair_info['ground_path']).name)
        shutil.copy2(pair_info['hubble_path'], pair_folder / Path(pair_info['hubble_path']).name)
        return True
    except: return False

# ============================================================================
# CORE PREPARATION LOGIC
# ============================================================================
def prepare_dataset(target_dir):
    print(f"\n📊 PREPARAZIONE DATASET: {target_dir.name}")
    SOURCE_PATH = target_dir / "6_patches_from_cropped" / "paired_patches_folders"
    OUTPUT_SPLITS = target_dir / "6_patches_from_cropped" / "splits"
    
    if not SOURCE_PATH.exists():
        print(f"❌ Errore: Cartella {SOURCE_PATH} non trovata.")
        return False

    pair_folders = sorted(list(SOURCE_PATH.glob("pair_*")))
    print(f"   Trovate {len(pair_folders)} cartelle di coppie.")
    
    valid_pairs = []
    invalid_pairs = []
    
    print("\n📐 Analisi Intelligente (Smart Sort)...")
    
    for p_dir in tqdm(pair_folders, desc="Scansione"):
        # 1. Trova TUTTI i fits
        all_fits = list(p_dir.glob("*.fits"))
        if len(all_fits) < 2:
            invalid_pairs.append((p_dir.name, "Meno di 2 file FITS"))
            continue
            
        # 2. Analizza Dimensioni per ogni file
        candidates = []
        for f in all_fits:
            h, w = get_fits_dims(f)
            area = h * w
            if area > 0:
                candidates.append({'path': f, 'h': h, 'w': w, 'area': area})
        
        if len(candidates) < 2:
            invalid_pairs.append((p_dir.name, "File FITS illeggibili"))
            continue
            
        # 3. ORDINA PER AREA (Piccolo -> Grande)
        candidates.sort(key=lambda x: x['area'])
        
        # 4. SELEZIONA ESTREMI
        lr_img = candidates[0]   # Il più piccolo
        hr_img = candidates[-1]  # Il più grande
        
        # 5. VALIDAZIONE
        if lr_img['area'] == hr_img['area']:
            invalid_pairs.append((p_dir.name, "LR e HR hanno stessa dimensione"))
            continue
            
        # Validazione dimensioni tollerante
        if not (60 <= lr_img['h'] <= 160): # Accetta range 128 +/-
             invalid_pairs.append((p_dir.name, f"LR size anomala {lr_img['h']}x{lr_img['w']}"))
             continue
             
        valid_pairs.append({
            "patch_id": p_dir.name,
            "ground_path": str(lr_img['path']),
            "hubble_path": str(hr_img['path']),
            "lr_shape": (lr_img['h'], lr_img['w']),
            "hr_shape": (hr_img['h'], hr_img['w'])
        })

    # ========================================
    # RISULTATI
    # ========================================
    print(f"\n📊 RISULTATI ANALISI:")
    print(f"   ✅ Coppie valide:   {len(valid_pairs)}")
    print(f"   ❌ Scartate:        {len(invalid_pairs)}")
    
    if not valid_pairs:
        print("❌ ERRORE: Nessuna coppia valida. Controlla che le patch siano state estratte.")
        return False

    # Splitting
    random.shuffle(valid_pairs)
    n = len(valid_pairs)
    n_tr = int(n * 0.8)
    n_val = int(n * 0.1)
    
    sets = {
        'train': valid_pairs[:n_tr],
        'val': valid_pairs[n_tr:n_tr+n_val],
        'test': valid_pairs[n_tr+n_val:]
    }
    
    # Pulizia e Scrittura
    if OUTPUT_SPLITS.exists(): shutil.rmtree(OUTPUT_SPLITS)
    OUTPUT_SPLITS.mkdir(parents=True)

    for split_name, pairs in sets.items():
        out_dir = OUTPUT_SPLITS / split_name
        out_dir.mkdir()
        print(f"   💾 Copia {split_name} ({len(pairs)})...")
        for idx, p in enumerate(tqdm(pairs, desc=split_name)):
            copy_pair_to_split(p, out_dir, idx)

    # Metadata
    metadata = {
        "target": target_dir.name,
        "target_lr_size": TARGET_LR_SIZE,
        "target_hr_size": TARGET_HR_SIZE,
        "scale_ratio": SCALE_RATIO,
        "train_pairs": len(sets['train']),
        "val_pairs": len(sets['val'])
    }
    with open(OUTPUT_SPLITS / "dataset_info.json", 'w') as f:
        json.dump(metadata, f, indent=2)
        
    return True

if __name__ == "__main__":
    td = select_target_directory()
    if td and prepare_dataset(td):
        print("\n✅ Preparazione completata! Ora puoi eseguire la Verifica (Step 3).")