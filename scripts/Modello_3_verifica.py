"""
MODELLO - STEP 3: VERIFICA DATASET (80x80 -> 512x512)
Verifica specifica per il nuovo formato.
"""

import sys
import json
from pathlib import Path
from astropy.io import fits
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.append(str(PROJECT_ROOT))
ROOT_DATA_DIR = PROJECT_ROOT / "data"

def select_target_directory():
    print("\nSELEZIONE DATASET DA VERIFICARE")
    # Cerca dataset che hanno la cartella splits dentro aligned
    valid = []
    for d in ROOT_DATA_DIR.iterdir():
        if (d / "6_patches_aligned" / "splits").exists():
            valid.append(d)
            
    if not valid: return None
    
    for i, d in enumerate(valid): print(f"{i+1}: {d.name}")
    try:
        s = int(input(">> ")) - 1
        return valid[s]
    except: return None

def check_dataset(target_dir):
    splits_dir = target_dir / "6_patches_aligned" / "splits"
    train_dir = splits_dir / "train"
    
    print(f"\n🔍 Controllo: {train_dir}")
    
    pairs = list(train_dir.glob("pair_*"))
    if not pairs:
        print("❌ Cartella vuota.")
        return False
        
    # Check random
    import random
    sample = random.choice(pairs)
    f_lr = sample / "observatory.fits"
    f_hr = sample / "hubble.fits"
    
    if not f_lr.exists() or not f_hr.exists():
        print("❌ File mancanti nella coppia.")
        return False
        
    with fits.open(f_lr) as l, fits.open(f_hr) as h:
        lr_shape = l[0].data.shape
        hr_shape = h[0].data.shape
        
        # Gestione dimensioni
        h_lr = lr_shape[-2] if len(lr_shape)==3 else lr_shape[0]
        h_hr = hr_shape[-2] if len(hr_shape)==3 else hr_shape[0]
        
        print(f"   Campione: {sample.name}")
        print(f"   LR Shape: {lr_shape} (Atteso ~80)")
        print(f"   HR Shape: {hr_shape} (Atteso ~512)")
        
        if h_lr == 80 and h_hr == 512:
            print("✅ DIMENSIONI PERFETTE.")
        else:
            print("⚠️  Dimensioni diverse dallo standard (80->512), ma potrebbe funzionare.")
            
    return True

if __name__ == "__main__":
    td = select_target_directory()
    if td: check_dataset(td)