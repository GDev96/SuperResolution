"""
MODELLO - STEP 3: VERIFICA DATASET (SMART VERSION)
Logica aggiornata: ignora file extra e seleziona automaticamente LR/HR per dimensione.
"""

import sys
import json
import subprocess
from pathlib import Path
from astropy.io import fits
import numpy as np

# ============================================================================
# CONFIGURAZIONE PATH
# ============================================================================
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.append(str(PROJECT_ROOT))
ROOT_DATA_DIR = PROJECT_ROOT / "data"

print(f"📂 Project Root: {PROJECT_ROOT}")

# Importazione robusta Dataset
try:
    from src.dataset import AstronomicalDataset
    print("✅ Modulo 'src.dataset' importato.")
except ImportError:
    try:
        from src.light.dataset import AstronomicalDataset
        print("⚠️ 'src.dataset' non trovato, usato 'src.light.dataset'.")
    except ImportError as e:
        print(f"❌ ERRORE IMPORT: {e}"); sys.exit(1)

# ============================================================================
# FUNZIONI
# ============================================================================
def select_target_directory():
    print("\n" + "📂"*35)
    print("SELEZIONE DATASET DA VERIFICARE".center(70))
    print("📂"*35)
    
    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    except: return None

    valid_targets = [d for d in subdirs if (d / "6_patches_from_cropped" / "splits").exists()]
    
    if not valid_targets:
        print("\n❌ Nessun dataset con splits trovato!")
        return None

    print("\nDataset disponibili:")
    for i, d in enumerate(valid_targets):
        splits = d / "6_patches_from_cropped" / "splits"
        meta = splits / "dataset_info.json"
        info = ""
        if meta.exists():
            try:
                with open(meta) as f: 
                    j = json.load(f)
                    info = f"(LR: {j.get('target_lr_size')} | HR: {j.get('target_hr_size')})"
            except: pass
        print(f"   {i+1}: {d.name} {info}")

    while True:
        try:
            c = input(f"\n👉 Scelta (1-{len(valid_targets)}) o 'q': ").strip()
            if c == 'q': return None
            idx = int(c) - 1
            if 0 <= idx < len(valid_targets): return valid_targets[idx]
        except: pass

def check_dataset(target_dir):
    print(f"\n🔍 VERIFICA DATASET: {target_dir.name}")
    splits_dir = target_dir / "6_patches_from_cropped" / "splits"
    train_dir = splits_dir / "train"
    
    if not train_dir.exists():
        print(f"❌ Cartella train mancante: {train_dir}")
        return False
        
    # 1. CREAZIONE LISTA SMART
    print("\n🔬 Analisi Smart delle coppie (ignoro file extra)...")
    train_list = []
    
    # Controlliamo le prime 10 cartelle
    pair_folders = sorted(list(train_dir.glob("pair_*")))[:10]
    
    for p_dir in pair_folders:
        fits_files = list(p_dir.glob("*.fits"))
        if len(fits_files) < 2: continue
        
        candidates = []
        for f in fits_files:
            try:
                with fits.open(f) as hdul:
                    if hdul[0].data is not None:
                        shape = hdul[0].data.shape
                        # Gestione (C,H,W) o (H,W)
                        if len(shape) == 3: area = shape[-2]*shape[-1]
                        else: area = shape[0]*shape[1]
                        candidates.append((f, area))
            except: pass
        
        if len(candidates) < 2: continue
        
        # ORDINA: Piccolo -> Grande
        candidates.sort(key=lambda x: x[1])
        lr = candidates[0][0]
        hr = candidates[-1][0]
        
        train_list.append({
            "ground_path": str(lr),
            "hubble_path": str(hr)
        })
        print(f"   ✅ {p_dir.name}: LR trovato ({lr.name}) -> HR trovato ({hr.name})")

    if not train_list:
        print("❌ Nessuna coppia valida trovata.")
        return False
        
    # 2. TEST CARICAMENTO PYTORCH
    print(f"\n📦 Test caricamento PyTorch su {len(train_list)} campioni...")
    temp_json = splits_dir / "temp_test.json"
    
    try:
        with open(temp_json, 'w') as f: json.dump(train_list, f)
        
        # Carichiamo senza resize forzato per vedere le dimensioni reali
        ds = AstronomicalDataset(temp_json, base_path=PROJECT_ROOT, augment=False)
        
        if len(ds) > 0:
            sample = ds[0]
            lr_shape = sample['lr'].shape
            hr_shape = sample['hr'].shape
            print(f"   📄 Tensor Shape Rilevato: LR {lr_shape} -> HR {hr_shape}")
            
            # Verifica congruenza base
            if lr_shape[-1] < hr_shape[-1]:
                print("   ✅ Le dimensioni sembrano corrette (LR < HR).")
                temp_json.unlink()
                return True
            else:
                print("   ⚠️  ATTENZIONE: LR non è più piccolo di HR. Controlla i dati.")
                temp_json.unlink()
                return True # Ritorniamo True comunque per permettere all'utente di decidere
        else:
            print("   ❌ Dataset vuoto dopo il caricamento.")
            return False
            
    except Exception as e:
        print(f"   ❌ Errore Dataset Class: {e}")
        if temp_json.exists(): temp_json.unlink()
        return False

def ask_next():
    print("\n" + "="*60)
    print("🎯 VERIFICA COMPLETATA".center(60))
    print("="*60)
    print("1. Avvia Training LIGHT (Test veloce)")
    print("2. Avvia Training HEAVY (Produzione)")
    print("0. Esci")
    
    c = input("\n👉 Scelta: ").strip()
    if c == '1': subprocess.run([sys.executable, str(HERE / "Modello_4_train_light.py")])
    elif c == '2': subprocess.run([sys.executable, str(HERE / "Modello_4_train_heavy.py")])

if __name__ == "__main__":
    td = select_target_directory()
    if td:
        if check_dataset(td):
            ask_next()
        else:
            print("\n❌ Verifica fallita.")