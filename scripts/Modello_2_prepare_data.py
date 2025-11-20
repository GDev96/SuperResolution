"""
MODELLO - STEP 2: PREPARAZIONE DATASET
Con menu selezione target e splits in sottocartelle (train/val/test) con foto

POSIZIONE: scripts/Modello_2_prepare_data.py
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
# CONFIGURAZIONE PATH (DA CARTELLA SCRIPTS)
# ============================================================================
HERE = Path(__file__).resolve().parent  # scripts/
PROJECT_ROOT = HERE.parent               # SuperResolution/
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# DIMENSIONI TARGET - Ottimizzate per RTX 2060 (6GB VRAM)
TARGET_HR_SIZE = 512  # Hubble HR patches
TARGET_LR_SIZE = 34   # Ground LR patches
SCALE_RATIO = 15      # Rapporto di scala (512/34 ≈ 15)

print(f"📂 Project Root: {PROJECT_ROOT}")
print(f"📂 Data Dir:     {ROOT_DATA_DIR}")

# ============================================================================
# FUNZIONI MENU E SELEZIONE (DA DATASET)
# ============================================================================
def select_target_directory():
    """Mostra un menu per selezionare UNA cartella target (NO batch per modello)."""
    print("\n" + "📂"*35)
    print("SELEZIONE DATASET TARGET".center(70))
    print("📂"*35)
    print(f"\nScansione sottocartelle in: {ROOT_DATA_DIR}")

    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() 
                   if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile leggere la cartella {ROOT_DATA_DIR}: {e}")
        return None

    if not subdirs:
        print(f"\n❌ ERRORE: Nessuna sottocartella trovata in {ROOT_DATA_DIR}")
        return None

    print("\nCartelle target disponibili:")
    for i, dir_path in enumerate(subdirs):
        print(f"   {i+1}: {dir_path.name}")

    while True:
        print("\n" + "─"*70)
        try:
            choice_str = input(f"👉 Seleziona un numero (1-{len(subdirs)}) o 'q' per uscire: ").strip()

            if choice_str.lower() == 'q':
                return None

            choice = int(choice_str)
            choice_idx = choice - 1
            
            if 0 <= choice_idx < len(subdirs):
                selected_dir = subdirs[choice_idx]
                print(f"\n✅ Dataset selezionato: {selected_dir.name}")
                return selected_dir
            else:
                print(f"❌ Scelta non valida.")
        except ValueError:
            print("❌ Input non valido.")

# ============================================================================
# FUNZIONI
# ============================================================================
def get_fits_dims(path):
    """Ottiene le dimensioni di un file FITS."""
    try:
        with fits.open(path, mode='readonly', memmap=True) as hdul:
            for hdu in hdul:
                if hasattr(hdu, 'shape') and len(hdu.shape) >= 2:
                    return hdu.shape[-2:]
    except Exception:
        pass
    return (0, 0)

def check_dimensions_compatibility(lr_shape, hr_shape):
    """Verifica che le dimensioni siano compatibili con il modello."""
    lr_h, lr_w = lr_shape
    hr_h, hr_w = hr_shape
    
    if lr_h == 0 or lr_w == 0 or hr_h == 0 or hr_w == 0:
        return False, "Dimensioni zero"
    
    if lr_h >= hr_h or lr_w >= hr_w:
        return False, f"LR ({lr_h}x{lr_w}) non è più piccola di HR ({hr_h}x{hr_w})"
    
    actual_scale_h = hr_h / lr_h
    actual_scale_w = hr_w / lr_w
    
    scale_tolerance = 0.2
    min_scale = SCALE_RATIO * (1 - scale_tolerance)
    max_scale = SCALE_RATIO * (1 + scale_tolerance)
    
    if not (min_scale <= actual_scale_h <= max_scale and min_scale <= actual_scale_w <= max_scale):
        return False, f"Rapporto scala {actual_scale_h:.1f}x fuori range [{min_scale:.1f}-{max_scale:.1f}]"
    
    size_tolerance = 50
    hr_in_range = (TARGET_HR_SIZE - size_tolerance <= hr_h <= TARGET_HR_SIZE + size_tolerance and
                   TARGET_HR_SIZE - size_tolerance <= hr_w <= TARGET_HR_SIZE + size_tolerance)
    lr_in_range = (TARGET_LR_SIZE - size_tolerance <= lr_h <= TARGET_LR_SIZE + size_tolerance and
                   TARGET_LR_SIZE - size_tolerance <= lr_w <= TARGET_LR_SIZE + size_tolerance)
    
    if not (hr_in_range and lr_in_range):
        return False, f"Dimensioni {lr_h}x{lr_w} -> {hr_h}x{hr_w} fuori target"
    
    return True, "OK"

def copy_pair_to_split(pair_info, split_dir, pair_id):
    """Copia una coppia di patch nella directory split corrispondente."""
    try:
        # Crea sottocartella pair nella split
        pair_folder = split_dir / f"pair_{pair_id:05d}"
        pair_folder.mkdir(parents=True, exist_ok=True)
        
        # Copia file LR e HR
        ground_src = Path(pair_info['ground_path'])
        hubble_src = Path(pair_info['hubble_path'])
        
        shutil.copy2(ground_src, pair_folder / ground_src.name)
        shutil.copy2(hubble_src, pair_folder / hubble_src.name)
        
        return True
    except Exception as e:
        print(f"⚠️  Errore copia {pair_id}: {e}")
        return False

def prepare_dataset(target_dir):
    """Prepara il dataset per un target specifico."""
    
    print("\n" + "="*70)
    print(f"📊 PREPARAZIONE DATASET: {target_dir.name}".center(70))
    print("="*70)
    print(f"\n🎯 Configurazione:")
    print(f"   Target LR size: {TARGET_LR_SIZE}x{TARGET_LR_SIZE} px")
    print(f"   Target HR size: {TARGET_HR_SIZE}x{TARGET_HR_SIZE} px")
    print(f"   Scale ratio: {SCALE_RATIO}x")
    print(f"   Hardware: RTX 2060 (6GB VRAM)")
    
    # Path dati
    SOURCE_PATH = target_dir / "6_patches_from_cropped" / "paired_patches_folders"
    
    # Output splits nella stessa posizione delle coppie
    OUTPUT_SPLITS_BASE = target_dir / "6_patches_from_cropped" / "splits"
    
    print(f"\n🔍 Scansione cartella: {SOURCE_PATH}")
    
    if not SOURCE_PATH.exists():
        print(f"❌ Errore: La cartella dati non esiste!")
        print(f"   Ho cercato in: {SOURCE_PATH}")
        print(f"\n💡 Esegui prima: Dataset_step3_analizzapatch_OPTIMIZED.py")
        return False

    pair_folders = sorted(list(SOURCE_PATH.glob("pair_*")))
    print(f"   Trovate {len(pair_folders)} cartelle di coppie.")
    
    if len(pair_folders) == 0:
        print(f"❌ Nessuna coppia trovata!")
        return False
    
    # ========================================
    # ANALISI E VALIDAZIONE
    # ========================================
    valid_pairs = []
    invalid_pairs = []
    
    print("\n📐 Analisi dimensioni coppie...")
    for p_dir in tqdm(pair_folders, desc="Scansione"):
        fits_files = list(p_dir.glob("*.fits"))
        if len(fits_files) != 2:
            fits_files = list(p_dir.glob("*.fits*"))
        if len(fits_files) != 2:
            invalid_pairs.append((p_dir.name, "Numero file != 2"))
            continue
        
        f1, f2 = fits_files[0], fits_files[1]
        
        try:
            dims1 = get_fits_dims(f1)
            dims2 = get_fits_dims(f2)
            area1, area2 = dims1[0]*dims1[1], dims2[0]*dims2[1]
            
            if area1 == 0 or area2 == 0:
                invalid_pairs.append((p_dir.name, "Dimensioni zero"))
                continue

            if area1 < area2:
                ground, hubble = f1, f2
                g_dim, h_dim = dims1, dims2
            else:
                ground, hubble = f2, f1
                g_dim, h_dim = dims2, dims1
            
            is_valid, message = check_dimensions_compatibility(g_dim, h_dim)
            
            if not is_valid:
                invalid_pairs.append((p_dir.name, message))
                continue
            
            valid_pairs.append({
                "patch_id": p_dir.name,
                "ground_path": str(ground),
                "hubble_path": str(hubble),
                "lr_shape": g_dim,
                "hr_shape": h_dim
            })
            
        except Exception as e:
            invalid_pairs.append((p_dir.name, f"Errore: {str(e)}"))
            continue

    # ========================================
    # STATISTICHE
    # ========================================
    print(f"\n📊 RISULTATI:")
    print(f"   ✅ Coppie valide:   {len(valid_pairs)}")
    print(f"   ❌ Coppie invalide: {len(invalid_pairs)}")
    
    if len(invalid_pairs) > 0 and len(invalid_pairs) <= 10:
        print(f"\n⚠️  Coppie scartate:")
        for pair_name, reason in invalid_pairs[:10]:
            print(f"      • {pair_name}: {reason}")
    elif len(invalid_pairs) > 10:
        print(f"\n⚠️  Prime 10 coppie scartate:")
        for pair_name, reason in invalid_pairs[:10]:
            print(f"      • {pair_name}: {reason}")
        print(f"      ... e altre {len(invalid_pairs)-10}")
    
    if not valid_pairs:
        print(f"\n❌ ERRORE: Nessuna coppia valida trovata!")
        return False

    # Statistiche dimensioni
    lr_shapes = [p['lr_shape'] for p in valid_pairs]
    hr_shapes = [p['hr_shape'] for p in valid_pairs]
    
    avg_lr_h = np.mean([s[0] for s in lr_shapes])
    avg_lr_w = np.mean([s[1] for s in lr_shapes])
    avg_hr_h = np.mean([s[0] for s in hr_shapes])
    avg_hr_w = np.mean([s[1] for s in hr_shapes])
    
    print(f"\n📐 Statistiche dimensioni:")
    print(f"   LR medio: {avg_lr_h:.0f}x{avg_lr_w:.0f} px")
    print(f"   HR medio: {avg_hr_h:.0f}x{avg_hr_w:.0f} px")
    print(f"   Rapporto: {avg_hr_h/avg_lr_h:.2f}x")

    # ========================================
    # SPLIT DATASET
    # ========================================
    print(f"\n🔀 Splitting dataset (80% train, 10% val, 10% test)...")
    random.shuffle(valid_pairs)
    n = len(valid_pairs)
    n_tr = int(n * 0.8)
    n_val = int(n * 0.1)
    
    train_pairs = valid_pairs[:n_tr]
    val_pairs = valid_pairs[n_tr:n_tr+n_val]
    test_pairs = valid_pairs[n_tr+n_val:]
    
    print(f"   Train: {len(train_pairs)} ({len(train_pairs)/n*100:.1f}%)")
    print(f"   Val:   {len(val_pairs)} ({len(val_pairs)/n*100:.1f}%)")
    print(f"   Test:  {len(test_pairs)} ({len(test_pairs)/n*100:.1f}%)")
    
    # ========================================
    # CREA DIRECTORY SPLITS E COPIA FILE
    # ========================================
    split_dirs = {
        'train': OUTPUT_SPLITS_BASE / 'train',
        'val': OUTPUT_SPLITS_BASE / 'val',
        'test': OUTPUT_SPLITS_BASE / 'test'
    }
    
    for split_dir in split_dirs.values():
        split_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Copia file nelle directory splits...")
    
    # Copia train
    print(f"\n📦 Train set ({len(train_pairs)} coppie)...")
    for idx, pair in enumerate(tqdm(train_pairs, desc="Train")):
        copy_pair_to_split(pair, split_dirs['train'], idx)
    
    # Copia val
    print(f"\n📦 Validation set ({len(val_pairs)} coppie)...")
    for idx, pair in enumerate(tqdm(val_pairs, desc="Val")):
        copy_pair_to_split(pair, split_dirs['val'], idx)
    
    # Copia test
    print(f"\n📦 Test set ({len(test_pairs)} coppie)...")
    for idx, pair in enumerate(tqdm(test_pairs, desc="Test")):
        copy_pair_to_split(pair, split_dirs['test'], idx)
    
    # ========================================
    # SALVA METADATA
    # ========================================
    metadata = {
        "target": target_dir.name,
        "target_lr_size": TARGET_LR_SIZE,
        "target_hr_size": TARGET_HR_SIZE,
        "scale_ratio": SCALE_RATIO,
        "hardware_optimization": "RTX 2060 (6GB VRAM)",
        "total_pairs": n,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "invalid_pairs": len(invalid_pairs),
        "avg_lr_shape": [int(avg_lr_h), int(avg_lr_w)],
        "avg_hr_shape": [int(avg_hr_h), int(avg_hr_w)],
        "splits_location": str(OUTPUT_SPLITS_BASE)
    }
    
    metadata_file = OUTPUT_SPLITS_BASE / "dataset_info.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Splits salvati in: {OUTPUT_SPLITS_BASE}")
    print(f"   📁 {split_dirs['train']}")
    print(f"   📁 {split_dirs['val']}")
    print(f"   📁 {split_dirs['test']}")
    print(f"   📄 {metadata_file}")
    
    return True, OUTPUT_SPLITS_BASE

def ask_continue_to_next_step():
    """Chiede se proseguire con il prossimo script."""
    next_script_name = 'Modello_3_verifica.py'
    HERE = Path(__file__).resolve().parent
    next_script_path = HERE / next_script_name

    print("\n" + "="*70)
    print("🎯 STEP 2 COMPLETATO!")
    print("="*70)
    print(f"\n📋 PROSSIMO STEP: Verifica Dataset ({next_script_name})")
    print("   Questo script verifica che il dataset sia caricabile correttamente.")
    
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
            print(f"\n👋 Ok. Esegui manualmente: python scripts\\{next_script_name}")
            return
        else:
            print("❌ Scelta non valida.")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Selezione target
    target_dir = select_target_directory()
    
    if not target_dir:
        print("\n👋 Operazione annullata.")
        sys.exit(0)
    
    # Prepara dataset
    success = prepare_dataset(target_dir)
    
    if success:
        ask_continue_to_next_step()
    else:
        print("\n❌ Preparazione dataset fallita.")
        print("   Controlla gli errori sopra e correggi prima di procedere.")