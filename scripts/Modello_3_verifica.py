"""
MODELLO - STEP 3: VERIFICA DATASET
Verifica dataset nelle splits del target selezionato

POSIZIONE: scripts/Modello_3_verifica.py
"""

import sys
import json
import subprocess
from pathlib import Path

# ============================================================================
# FIX IMPORT: scripts -> src nella cartella PADRE
# ============================================================================
HERE = Path(__file__).resolve().parent  # scripts/
PROJECT_ROOT = HERE.parent               # SuperResolution/
sys.path.append(str(PROJECT_ROOT))

print(f"📂 Project Root: {PROJECT_ROOT}")

ROOT_DATA_DIR = PROJECT_ROOT / "data"

try:
    from src.dataset import AstronomicalDataset
    print("✅ Modulo 'src.dataset' importato correttamente.")
except ImportError as e:
    print(f"❌ ERRORE IMPORT: {e}")
    print(f"   Assicurati che la cartella 'src' sia dentro {PROJECT_ROOT}")
    sys.exit(1)

# ============================================================================
# FUNZIONI MENU
# ============================================================================
def select_target_directory():
    """Mostra menu per selezionare dataset target da verificare."""
    print("\n" + "📂"*35)
    print("SELEZIONE DATASET DA VERIFICARE".center(70))
    print("📂"*35)
    print(f"\nScansione sottocartelle in: {ROOT_DATA_DIR}")

    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() 
                   if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile leggere {ROOT_DATA_DIR}: {e}")
        return None

    if not subdirs:
        print(f"\n❌ ERRORE: Nessuna sottocartella trovata")
        return None

    # Filtra solo quelli con splits create
    valid_targets = []
    for d in subdirs:
        splits_dir = d / "6_patches_from_cropped" / "splits"
        if splits_dir.exists():
            valid_targets.append(d)
    
    if not valid_targets:
        print(f"\n❌ Nessun dataset con splits trovato!")
        print(f"\n💡 Esegui prima: Modello_2_prepare_data.py")
        return None

    print("\nDataset disponibili per verifica:")
    for i, dir_path in enumerate(valid_targets):
        splits_dir = dir_path / "6_patches_from_cropped" / "splits"
        metadata_file = splits_dir / "dataset_info.json"
        
        if metadata_file.exists():
            with open(metadata_file) as f:
                info = json.load(f)
                train_count = info.get('train_pairs', '?')
                print(f"   {i+1}: {dir_path.name} ({train_count} training pairs)")
        else:
            print(f"   {i+1}: {dir_path.name}")

    while True:
        print("\n" + "─"*70)
        try:
            choice_str = input(f"👉 Seleziona un numero (1-{len(valid_targets)}) o 'q' per uscire: ").strip()

            if choice_str.lower() == 'q':
                return None

            choice = int(choice_str)
            choice_idx = choice - 1
            
            if 0 <= choice_idx < len(valid_targets):
                selected_dir = valid_targets[choice_idx]
                print(f"\n✅ Dataset selezionato: {selected_dir.name}")
                return selected_dir
            else:
                print(f"❌ Scelta non valida.")
        except ValueError:
            print("❌ Input non valido.")

# ============================================================================
# VERIFICA DATASET
# ============================================================================
def check_dataset(target_dir):
    """Verifica il dataset di un target specifico."""
    
    print("\n" + "="*70)
    print(f"🔍 VERIFICA DATASET: {target_dir.name}".center(70))
    print("="*70)
    
    splits_dir = target_dir / "6_patches_from_cropped" / "splits"
    
    # Verifica struttura
    train_dir = splits_dir / "train"
    val_dir = splits_dir / "val"
    test_dir = splits_dir / "test"
    metadata_file = splits_dir / "dataset_info.json"

    print(f"\n📂 Cartella Splits: {splits_dir}")
    
    # Verifica esistenza directory
    if not train_dir.exists():
        print(f"❌ Cartella train mancante: {train_dir}")
        print("   👉 Esegui prima 'Modello_2_prepare_data.py'")
        return False
    
    if not val_dir.exists():
        print(f"❌ Cartella val mancante: {val_dir}")
        return False
    
    # Conta coppie
    train_pairs = list(train_dir.glob("pair_*"))
    val_pairs = list(val_dir.glob("pair_*"))
    test_pairs = list(test_dir.glob("pair_*")) if test_dir.exists() else []
    
    print(f"\n📊 Struttura dataset:")
    print(f"   Train: {len(train_pairs)} coppie")
    print(f"   Val:   {len(val_pairs)} coppie")
    print(f"   Test:  {len(test_pairs)} coppie")
    
    if len(train_pairs) == 0:
        print(f"\n❌ Nessuna coppia trovata in train!")
        return False
    
    # Carica metadata
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
            print(f"\n📋 Info dataset:")
            print(f"   Target: {metadata.get('target', 'N/A')}")
            print(f"   LR size: {metadata.get('target_lr_size', 'N/A')}x{metadata.get('target_lr_size', 'N/A')}")
            print(f"   HR size: {metadata.get('target_hr_size', 'N/A')}x{metadata.get('target_hr_size', 'N/A')}")
            print(f"   Scale: {metadata.get('scale_ratio', 'N/A')}x")
    
    # Test caricamento con AstronomicalDataset
    print(f"\n📦 Test caricamento con AstronomicalDataset...")
    
    try:
        # Crea un finto JSON con path delle coppie
        train_list = []
        for pair_dir in list(train_dir.glob("pair_*"))[:10]:  # Test primi 10
            fits_files = list(pair_dir.glob("*.fits"))
            if len(fits_files) == 2:
                # Identifica LR e HR per dimensione
                from astropy.io import fits as astro_fits
                dims = []
                for f in fits_files:
                    with astro_fits.open(f) as hdul:
                        dims.append((f, hdul[0].data.shape[-2:]))
                
                if dims[0][1][0] * dims[0][1][1] < dims[1][1][0] * dims[1][1][1]:
                    lr_file, hr_file = dims[0][0], dims[1][0]
                else:
                    lr_file, hr_file = dims[1][0], dims[0][0]
                
                train_list.append({
                    "ground_path": str(lr_file),
                    "hubble_path": str(hr_file)
                })
        
        if not train_list:
            print("❌ Nessuna coppia valida trovata per il test!")
            return False
        
        # Test caricamento
        print(f"   Test su {len(train_list)} coppie...")
        
        # Crea JSON temporaneo
        temp_json = splits_dir / "temp_test.json"
        with open(temp_json, 'w') as f:
            json.dump(train_list, f)
        
        try:
            ds = AstronomicalDataset(temp_json, base_path=PROJECT_ROOT, augment=False)
            print(f"   ✅ Dataset inizializzato: {len(ds)} campioni")
            
            if len(ds) > 0:
                sample = ds[0]
                print(f"   📄 Test Shape: LR {sample['lr'].shape} -> HR {sample['hr'].shape}")
                print("   ✅ Immagini caricate correttamente!")
                
                # Cleanup
                temp_json.unlink()
                
                return True
            else:
                print("   ⚠️  Dataset vuoto!")
                temp_json.unlink()
                return False
                
        except Exception as e:
            print(f"   ❌ Errore caricamento: {e}")
            if temp_json.exists():
                temp_json.unlink()
            return False

    except Exception as e:
        print(f"   ❌ Errore durante il test: {e}")
        return False

def ask_continue_to_next_step():
    """Chiede se proseguire con il training."""
    next_script_name = 'Modello_4_train.py'
    HERE = Path(__file__).resolve().parent
    next_script_path = HERE / next_script_name

    print("\n" + "="*70)
    print("🎯 STEP 3 (Verifica) COMPLETATO!")
    print("="*70)
    print(f"\n📋 PROSSIMO STEP: Training ({next_script_name})")
    print("   Questo avvierà l'addestramento del modello.")
    print("\n⚠️  NOTA: Il training richiede diverse ore!")
    
    while True:
        print("\n" + "─"*70)
        choice = input(f"👉 Vuoi avviare '{next_script_name}' ora? [S/n]: ").strip().lower()
        if choice in ('', 's', 'si', 'y', 'yes'):
            if next_script_path.exists():
                print(f"\n🚀 Avvio {next_script_name}...")
                try:
                    subprocess.run([sys.executable, str(next_script_path)])
                except Exception as e:
                    print(f"❌ Errore durante l'avvio: {e}")
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
    
    # Verifica dataset
    success = check_dataset(target_dir)
    
    if success:
        ask_continue_to_next_step()
    else:
        print("\n❌ Verifica fallita. Correggi gli errori prima di proseguire col training.")