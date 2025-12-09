import json
import random
import sys
from pathlib import Path
from tqdm import tqdm

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

def prepare_dataset(target_dir_path):
    target_dir = Path(target_dir_path)
    print(f"\n🧪 PREPARAZIONE OVERFITTING TEST: {target_dir.name}")
    
    SOURCE = target_dir / "7_dataset_ready_LOG"
    SPLITS = target_dir / "8_dataset_split" / "splits_json"
    
    if not SOURCE.exists():
        SOURCE = target_dir / "7_dataset_ready"

    pair_folders = sorted(list(SOURCE.glob("pair_*")))
    
    if not pair_folders:
        print("❌ Nessuna patch trovata.")
        return

    # --- MODIFICA RADICALE PER OVERFITTING ---
    # Prendiamo SOLO LA PRIMA COPPIA
    selected_pair = pair_folders[0] 
    
    h_file = selected_pair / "hubble.tiff"
    o_file = selected_pair / "observatory.tiff"
    
    if not (h_file.exists() and o_file.exists()):
        print("❌ La prima coppia è incompleta/corrotta.")
        return

    single_entry = [{
        "patch_id": selected_pair.name, 
        "hubble_path": str(h_file.resolve()), 
        "ground_path": str(o_file.resolve())
    }]

    print(f"   ⚠️ OVERFITTING MODE ATTIVA")
    print(f"   🎯 Uso solo: {selected_pair.name}")

    SPLITS.mkdir(parents=True, exist_ok=True)
    
    # Salviamo LO STESSO file in tutti i json
    with open(SPLITS / 'train.json', 'w') as f: json.dump(single_entry, f, indent=4)
    with open(SPLITS / 'val.json', 'w') as f: json.dump(single_entry, f, indent=4)
    with open(SPLITS / 'test.json', 'w') as f: json.dump(single_entry, f, indent=4)

    print(f"✅ JSON Generati. Il modello vedrà solo questa immagine all'infinito.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prepare_dataset(Path(sys.argv[1]))
    else:
        # Auto-detect veloce
        subs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir()]
        for d in subs: 
            if (d/'7_dataset_ready_LOG').exists():
                prepare_dataset(d)
                break