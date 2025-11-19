import sys
import json
import subprocess
from pathlib import Path

# ============================================================
# FIX IMPORT: Siamo in Modello, src è nella cartella PADRE
# ============================================================
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.append(str(PROJECT_ROOT))
# ============================================================

try:
    from src.dataset import AstronomicalDataset
    print("✅ Modulo 'src' importato correttamente.")
except ImportError as e:
    print(f"❌ ERRORE IMPORT: {e}")
    print(f"   Assicurati che la cartella 'src' sia dentro {PROJECT_ROOT}")
    sys.exit(1)

def check():
    print("\n" + "="*60)
    print("🔍 VERIFICA DATASET (Da cartella Modello)")
    print("="*60)
    
    splits_dir = PROJECT_ROOT / "data" / "splits"
    train_json = splits_dir / "train.json"

    print(f"📂 Cartella Splits: {splits_dir}")
    if not train_json.exists():
        print(f"❌ File mancante: {train_json}")
        print("   👉 Esegui prima '2_prepare_data.py'")
        return False # Ritorna False se fallisce

    try:
        print("\n📦 Test caricamento Dataset...")
        ds = AstronomicalDataset(train_json, base_path=PROJECT_ROOT, augment=False)
        print(f"   ✅ Dataset inizializzato: {len(ds)} campioni")
        
        if len(ds) > 0:
            sample = ds[0]
            print(f"   🔄 Test Shape: LR {sample['lr'].shape} -> HR {sample['hr'].shape}")
            print("   ✅ Immagini caricate correttamente.")
            return True # Successo
        else:
            print("   ⚠️  Dataset vuoto!")
            return False

    except Exception as e:
        print(f"   ❌ Errore caricamento dati: {e}")
        return False

def ask_continue_to_next_step():
    """Chiede se proseguire con il prossimo script."""
    next_script_name = 'Modello_4_train.py'
    HERE = Path(__file__).resolve().parent
    next_script_path = HERE / next_script_name

    print("\n" + "="*70)
    print("🎯 STEP 3 COMPLETATO!")
    print("="*70)
    print(f"\n📋 PROSSIMO STEP: Training ({next_script_name})")
    print("   Questo avvierà l'addestramento del modello.")
    
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
    success = check()
    if success:
        ask_continue_to_next_step()
    else:
        print("\n❌ Verifica fallita. Correggi gli errori prima di proseguire col training.")