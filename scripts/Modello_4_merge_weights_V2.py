"""
MERGE WORKER MODELS (Weight Averaging)
Unisce i pesi dei modelli addestrati dai 7 worker paralleli (outputs/TARGET_GPU_*)
e salva il risultato direttamente nella cartella principale dei checkpoint (outputs/TARGET/checkpoints/best.pth).

Input: outputs/TARGET_GPU_0 ... TARGET_GPU_6
Output: outputs/TARGET/checkpoints/best.pth (SOVRASCRITTO)
"""

import os
import sys
import torch
import copy
from pathlib import Path
import argparse

# Configurazione Path
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # L'architettura deve essere importata per garantire la compatibilità dello state_dict
    from src.architecture import HybridSuperResolutionModel 
except ImportError:
    sys.exit("❌ Errore import src. Assicurati di avere la classe HybridSuperResolutionModel.")

def merge_models(target_base_name):
    print("="*60)
    print(f"🔄 MERGING PESI DEI WORKER PER TARGET: {target_base_name}")
    print("="*60)
    
    outputs_dir = PROJECT_ROOT / "outputs"
    # Cerca tutte le cartelle generate dai worker indipendenti
    worker_dirs = sorted(list(outputs_dir.glob(f"{target_base_name}_GPU_*")))
    
    if not worker_dirs:
        print(f"❌ Nessuna cartella worker trovata per {target_base_name}_GPU_*.")
        print("   Verificare che il LAUNCHER abbia completato l'esecuzione.")
        return

    # Percorso finale del checkpoint da sovrascrivere
    final_checkpoint_dir = outputs_dir / target_base_name / "checkpoints"
    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_best_path = final_checkpoint_dir / "best.pth"
    
    print(f"   Trovati {len(worker_dirs)} worker.")
    print(f"   Destinazione Finale: {final_best_path}")

    # Lista per contenere gli state_dict
    models_state_dicts = []
    valid_workers = 0

    for w_dir in worker_dirs:
        # Cerca il modello migliore salvato dal worker (best_model.pth o last.pth)
        ckpt_path = w_dir / "checkpoints" / "best_model.pth"
        if not ckpt_path.exists():
            ckpt_path = w_dir / "checkpoints" / "last.pth"
        
        if ckpt_path.exists():
            print(f"   Load: {w_dir.name} -> {ckpt_path.name}")
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                # Estrai solo lo state_dict se è un checkpoint completo
                state_dict = ckpt.get('model_state_dict', ckpt) 
                models_state_dicts.append(state_dict)
                valid_workers += 1
            except Exception as e:
                print(f"   ⚠️ Errore caricamento {ckpt_path}: {e}")
        else:
            print(f"   ⚠️ Nessun checkpoint valido in {w_dir.name}")

    if valid_workers == 0:
        print("❌ Nessun modello valido caricato. Merging annullato.")
        return

    print(f"\n   ⚗️  Calcolo media dei pesi di {valid_workers} modelli...")
    
    # Prendi il primo come base (deep copy per non modificare l'originale)
    avg_state_dict = copy.deepcopy(models_state_dicts[0])
    
    # Esegui la media (Weight Averaging)
    for key in avg_state_dict:
        # Salta i buffer interi non necessari per la media (es. num_batches_tracked)
        if 'num_batches_tracked' in key:
            continue 

        for i in range(1, valid_workers):
            avg_state_dict[key] += models_state_dicts[i][key]
        
        # Dividi per il numero totale per fare la media
        if torch.is_floating_point(avg_state_dict[key]):
            avg_state_dict[key] = avg_state_dict[key] / valid_workers
        # else: Mantieni il valore intero (per i buffer non floating point)

    # Salva il modello fuso (solo i pesi)
    print(f"\n   💾 Sovrascrittura {final_best_path.name}...")
    torch.save(avg_state_dict, final_best_path)
    
    print("\n✅ MERGING COMPLETATO CON SUCCESSO!")
    print(f"   Il modello fuso è ora in: {final_best_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Qui il target deve essere M33 (il nome del target, non il nome della cartella merged)
    parser.add_argument('--target', type=str, required=True, help="Nome base del target (es. M33)")
    args = parser.parse_args()
    
    merge_models(args.target)