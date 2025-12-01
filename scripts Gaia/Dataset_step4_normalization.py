import os
import sys
import shutil
import numpy as np
from astropy.io import fits
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURAZIONE =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# Impostazioni Normalizzazione
# Tagliamo l'1% più scuro (fondo cielo sporco) e lo 0.1% più luminoso (stelle bruciate)
LOWER_PERCENTILE = 1.0 
UPPER_PERCENTILE = 99.9 

def select_target_directory():
    print("\n" + "⚖️"*35)
    print("NORMALIZZAZIONE DATASET (FITS -> TIFF 16-bit)".center(70))
    print("⚖️"*35)
    
    subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    if not subdirs: return None
    
    print("\nTarget disponibili (con cartella '6_patches_final'):")
    valid_targets = []
    for i, d in enumerate(subdirs):
        if (d / '6_patches_final').exists():
            print(f" {len(valid_targets)+1}: {d.name}")
            valid_targets.append(d)
            
    if not valid_targets:
        print("❌ Nessun target ha le patch estratte. Esegui lo Step 3 prima.")
        return None

    try:
        idx = int(input("Scelta: ")) - 1
        return valid_targets[idx] if 0 <= idx < len(valid_targets) else None
    except: return None

def normalize_to_uint16(data):
    """
    Normalizza un array float (qualsiasi range) in uint16 (0-65535).
    Usa clipping basato sui percentili per massimizzare il contrasto utile.
    """
    # 1. Gestione NaN e Infinti
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Calcolo range dinamico utile
    vmin = np.percentile(data, LOWER_PERCENTILE)
    vmax = np.percentile(data, UPPER_PERCENTILE)
    
    # Protezione da immagini piatte/nere
    if vmax <= vmin:
        return np.zeros(data.shape, dtype=np.uint16)
    
    # 3. Normalizzazione 0.0 - 1.0 con clipping
    # I valori sotto vmin diventano 0, sopra vmax diventano 1
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    
    # 4. Espansione a 16-bit
    return (norm * 65535).astype(np.uint16)

def main():
    target_dir = select_target_directory()
    if not target_dir: return

    input_dir = target_dir / '6_patches_final'
    output_dir = target_dir / '7_dataset_ready'

    # Pulizia e creazione cartelle output
    if output_dir.exists(): shutil.rmtree(output_dir)
    
    path_hr = output_dir / "HR" # High Res (Hubble)
    path_lr = output_dir / "LR" # Low Res (Osservatorio)
    
    path_hr.mkdir(parents=True)
    path_lr.mkdir(parents=True)

    pairs = sorted(list(input_dir.glob("pair_*")))
    print(f"\n🚀 Avvio normalizzazione su {len(pairs)} coppie...")
    print(f"   Input:  {input_dir}")
    print(f"   Output: {output_dir}")

    count = 0
    for pair_folder in tqdm(pairs, cols=100):
        try:
            p_id = pair_folder.name # es. pair_000001
            
            f_hubble = pair_folder / "hubble.fits"
            f_obs = pair_folder / "observatory.fits"
            
            if not f_hubble.exists() or not f_obs.exists(): continue

            # 1. Caricamento FITS
            with fits.open(f_hubble) as h: d_hr = h[0].data
            with fits.open(f_obs) as o:    d_lr = o[0].data

            # 2. Normalizzazione Indipendente
            # Normalizziamo HR e LR separatamente per portare entrambi nel range 0-65535
            # L'AI imparerà a mappare le strutture, non la luminosità assoluta.
            img_hr_u16 = normalize_to_uint16(d_hr)
            img_lr_u16 = normalize_to_uint16(d_lr)

            # 3. Salvataggio TIFF
            # mode='I;16' è specifico di PIL per i TIFF a 16-bit monocromatici
            Image.fromarray(img_hr_u16, mode='I;16').save(path_hr / f"{p_id}.tiff")
            Image.fromarray(img_lr_u16, mode='I;16').save(path_lr / f"{p_id}.tiff")
            
            count += 1
            
        except Exception as e:
            print(f"⚠️ Errore su {pair_folder.name}: {e}")

    print(f"\n✅ DATASET COMPLETATO!")
    print(f"   Patch salvate: {count}")
    print(f"   Posizione HR:  {path_hr}")
    print(f"   Posizione LR:  {path_lr}")
    print("\n   Ora puoi usare queste cartelle per il training del modello.")

if __name__ == "__main__":
    main()