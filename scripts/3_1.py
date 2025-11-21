"""
STEP 3: ESTRAZIONE PATCH (MASSIVA - NO LIMIT)
Logica: Estrae Hubble e Osservatorio a 512x512 dai Mosaici.
SBLOCCATO: Estrae tutte le patch possibili per il training.
"""

import sys
import os
import shutil
import logging
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import reproject
try:
    from reproject import reproject_interp
except ImportError:
    print("❌ ERRORE: Manca 'reproject'. Installa con: pip install reproject")
    sys.exit(1)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE PRODUZIONE
# ============================================================================
SIZE_CROP = 512         
STRIDE = 64             # Sovrapposizione per aumentare i dati
VALID_THRESHOLD = 0.95  # Scarta se > 5% di pixel sono neri/NaN
MIN_SIGMA_THRESHOLD = 0.005 

# 🛑 LIMITE RIMOSSO PER TRAINING REALE 🛑
LIMIT_SAVE = float('inf')  
NUM_WORKERS = min(32, (os.cpu_count() or 1) * 2)

# ============================================================================
# PERCORSI (RunPod Safe)
# ============================================================================
PROJECT_ROOT = Path("/root/SuperResolution")
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROOT_DATA_DIR = PROJECT_ROOT / "data"

# ============================================================================
# UTILS
# ============================================================================
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def is_valid_patch(data):
    if data is None or data.size == 0: return False
    
    # 1. Controllo NaN/Zeri
    invalid = np.isnan(data) | (data == 0)
    valid_ratio = 1.0 - (np.sum(invalid) / data.size)
    if valid_ratio < VALID_THRESHOLD: return False

    # 2. Controllo Struttura (Sigma) - Evita patch completamente piatte (cielo vuoto)
    try:
        valid_data = data[~np.isnan(data) & (data != 0)]
        if valid_data.size < 100: return False
        min_v, max_v = valid_data.min(), valid_data.max()
        if max_v - min_v > 1e-8:
             valid_data_norm = (valid_data - min_v) / (max_v - min_v)
             if valid_data_norm.std() < MIN_SIGMA_THRESHOLD: return False 
        else: return False
    except: return False
        
    return True

# ============================================================================
# WORKER THREAD
# ============================================================================
def process_single_patch(args):
    y, x, data_h, header_h, data_o, wcs_o = args
    
    # 1. Estrazione Hubble
    patch_h = data_h[y:y+SIZE_CROP, x:x+SIZE_CROP].copy()
    if not is_valid_patch(patch_h): return False, None, None, None, None

    # Header Crop
    header_crop = header_h.copy()
    header_crop['NAXIS1'] = SIZE_CROP
    header_crop['NAXIS2'] = SIZE_CROP
    header_crop['CRPIX1'] -= x
    header_crop['CRPIX2'] -= y
    for k in ['LTV1', 'LTV2', 'LTM1_1']: 
        if k in header_crop: del header_crop[k]

    # 2. Riproiezione Osservatorio (Allineamento WCS preciso)
    try:
        patch_o, footprint = reproject_interp(
            (data_o, wcs_o), 
            header_crop, 
            shape_out=(SIZE_CROP, SIZE_CROP),
            order='bilinear'
        )
    except: return False, None, None, None, None

    if not is_valid_patch(patch_o): return False, None, None, None, None
    
    return True, patch_h, patch_o, header_crop, header_crop

# ============================================================================
# CORE PROCESS
# ============================================================================
def process_target(base_dir, logger):
    target_name = base_dir.name
    mosaic_dir = base_dir / '5_mosaics'
    output_dir = base_dir / '6_patches_aligned'
    
    f_hub = mosaic_dir / 'final_mosaic_hubble.fits'
    f_obs = mosaic_dir / 'final_mosaic_observatory.fits'
    
    # Fallback: Se i file specifici non esistono, cerca il mosaico unico generato da step precedenti
    if not f_hub.exists(): f_hub = mosaic_dir / 'final_mosaic.fits' 

    print("\n" + "✂️ "*35)
    print(f"TARGET: {target_name}".center(70))
    print("✂️ "*35)

    if not f_hub.exists():
        logger.error(f"❌ Mosaico Hubble non trovato in {mosaic_dir}")
        print(f"   (Hai eseguito lo Step 2 'python scripts/2_42.py' per creare il mosaico?)")
        return
    
    # Se f_obs non c'è, usiamo f_hub (per test) o cerchiamo meglio, 
    # ma assumiamo che se siamo qui, i mosaici esistono.
    if not f_obs.exists():
        # Fallback critico: se c'è solo un mosaico (es. 2_42.py ne crea uno solo 'final_mosaic.fits')
        # allora quel mosaico contiene GIA' i dati fusi. 
        # Ma lo script 3_1 si aspetta DUE file sorgente (Hubble e Obs) per creare le coppie.
        # Se hai usato 2_42.py, esso crea un mosaico UNICO. 
        # PER IL TRAINING SERVE LA COPPIA (Input, Target).
        # Se hai solo final_mosaic.fits, non puoi fare training supervisionato SR.
        # DEVI AVERE: final_mosaic_hubble.fits E final_mosaic_observatory.fits.
        pass

    print("   ⏳ Caricamento Mosaici...")
    try:
        with fits.open(f_hub) as h: 
            data_h = h[0].data.astype(np.float32)
            header_h = h[0].header
        if data_h.ndim == 3: data_h = data_h[0]

        # Se manca l'osservatorio separato, fermiamo e avvisiamo
        if not f_obs.exists():
             logger.error("❌ Mosaico Osservatorio non trovato!")
             print("   Devi avere entrambi i mosaici (Hubble e Osservatorio) per creare le coppie.")
             return

        with fits.open(f_obs) as h: 
            data_o = h[0].data.astype(np.float32)
            header_o = h[0].header
            wcs_o = WCS(header_o)
        if data_o.ndim == 3: data_o = data_o[0]
    except Exception as e:
        logger.error(f"Errore lettura FITS: {e}")
        return

    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    ny, nx = data_h.shape
    y_range = range(0, ny - SIZE_CROP + 1, STRIDE)
    x_range = range(0, nx - SIZE_CROP + 1, STRIDE)
    
    tasks = []
    for y in y_range:
        for x in x_range:
            tasks.append((y, x, data_h, header_h, data_o, wcs_o))
        
    print(f"   Patch potenziali: {len(tasks)}")
    print(f"   🚀 Avvio Estrazione Multi-Thread...")

    saved_count = 0
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_patch, t) for t in tasks]
        
        with tqdm(total=len(tasks), unit="patch") as pbar:
            for future in as_completed(futures):
                try:
                    valid, p_h, p_o, h_hr, _ = future.result()
                    if valid:
                        pair_folder = output_dir / f"pair_{saved_count:05d}"
                        pair_folder.mkdir(exist_ok=True)
                        
                        fits.PrimaryHDU(data=p_h, header=h_hr).writeto(pair_folder / "hubble.fits", overwrite=True)
                        fits.PrimaryHDU(data=p_o, header=h_hr).writeto(pair_folder / "observatory.fits", overwrite=True)
                        
                        saved_count += 1
                except: pass
                pbar.update(1)

    print(f"\n✅ Completato. Salvate {saved_count} coppie in {output_dir}")

def main():
    logger = setup_logging()
    
    # Controllo cartella dati
    if not ROOT_DATA_DIR.exists():
        print(f"❌ Errore: Cartella dati non trovata: {ROOT_DATA_DIR}")
        return

    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    except: subdirs = []

    if not subdirs: 
        print(f"❌ Nessuna sottocartella trovata in {ROOT_DATA_DIR}")
        return
        
    print("Seleziona Target:")
    print("   0: TUTTI")
    for i, d in enumerate(subdirs): print(f"   {i+1}: {d.name}")
    
    try:
        sel_str = input(">> ").strip()
        if not sel_str: return
        sel = int(sel_str)
        
        if sel == 0:
            for d in subdirs: process_target(d, logger)
        elif 0 < sel <= len(subdirs): 
            process_target(subdirs[sel-1], logger)
    except ValueError: print("Input non valido")
    except Exception as e: print(f"Errore: {e}")

if __name__ == "__main__":
    main()