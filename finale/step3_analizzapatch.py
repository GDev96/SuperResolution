"""
PIPELINE SUPER-REVOLT GAIA - STEP 6 & REPORTING
"""

import os
import sys
import json
import time
import logging
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE PATH ASSOLUTI
# ============================================================================
# Definizione della radice del progetto
PROJECT_ROOT = Path(r"F:\Super Revolt Gaia\SuperResolution")

# Percorsi assoluti derivati
LOGS_DIR = PROJECT_ROOT / "logs"
# ============================================================================

# --- PARAMETRI DATASET ---
TARGET_FOV_ARCMIN = 1       # Dimensione angolare della patch (1 arcominuto)
OVERLAP_PERCENT = 25        # Sovrapposizione tra patch adiacenti
MIN_VALID_PIXELS = 50       # % minima di pixel validi (non-NaN) per accettare la patch
MATCH_THRESHOLD = 0.5 / 60.0 # Tolleranza accoppiamento (0.5 arcsec in gradi)
NUM_THREADS = 7

# --- LOGGING & UTILS ---

def setup_logger(base_dir):
    # Usa il percorso assoluto definito in alto
    log_dir = LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / f'patches_{base_dir.name}_{int(time.time())}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger()

# -----------------------------------------------------------------------------
# ESTRAZIONE PATCHES (Invariata)
# -----------------------------------------------------------------------------

def extract_patches_worker(fits_path, output_dir, prefix, logger):
    """
    Estrae patch da una singola immagine FITS.
    Calcola la dimensione in pixel della patch basandosi sulla scala WCS 
    per garantire che copra 'TARGET_FOV_ARCMIN'.
    """
    extracted = []
    try:
        with fits.open(fits_path) as hdul:
            # Ricerca HDU valido
            hdu = next((h for h in hdul if h.data is not None and h.data.ndim >= 2), None)
            if not hdu: return []
            
            data = hdu.data[0] if hdu.data.ndim == 3 else hdu.data
            header = hdu.header
            wcs = WCS(header)
            
            if not wcs.has_celestial: return []

            # 1. Calcolo Scala Pixel (gradi/px)
            if hasattr(wcs.wcs, 'cd'):
                scale_deg = np.sqrt(wcs.wcs.cd[0,0]**2 + wcs.wcs.cd[0,1]**2)
            else:
                scale_deg = abs(wcs.wcs.cdelt[0])
            
            # 2. Calcolo dimensione patch in pixel
            target_deg = TARGET_FOV_ARCMIN / 60.0
            patch_size = int(target_deg / scale_deg)
            # Arrotondamento a multiplo di 8 (utile per CNN/Deep Learning)
            patch_size = ((patch_size + 7) // 8) * 8
            
            # 3. Sliding Window
            step = int(patch_size * (1 - OVERLAP_PERCENT / 100.0))
            ny, nx = data.shape
            
            for y in range(0, ny - patch_size + 1, step):
                for x in range(0, nx - patch_size + 1, step):
                    # Estrazione numpy slice
                    patch_data = data[y : y+patch_size, x : x+patch_size]
                    
                    # Check validità (evita patch vuote/bordi neri)
                    if (np.isfinite(patch_data).sum() / patch_data.size * 100) < MIN_VALID_PIXELS:
                        continue

                    # Calcolo centro astrometrico
                    center_sky = wcs.pixel_to_world(x + patch_size//2, y + patch_size//2)
                    
                    # Salvataggio patch
                    fname = f"{prefix}_{fits_path.stem}_p{len(extracted):04d}.fits"
                    out_path = output_dir / fname
                    
                    # Header minimale per la patch
                    p_head = header.copy()
                    p_head['NAXIS1'] = patch_size
                    p_head['NAXIS2'] = patch_size
                    fits.PrimaryHDU(patch_data, header=p_head).writeto(out_path, overwrite=True)
                    
                    extracted.append({
                        'filename': fname,
                        'ra': center_sky.ra.deg,
                        'dec': center_sky.dec.deg,
                        'path': str(out_path)
                    })
                    
        return extracted
    except Exception as e:
        logger.error(f"Errore estrazione {fits_path.name}: {e}")
        return []

# -----------------------------------------------------------------------------
# ACCOPPIAMENTO (PAIRING) (Invariata)
# -----------------------------------------------------------------------------

def pair_patches(h_patches, o_patches, output_dir):
    """
    Trova per ogni patch di Hubble (HR) la corrispondente patch dell'Osservatorio (LR).
    Usa la distanza angolare tra i centri (SkyCoord separation).
    """
    pairs = []
    pair_dir = output_dir / 'paired_patches_folders'
    pair_dir.mkdir(parents=True, exist_ok=True)
    
    # Creazione cataloghi SkyCoord per matching vettoriale veloce
    if not h_patches or not o_patches: return []

    print("Indicizzazione coordinate per il matching...")
    # Ottimizzazione: Per dataset grandi usare KDTree, qui usiamo loop semplice o search astropy
    
    # Logica semplificata per chiarezza (O(N*M) - ottimizzabile ma ok per <10k patch)
    for h_p in tqdm(h_patches, desc="Matching HR->LR"):
        h_coord = SkyCoord(h_p['ra'], h_p['dec'], unit='deg')
        best_match = None
        min_dist = float('inf')
        
        # Cerca patch osservatorio più vicina
        for o_p in o_patches:
            o_coord = SkyCoord(o_p['ra'], o_p['dec'], unit='deg')
            dist = h_coord.separation(o_coord).deg
            
            if dist < min_dist:
                min_dist = dist
                best_match = o_p
        
        # Se entro la soglia, salva la coppia
        if min_dist < MATCH_THRESHOLD and best_match:
            pair_id = f"pair_{len(pairs):05d}"
            dest = pair_dir / pair_id
            dest.mkdir()
            
            # Copia fisica file
            shutil.copy(h_p['path'], dest / h_p['filename'])
            shutil.copy(best_match['path'], dest / best_match['filename'])
            
            pairs.append({
                'id': pair_id,
                'hr': h_p['filename'],
                'lr': best_match['filename'],
                'dist_deg': min_dist
            })
            
    return pairs

# -----------------------------------------------------------------------------
# MAIN PROCESS
# -----------------------------------------------------------------------------

def run_patch_pipeline(base_dir):
    logger = setup_logger(base_dir)
    print(f"\n--- Generazione Patches per {base_dir.name} ---")

    # 1. Setup Paths
    # Si può scegliere se usare le immagini 'cropped' (uguali dimensioni) o 'registered' (native)
    # Default: Cropped per consistenza geometrica
    src_h = base_dir / '4_cropped' / 'hubble'
    src_o = base_dir / '4_cropped' / 'observatory'
    out_root = base_dir / '6_patches'
    
    out_h = out_root / 'hubble_native'
    out_o = out_root / 'observatory_native'
    out_h.mkdir(parents=True, exist_ok=True)
    out_o.mkdir(parents=True, exist_ok=True)

    # 2. Estrazione Parallela
    all_h_patches = []
    all_o_patches = []

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as exc:
        # Submit Hubble jobs
        futures_h = [exc.submit(extract_patches_worker, f, out_h, 'hr', logger) for f in src_h.glob('*.fits')]
        # Submit Obs jobs
        futures_o = [exc.submit(extract_patches_worker, f, out_o, 'lr', logger) for f in src_o.glob('*.fits')]
        
        for f in tqdm(as_completed(futures_h), total=len(futures_h), desc="Extract Hubble"):
            all_h_patches.extend(f.result())
            
        for f in tqdm(as_completed(futures_o), total=len(futures_o), desc="Extract Obs"):
            all_o_patches.extend(f.result())

    print(f"Estratte: {len(all_h_patches)} HR, {len(all_o_patches)} LR")

    # 3. Matching
    pairs = pair_patches(all_h_patches, all_o_patches, out_root)
    print(f"Coppie validate create: {len(pairs)}")
    
    # 4. Report JSON (per dataset loader)
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {'fov': TARGET_FOV_ARCMIN, 'overlap': OVERLAP_PERCENT},
        'stats': {'hr_count': len(all_h_patches), 'lr_count': len(all_o_patches), 'pairs': len(pairs)},
        'pairs': pairs
    }
    with open(out_root / 'dataset.json', 'w') as f:
        json.dump(report, f, indent=2)

def main():
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if target.exists():
            run_patch_pipeline(target)
    else:
        print("Uso: python step3_analizzapatch.py <path_to_target_folder>")

if __name__ == "__main__":
    main()