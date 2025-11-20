"""
STEP 5: ESTRAZIONE PATCH - VERSIONE CUSTOM 128x128 / 512x512
Features:
- Menu interattivo per selezione target
- Multithreading avanzato per estrazione e matching
- Ottimizzato per RTX 2060 (6GB VRAM) + 64GB RAM
- Dimensioni target: 
    * Hubble HR: 512x512
    * Observatory LR: 128x128 (Scale factor: 4.0x)
- Parallelizzazione completa del matching
- Progress tracking dettagliato

INPUT: Cartelle '4_cropped/hubble' e '4_cropped/observatory'
OUTPUT: Cartella '6_patches_from_cropped' con:
  - hubble_patches/ (512x512px)
  - observatory_patches/ (128x128px)
  - paired_patches_folders/ (coppie matched)
"""

import sys
import time
import json
import logging
import shutil
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
from astropy.io import fits
from astropy.wcs import WCS
from datetime import datetime
import multiprocessing as mp
import subprocess

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE PATH DINAMICI
# ============================================================================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR_ROOT = ROOT_DATA_DIR / "logs"
SCRIPTS_DIR = CURRENT_SCRIPT_DIR

print(f"📂 Project Root: {PROJECT_ROOT}")
print(f"📂 Data Dir:     {ROOT_DATA_DIR}")

# ============================================================================
# PARAMETRI PATCH - CONFIGURAZIONE AGGIORNATA
# ============================================================================
# Hubble rimane 512x512 (HR)
TARGET_SIZE_HUBBLE = 512    
# Observatory impostato a 128x128 (LR)
TARGET_SIZE_GROUND = 128     

# Nota sul fattore di scala: 512 / 128 = 4.0x
OVERLAP_PERCENT = 25        # Overlap tra patch adiacenti
MATCH_THRESHOLD = 0.5 / 60.0  # Soglia matching in gradi (0.5 arcmin)

# Configurazione Hardware
NUM_WORKERS = mp.cpu_count()
IO_WORKERS = 24
USE_PROCESS_POOL = True

print(f"⚙️  Configurazione Patch:")
print(f"    🛰️  Hubble (HR):      {TARGET_SIZE_HUBBLE}x{TARGET_SIZE_HUBBLE} px")
print(f"    🔭 Observatory (LR): {TARGET_SIZE_GROUND}x{TARGET_SIZE_GROUND} px")
print(f"    📐 Scale Factor:     {TARGET_SIZE_HUBBLE/TARGET_SIZE_GROUND:.2f}x")
print(f"⚙️  Workers: {NUM_WORKERS} (extraction), {IO_WORKERS} (I/O)")

# ============================================================================
# SETUP LOGGING
# ============================================================================
def setup_logging(base_dir_name="batch"):
    LOG_DIR_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR_ROOT / f'patch_extraction_{base_dir_name}_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ============================================================================
# FUNZIONI MENU
# ============================================================================
def select_target_directory():
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET".center(70))
    print("📂"*35)
    print(f"\nScansione sottocartelle in: {ROOT_DATA_DIR}")

    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() 
                   if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        return []

    if not subdirs:
        print(f"\n❌ ERRORE: Nessuna sottocartella trovata")
        return []

    print("\nCartelle target disponibili:")
    print(f"   0: ✨ Processa TUTTI i {len(subdirs)} target")
    print("   " + "─"*30)
    for i, dir_path in enumerate(subdirs):
        print(f"   {i+1}: {dir_path.name}")

    while True:
        print("\n" + "─"*70)
        try:
            choice = int(input(f"👉 Seleziona (0-{len(subdirs)}): ").strip())
            if choice == 0: return subdirs
            if 0 < choice <= len(subdirs): return [subdirs[choice-1]]
        except ValueError: pass
        print("❌ Input non valido.")

def ask_continue_to_next_step():
    print("\n" + "="*70)
    print(f"🎯 STEP 5 COMPLETATO! (LR: {TARGET_SIZE_GROUND}px | HR: {TARGET_SIZE_HUBBLE}px)")
    print("="*70)
    while True:
        choice = input(f"👉 Vuoi avviare Modello_2_prepare_data.py? [S/n]: ").strip().lower()
        if choice in ('', 's', 'y'): return True
        if choice in ('n', 'no'): return False

# ============================================================================
# FUNZIONI ESTRAZIONE
# ============================================================================
def extract_patches_from_file(file_path, output_dir, prefix, target_size, overlap_percent):
    patches = []
    try:
        with fits.open(file_path, memmap=True) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()
            if len(data.shape) == 3: data = data[0]
            
            wcs = WCS(header)
            if not wcs.has_celestial: return []
            
            patch_size = target_size
            step = int(patch_size * (1 - overlap_percent / 100))
            ny, nx = data.shape
            
            patch_count = 0
            for y in range(0, ny - patch_size + 1, step):
                for x in range(0, nx - patch_size + 1, step):
                    patch_data = data[y:y+patch_size, x:x+patch_size].copy()
                    
                    if np.isnan(patch_data).mean() > 0.5: continue
                    valid_data = patch_data[~np.isnan(patch_data)]
                    if len(valid_data) > 0 and np.std(valid_data) < 1e-6: continue
                    
                    try:
                        center_coord = wcs.pixel_to_world(x + patch_size/2, y + patch_size/2)
                        ra, dec = center_coord.ra.deg, center_coord.dec.deg
                    except: continue
                    
                    patch_name = f"{prefix}_{file_path.stem}_p{patch_count:04d}.fits"
                    patch_path = output_dir / patch_name
                    
                    new_header = header.copy()
                    new_header['NAXIS1'] = patch_size
                    new_header['NAXIS2'] = patch_size
                    if 'CRPIX1' in new_header: new_header['CRPIX1'] -= x
                    if 'CRPIX2' in new_header: new_header['CRPIX2'] -= y
                    
                    fits.PrimaryHDU(data=patch_data, header=new_header).writeto(patch_path, overwrite=True)
                    
                    patches.append({
                        'file': patch_name, 'path': str(patch_path),
                        'ra': ra, 'dec': dec,
                        'size': patch_size
                    })
                    patch_count += 1
        return patches
    except Exception as e:
        logging.error(f"Errore estrazione {file_path.name}: {e}")
        return []

# ============================================================================
# MATCHING
# ============================================================================
def find_matches_chunk(args):
    hubble_chunk, observatory_patches, obs_coords, match_threshold = args
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(obs_coords)
        chunk_matches = []
        for h_patch in hubble_chunk:
            dist, idx = tree.query([h_patch['ra'], h_patch['dec']], k=1)
            if dist < match_threshold:
                chunk_matches.append((h_patch, observatory_patches[idx], dist))
        return chunk_matches
    except: return []

def parallel_matching(hubble_patches, observatory_patches, match_threshold, num_workers):
    print(f"   🧮 Preparazione KDTree...")
    obs_coords = np.array([[p['ra'], p['dec']] for p in observatory_patches])
    
    chunk_size = max(1, len(hubble_patches) // (num_workers * 4))
    hubble_chunks = [hubble_patches[i:i+chunk_size] for i in range(0, len(hubble_patches), chunk_size)]
    
    worker_args = [(chunk, observatory_patches, obs_coords, match_threshold) for chunk in hubble_chunks]
    matches = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(find_matches_chunk, args) for args in worker_args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="🔗 Matching"):
            matches.extend(future.result())
    return matches

def copy_patch_pair(args):
    pair_id, h_patch, o_patch, pairs_dir = args
    try:
        pair_dir = pairs_dir / f"pair_{pair_id:05d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(h_patch['path'], pair_dir / h_patch['file'])
        shutil.copy2(o_patch['path'], pair_dir / o_patch['file'])
        return True
    except: return False

# ============================================================================
# PIPELINE
# ============================================================================
def process_single_target(base_dir, logger):
    print("\n" + "🎯"*35)
    print(f"ESTRAZIONE PATCH: {base_dir.name}".center(70))
    print(f"LR: {TARGET_SIZE_GROUND}px | HR: {TARGET_SIZE_HUBBLE}px".center(70))
    print("🎯"*35)
    
   # MODIFICA: Prende i file direttamente dalla cartella '3_registered_native'
    input_hubble = base_dir / '3_registered_native' / 'hubble'
    input_observatory = base_dir / '3_registered_native' / 'observatory'
    output_base = base_dir / '6_patches_from_cropped'
    
    if output_base.exists(): shutil.rmtree(output_base)
    
    output_hubble = output_base / 'hubble_patches'
    output_observatory = output_base / 'observatory_patches'
    pairs_dir = output_base / 'paired_patches_folders'
    
    output_hubble.mkdir(parents=True, exist_ok=True)
    output_observatory.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)
    
    hubble_files = sorted(list(input_hubble.glob('*.fits')))
    observatory_files = sorted(list(input_observatory.glob('*.fits')))
    
    if not hubble_files or not observatory_files:
        print("❌ Input incompleti.")
        return False
        
    # HUBBLE (512x512)
    print(f"\n🔵 Estrazione Hubble ({TARGET_SIZE_HUBBLE}px)...")
    h_patches = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        fs = [ex.submit(extract_patches_from_file, f, output_hubble, 'hr', TARGET_SIZE_HUBBLE, OVERLAP_PERCENT) for f in hubble_files]
        for f in tqdm(as_completed(fs), total=len(fs), desc="Processing"): h_patches.extend(f.result())
        
    # OBSERVATORY (128x128)
    print(f"\n🟢 Estrazione Observatory ({TARGET_SIZE_GROUND}px)...")
    o_patches = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        fs = [ex.submit(extract_patches_from_file, f, output_observatory, 'lr', TARGET_SIZE_GROUND, OVERLAP_PERCENT) for f in observatory_files]
        for f in tqdm(as_completed(fs), total=len(fs), desc="Processing"): o_patches.extend(f.result())

    if not h_patches or not o_patches: return False

    # MATCHING
    print(f"\n🔗 Matching...")
    matches = parallel_matching(h_patches, o_patches, MATCH_THRESHOLD, NUM_WORKERS)
    print(f"   ✅ Matches: {len(matches)}")

    # COPY
    print(f"\n💾 Salvataggio coppie...")
    count = 0
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as ex:
        fs = [ex.submit(copy_patch_pair, (i, m[0], m[1], pairs_dir)) for i, m in enumerate(matches)]
        for f in tqdm(as_completed(fs), total=len(fs), desc="Copying"): 
            if f.result(): count += 1
            
    # METADATA
    with open(output_base / 'dataset_info.json', 'w') as f:
        json.dump({
            'target_lr_size': TARGET_SIZE_GROUND,
            'target_hr_size': TARGET_SIZE_HUBBLE,
            'scale_ratio': TARGET_SIZE_HUBBLE / TARGET_SIZE_GROUND,
            'pairs': count
        }, f, indent=2)

    print(f"\n✅ Finito: {count} coppie salvate in {pairs_dir}")
    return True

def main():
    if len(sys.argv) > 1: target_dirs = [Path(sys.argv[1]).resolve()]
    else: target_dirs = select_target_directory()
    if not target_dirs: return

    logger = setup_logging()
    
    for base_dir in target_dirs:
        process_single_target(base_dir, logger)

    if ask_continue_to_next_step():
        next_script = SCRIPTS_DIR / 'Modello_2_prepare_data.py'
        if next_script.exists(): subprocess.run([sys.executable, str(next_script)])

if __name__ == "__main__":
    main()