import sys
import shutil
import random
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from skimage.transform import resize
import warnings

warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE =================
ROOT_DATA_DIR = Path(r"c:\Users\fratt\Documents\GitHub\SuperResolution\data")

# Parametri Patch
TARGET_HR_SIZE = 512       
OVERLAP_PERCENT = 90       
AI_INPUT_SIZE = 128        
MIN_VALID_PIXELS = 0.5     

# Percentuali Split Dataset
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Nome cartella output
OUTPUT_FOLDER_NAME = "6_patches_aligned"
# ==================================================

def debug_corner_coordinates(task, h_wcs):
    """
    Stampa un confronto dettagliato delle coordinate dei 4 angoli
    tra la patch Hubble e la patch Osservatorio prevista.
    """
    print(f"\n🔍 VERIFICA COORDINATE ANGOLI (Coppia #{task['pair_id']:05d})")
    print(f"   File Obs: {task['obs_file'].name}")
    
    # --- 1. HUBBLE CORNERS ---
    h_x, h_y = task['h_crop_x'], task['h_crop_y']
    h_size = TARGET_HR_SIZE
    
    # Ordine: Top-Left, Top-Right, Bottom-Left, Bottom-Right
    # Nota: In FITS/WCS, (0,0) è il centro del primo pixel. 
    # Usiamo coordinate pixel 0-based.
    h_corners_pix = np.array([
        [h_x, h_y],                   # TL
        [h_x + h_size, h_y],          # TR
        [h_x, h_y + h_size],          # BL
        [h_x + h_size, h_y + h_size]  # BR
    ])
    h_corners_world = h_wcs.pixel_to_world_values(h_corners_pix)
    
    # --- 2. OBSERVATORY CORNERS ---
    try:
        with fits.open(task['obs_file']) as hdul:
            o_wcs = WCS(hdul[0].header)
            
        o_cx, o_cy = task['obs_center_x'], task['obs_center_y']
        o_size = task['raw_patch_size']
        
        # Replichiamo la logica di ritaglio esatta
        o_x_start = int(o_cx - o_size / 2)
        o_y_start = int(o_cy - o_size / 2)
        
        o_corners_pix = np.array([
            [o_x_start, o_y_start],
            [o_x_start + o_size, o_y_start],
            [o_x_start, o_y_start + o_size],
            [o_x_start + o_size, o_y_start + o_size]
        ])
        o_corners_world = o_wcs.pixel_to_world_values(o_corners_pix)
        
    except Exception as e:
        print(f"   ❌ Errore lettura WCS Obs: {e}")
        return

    # --- 3. CONFRONTO ---
    labels = ["Top-Left ", "Top-Right", "Bot-Left ", "Bot-Right"]
    print("-" * 85)
    print(f"   {'Angolo':<10} | {'Hubble (RA, DEC)':<25} | {'Observatory (RA, DEC)':<25} | {'Diff (\")':<8}")
    print("-" * 85)
    
    max_diff = 0
    for i in range(4):
        hra, hdec = h_corners_world[i]
        ora, odec = o_corners_world[i]
        
        # Distanza euclidea approssimata in gradi
        diff_deg = np.sqrt((hra - ora)**2 + (hdec - odec)**2)
        diff_arcsec = diff_deg * 3600
        max_diff = max(max_diff, diff_arcsec)
        
        print(f"   {labels[i]:<10} | {hra:.5f}, {hdec:.5f}   | {ora:.5f}, {odec:.5f}   | {diff_arcsec:.2f}\"")
    
    print("-" * 85)
    if max_diff < 2.0: # Tolleranza: 2 arcsec (circa 1 pixel LR)
        print("   ✅ ALLINEAMENTO GEOMETRICO: OTTIMO")
    elif max_diff < 5.0:
        print("   ⚠️ ALLINEAMENTO GEOMETRICO: ACCETTABILE (Leggero shift)")
    else:
        print("   ❌ ALLINEAMENTO GEOMETRICO: CRITICO (Discrepanza elevata)")

def get_pixel_scale_deg(wcs):
    scales = proj_plane_pixel_scales(wcs)
    return np.mean(scales)

def get_virtual_hubble_grid(h_file):
    """Genera coordinate Hubble virtuali."""
    coords = []
    try:
        with fits.open(h_file) as hdul:
            header = hdul[0].header
            wcs = WCS(header)
            shape = hdul[0].data.shape
            if len(shape) == 3: shape = shape[1:]
            ny, nx = shape

        step = int(TARGET_HR_SIZE * (1 - OVERLAP_PERCENT / 100))
        if step < 1: step = 1

        for y in range(0, ny - TARGET_HR_SIZE + 1, step):
            for x in range(0, nx - TARGET_HR_SIZE + 1, step):
                center_x = x + TARGET_HR_SIZE / 2
                center_y = y + TARGET_HR_SIZE / 2
                coords.append({
                    'x_hr': x,
                    'y_hr': y,
                    'center_x': center_x,
                    'center_y': center_y
                })
        
        pixel_coords = np.array([[c['center_x'], c['center_y']] for c in coords])
        world_coords = wcs.pixel_to_world_values(pixel_coords)
        
        for i, c in enumerate(coords):
            c['ra'] = world_coords[i][0]
            c['dec'] = world_coords[i][1]
            
        return coords, wcs
    
    except Exception as e:
        print(f"Errore lettura Hubble {h_file.name}: {e}")
        return [], None

def check_observatory_overlap(obs_file, hubble_coords, h_fov_deg):
    """Controlla overlap geometrico senza estrarre."""
    valid_matches = []
    try:
        with fits.open(obs_file) as hdul:
            header = hdul[0].header
            wcs = WCS(header)
            shape = hdul[0].data.shape
            if len(shape) == 3: shape = shape[1:]
            ny, nx = shape
            obs_scale = get_pixel_scale_deg(wcs)

        raw_patch_size = int(round(h_fov_deg / obs_scale))
        if raw_patch_size < 4: return []

        ra_dec = np.array([[c['ra'], c['dec']] for c in hubble_coords])
        obs_pixels = wcs.world_to_pixel_values(ra_dec)
        margin = raw_patch_size / 2
        
        for i, (ox, oy) in enumerate(obs_pixels):
            if (ox - margin >= 0) and (ox + margin < nx) and \
               (oy - margin >= 0) and (oy + margin < ny):
                valid_matches.append({
                    'h_idx': i,
                    'obs_file': obs_file,
                    'obs_center_x': ox,
                    'obs_center_y': oy,
                    'raw_patch_size': raw_patch_size,
                    'ra': hubble_coords[i]['ra'],
                    'dec': hubble_coords[i]['dec']
                })
    except: pass
    return valid_matches

def extract_and_save_pair_folder(task, h_file, output_root):
    try:
        pair_id = task['pair_id']
        pair_dir = output_root / f"pair_{pair_id:06d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. ESTRAZIONE HUBBLE
        with fits.open(h_file) as h_hdul:
            h_data = h_hdul[0].data
            if h_data.ndim == 3: h_data = h_data[0]
            x, y = task['h_crop_x'], task['h_crop_y']
            hr_patch = h_data[y : y + TARGET_HR_SIZE, x : x + TARGET_HR_SIZE]
            
            if np.isnan(hr_patch).mean() > (1 - MIN_VALID_PIXELS): return False
            if np.mean(hr_patch) == 0: return False

            h_header = h_hdul[0].header.copy()
            h_header['NAXIS1'] = TARGET_HR_SIZE
            h_header['NAXIS2'] = TARGET_HR_SIZE

        # 2. ESTRAZIONE OBS
        with fits.open(task['obs_file']) as o_hdul:
            o_data = o_hdul[0].data
            if o_data.ndim == 3: o_data = o_data[0]
            ox, oy = task['obs_center_x'], task['obs_center_y']
            size = task['raw_patch_size']
            x_start = int(ox - size / 2)
            y_start = int(oy - size / 2)
            lr_raw = o_data[y_start : y_start + size, x_start : x_start + size]

            if np.isnan(lr_raw).mean() > (1 - MIN_VALID_PIXELS): return False
            if np.mean(lr_raw) == 0: return False

            lr_resized = resize(lr_raw, (AI_INPUT_SIZE, AI_INPUT_SIZE), 
                                anti_aliasing=True, preserve_range=True).astype(np.float32)
            
            o_header = o_hdul[0].header.copy()
            o_header['NAXIS1'] = AI_INPUT_SIZE
            o_header['NAXIS2'] = AI_INPUT_SIZE

        # 3. SALVATAGGIO
        fits.PrimaryHDU(data=hr_patch, header=h_header).writeto(pair_dir / "hubble.fits", overwrite=True)
        fits.PrimaryHDU(data=lr_resized, header=o_header).writeto(pair_dir / "observatory.fits", overwrite=True)
        
        return True
    except: return False

def perform_splitting(dataset_root):
    print("\n✂️  Eseguendo lo split del dataset (Train/Val/Test)...")
    
    all_pairs = sorted(list(dataset_root.glob("pair_*")))
    random.shuffle(all_pairs) 
    
    total = len(all_pairs)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    
    train_dirs = all_pairs[:n_train]
    val_dirs = all_pairs[n_train : n_train + n_val]
    test_dirs = all_pairs[n_train + n_val:]
    
    splits_root = dataset_root / "splits"
    if splits_root.exists(): shutil.rmtree(splits_root)
    
    for split_name, dirs in [("train", train_dirs), ("val", val_dirs), ("test", test_dirs)]:
        target_dir = splits_root / split_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for d in tqdm(dirs, desc=f"Moving to {split_name}"):
            shutil.move(str(d), str(target_dir / d.name))

    print(f"   ✅ Split completato: {len(train_dirs)} Train, {len(val_dirs)} Val, {len(test_dirs)} Test.")

def estimate_time(total_pairs):
    """Stima il tempo necessario basandosi sui core disponibili."""
    cpu_count = os.cpu_count() or 4
    estimated_speed = 30 # coppie/sec
    
    total_seconds = total_pairs / estimated_speed
    
    if total_seconds < 60:
        time_str = f"{total_seconds:.0f} secondi"
    elif total_seconds < 3600:
        time_str = f"{total_seconds/60:.1f} minuti"
    else:
        time_str = f"{total_seconds/3600:.1f} ore"
        
    return time_str, cpu_count

def main():
    print(f"🚀 PIPELINE INTELLIGENTE: ESTRAZIONE + SPLIT")
    
    for target_dir in ROOT_DATA_DIR.iterdir():
        if not target_dir.is_dir() or target_dir.name in ['logs', '__pycache__']: continue
        
        print(f"\n📂 Target: {target_dir.name}")
        
        input_h = target_dir / '3_registered_native' / 'hubble'
        input_o = target_dir / '3_registered_native' / 'observatory'
        h_files = sorted(list(input_h.glob("*.fits")))
        o_files = sorted(list(input_o.glob("*.fits")))

        if not h_files or not o_files: continue
        h_master = h_files[0]
        
        # 1. Virtual Matching
        print("   🔍 Matching Virtuale in corso...")
        h_grid, h_wcs = get_virtual_hubble_grid(h_master)
        if not h_grid: continue
        h_fov_deg = get_pixel_scale_deg(h_wcs) * TARGET_HR_SIZE
        
        extraction_tasks = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(check_observatory_overlap, f, h_grid, h_fov_deg) for f in o_files]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating"):
                extraction_tasks.extend(future.result())
        
        for i, task in enumerate(extraction_tasks):
            task['pair_id'] = i
            pt = h_grid[task['h_idx']]
            task['h_crop_x'] = pt['x_hr']
            task['h_crop_y'] = pt['y_hr']

        # 2. Report e Conferma con Tempi, Dimensioni e Esempi
        total_pairs = len(extraction_tasks)
        est_size_mb = (total_pairs * (TARGET_HR_SIZE**2 + AI_INPUT_SIZE**2) * 4) / (1024**2)
        time_est, cores = estimate_time(total_pairs)
        
        # Prepara dati esempio
        sample_raw_size = extraction_tasks[0]['raw_patch_size'] if total_pairs > 0 else 0
        
        print("\n" + "="*60)
        print("📊 REPORT PIANO DI ESTRAZIONE".center(60))
        print("="*60)
        print(f"   • Immagine Hubble (Master): {h_master.name}")
        print(f"   • Files Osservatorio:       {len(o_files)}")
        print(f"   ------------------------------------------------")
        print(f"   📏 DETTAGLI DIMENSIONALI:")
        print(f"      Target HR:      {TARGET_HR_SIZE}x{TARGET_HR_SIZE} px (Hubble)")
        print(f"      Target LR (AI): {AI_INPUT_SIZE}x{AI_INPUT_SIZE} px (Interpolato)")
        print(f"      Raw Crop Obs:   ~{sample_raw_size}x{sample_raw_size} px (Dato Fisico)")
        print(f"   ------------------------------------------------")
        
        # === 🔴 QUI AGGIUNGIAMO LA CHIAMATA ALLA FUNZIONE DEBUG ===
        if total_pairs > 0:
            # Prendi 3 indici casuali
            sample_indices = np.linspace(0, total_pairs - 1, 3, dtype=int)
            
            # Passiamo anche l'header WCS di Hubble che avevamo calcolato prima
            _, h_wcs_obj = get_virtual_hubble_grid(h_master)
            
            print(f"   📍 VERIFICA COORDINATE ANGOLI (3 Campioni):")
            for idx in sample_indices:
                # Chiamata alla tua funzione
                debug_corner_coordinates(extraction_tasks[idx], h_wcs_obj)
        else:
            print("      (Nessun match trovato)")
        # ==========================================================

        print(f"   ------------------------------------------------")
        print(f"   🛰️  PATCH HUBBLE UNICHE:    {len(h_grid)} (Posizioni fisiche)")
        print(f"   🔭 COPPIE TOTALI (MATCH):   {total_pairs} (Dataset finale)")
        print(f"   ------------------------------------------------")
        print(f"   💾 Spazio stimato su disco: ~{est_size_mb:.1f} MB")
        print(f"   ⏱️  TEMPO STIMATO:          ~{time_est} (su {cores} core)")
        print(f"   📂 Output:                  {target_dir / OUTPUT_FOLDER_NAME}")
        print("="*60)
        
        if total_pairs == 0:
            print("   ⚠️ Nessuna sovrapposizione trovata. Controlla i WCS.")
            continue

        choice = input("\n👉 Confermi l'estrazione e il salvataggio? [S/n]: ").lower().strip()
        if choice not in ['s', 'y', '']:
            print("❌ Annullato.")
            continue

        # 3. Estrazione Fisica
        out_root = target_dir / OUTPUT_FOLDER_NAME
        if out_root.exists(): shutil.rmtree(out_root)
        out_root.mkdir(parents=True)
        
        print(f"\n   🚀 3. Estrazione Fisica in corso...")
        
        success = 0
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(extract_and_save_pair_folder, t, h_master, out_root) for t in extraction_tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Writing FITS"):
                if f.result(): success += 1
        
        # 4. Splitting
        perform_splitting(out_root)
        
        # 5. Metadata
        print("   📝 Salvataggio dataset_map.csv...")
        meta = [{'id': t['pair_id'], 'ra': t['ra'], 'dec': t['dec'], 'src': t['obs_file'].name} for t in extraction_tasks]
        pd.DataFrame(meta).to_csv(out_root / 'dataset_map.csv', index=False)
        
        print(f"\n✅ FINITO: {success} coppie processate e divise.")

if __name__ == "__main__":
    main()