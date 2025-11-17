#!/usr/bin/env python3
"""
STEP 6 e REPORT: ESTRAZIONE PATCHES e REPORT COMPLETO

MODIFICATO:
- Rimossa completamente l'opzione 'Analisi Overlap' (Step 5).
- Il menu ora offre solo 'Estrazione Patches' (Step 6) e 'Report Stato Completo'.
- AGGIUNTO: Opzione 'Processa TUTTI' nel menu di selezione.
- AGGIUNTO: Loop principale in main() per processare uno o più target.
- Accetta BASE_DIR come argomento da riga di comando (da step2).
- AGGIUNTO: Multi-threading per l'estrazione patches (Step 6).
- AGGIUNTO: Multi-threading per l'accoppiamento patches (Step 6).
- AGGIUNTO: Report .md per i metadati delle patches.
- AGGIUNTO: "Report Stato Completo" ora analizza OGNI FILE singolarmente.
"""

import os
import glob
import time
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
# import matplotlib.pyplot as plt # Non più necessario
from tqdm import tqdm
import warnings
import shutil
import sys 
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

LOG_DIR_ROOT = Path(r'F:\Super Revolt Gaia\logs')

# Parametri Step 6
TARGET_FOV_ARCMIN = 1
OVERLAP_PERCENT = 25
MIN_VALID_PERCENT = 50
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_THREADS = 7

# ============================================================================
# FUNZIONE SELEZIONE TARGET (MODIFICATA PER "TUTTI")
# ============================================================================

ROOT_DATA_DIR = Path(r'F:\Super Revolt Gaia\SuperResolution\data')

def select_target_directory():
    """
    Mostra un menu per selezionare una o TUTTE le cartelle target.
    Restituisce: Elenco di Path (uno o più) o elenco vuoto.
    """
    print("\n" + "📂"*35); print("SELEZIONE CARTELLA TARGET".center(70)); print("📂"*35)
    print(f"\nScansione sottocartelle in: {ROOT_DATA_DIR}")

    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir()]
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile leggere la cartella {ROOT_DATA_DIR}: {e}"); return []
    if not subdirs:
        print(f"\n❌ ERRORE: Nessuna sottocartella trovata in {ROOT_DATA_DIR}"); return []

    print("\nCartelle target disponibili:")
    print(f"   0: ✨ Processa TUTTI i {len(subdirs)} target")
    print("   " + "─"*30)
    for i, dir_path in enumerate(subdirs): print(f"   {i+1}: {dir_path.name}")

    while True:
        print("\n" + "─"*70)
        try:
            choice_str = input(f"👉 Seleziona un numero (0-{len(subdirs)}) o 'q' per uscire: ").strip()
            if choice_str.lower() == 'q': print("👋 Uscita."); return []
            choice = int(choice_str)
            if choice == 0:
                print(f"\n✅ Selezionati TUTTI i {len(subdirs)} target."); return subdirs
            choice_idx = choice - 1
            if 0 <= choice_idx < len(subdirs):
                selected_dir = subdirs[choice_idx]
                print(f"\n✅ Cartella selezionata: {selected_dir.name}"); return [selected_dir]
            else:
                print(f"❌ Scelta non valida. Inserisci un numero tra 0 e {len(subdirs)}.")
        except ValueError: print("❌ Input non valido. Inserisci un numero.")
        except Exception as e: print(f"❌ Errore: {e}"); return []

# ============================================================================
# MENU PRINCIPALE (SEMPLIFICATO)
# ============================================================================

def main_menu():
    print("\n" + "🌟"*35); print("MENU PRINCIPALE".center(70)); print("🌟"*35)
    print(f"\n📋 SCEGLI OPERAZIONE (verrà applicata a tutti i target selezionati):")
    print(f"\n1️⃣  ESTRAZIONE PATCHES (Step 6)")
    print(f"\n2️⃣ NON ANCORA FUNZIONANTE, FUNZIONA MALE LEGGE LE CARTELLE NON I FILE   REPORT STATO COMPLETO (Scansiona tutti i file)")
    
    while True:
        print(f"\n" + "─"*70); choice = input("👉 Scegli opzione [1/2, default=1]: ").strip()
        if choice in ('', '1'): print(f"\n✅ Selezionato: ESTRAZIONE PATCHES"); return 'patches'
        elif choice == '2': print(f"\n✅ Selezionato: REPORT STATO COMPLETO"); return 'status_report'
        else: print(f"❌ Scelta non valida. Inserisci 1 o 2.")

# ============================================================================
# FUNZIONI REPORT STATO COMPLETO (MODIFICATE PER FILE SINGOLI)
# ============================================================================

def scan_directory_for_files(path):
    """
    Scansiona una directory e restituisce un elenco di dettagli per OGNI file.
    """
    file_details_list = []
    try:
        if not path.exists():
            return [] # La cartella non esiste
        
        files = list(path.glob('*.fits')) + list(path.glob('*.fit'))
        if not files:
            return [] # La cartella esiste ma è vuota
        
        # Usa tqdm se la lista è lunga
        file_iterator = files
        if len(files) > 10:
             file_iterator = tqdm(files, desc=f"   Scansione {path.name}", ncols=70, leave=False)

        for file_path in file_iterator:
            details = {
                'filename': file_path.name,
                'size_mb': round(file_path.stat().st_size / (1024**2), 2),
                'dimensions_px': 'N/A',
                'wcs_ok': False
            }

            try:
                with fits.open(file_path) as hdul:
                    wcs_found = False
                    dim_found = False
                    for hdu in hdul:
                        if hdu.data is not None and len(hdu.data.shape) >= 2:
                            if not dim_found:
                                # Prendi le dimensioni dal primo HDU con dati
                                shape = hdu.data.shape
                                if len(shape) == 3: # Cubo di dati
                                    details['dimensions_px'] = f"{shape[2]}x{shape[1]} (Cubo {shape[0]})"
                                else: # Immagine 2D
                                    details['dimensions_px'] = f"{shape[1]}x{shape[0]}"
                                dim_found = True
                            
                            # Controlla il WCS
                            try:
                                wcs = WCS(hdu.header)
                                if wcs.has_celestial:
                                    wcs_found = True
                            except Exception:
                                continue # Header WCS non valido in questo HDU
                    
                    details['wcs_ok'] = wcs_found
            except Exception:
                details['dimensions_px'] = 'Errore Lettura'
                details['wcs_ok'] = False
            
            file_details_list.append(details)
            
        return file_details_list
    
    except Exception as e:
        print(f"Errore scansione directory {path}: {e}")
        return []

def save_comprehensive_report_md(results, output_path, base_dir_name):
    """Salva il report di stato completo e dettagliato in Markdown."""
    print(f"   ✍️  Scrittura Report Stato Completo: {output_path.name}")
    def fmt_bool(val):
        if val is True: return "✅ Sì"
        if val is False: return "❌ No"
        return "N/A"

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Report Stato Completo: {base_dir_name}\n\n")
            f.write(f"**Generato il:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Scansione dettagliata di tutti i file FITS in ogni cartella della pipeline.\n")
            
            for stage, folders in results.items():
                f.write(f"\n---\n\n## {stage}\n\n")
                if not folders:
                    f.write("*(Nessuna cartella o file trovato)*\n")
                    continue
                
                for folder in folders:
                    path_str = str(folder['path']).replace('\\', '/')
                    files = folder['files']
                    f.write(f"### 📁 Cartella: `{path_str}`\n")
                    f.write(f"**Totale File:** {len(files)}\n\n")
                    
                    if not files:
                        f.write("*(Cartella vuota)*\n\n")
                        continue
                        
                    # Scrivi la tabella per questa cartella
                    f.write("| File | Dimensioni (px) | Dim. (MB) | WCS Valido |\n")
                    f.write("| :--- | :--- | :--- | :--- |\n")
                    
                    total_mb = 0.0
                    for file in files:
                        total_mb += file['size_mb']
                        f.write(f"| `{file['filename']}` | {file['dimensions_px']} | {file['size_mb']:.2f} | {fmt_bool(file['wcs_ok'])} |\n")
                    
                    f.write(f"| **Totale Cartella** | | **{total_mb:.2f} MB** | |\n\n")

    except Exception as e:
        print(f"   ❌ ERRORE: Impossibile scrivere il report MD {output_path.name}: {e}")

def run_comprehensive_analysis(BASE_DIR):
    """Esegue la scansione di tutte le cartelle e genera un report MD."""
    print(f"\n🔬 Scansione completa del target: {BASE_DIR.name}...")
    print(f"   (potrebbe richiedere tempo, apro ogni file FITS...)")
    
    # Definisce tutti i percorsi standard della pipeline
    paths_to_scan = {
        "1. Originarie": [
            BASE_DIR / '1_originarie' / 'local_raw',
            BASE_DIR / '1_originarie' / 'img_lights',
        ],
        "2. WCS Aggiunto (Step 1)": [
            BASE_DIR / '2_wcs' / 'osservatorio',
            BASE_DIR / '2_wcs' / 'hubble',
        ],
        "3. Registrate (Step 2)": [
            BASE_DIR / '3_registered_native' / 'osservatorio',
            BASE_DIR / '3_registered_native' / 'hubble',
        ],
        "4. Cropped (Step 3)": [
            BASE_DIR / '4_cropped' / 'osservatorio',
            BASE_DIR / '4_cropped' / 'hubble',
        ],
        "5. Mosaici (Step 4)": [
            BASE_DIR / '5_mosaics',
        ],
        "6. Patches (da Cropped)": [
            BASE_DIR / '6_patches_from_cropped' / 'observatory_native',
            BASE_DIR / '6_patches_from_cropped' / 'hubble_native',
            BASE_DIR / '6_patches_from_cropped' / 'paired_patches_folders',
        ],
        "6. Patches (da Registered)": [
            BASE_DIR / '6_patches_from_registered' / 'observatory_native',
            BASE_DIR / '6_patches_from_registered' / 'hubble_native',
            BASE_DIR / '6_patches_from_registered' / 'paired_patches_folders',
        ]
    }

    results = {}
    
    # Non si può usare tqdm qui perché la funzione interna ha il suo
    for stage, paths in paths_to_scan.items():
        print(f"\n--- Fase: {stage} ---")
        stage_results = []
        for path in paths:
            file_list = scan_directory_for_files(path)
            # Aggiungi solo se la cartella esiste e/o ha file
            if file_list or path.exists():
                stage_results.append({
                    'path': path.relative_to(BASE_DIR.parent),
                    'files': file_list
                })
        
        if stage_results:
            results[stage] = stage_results

    output_path = BASE_DIR / "report_stato_completo.md"
    save_comprehensive_report_md(results, output_path, BASE_DIR.name)
    
    print(f"\n✅ Report Stato Completo salvato in: {output_path}")
    return True

# ============================================================================
# STEP 6: ESTRAZIONE PATCHES
# ============================================================================

def select_input_type(BASE_DIR, INPUT_CROPPED_HUBBLE, INPUT_CROPPED_OBSERVATORY, INPUT_REGISTERED_HUBBLE, INPUT_REGISTERED_OBSERVATORY):
    print("\n" + "🎯"*35); print("SELEZIONE TIPO IMMAGINI INPUT (PER STEP 6)".center(70)); print("🎯"*35)
    print(f"\n📋 OPZIONI DISPONIBILI:\n\n1️⃣  IMMAGINI CROPPED (CONSIGLIATO)\n   ✓ Tutte le immagini hanno le stesse dimensioni\n   📁 Input: {INPUT_CROPPED_HUBBLE.parent}")
    print(f"\n2️⃣  IMMAGINI REGISTERED\n   • Mantiene dimensioni originali\n   📁 Input: {INPUT_REGISTERED_HUBBLE.parent}")
    INPUT_HUBBLE, INPUT_OBSERVATORY, OUTPUT_DIR_PATCHES, source_type = None, None, None, ""
    while True:
        print(f"\n" + "─"*70); choice = input("👉 Scegli opzione [1/2, default=1]: ").strip()
        if choice == '' or choice == '1':
            INPUT_HUBBLE, INPUT_OBSERVATORY, OUTPUT_DIR_PATCHES, source_type = INPUT_CROPPED_HUBBLE, INPUT_CROPPED_OBSERVATORY, BASE_DIR / '6_patches_from_cropped', "cropped"
            print(f"\n✅ Selezionato: IMMAGINI CROPPED"); break
        elif choice == '2':
            INPUT_HUBBLE, INPUT_OBSERVATORY, OUTPUT_DIR_PATCHES, source_type = INPUT_REGISTERED_HUBBLE, INPUT_REGISTERED_OBSERVATORY, BASE_DIR / '6_patches_from_registered', "registered"
            print(f"\n✅ Selezionato: IMMAGINI REGISTERED"); break
        else: print(f"❌ Scelta non valida. Inserisci 1 o 2.")
    OUTPUT_HUBBLE_PATCHES, OUTPUT_OBS_PATCHES = OUTPUT_DIR_PATCHES / 'hubble_native', OUTPUT_DIR_PATCHES / 'observatory_native'
    print(f"\n🔍 Verifica directory...")
    if INPUT_HUBBLE.exists(): print(f"   ✓ Hubble: {len(list(INPUT_HUBBLE.glob('*.fits')) + list(INPUT_HUBBLE.glob('*.fit')))} file trovati")
    else: print(f"   ⚠️  Directory Hubble non trovata: {INPUT_HUBBLE}")
    if INPUT_OBSERVATORY.exists(): print(f"   ✓ Observatory: {len(list(INPUT_OBSERVATORY.glob('*.fits')) + list(INPUT_OBSERVATORY.glob('*.fit')))} file trovati")
    else: print(f"   ⚠️  Directory Observatory non trovata: {INPUT_OBSERVATORY}")
    print(f"\n📂 Output patches: {OUTPUT_DIR_PATCHES}")
    return (source_type, INPUT_HUBBLE, INPUT_OBSERVATORY, OUTPUT_DIR_PATCHES, OUTPUT_HUBBLE_PATCHES, OUTPUT_OBS_PATCHES)

def setup_logging():
    LOG_DIR_ROOT.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR_ROOT / f'patch_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()])
    return logging.getLogger(__name__)

def extract_patches_from_image(fits_path, output_dir, source_label, logger):
    try:
        with fits.open(fits_path) as hdul:
            data_hdu = None
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) >= 2: data_hdu = hdu; break
            if data_hdu is None: return []
            data, header = data_hdu.data, data_hdu.header
            if len(data.shape) == 3: data = data[0]
            try: wcs = WCS(header); assert wcs.has_celestial
            except: return []
            try:
                if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None: pixel_scale_deg = np.sqrt(wcs.wcs.cd[0,0]**2 + wcs.wcs.cd[0,1]**2)
                elif hasattr(wcs.wcs, 'cdelt'): pixel_scale_deg = abs(wcs.wcs.cdelt[0])
                else: p1 = wcs.pixel_to_world(0, 0); p2 = wcs.pixel_to_world(1, 0); pixel_scale_deg = p1.separation(p2).deg
                pixel_scale_arcsec = pixel_scale_deg * 3600
            except: return []
            target_arcsec = TARGET_FOV_ARCMIN * 60
            patch_size_px = int(target_arcsec / pixel_scale_arcsec); patch_size_px = ((patch_size_px + 7) // 8) * 8
            step = int(patch_size_px * (1 - OVERLAP_PERCENT/100))
            ny, nx = data.shape; patches_info = []; patch_idx = 0; y = 0
            while y + patch_size_px <= ny:
                x = 0
                while x + patch_size_px <= nx:
                    patch_data = data[y:y+patch_size_px, x:x+patch_size_px]
                    valid_mask = np.isfinite(patch_data); valid_percent = 100 * valid_mask.sum() / patch_data.size
                    if valid_percent >= MIN_VALID_PERCENT:
                        center_x, center_y = x + patch_size_px // 2, y + patch_size_px // 2
                        center_coord = wcs.pixel_to_world(center_x, center_y)
                        filename = f'{source_label}_{Path(fits_path).stem}_p{patch_idx:04d}.fits'
                        output_path = output_dir / filename
                        patch_header = header.copy(); patch_header['NAXIS1'] = patch_size_px; patch_header['NAXIS2'] = patch_size_px
                        patch_hdu = fits.PrimaryHDU(patch_data, header=patch_header); patch_hdu.writeto(output_path, overwrite=True)
                        patches_info.append({ 'filename': filename, 'source_file': Path(fits_path).name, 'patch_index': patch_idx, 'position_px': (x, y), 'patch_size_px': patch_size_px,
                                              'center_ra': center_coord.ra.deg, 'center_dec': center_coord.dec.deg, 'pixel_scale_arcsec': pixel_scale_arcsec, 'valid_percent': valid_percent })
                        patch_idx += 1
                    x += step
                y += step
            return patches_info
    except Exception as e:
        logger.error(f"Errore in {fits_path.name}: {e}"); return []

def save_patch_metadata_md(patches_info, output_path, label):
    """Salva i metadati delle patch in un file Markdown leggibile."""
    print(f"   ✍️  Scrittura report MD: {output_path.name}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Report Metadati Patches: {label}\n\n")
            f.write(f"**Totale patches estratte:** {len(patches_info)}\n")
            f.write(f"**Report generato il:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("| Immagine Sorgente | Nome Patch | Posizione (X, Y) | Dimensione (px) | % Valida | Centro RA (deg) | Centro Dec (deg) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for patch in patches_info:
                pos = patch.get('position_px', ('N/A', 'N/A'))
                size = patch.get('patch_size_px', 'N/A')
                valid = f"{patch.get('valid_percent', 0):.2f}%"
                ra = f"{patch.get('center_ra', 0):.6f}"
                dec = f"{patch.get('center_dec', 0):.6f}"
                f.write(f"| {patch.get('source_file', 'N/A')} | `{patch.get('filename', 'N/A')}` | ({pos[0]}, {pos[1]}) | {size}x{size} | {valid} | {ra} | {dec} |\n")
    except Exception as e:
        print(f"   ❌ ERRORE: Impossibile scrivere il report MD {output_path.name}: {e}")

def save_pairs_metadata_md(pairs_info, output_path):
    """Salva i metadati delle coppie in un file Markdown leggibile."""
    print(f"   ✍️  Scrittura report MD: {output_path.name}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Report Accoppiamento Patches\n\n")
            f.write(f"**Totale coppie create:** {len(pairs_info)}\n")
            f.write(f"**Report generato il:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("| ID Coppia (Folder) | Patch Hubble | Patch Observatory | Distanza (arcmin) |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for i, pair in enumerate(pairs_info):
                pair_id = f"`pair_{i:05d}`"
                h_patch = f"`{pair.get('hubble_patch', 'N/A')}`"
                o_patch = f"`{pair.get('observatory_patch', 'N/A')}`"
                dist = f"{pair.get('separation_arcmin', 0):.4f}"
                f.write(f"| {pair_id} | {h_patch} | {o_patch} | {dist} |\n")
    except Exception as e:
        print(f"   ❌ ERRORE: Impossibile scrivere il report MD {output_path.name}: {e}")

def save_summary_metadata_md(metadata, output_path):
    """Salva il report MD di riepilogo del dataset."""
    print(f"   ✍️  Scrittura report MD: {output_path.name}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Report Riepilogo Dataset\n\n")
            f.write(f"**Generato il:** {metadata.get('timestamp', 'N/A')}\n\n")
            f.write("## Configurazione\n")
            f.write(f"- **Tipo Immagini Sorgente:** `{metadata.get('source_type', 'N/A')}`\n")
            f.write(f"- **FOV Target:** {metadata.get('target_fov_arcmin', 'N/A')} arcmin\n")
            f.write(f"- **Overlap:** {metadata.get('overlap_percent', 'N/A')}%\n")
            f.write("\n## Riepilogo Patch\n")
            f.write(f"- **Patches Hubble Estratte:** {metadata.get('hubble', {}).get('num_patches', 0)}\n")
            f.write(f"- **Patches Observatory Estratte:** {metadata.get('observatory', {}).get('num_patches', 0)}\n")
            f.write(f"- **Coppie Create:** {metadata.get('pairs', {}).get('num_pairs', 0)}\n")
    except Exception as e:
        print(f"   ❌ ERRORE: Impossibile scrivere il report MD {output_path.name}: {e}")

def extract_all_patches(input_dir, output_dir, label, logger):
    """Estrae patches da tutte le immagini (con multithreading)"""
    print(f"\n🔪 Estrazione patches {label.upper()} (con {NUM_THREADS} threads)...")
    files = list(input_dir.glob('*.fits')) + list(input_dir.glob('*.fit'))
    if not files: print(f"   ⚠️  Nessun file in {input_dir}"); return []
    all_patches = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(extract_patches_from_image, fpath, output_dir, label, logger): fpath for fpath in files}
        with tqdm(total=len(files), desc=f"   {label}", ncols=70) as pbar:
            for future in as_completed(futures):
                fpath = futures[future]
                try:
                    patches_from_file = future.result()
                    if patches_from_file: all_patches.extend(patches_from_file)
                except Exception as e:
                    logger.error(f"Errore nel thread processando {fpath.name}: {e}")
                pbar.update(1)
    print(f"   ✓ Estratte {len(all_patches)} patches da {len(files)} immagini")
    metadata_file_md = output_dir / f'{label}_patches_report.md'
    save_patch_metadata_md(all_patches, metadata_file_md, label.upper())
    return all_patches


def find_best_match_for_hubble_patch(h_patch, obs_patches_list, threshold_arcmin):
    """Task per un singolo thread: Trova la migliore patch observatory"""
    h_coord = SkyCoord(h_patch['center_ra']*u.deg, h_patch['center_dec']*u.deg)
    best_dist = float('inf')
    best_obs = None
    for o_patch in obs_patches_list:
        o_coord = SkyCoord(o_patch['center_ra']*u.deg, o_patch['center_dec']*u.deg)
        dist_arcmin = h_coord.separation(o_coord).arcmin
        if dist_arcmin < best_dist:
            best_dist = dist_arcmin
            best_obs = o_patch
    if best_dist < threshold_arcmin and best_obs:
        return (h_patch, best_obs, best_dist)
    return None


def create_patch_pairs(hubble_patches, obs_patches, 
                       OUTPUT_DIR_PATCHES, 
                       OUTPUT_HUBBLE_PATCHES, 
                       OUTPUT_OBS_PATCHES, 
                       logger):
    """Crea coppie di patches (con multi-threading) e salva in cartelle dedicate."""
    print(f"\n🔗 Creazione coppie patches (con {NUM_THREADS} threads)...")
    hubble_patch_dir = OUTPUT_HUBBLE_PATCHES
    obs_patch_dir = OUTPUT_OBS_PATCHES
    pairs_folders_dir = OUTPUT_DIR_PATCHES / 'paired_patches_folders'
    pairs_folders_dir.mkdir(parents=True, exist_ok=True)
    print(f"   📂 Creazione cartelle coppie in: {pairs_folders_dir}")
    threshold_arcmin = 0.5
    raw_pairs_results = []
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = { executor.submit(find_best_match_for_hubble_patch, h_patch, obs_patches, threshold_arcmin): h_patch for h_patch in hubble_patches }
        print("   (Fase 1: Calcolo abbinamenti in parallelo...)")
        with tqdm(total=len(hubble_patches), desc="   Pairing (CPU)", ncols=70) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result: raw_pairs_results.append(result)
                pbar.update(1)

    print(f"\n   (Fase 2: Copia file e creazione cartelle...)")
    raw_pairs_results.sort(key=lambda x: x[0]['filename'])
    final_pairs_metadata = []
    
    for i, (h_patch, best_obs, best_dist) in enumerate(tqdm(raw_pairs_results, desc="   Copia (I/O)", ncols=70)):
        pair_folder_name = f'pair_{i:05d}'
        pair_dest_dir = pairs_folders_dir / pair_folder_name
        pair_dest_dir.mkdir(exist_ok=True)
        h_filename, o_filename = h_patch['filename'], best_obs['filename']
        h_src_path, o_src_path = hubble_patch_dir / h_filename, obs_patch_dir / o_filename
        try:
            shutil.copy2(h_src_path, pair_dest_dir / h_filename)
            shutil.copy2(o_src_path, pair_dest_dir / o_filename)
        except FileNotFoundError as e: logger.warning(f"File non trovato durante copia coppia: {e}")
        except Exception as e: logger.error(f"Errore copiando {pair_folder_name}: {e}")
        final_pairs_metadata.append({ 'hubble_patch': h_filename, 'observatory_patch': o_filename, 'separation_arcmin': best_dist })
    
    pairs_file_md = OUTPUT_DIR_PATCHES / 'patch_pairs_report.md'
    save_pairs_metadata_md(final_pairs_metadata, pairs_file_md)
    
    num_pairs_found = len(final_pairs_metadata)
    print(f"   ✓ {num_pairs_found} coppie create (report MD)")
    print(f"   ✓ {num_pairs_found} cartelle coppie copiate in {pairs_folders_dir}")
    return final_pairs_metadata


def create_dataset_split(patches, output_dir, logger):
    """Crea split train/val/test (MANTIENE JSON PERCHÉ È FUNZIONALE)"""
    if not patches: return None
    np.random.shuffle(patches); n = len(patches); n_train, n_val = int(n * TRAIN_RATIO), int(n * VAL_RATIO)
    splits = { 'train': patches[:n_train], 'val': patches[n_train:n_train+n_val], 'test': patches[n_train+n_val:] }
    split_file = output_dir / 'dataset_split.json'
    print(f"   ✍️  Scrittura split JSON: {split_file.name}")
    with open(split_file, 'w') as f:
        json.dump({ 'timestamp': datetime.now().isoformat(), 'train_ratio': TRAIN_RATIO, 'val_ratio': VAL_RATIO, 'test_ratio': TEST_RATIO,
                    'splits': {k: [p['filename'] for p in v] for k, v in splits.items()} }, f, indent=2)
    return splits


def run_patches(BASE_DIR, INPUT_CROPPED_HUBBLE, INPUT_CROPPED_OBSERVATORY, INPUT_REGISTERED_HUBBLE, INPUT_REGISTERED_OBSERVATORY):
    (source_type, INPUT_HUBBLE, INPUT_OBSERVATORY, OUTPUT_DIR_PATCHES, OUTPUT_HUBBLE_PATCHES, OUTPUT_OBS_PATCHES) = select_input_type(
         BASE_DIR, INPUT_CROPPED_HUBBLE, INPUT_CROPPED_OBSERVATORY, INPUT_REGISTERED_HUBBLE, INPUT_REGISTERED_OBSERVATORY)
    logger = setup_logging()
    print("\n" + "✂️ "*35); print(f"STEP 6: ESTRAZIONE PATCHES".center(70)); print("✂️ "*35)
    print(f"\n📋 CONFIGURAZIONE:\n   Tipo input: {source_type.upper()}\n   FOV target: {TARGET_FOV_ARCMIN} arcmin\n   Overlap: {OVERLAP_PERCENT}%")
    print(f"\n📂 INPUT:\n   Hubble: {INPUT_HUBBLE}\n   Observatory: {INPUT_OBSERVATORY}")
    print(f"\n📂 OUTPUT:\n   {OUTPUT_DIR_PATCHES}")
    OUTPUT_HUBBLE_PATCHES.mkdir(parents=True, exist_ok=True); OUTPUT_OBS_PATCHES.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\nESTRAZIONE PATCHES\n{'='*70}")
    hubble_patches, obs_patches = [], []
    if INPUT_HUBBLE.exists(): hubble_patches = extract_all_patches(INPUT_HUBBLE, OUTPUT_HUBBLE_PATCHES, 'hubble', logger)
    if INPUT_OBSERVATORY.exists(): obs_patches = extract_all_patches(INPUT_OBSERVATORY, OUTPUT_OBS_PATCHES, 'observatory', logger)
    
    # --- CORREZIONE TYPO (da 'Clock' a '*70') ---
    print(f"\n{'='*70}\n📊 RIEPILOGO ESTRAZIONE\n{'='*70}")
    # --- FINE CORREZIONE ---
    
    print(f"\n   Hubble patches: {len(hubble_patches)}\n   Observatory patches: {len(obs_patches)}\n   TOTALE: {len(hubble_patches) + len(obs_patches)}")
    pairs = []
    if hubble_patches and obs_patches:
        pairs = create_patch_pairs(hubble_patches, obs_patches, OUTPUT_DIR_PATCHES, OUTPUT_HUBBLE_PATCHES, OUTPUT_OBS_PATCHES, logger)
    print(f"\n{'='*70}\nSPLIT DATASET\n{'='*70}")
    if hubble_patches:
        print(f"\n📊 Hubble:"); hubble_splits = create_dataset_split(hubble_patches, OUTPUT_HUBBLE_PATCHES, logger)
        if hubble_splits: print(f"   Train: {len(hubble_splits['train'])}\n   Val: {len(hubble_splits['val'])}\n   Test: {len(hubble_splits['test'])}")
    if obs_patches:
        print(f"\n📊 Observatory:"); obs_splits = create_dataset_split(obs_patches, OUTPUT_OBS_PATCHES, logger)
        if obs_splits: print(f"   Train: {len(obs_splits['train'])}\n   Val: {len(obs_splits['val'])}\n   Test: {len(obs_splits['test'])}")
    
    metadata = { 'timestamp': datetime.now().isoformat(), 'source_type': source_type, 'target_fov_arcmin': TARGET_FOV_ARCMIN, 'overlap_percent': OVERLAP_PERCENT,
                 'hubble': {'num_patches': len(hubble_patches)}, 'observatory': {'num_patches': len(obs_patches)}, 'pairs': {'num_pairs': len(pairs)} }
    
    metadata_file_md = OUTPUT_DIR_PATCHES / 'dataset_summary_report.md'
    save_summary_metadata_md(metadata, metadata_file_md)

    print(f"\n{'='*70}\n✅ ESTRAZIONE (Step 6) COMPLETATA\n{'='*70}\n📁 Output: {OUTPUT_DIR_PATCHES}")
    return True

# ============================================================================
# MAIN (SEMPLIFICATO)
# ============================================================================

def main():
    """Main unificato"""
    
    target_dirs = []
    if len(sys.argv) > 1:
        target_dirs = [Path(sys.argv[1])]
        print(f"🚀 Avviato da script precedente. Target: {target_dirs[0].name}")
    else:
        target_dirs = select_target_directory()
    
    if not target_dirs:
        print("Nessun target selezionato. Uscita."); return

    if len(target_dirs) > 1:
        print(f"Modalità Batch: {len(target_dirs)} target".center(70))
    print("="*70)

    start_time_total = time.time()
    mode = main_menu()

    # Liste per tenere traccia di cosa fare
    targets_to_patch = []
    targets_to_report = []
    
    if mode == 'patches':
        targets_to_patch = list(target_dirs)
    elif mode == 'status_report':
        targets_to_report = list(target_dirs)
    
    # --- FASE 1: PATCHES (se richiesta) ---
    successful_patch_targets = []
    failed_patch_targets = []
    
    if targets_to_patch:
        print("\n" + "="*70); print("INIZIO FASE: ESTRAZIONE PATCHES (Step 6)"); print("="*70)
        for BASE_DIR in targets_to_patch:
            print("\n" + "🚀"*35); print(f"TARGET PATCH: {BASE_DIR.name}".center(70)); print("🚀"*35)
            start_time_target = time.time()
            INPUT_CROPPED_HUBBLE = BASE_DIR / '4_cropped' / 'hubble'
            INPUT_CROPPED_OBSERVATORY = BASE_DIR / '4_cropped' / 'observatory'
            INPUT_REGISTERED_HUBBLE = BASE_DIR / '3_registered_native' / 'hubble'
            INPUT_REGISTERED_OBSERVATORY = BASE_DIR / '3_registered_native' / 'observatory'
            try:
                success = run_patches(BASE_DIR, INPUT_CROPPED_HUBBLE, INPUT_CROPPED_OBSERVATORY, INPUT_REGISTERED_HUBBLE, INPUT_REGISTERED_OBSERVATORY)
                if success:
                    successful_patch_targets.append(BASE_DIR)
                else:
                    failed_patch_targets.append(BASE_DIR)
            except Exception as e:
                print(f"\n❌ ERRORE CRITICO (Patches) su {BASE_DIR.name}: {e}")
                import traceback; traceback.print_exc()
                failed_patch_targets.append(BASE_DIR)
            elapsed_target = time.time() - start_time_target
            print(f"⏱️  Tempo patches per {BASE_DIR.name}: {elapsed_target:.1f} secondi")

    # --- FASE 2: REPORT STATO COMPLETO (se richiesta) ---
    successful_report_targets = []
    failed_report_targets = []
    
    if targets_to_report:
        print("\n" + "="*70); print("INIZIO FASE: REPORT STATO COMPLETO"); print("="*70)
        for BASE_DIR in targets_to_report:
            print("\n" + "🚀"*35); print(f"TARGET REPORT: {BASE_DIR.name}".center(70)); print("🚀"*35)
            start_time_target = time.time()
            try:
                success = run_comprehensive_analysis(BASE_DIR)
                if success:
                    successful_report_targets.append(BASE_DIR)
                else:
                    failed_report_targets.append(BASE_DIR)
            except Exception as e:
                print(f"\n❌ ERRORE CRITICO (Report) su {BASE_DIR.name}: {e}")
                failed_report_targets.append(BASE_DIR)
            elapsed_target = time.time() - start_time_target
            print(f"⏱️  Tempo report per {BASE_DIR.name}: {elapsed_target:.1f} secondi")

    # --- RIEPILOGO FINALE ---
    elapsed_total = time.time() - start_time_total
    print(f"\n" + "="*70); print("COMPLETAMENTO OPERAZIONI"); print("="*70)
    print(f"   Modalità eseguita: {mode.upper()}")
    print(f"   Target totali selezionati: {len(target_dirs)}")
    if targets_to_patch:
        print(f"   Patches (Step 6): {len(successful_patch_targets)} success, {len(failed_patch_targets)} fail")
    if targets_to_report:
        print(f"   Report Stato: {len(successful_report_targets)} success, {len(failed_report_targets)} fail")
    print(f"\n   ⏱️  Tempo totale batch: {elapsed_total:.1f} secondi ({elapsed_total/60:.1f} minuti)")
    print(f"\n{'='*70}\nGRAZIE PER AVER USATO LO SCRIPT!\n{'='*70}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()