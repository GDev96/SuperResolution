"""
STEP 4: ESTRAZIONE PATCHES (MIGLIORATO - RISOLUZIONE NATIVA)
Estrae patches dalle immagini registrate mantenendo la risoluzione nativa di ogni immagine.

MIGLIORIA CHIAVE:
- Non forza tutte le patches a 0.1 arcsec/pixel
- Ogni patch mantiene la risoluzione dell'immagine sorgente
- Le patches di Hubble saranno ad alta risoluzione (~0.04"/px)
- Le patches dell'Osservatorio manterranno la loro risoluzione nativa
- La dimensione in pixel viene adattata per mantenere un FOV costante in arcmin
"""

import os
import glob
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE (DINAMICA)
# ============================================================================
from pathlib import Path

# Ottieni il percorso assoluto della directory contenente questo script
SCRIPT_DIR = Path(__file__).resolve().parent

# Cerca la cartella 'data'
if (SCRIPT_DIR / 'data').exists():
    # Caso 1: Lo script è nella root del progetto
    PROJECT_ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / 'data').exists():
    # Caso 2: Lo script è in una sottocartella
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    raise FileNotFoundError(
        f"Impossibile trovare la directory 'data' relativa a {SCRIPT_DIR}. "
        "Assicurati che 'data' sia nella cartella principale del progetto."
    )

# Definisci i percorsi principali
BASE_DIR = PROJECT_ROOT / 'data'
LOG_DIR = PROJECT_ROOT / 'logs' 

# --- Percorsi per Riepilogo Pipeline ---
# Input: Originali (da Step 0/1)
INPUT_ORIG_HUBBLE = BASE_DIR / 'img_lights_1'
INPUT_ORIG_OBSERVATORY = BASE_DIR / 'local_raw'

# Input: WCS (da Step 1)
INPUT_WCS_HUBBLE = BASE_DIR / 'lith_con_wcs'
INPUT_WCS_OBSERVATORY = BASE_DIR / 'osservatorio_con_wcs'

# Input: immagini registrate con risoluzione nativa (da Step 3)
INPUT_HUBBLE = BASE_DIR / '3_registered_native' / 'hubble'
INPUT_OBSERVATORY = BASE_DIR / '3_registered_native' / 'observatory'

# Output: patches (da Step 4)
OUTPUT_DIR = BASE_DIR / '4_patches_native'
OUTPUT_HUBBLE_PATCHES = OUTPUT_DIR / 'hubble_native'
OUTPUT_OBS_PATCHES = OUTPUT_DIR / 'observatory_native'

# --- PARAMETRI PATCHES ---
# FOV target in arcmin - questo rimane costante per tutte le patches
TARGET_FOV_ARCMIN = 0.85  # ~51 arcsec, simile alla vecchia 512px @ 0.1"/px

OVERLAP_PERCENT = 25  # % overlap tra patches vicine
MIN_VALID_PERCENT = 50  # % minima pixel validi (non-NaN) in una patch

# --- SPLIT DATASET ---
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
# ============================================================================
# SETUP
# ============================================================================

def setup_logging():
    """Configura logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'patch_extraction_native_{timestamp}.log'
    
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
# FUNZIONE DI RIEPILOGO PIPELINE
# ============================================================================

def analyze_directory(directory: Path, step_title: str, recursive: bool, scale_key: str = None, logger=None):
    """
    Analizza una directory e stampa un riepilogo di file, dimensioni e scale.
    
    Args:
        directory: Path alla directory
        step_title: Titolo da stampare (es. "Step 1: Input Originale")
        recursive: True per cercare in sottocartelle (usato per Step 1)
        scale_key: Header FITS da cui leggere la scala (es. 'NATIVESC', 'PIXSCALE')
                   o 'WCS' per calcolarlo dal WCS.
        logger: Logger (non usato in questa versione, stampa su stdout)
    """
    print(f"\n{'-'*70}")
    print(f"📂 {step_title.upper()}")
    print(f"   Directory: {directory}")
    print(f"{'-'*70}")
    
    if not directory.exists():
        print("   ⚠️ Directory non trovata.")
        return 0, 0.0

    if recursive:
        files = list(directory.rglob('*.fits')) + list(directory.rglob('*.fit'))
    else:
        files = list(directory.glob('*.fits')) + list(directory.glob('*.fit'))
    
    if not files:
        print("   ℹ️ Nessun file .fits/.fit trovato.")
        return 0, 0.0
    
    total_files = len(files)
    total_mb_calculated = 0.0  # Solo per i file analizzati
    scales = set()
    shapes = set()
    
    max_files_to_show = 5
    print(f"   Analisi di {total_files} file (mostro i primi {max_files_to_show}):")
    
    for i, f_path in enumerate(files[:max_files_to_show]):
        try:
            with fits.open(f_path) as hdul:
                data_hdu = None
                for hdu in hdul:
                    # Trova il primo HDU con dati 2D o più
                    if hdu.data is not None and len(hdu.data.shape) >= 2:
                        data_hdu = hdu
                        break
                
                if data_hdu is None:
                    print(f"      {i+1}. {f_path.name[:40]:40s} → (Nessun dato 2D+ trovato)")
                    continue
                
                data = data_hdu.data
                header = data_hdu.header
                
                # Gestisci dati 3D (es. cubi) prendendo il primo slice
                if len(data.shape) == 3:
                    data = data[0]
                
                ny, nx = data.shape
                shapes.add(f"{nx}x{ny}")
                size_mb = data.nbytes / (1024**2)
                total_mb_calculated += size_mb
                
                scale_str = ""
                if scale_key:
                    if scale_key == 'WCS': # Caso speciale: calcola da WCS (Step 2)
                        try:
                            wcs = WCS(header)
                            if wcs.has_celestial:
                                if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
                                    cd = wcs.wcs.cd
                                    pixel_scale_deg = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
                                else:
                                    pixel_scale_deg = abs(wcs.wcs.cdelt[0])
                                scale_val = pixel_scale_deg * 3600.0
                                scales.add(round(scale_val, 4))
                                scale_str = f"@ {scale_val:.4f}\"/px"
                            else:
                                scale_str = "(WCS non cel.)"
                        except:
                            scale_str = "(WCS err)"
                    else: # Cerca chiave header (Step 3 e 4)
                        scale_val = header.get(scale_key)
                        if scale_val is not None:
                            scales.add(round(float(scale_val), 4))
                            scale_str = f"@ {float(scale_val):.4f}\"/px"
                        else:
                            scale_str = f"({scale_key} assente)"
                
                print(f"      {i+1}. {f_path.name[:40]:40s} → {nx:5d} × {ny:5d} px {scale_str} ({size_mb:7.2f} MB)")
        
        except Exception as e:
            print(f"      {i+1}. {f_path.name[:40]:40s} → Errore: {e}")

    # Calcolo MB totali (stimato dagli altri file)
    if total_files > max_files_to_show:
        avg_mb = total_mb_calculated / min(max_files_to_show, total_files)
        estimated_total_mb = total_mb_calculated + (avg_mb * (total_files - max_files_to_show))
        print(f"      ... e altri {total_files - max_files_to_show} file.")
    else:
        estimated_total_mb = total_mb_calculated

    print(f"\n   --- Riepilogo directory ---")
    print(f"   Numero totale file: {total_files}")
    print(f"   Dimensione totale stimata: {estimated_total_mb:.2f} MB (~{estimated_total_mb/1024:.2f} GB)")
    if shapes:
        unique_shapes = sorted(list(shapes))
        if len(unique_shapes) > 10:
             print(f"   Dimensioni (px) uniche: {len(unique_shapes)} (troppe da mostrare)")
        else:
             print(f"   Dimensioni (px) uniche: {unique_shapes}")
    if scales:
        unique_scales = sorted(list(scales))
        if len(unique_scales) > 10:
             print(f"   Scale (arcsec/px) uniche: {len(unique_scales)} (troppe da mostrare)")
        else:
            print(f"   Scale (arcsec/px) uniche: {unique_scales}")
    
    return total_files, estimated_total_mb


def print_pipeline_summary(logger):
    """Stampa un riepilogo completo delle dimensioni dei file in tutta la pipeline."""
    print("\n" + "📊"*35)
    print(f"RIEPILOGO COMPLETO DIMENSIONI PIPELINE".center(70))
    print("📊"*35)
    
    # --- HUBBLE ---
    print("\n\n" + "="*70)
    print("🛰️  HUBBLE (LITH)")
    print("="*70)
    
    analyze_directory(INPUT_ORIG_HUBBLE, "Step 1: Input Originale (LITH)", recursive=True, scale_key=None, logger=logger)
    analyze_directory(INPUT_WCS_HUBBLE, "Step 2: Con WCS (da Step 1)", recursive=False, scale_key='WCS', logger=logger)
    analyze_directory(INPUT_HUBBLE, "Step 3: Registrate (da Step 3)", recursive=False, scale_key='NATIVESC', logger=logger)
    analyze_directory(OUTPUT_HUBBLE_PATCHES, "Step 4: Patches Finali (da Step 4)", recursive=False, scale_key='PIXSCALE', logger=logger)

    # --- OBSERVATORY ---
    print("\n\n" + "="*70)
    print("📡 OBSERVATORY")
    print("="*70)
    
    analyze_directory(INPUT_ORIG_OBSERVATORY, "Step 1: Input Originale (Osservatorio)", recursive=True, scale_key=None, logger=logger)
    analyze_directory(INPUT_WCS_OBSERVATORY, "Step 2: Con WCS (da Step 1)", recursive=False, scale_key='WCS', logger=logger)
    analyze_directory(INPUT_OBSERVATORY, "Step 3: Registrate (da Step 3)", recursive=False, scale_key='NATIVESC', logger=logger)
    analyze_directory(OUTPUT_OBS_PATCHES, "Step 4: Patches Finali (da Step 4)", recursive=False, scale_key='PIXSCALE', logger=logger)
    
    print("\n" + "📊"*35)
    print(f"FINE RIEPILOGO PIPELINE".center(70))
    print("📊"*35)

# ============================================================================
# FUNZIONI ESTRAZIONE PATCHES CON RISOLUZIONE NATIVA
# ============================================================================

def get_native_pixel_scale(header):
    """
    Estrae la risoluzione nativa dell'immagine dall'header.
    Returns: pixel_scale in arcsec/pixel
    """
    # Prima prova a leggere dal nostro metadato personalizzato
    if 'NATIVESC' in header:
        return header['NATIVESC']
    
    # Altrimenti calcola dal WCS
    try:
        wcs = WCS(header)
        if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
            cd = wcs.wcs.cd
            pixel_scale_deg = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
        elif hasattr(wcs.wcs, 'cdelt'):
            pixel_scale_deg = abs(wcs.wcs.cdelt[0])
        else:
            return None
        return pixel_scale_deg * 3600.0
    except:
        return None


def calculate_patch_size_for_fov(pixel_scale_arcsec, target_fov_arcmin):
    """
    Calcola la dimensione della patch in pixel per ottenere un FOV target.
    
    Args:
        pixel_scale_arcsec: Risoluzione in arcsec/pixel
        target_fov_arcmin: FOV desiderato in arcmin
        
    Returns:
        patch_size_px: Dimensione patch in pixel (arrotondata a multiplo di 8)
    """
    target_fov_arcsec = target_fov_arcmin * 60
    patch_size_px = int(target_fov_arcsec / pixel_scale_arcsec)
    
    # Arrotonda a multiplo di 8 per compatibilità con reti neurali
    patch_size_px = ((patch_size_px + 7) // 8) * 8
    
    # Limiti di sicurezza
    patch_size_px = max(64, min(2048, patch_size_px))
    
    return patch_size_px


def extract_patches_from_image(image_path, output_dir, source_name, logger):
    """
    Estrae patches da una singola immagine MANTENENDO LA RISOLUZIONE NATIVA.
    
    Differenza chiave: la dimensione in pixel viene calcolata per mantenere
    un FOV costante, non una dimensione in pixel costante.
    """
    try:
        with fits.open(image_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            
            if data is None:
                logger.warning(f"Nessun dato in {image_path.name}")
                return []
            
            # Se 3D, usa primo canale
            if len(data.shape) == 3:
                data = data[0]
            
            ny, nx = data.shape
            
            # Ottieni risoluzione nativa
            pixel_scale = get_native_pixel_scale(header)
            if pixel_scale is None:
                logger.warning(f"Impossibile determinare pixel scale per {image_path.name}")
                return []
            
            # Calcola dimensione patch per mantenere FOV costante
            patch_size = calculate_patch_size_for_fov(pixel_scale, TARGET_FOV_ARCMIN)
            
            # FOV effettivo
            actual_fov_arcsec = patch_size * pixel_scale
            actual_fov_arcmin = actual_fov_arcsec / 60
            
            logger.info(f"  {image_path.name}:")
            logger.info(f"    Risoluzione nativa: {pixel_scale:.4f}\"/px")
            logger.info(f"    Patch size: {patch_size}×{patch_size} px")
            logger.info(f"    FOV effettivo: {actual_fov_arcmin:.4f} arcmin")
            
            # WCS
            try:
                wcs = WCS(header)
                has_wcs = wcs.has_celestial
            except:
                has_wcs = False
                logger.warning(f"  WCS non valido per {image_path.name}")
            
            # Calcola step per overlap
            step = int(patch_size * (1 - OVERLAP_PERCENT / 100))
            if step <= 0: step = 1
            
            # Verifica se l'immagine è abbastanza grande
            if nx < patch_size or ny < patch_size:
                logger.warning(f"  Immagine troppo piccola: {nx}×{ny} < {patch_size}×{patch_size}")
                return []
            
            # Genera posizioni patches
            patches_info = []
            patch_id = 0
            
            for y in range(0, ny - patch_size + 1, step):
                for x in range(0, nx - patch_size + 1, step):
                    # Estrai patch
                    patch_data = data[y:y+patch_size, x:x+patch_size]
                    
                    # Verifica validità
                    valid_mask = np.isfinite(patch_data)
                    valid_percent = (np.sum(valid_mask) / patch_data.size) * 100
                    
                    if valid_percent < MIN_VALID_PERCENT:
                        continue  # Salta patch con troppi NaN
                    
                    # Coordinate centro patch
                    center_x = x + patch_size // 2
                    center_y = y + patch_size // 2
                    
                    # Coordinate celesti (se disponibili)
                    if has_wcs:
                        try:
                            center_world = wcs.pixel_to_world(center_x, center_y)
                            ra = center_world.ra.deg
                            dec = center_world.dec.deg
                        except:
                            ra = None
                            dec = None
                    else:
                        ra = None
                        dec = None
                    
                    # Statistiche patch
                    valid_data = patch_data[valid_mask]
                    if len(valid_data) == 0:
                        stats = {
                            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0
                        }
                    else:
                        stats = {
                            'mean': float(np.mean(valid_data)),
                            'std': float(np.std(valid_data)),
                            'min': float(np.min(valid_data)),
                            'max': float(np.max(valid_data)),
                            'median': float(np.median(valid_data)),
                        }
                    
                    # Nome file patch
                    image_stem = image_path.stem
                    patch_filename = f"{image_stem}_patch_{patch_id:04d}.fits"
                    patch_path = output_dir / patch_filename
                    
                    # Salva patch
                    patch_header = fits.Header()
                    patch_header['SOURCE'] = source_name
                    patch_header['ORIGFILE'] = image_path.name
                    patch_header['PATCHID'] = patch_id
                    patch_header['PATCHX'] = x
                    patch_header['PATCHY'] = y
                    patch_header['PATCHCX'] = center_x
                    patch_header['PATCHCY'] = center_y
                    patch_header['PATCHSZ'] = patch_size
                    patch_header['VALIDPCT'] = valid_percent
                    
                    # IMPORTANTE: Salva risoluzione nativa
                    patch_header['PIXSCALE'] = (pixel_scale, "Native pixel scale (arcsec/px)")
                    patch_header['FOVARCM'] = (actual_fov_arcmin, "Field of View (arcmin)")
                    patch_header['FOVARCS'] = (actual_fov_arcsec, "Field of View (arcsec)")
                    
                    if ra is not None:
                        patch_header['RA'] = ra
                        patch_header['DEC'] = dec
                    
                    # Copia WCS se disponibile
                    if has_wcs:
                        try:
                            # Aggiusta WCS per la patch
                            patch_wcs = wcs.deepcopy()
                            patch_wcs.wcs.crpix[0] -= x
                            patch_wcs.wcs.crpix[1] -= y
                            patch_header.update(patch_wcs.to_header())
                        except:
                            pass
                    
                    # Aggiungi statistiche
                    for key, val in stats.items():
                        patch_header[f'STAT_{key.upper()}'] = val
                    
                    # Salva
                    fits.PrimaryHDU(data=patch_data, header=patch_header).writeto(
                        patch_path, overwrite=True
                    )
                    
                    # Info per metadata
                    patch_info = {
                        'filename': patch_filename,
                        'patch_id': patch_id,
                        'position': {'x': x, 'y': y},
                        'center': {'x': center_x, 'y': center_y},
                        'patch_size_px': patch_size,
                        'pixel_scale_arcsec': pixel_scale,
                        'fov_arcmin': actual_fov_arcmin,
                        'valid_percent': valid_percent,
                        'stats': stats,
                    }
                    
                    if ra is not None:
                        patch_info['ra'] = ra
                        patch_info['dec'] = dec
                    
                    patches_info.append(patch_info)
                    patch_id += 1
            
            logger.info(f"  ✓ Estratte {len(patches_info)} patches da {image_path.name}")
            return patches_info
            
    except Exception as e:
        logger.error(f"Errore {image_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return []


def extract_all_patches(input_dir, output_dir, source_name, logger):
    """Estrae patches da tutte le immagini in una directory."""
    if not input_dir.exists():
        logger.warning(f"Directory non trovata: {input_dir}")
        return []
    
    fits_files = list(input_dir.glob('*.fits')) + list(input_dir.glob('*.fit'))
    
    if not fits_files:
        logger.warning(f"Nessun file in {input_dir}")
        return []
    
    print(f"\n📂 {source_name.upper()}: {len(fits_files)} immagini")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_patches = []
    
    with tqdm(total=len(fits_files), desc=f"  Estrazione {source_name}") as pbar:
        for image_path in fits_files:
            patches = extract_patches_from_image(image_path, output_dir, source_name, logger)
            all_patches.extend(patches)
            pbar.update(1)
    
    print(f"   ✓ Estratte {len(all_patches)} patches totali")
    
    # Salva metadata
    metadata_file = output_dir / f'{source_name}_patches_metadata.json'
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'source': source_name,
        'num_images': len(fits_files),
        'num_patches': len(all_patches),
        'target_fov_arcmin': TARGET_FOV_ARCMIN,
        'overlap_percent': OVERLAP_PERCENT,
        'min_valid_percent': MIN_VALID_PERCENT,
        'patches': all_patches
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata salvato: {metadata_file}")
    
    return all_patches


def create_dataset_split(patches_info, output_dir, logger):
    """Crea split train/val/test."""
    if not patches_info:
        return None
    
    n_total = len(patches_info)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    
    # Shuffle
    np.random.seed(42)
    indices = np.random.permutation(n_total)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    splits = {
        'train': [patches_info[i]['filename'] for i in train_idx],
        'val': [patches_info[i]['filename'] for i in val_idx],
        'test': [patches_info[i]['filename'] for i in test_idx],
    }
    
    # Salva
    for split_name, filenames in splits.items():
        split_file = output_dir / f'{split_name}_split.json'
        with open(split_file, 'w') as f:
            json.dump({
                'split': split_name,
                'count': len(filenames),
                'files': filenames
            }, f, indent=2)
    
    logger.info(f"Split creato: train={len(splits['train'])}, "
               f"val={len(splits['val'])}, test={len(splits['test'])}")
    
    return splits


def create_patch_pairs(hubble_patches, obs_patches, output_dir, logger):
    """Crea coppie Hubble-Observatory basate su coordinate."""
    if not hubble_patches or not obs_patches:
        return []
    
    pairs = []
    
    print(f"\n🔗 Creazione coppie Hubble-Observatory...")
    
    # Filtra patches con coordinate
    hubble_with_coords = [p for p in hubble_patches if 'ra' in p and 'dec' in p]
    obs_with_coords = [p for p in obs_patches if 'ra' in p and 'dec' in p]
    
    if not hubble_with_coords or not obs_with_coords:
        print(f"   ⚠️  Nessuna patch con coordinate WCS")
        return []
    
    # Match basato su distanza angolare
    threshold_arcmin = TARGET_FOV_ARCMIN * 0.5  # Metà del FOV
    
    for h_patch in tqdm(hubble_with_coords, desc="  Matching"):
        h_ra, h_dec = h_patch['ra'], h_patch['dec']
        
        best_match = None
        best_dist = float('inf')
        
        for o_patch in obs_with_coords:
            o_ra, o_dec = o_patch['ra'], o_patch['dec']
            
            # Distanza angolare approssimata
            delta_ra = (h_ra - o_ra) * np.cos(np.radians(h_dec))
            delta_dec = h_dec - o_dec
            dist_deg = np.sqrt(delta_ra**2 + delta_dec**2)
            dist_arcmin = dist_deg * 60
            
            if dist_arcmin < threshold_arcmin and dist_arcmin < best_dist:
                best_dist = dist_arcmin
                best_match = o_patch
        
        if best_match:
            pairs.append({
                'hubble': h_patch['filename'],
                'observatory': best_match['filename'],
                'hubble_ra': h_ra,
                'hubble_dec': h_dec,
                'obs_ra': best_match['ra'],
                'obs_dec': best_match['dec'],
                'separation_arcmin': best_dist
            })
    
    # Salva pairs
    pairs_file = output_dir / 'patch_pairs.json'
    with open(pairs_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_pairs': len(pairs),
            'threshold_arcmin': threshold_arcmin,
            'pairs': pairs
        }, f, indent=2)
    
    print(f"   ✓ {len(pairs)} coppie create")
    logger.info(f"Coppie Hubble-Observatory: {len(pairs)}")
    
    return pairs


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info(f"ESTRAZIONE PATCHES CON RISOLUZIONE NATIVA")
    logger.info("=" * 60)
    
    print("\n" + "✂️ "*35)
    print(f"STEP 4: ESTRAZIONE PATCHES (RISOLUZIONE NATIVA)".center(70))
    print("✂️ "*35)
    
    print(f"\n📋 CONFIGURAZIONE:")
    print(f"   FOV target: {TARGET_FOV_ARCMIN} arcmin (costante per tutte le patches)")
    print(f"   Overlap: {OVERLAP_PERCENT}%")
    print(f"   Validità minima: {MIN_VALID_PERCENT}%")
    print(f"   ⭐ La dimensione in pixel si adatta alla risoluzione di ogni immagine!")
    
    print(f"\n📂 INPUT (da Step 3):")
    print(f"   Hubble: {INPUT_HUBBLE}")
    print(f"   Observatory: {INPUT_OBSERVATORY}")
    
    print(f"\n📂 OUTPUT (Patches):")
    print(f"   {OUTPUT_DIR}")
    
    # Crea directories output
    OUTPUT_HUBBLE_PATCHES.mkdir(parents=True, exist_ok=True)
    OUTPUT_OBS_PATCHES.mkdir(parents=True, exist_ok=True)
    
    # Estrazione patches
    print(f"\n{'='*70}")
    print("ESTRAZIONE PATCHES")
    print(f"{'='*70}")
    
    # === BLOCCO DIMENSIONI INPUT RIMOSSO ===
    # Verrà mostrato nel riepilogo finale completo
    
    print(f"\n{'='*70}\n")
    
    hubble_patches = []
    obs_patches = []
    
    # Hubble
    if INPUT_HUBBLE.exists():
        hubble_patches = extract_all_patches(
            INPUT_HUBBLE,
            OUTPUT_HUBBLE_PATCHES,
            'hubble',
            logger
        )
    else:
        print(f"\n⚠️  Directory Hubble non trovata: {INPUT_HUBBLE}")
    
    # Observatory
    if INPUT_OBSERVATORY.exists():
        obs_patches = extract_all_patches(
            INPUT_OBSERVATORY,
            OUTPUT_OBS_PATCHES,
            'observatory',
            logger
        )
    else:
        print(f"\n⚠️  Directory Observatory non trovata: {INPUT_OBSERVATORY}")
    
    # Riepilogo estrazione
    print(f"\n{'='*70}")
    print("📊 RIEPILOGO ESTRAZIONE (Questo Step)")
    print(f"{'='*70}")
    print(f"\n   Hubble patches: {len(hubble_patches)}")
    print(f"   Observatory patches: {len(obs_patches)}")
    print(f"   TOTALE: {len(hubble_patches) + len(obs_patches)}")
    
    if hubble_patches:
        sizes = set([p['patch_size_px'] for p in hubble_patches])
        scales = set([round(p['pixel_scale_arcsec'], 4) for p in hubble_patches])
        print(f"\n   📊 Hubble:")
        print(f"      Dimensioni patches: {sorted(sizes)} px")
        print(f"      Risoluzioni: {sorted(scales)} \"/px")
    
    if obs_patches:
        sizes = set([p['patch_size_px'] for p in obs_patches])
        scales = set([round(p['pixel_scale_arcsec'], 4) for p in obs_patches])
        print(f"\n   📊 Observatory:")
        print(f"      Dimensioni patches: {sorted(sizes)} px")
        print(f"      Risoluzioni: {sorted(scales)} \"/px")
    
    logger.info(f"Patches estratte: Hubble={len(hubble_patches)}, Observatory={len(obs_patches)}")
    
    if len(hubble_patches) == 0 and len(obs_patches) == 0:
        print(f"\n❌ Nessuna patch estratta!")
        # Non esce, così può stampare il riepilogo della pipeline
    
    # Crea coppie
    if hubble_patches and obs_patches:
        pairs = create_patch_pairs(hubble_patches, obs_patches, OUTPUT_DIR, logger)
    else:
        pairs = []
    
    # Split dataset
    print(f"\n{'='*70}")
    print("SPLIT DATASET")
    print(f"{'='*70}")
    
    if hubble_patches:
        print(f"\n📊 Hubble:")
        hubble_splits = create_dataset_split(hubble_patches, OUTPUT_HUBBLE_PATCHES, logger)
        if hubble_splits:
            print(f"   Train: {len(hubble_splits['train'])}")
            print(f"   Val: {len(hubble_splits['val'])}")
            print(f"   Test: {len(hubble_splits['test'])}")
    
    if obs_patches:
        print(f"\n📊 Observatory:")
        obs_splits = create_dataset_split(obs_patches, OUTPUT_OBS_PATCHES, logger)
        if obs_splits:
            print(f"   Train: {len(obs_splits['train'])}")
            print(f"   Val: {len(obs_splits['val'])}")
            print(f"   Test: {len(obs_splits['test'])}")
    
    # Salva metadata generale
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'target': 'all',
        'target_fov_arcmin': TARGET_FOV_ARCMIN,
        'overlap_percent': OVERLAP_PERCENT,
        'min_valid_percent': MIN_VALID_PERCENT,
        'mode': 'native_resolution',
        'hubble': {
            'num_patches': len(hubble_patches),
            'output_dir': str(OUTPUT_HUBBLE_PATCHES),
        },
        'observatory': {
            'num_patches': len(obs_patches),
            'output_dir': str(OUTPUT_OBS_PATCHES),
        },
        'pairs': {
            'num_pairs': len(pairs) if pairs else 0,
        },
        'splits': {
            'train_ratio': TRAIN_RATIO,
            'val_ratio': VAL_RATIO,
            'test_ratio': TEST_RATIO,
        }
    }
    
    metadata_file = OUTPUT_DIR / 'dataset_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Metadata salvato: {metadata_file}")
    
    # Summary finale
    print(f"\n{'='*70}")
    print(f"✅ ESTRAZIONE COMPLETATA")
    print(f"{'='*70}")
    
    print(f"\n📁 OUTPUT DIRECTORY:")
    print(f"   {OUTPUT_DIR}")
    
    # ========================================================================
    # STATISTICHE DIMENSIONI DETTAGLIATE (RIMOSSO)
    # Sostituito con il riepilogo completo della pipeline
    # ========================================================================
    
    # ========================================================================
    # RIEPILOGO COMPLETO PIPELINE
    # ========================================================================
    print_pipeline_summary(logger)
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(f"⭐ VANTAGGI RISOLUZIONE NATIVA")
    print(f"{'='*70}")
    print(f"   • Hubble: massima qualità preservata")
    print(f"   • Observatory: nessun degrado dalla risoluzione originale")
    print(f"   • FOV costante: {TARGET_FOV_ARCMIN} arcmin per tutte le patches")
    print(f"   • Compatibilità training: stessa copertura celeste")
    
    print(f"\n➡️  PROSSIMI PASSI:")
    print(f"   1. Verifica patches in: {OUTPUT_DIR}")
    print(f"   2. Le patches hanno dimensioni diverse ma FOV identico")
    print(f"   3. Per il training, considera di ridimensionare tutte a dimensione comune")
    print(f"   4. Oppure usa architetture che accettano input di dimensioni variabili")
    
    logger.info("Estrazione patches completata")


if __name__ == "__main__":
    print(f"\n✂️  ESTRAZIONE PATCHES (RISOLUZIONE NATIVA)")
    print(f"FOV target: {TARGET_FOV_ARCMIN} arcmin (costante)")
    print(f"Dimensione in pixel: variabile (adattata alla risoluzione)\n")
    
    start_time = time.time()
    
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Tempo totale: {elapsed:.1f} secondi")