"""
STEP 3+4: ANALISI STATISTICHE + ESTRAZIONE PATCHES (CON MENU INTERATTIVO)
Analizza mosaici, calcola dimensioni patches ottimali, chiede conferma all'utente
e procede con l'estrazione fisica delle patches.

FLUSSO:
1. Selezione mosaico(i) da menu
2. Analisi statistiche per ogni dimensione patch
3. Calcolo dimensione ottimale
4. CONFERMA UTENTE per estrazione
5. Estrazione fisica patches con risoluzione nativa
6. Creazione split train/val/test
"""

import os
import sys
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

warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)

# ============================================================================
# CONFIGURAZIONE DINAMICA
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.isdir(os.path.join(SCRIPT_DIR, 'data')):
    PROJECT_ROOT = SCRIPT_DIR
elif os.path.isdir(os.path.join(os.path.dirname(SCRIPT_DIR), 'data')):
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    raise FileNotFoundError("Impossibile trovare la directory 'data'.")

BASE_DIR = os.path.join(PROJECT_ROOT, 'data')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# INPUT (mosaici dallo step precedente)
INPUT_MOSAIC_DIR = os.path.join(BASE_DIR, 'mosaics')

# OUTPUT
OUTPUT_ANALYSIS_DIR = os.path.join(BASE_DIR, 'patch_analysis')
OUTPUT_PATCHES_DIR = os.path.join(BASE_DIR, 'dataset_patches')

# Parametri analisi
PATCH_SIZES = [64, 128, 256, 512]  # Dimensioni patches da analizzare
MIN_VALID_PERCENTAGE = 80.0  # Minima percentuale pixel validi in patch

# Parametri estrazione
OVERLAP_PERCENT = 25  # Overlap tra patches vicine
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================================
# MENU INTERATTIVO
# ============================================================================

def list_available_mosaics():
    """Lista mosaici disponibili."""
    if not os.path.exists(INPUT_MOSAIC_DIR):
        return []
    
    mosaic_files = glob.glob(os.path.join(INPUT_MOSAIC_DIR, 'mosaic_*.fits'))
    
    mosaics_info = []
    for filepath in mosaic_files:
        try:
            with fits.open(filepath) as hdul:
                header = hdul[0].header
                data = hdul[0].data
                
                source = header.get('SOURCE', 'unknown')
                obj_name = header.get('OBJECT', 'unknown')
                nimages = header.get('NIMAGES', 0)
                coverage = header.get('COVERAGE', 0.0)
                pixel_scale = header.get('PIXSCALE', None)
                
                shape = data.shape
                size_mb = (data.nbytes) / (1024**2)
                
                mosaics_info.append({
                    'path': filepath,
                    'filename': os.path.basename(filepath),
                    'source': source,
                    'object': obj_name,
                    'nimages': nimages,
                    'coverage': coverage,
                    'pixel_scale': pixel_scale,
                    'shape': shape,
                    'size_mb': size_mb
                })
        except Exception as e:
            continue
    
    return sorted(mosaics_info, key=lambda x: (x['source'], x['object']))


def interactive_menu():
    """Menu interattivo per selezionare mosaico."""
    print("\n" + "=" * 70)
    print("📊 ANALISI + ESTRAZIONE PATCHES".center(70))
    print("=" * 70)
    
    mosaics = list_available_mosaics()
    
    if not mosaics:
        print("\n❌ Nessun mosaico trovato!")
        print(f"   Verifica che esistano mosaici in:")
        print(f"   {INPUT_MOSAIC_DIR}")
        print(f"\n   💡 Esegui prima: python scriptsintegrati/step2_croppedmosaico.py")
        return None
    
    # === Selezione Mosaico ===
    print("\n🖼️  MOSAICI DISPONIBILI:")
    print("-" * 70)
    print(f"{'#':<4} {'Source':<10} {'Object':<15} {'Imgs':<5} {'Shape':<15} {'Size':<10} {'Scale':<8} {'Cov%':<6}")
    print("-" * 70)
    
    for i, mosaic in enumerate(mosaics, 1):
        shape_str = f"{mosaic['shape'][1]}×{mosaic['shape'][0]}"
        size_str = f"{mosaic['size_mb']:.1f} MB"
        scale_str = f"{mosaic['pixel_scale']:.3f}\"" if mosaic['pixel_scale'] else "N/A"
        cov_str = f"{mosaic['coverage']:.1f}%"
        
        print(f"{i:<4} {mosaic['source']:<10} {mosaic['object']:<15} "
              f"{mosaic['nimages']:<5} {shape_str:<15} {size_str:<10} {scale_str:<8} {cov_str:<6}")
    
    print(f"{len(mosaics)+1:<4} TUTTI (processa tutti i mosaici)")
    print("-" * 70)
    
    # Input mosaico
    while True:
        try:
            choice = input(f"\n➤ Scegli mosaico [1-{len(mosaics)+1}]: ").strip()
            mosaic_idx = int(choice) - 1
            
            if mosaic_idx == len(mosaics):
                # TUTTI
                selected_mosaics = mosaics
                print(f"\n✓ Selezionati: TUTTI ({len(mosaics)} mosaici)")
                break
            elif 0 <= mosaic_idx < len(mosaics):
                selected_mosaics = [mosaics[mosaic_idx]]
                print(f"\n✓ Selezionato: {mosaics[mosaic_idx]['filename']}")
                break
            else:
                print(f"⚠️  Inserisci un numero tra 1 e {len(mosaics)+1}")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Operazione annullata.")
            return None
    
    # === RIEPILOGO ===
    print("\n" + "=" * 70)
    print("📋 RIEPILOGO")
    print("=" * 70)
    print(f"   Mosaici da processare: {len(selected_mosaics)}")
    print(f"   Dimensioni patches test: {PATCH_SIZES}")
    print(f"   Min valid %: {MIN_VALID_PERCENTAGE}%")
    print(f"   Overlap estrazione: {OVERLAP_PERCENT}%")
    print("=" * 70)
    
    confirm = input("\n➤ Confermi e procedi con ANALISI? [S/n]: ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ Operazione annullata.")
        return None
    
    return selected_mosaics


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(mosaic_info=None):
    """Setup logging."""
    if mosaic_info:
        log_subdir = os.path.join(LOG_DIR, 'patch_extraction', mosaic_info['source'], mosaic_info['object'])
        log_prefix = f"extraction_{mosaic_info['source']}_{mosaic_info['object']}"
    else:
        log_subdir = os.path.join(LOG_DIR, 'patch_extraction')
        log_prefix = "extraction_all"
    
    os.makedirs(log_subdir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_subdir, f'{log_prefix}_{timestamp}.log')
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"LOG FILE: {log_filename}")
    logger.info("=" * 80)
    
    return logger


# ============================================================================
# STEP 1: ANALISI STATISTICHE
# ============================================================================

def analyze_patches_statistics(data, patch_size, min_valid_pct, logger):
    """Analizza patch per una data dimensione (SOLO statistiche, NO estrazione)."""
    height, width = data.shape
    
    # Calcola numero di patches
    n_patches_y = height // patch_size
    n_patches_x = width // patch_size
    
    if n_patches_x == 0 or n_patches_y == 0:
        logger.warning(f"Mosaico troppo piccolo per patch {patch_size}×{patch_size}")
        return None
    
    total_patches = n_patches_y * n_patches_x
    valid_patches = 0
    partially_valid = 0
    empty_patches = 0
    
    valid_percentages = []
    
    # Analizza ogni patch
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size
            
            patch = data[y_start:y_end, x_start:x_end]
            
            # Conta pixel validi
            valid_mask = np.isfinite(patch) & (patch != 0)
            n_valid = valid_mask.sum()
            valid_pct = (n_valid / (patch_size * patch_size)) * 100
            
            valid_percentages.append(valid_pct)
            
            if valid_pct >= min_valid_pct:
                valid_patches += 1
            elif valid_pct > 0:
                partially_valid += 1
            else:
                empty_patches += 1
    
    return {
        'patch_size': patch_size,
        'total_patches': total_patches,
        'valid_patches': valid_patches,
        'partially_valid': partially_valid,
        'empty_patches': empty_patches,
        'valid_percentage_mean': np.mean(valid_percentages),
        'valid_percentage_median': np.median(valid_percentages),
        'valid_percentage_std': np.std(valid_percentages),
        'usable_percentage': (valid_patches / total_patches) * 100
    }


def analyze_mosaic_statistics(mosaic_info, logger):
    """Analizza singolo mosaico per tutte le dimensioni patch."""
    filepath = mosaic_info['path']
    filename = mosaic_info['filename']
    
    logger.info("=" * 80)
    logger.info(f"ANALISI MOSAICO: {filename}")
    logger.info("=" * 80)
    
    try:
        print(f"\n📊 Analisi: {filename}")
        
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            header = hdul[0].header
        
        logger.info(f"Shape: {data.shape}")
        logger.info(f"Source: {mosaic_info['source']}")
        logger.info(f"Object: {mosaic_info['object']}")
        logger.info(f"Coverage: {mosaic_info['coverage']:.1f}%")
        logger.info("")
        
        # Statistiche globali
        valid_pixels = np.isfinite(data) & (data != 0)
        n_valid = valid_pixels.sum()
        total_pixels = data.size
        
        global_stats = {
            'filename': filename,
            'source': mosaic_info['source'],
            'object': mosaic_info['object'],
            'shape': data.shape,
            'total_pixels': total_pixels,
            'valid_pixels': int(n_valid),
            'valid_percentage': (n_valid / total_pixels) * 100,
            'nimages': mosaic_info['nimages'],
            'coverage': mosaic_info['coverage'],
            'pixel_scale': mosaic_info['pixel_scale']
        }
        
        if n_valid > 0:
            valid_data = data[valid_pixels]
            global_stats.update({
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'mean': float(np.mean(valid_data)),
                'median': float(np.median(valid_data)),
                'std': float(np.std(valid_data))
            })
        
        logger.info("STATISTICHE GLOBALI:")
        logger.info(f"  Pixel validi: {n_valid:,} / {total_pixels:,} ({global_stats['valid_percentage']:.1f}%)")
        if n_valid > 0:
            logger.info(f"  Range: {global_stats['min']:.2e} - {global_stats['max']:.2e}")
            logger.info(f"  Media: {global_stats['mean']:.2e}")
        logger.info("")
        
        print(f"  ✓ Shape: {data.shape[1]}×{data.shape[0]}px")
        print(f"  ✓ Pixel validi: {global_stats['valid_percentage']:.1f}%")
        if mosaic_info['pixel_scale']:
            print(f"  ✓ Pixel scale: {mosaic_info['pixel_scale']:.4f}\"/px")
        
        # Analisi per ogni dimensione patch
        patch_results = []
        
        print(f"\n  Analisi patches:")
        for patch_size in PATCH_SIZES:
            result = analyze_patches_statistics(data, patch_size, MIN_VALID_PERCENTAGE, logger)
            
            if result:
                patch_results.append(result)
                
                logger.info(f"PATCH {patch_size}×{patch_size}:")
                logger.info(f"  Totali: {result['total_patches']:,}")
                logger.info(f"  Valide (≥{MIN_VALID_PERCENTAGE}%): {result['valid_patches']:,} ({result['usable_percentage']:.1f}%)")
                logger.info(f"  Parziali: {result['partially_valid']:,}")
                logger.info(f"  Vuote: {result['empty_patches']:,}")
                logger.info(f"  Valid% media: {result['valid_percentage_mean']:.1f}%")
                logger.info("")
                
                print(f"    {patch_size}×{patch_size}: {result['valid_patches']:,}/{result['total_patches']:,} "
                      f"({result['usable_percentage']:.1f}%) valide")
        
        # Salva risultati analisi JSON
        output_subdir = os.path.join(OUTPUT_ANALYSIS_DIR, mosaic_info['source'], mosaic_info['object'])
        os.makedirs(output_subdir, exist_ok=True)
        
        base_name = os.path.splitext(filename)[0]
        json_path = os.path.join(output_subdir, f"{base_name}_analysis.json")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'mosaic_file': filename,
            'global_stats': global_stats,
            'patch_analysis': patch_results,
            'parameters': {
                'patch_sizes': PATCH_SIZES,
                'min_valid_percentage': MIN_VALID_PERCENTAGE
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Analisi salvata: {json_path}")
        print(f"  ✓ Analisi salvata: {os.path.basename(json_path)}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Errore analisi {filename}: {e}")
        import traceback
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return None


def find_optimal_patch_size(analysis_results):
    """Determina dimensione patch ottimale basandosi sulle statistiche."""
    if not analysis_results or 'patch_analysis' not in analysis_results:
        return None
    
    patch_analysis = analysis_results['patch_analysis']
    
    # Trova dimensione con miglior compromesso (maggior numero patches valide)
    best = max(patch_analysis, key=lambda x: x['valid_patches'])
    
    return best


# ============================================================================
# STEP 2: ESTRAZIONE FISICA PATCHES
# ============================================================================

def extract_patches_from_mosaic(mosaic_info, patch_size, logger):
    """Estrae fisicamente le patches dal mosaico."""
    filepath = mosaic_info['path']
    filename = mosaic_info['filename']
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"ESTRAZIONE PATCHES: {filename}")
    logger.info(f"Dimensione: {patch_size}×{patch_size}px")
    logger.info("=" * 80)
    
    try:
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            header = hdul[0].header
        
        # WCS
        try:
            wcs = WCS(header)
            has_wcs = wcs.has_celestial
        except:
            has_wcs = False
            logger.warning("WCS non valido")
        
        height, width = data.shape
        
        # Calcola step per overlap
        step = int(patch_size * (1 - OVERLAP_PERCENT / 100))
        
        # Output directory
        output_subdir = os.path.join(
            OUTPUT_PATCHES_DIR,
            mosaic_info['source'],
            mosaic_info['object']
        )
        os.makedirs(output_subdir, exist_ok=True)
        
        patches_info = []
        patch_id = 0
        
        print(f"\n  Estrazione patches {patch_size}×{patch_size}px (overlap={OVERLAP_PERCENT}%)...")
        
        # Conta patches totali
        n_y = (height - patch_size) // step + 1
        n_x = (width - patch_size) // step + 1
        total_positions = n_y * n_x
        
        for y in tqdm(range(0, height - patch_size + 1, step), desc=f"  Estrazione", leave=False):
            for x in range(0, width - patch_size + 1, step):
                # Estrai patch
                patch_data = data[y:y+patch_size, x:x+patch_size]
                
                # Valida
                valid_mask = np.isfinite(patch_data) & (patch_data != 0)
                valid_pct = (valid_mask.sum() / patch_data.size) * 100
                
                if valid_pct < MIN_VALID_PERCENTAGE:
                    continue
                
                # Coordinate centro
                center_x = x + patch_size // 2
                center_y = y + patch_size // 2
                
                # Coordinate celesti
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
                
                # Header patch
                patch_header = fits.Header()
                patch_header['SOURCE'] = mosaic_info['source']
                patch_header['OBJECT'] = mosaic_info['object']
                patch_header['ORIGFILE'] = filename
                patch_header['PATCHID'] = patch_id
                patch_header['PATCHX'] = x
                patch_header['PATCHY'] = y
                patch_header['PATCHCX'] = center_x
                patch_header['PATCHCY'] = center_y
                patch_header['PATCHSZ'] = patch_size
                patch_header['VALIDPCT'] = valid_pct
                
                if mosaic_info['pixel_scale']:
                    patch_header['PIXSCALE'] = (mosaic_info['pixel_scale'], "Pixel scale (arcsec/px)")
                    fov_arcmin = (patch_size * mosaic_info['pixel_scale']) / 60.0
                    patch_header['FOVARCM'] = (fov_arcmin, "Field of View (arcmin)")
                
                if ra is not None:
                    patch_header['RA'] = ra
                    patch_header['DEC'] = dec
                
                # Statistiche
                if valid_mask.sum() > 0:
                    valid_data = patch_data[valid_mask]
                    patch_header['STATMEAN'] = float(np.mean(valid_data))
                    patch_header['STATSTD'] = float(np.std(valid_data))
                    patch_header['STATMIN'] = float(np.min(valid_data))
                    patch_header['STATMAX'] = float(np.max(valid_data))
                
                # Copia WCS se disponibile
                if has_wcs:
                    try:
                        patch_wcs = wcs.deepcopy()
                        patch_wcs.wcs.crpix[0] -= x
                        patch_wcs.wcs.crpix[1] -= y
                        patch_header.update(patch_wcs.to_header())
                    except:
                        pass
                
                # Salva patch
                mosaic_stem = os.path.splitext(filename)[0]
                patch_filename = f"{mosaic_stem}_patch_{patch_id:04d}.fits"
                patch_path = os.path.join(output_subdir, patch_filename)
                
                fits.PrimaryHDU(data=patch_data, header=patch_header).writeto(
                    patch_path, overwrite=True, output_verify='silentfix'
                )
                
                # Metadata
                patch_info = {
                    'filename': patch_filename,
                    'patch_id': patch_id,
                    'position': {'x': x, 'y': y},
                    'center': {'x': center_x, 'y': center_y},
                    'patch_size': patch_size,
                    'valid_percent': valid_pct
                }
                
                if ra is not None:
                    patch_info['ra'] = ra
                    patch_info['dec'] = dec
                
                patches_info.append(patch_info)
                patch_id += 1
        
        logger.info(f"✅ Estratte {len(patches_info)} patches valide")
        print(f"  ✓ Estratte: {len(patches_info)} patches")
        
        # Salva metadata
        metadata_file = os.path.join(output_subdir, f'{mosaic_info["object"]}_patches_metadata.json')
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'mosaic_file': filename,
            'source': mosaic_info['source'],
            'object': mosaic_info['object'],
            'patch_size': patch_size,
            'overlap_percent': OVERLAP_PERCENT,
            'min_valid_percent': MIN_VALID_PERCENTAGE,
            'num_patches': len(patches_info),
            'patches': patches_info
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return patches_info, output_subdir
        
    except Exception as e:
        logger.error(f"❌ Errore estrazione patches: {e}")
        import traceback
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return [], None


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
        split_file = os.path.join(output_dir, f'{split_name}_split.json')
        with open(split_file, 'w') as f:
            json.dump({
                'split': split_name,
                'count': len(filenames),
                'files': filenames
            }, f, indent=2)
    
    logger.info(f"Split creato: train={len(splits['train'])}, "
               f"val={len(splits['val'])}, test={len(splits['test'])}")
    
    print(f"  ✓ Split: train={len(splits['train'])}, "
          f"val={len(splits['val'])}, test={len(splits['test'])}")
    
    return splits


# ============================================================================
# MAIN
# ============================================================================

def process_single_mosaic(mosaic_info, logger):
    """Processa singolo mosaico: analisi + conferma + estrazione."""
    print("\n" + "=" * 70)
    print(f"📦 {mosaic_info['source'].upper()}/{mosaic_info['object']}")
    print("=" * 70)
    
    # === STEP 1: ANALISI STATISTICHE ===
    analysis_results = analyze_mosaic_statistics(mosaic_info, logger)
    
    if not analysis_results:
        print(f"\n⚠️  Errore analisi, skip mosaico")
        return None
    
    # === TROVA DIMENSIONE OTTIMALE ===
    optimal = find_optimal_patch_size(analysis_results)
    
    if not optimal:
        print(f"\n⚠️  Nessuna dimensione patch valida")
        return None
    
    print(f"\n  💡 RACCOMANDAZIONE:")
    print(f"     Dimensione ottimale: {optimal['patch_size']}×{optimal['patch_size']}px")
    print(f"     Patches valide stimate: {optimal['valid_patches']:,}")
    print(f"     Coverage: {optimal['usable_percentage']:.1f}%")
    
    # === MOSTRA TUTTE LE OPZIONI ===
    print(f"\n  📊 TUTTE LE OPZIONI:")
    for result in analysis_results['patch_analysis']:
        print(f"     {result['patch_size']:3d}×{result['patch_size']:3d}px: "
              f"{result['valid_patches']:5,} patches ({result['usable_percentage']:5.1f}%)")
    
    # === CONFERMA UTENTE ===
    print(f"\n  ❓ VUOI ESTRARRE LE PATCHES?")
    
    # Usa dimensione ottimale come default
    default_size = optimal['patch_size']
    
    choice = input(f"\n  ➤ Dimensione patch [{default_size}px] (o n per skip): ").strip()
    
    if choice.lower() in ['n', 'no', 'skip']:
        print(f"  ⏭️  Skip estrazione patches")
        return None
    
    # Parse scelta
    if choice == '':
        patch_size = default_size
    else:
        try:
            patch_size = int(choice)
            # Verifica che sia tra le opzioni analizzate
            valid_sizes = [r['patch_size'] for r in analysis_results['patch_analysis']]
            if patch_size not in valid_sizes:
                print(f"  ⚠️  Dimensione non valida, uso {default_size}px")
                patch_size = default_size
        except ValueError:
            print(f"  ⚠️  Input non valido, uso {default_size}px")
            patch_size = default_size
    
    print(f"\n  ✓ Procedo con estrazione {patch_size}×{patch_size}px")
    
    # === STEP 2: ESTRAZIONE PATCHES ===
    patches_info, output_dir = extract_patches_from_mosaic(mosaic_info, patch_size, logger)
    
    if not patches_info:
        print(f"  ⚠️  Nessuna patch estratta")
        return None
    
    # === STEP 3: SPLIT DATASET ===
    splits = create_dataset_split(patches_info, output_dir, logger)
    
    return {
        'mosaic': mosaic_info['filename'],
        'source': mosaic_info['source'],
        'object': mosaic_info['object'],
        'patch_size': patch_size,
        'num_patches': len(patches_info),
        'output_dir': output_dir,
        'splits': splits
    }


def main():
    """Funzione principale."""
    print("=" * 70)
    print("📊 ANALISI + ESTRAZIONE PATCHES".center(70))
    print("=" * 70)
    
    # Menu interattivo
    selected_mosaics = interactive_menu()
    
    if not selected_mosaics:
        return
    
    # Setup logging
    if len(selected_mosaics) == 1:
        logger = setup_logging(selected_mosaics[0])
    else:
        logger = setup_logging(None)
    
    logger.info("=" * 80)
    logger.info("ANALISI + ESTRAZIONE PATCHES")
    logger.info("=" * 80)
    logger.info(f"Mosaici da processare: {len(selected_mosaics)}")
    logger.info(f"Patch sizes test: {PATCH_SIZES}")
    logger.info(f"Min valid %: {MIN_VALID_PERCENTAGE}%")
    logger.info(f"Overlap: {OVERLAP_PERCENT}%")
    logger.info("")
    
    # Processa mosaici
    results_all = []
    
    for mosaic_info in selected_mosaics:
        result = process_single_mosaic(mosaic_info, logger)
        if result:
            results_all.append(result)
    
    # Riepilogo finale
    print("\n" + "=" * 70)
    print("📊 RIEPILOGO FINALE")
    print("=" * 70)
    print(f"\n   Mosaici processati: {len(results_all)}/{len(selected_mosaics)}")
    
    if results_all:
        total_patches = sum(r['num_patches'] for r in results_all)
        
        print(f"\n   📦 PATCHES ESTRATTE PER FONTE:")
        
        by_source = {}
        for result in results_all:
            source = result['source']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(result)
        
        for source, source_results in by_source.items():
            source_patches = sum(r['num_patches'] for r in source_results)
            print(f"      {source}: {source_patches:,} patches da {len(source_results)} mosaico(i)")
            for res in source_results:
                print(f"         • {res['object']}: {res['num_patches']:,} patches ({res['patch_size']}×{res['patch_size']}px)")
        
        print(f"\n   📊 TOTALE PATCHES: {total_patches:,}")
        
        print(f"\n✅ ESTRAZIONE COMPLETATA!")
        print(f"\n   📁 Output: {OUTPUT_PATCHES_DIR}/")
        print(f"   📄 Analisi: {OUTPUT_ANALYSIS_DIR}/")
        
        print(f"\n   💡 PROSSIMI PASSI:")
        print(f"      1. Verifica patches estratte")
        print(f"      2. Usa split train/val/test per training")
        print(f"      3. Patches mantengono risoluzione nativa del mosaico")
    else:
        print(f"\n⚠️  Nessuna patch estratta")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("PROCESSO COMPLETATO")
    logger.info("=" * 80)
    if results_all:
        logger.info(f"Patches estratte: {sum(r['num_patches'] for r in results_all)}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️ Tempo totale: {elapsed:.2f}s")