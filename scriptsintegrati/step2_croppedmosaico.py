"""
STEP 2: CROP + MOSAICO (CON MENU INTERATTIVO)
Croppa le immagini registrate e crea mosaico finale.
Aggiunto: Menu interattivo per scegliere fonte e oggetto.
"""

import os
import sys
import glob
import time
import logging
import gc
from datetime import datetime
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)

try:
    from reproject import reproject_interp
    REPROJECT_AVAILABLE = True
except ImportError:
    REPROJECT_AVAILABLE = False
    print("❌ Libreria 'reproject' non trovata!")
    print("   Installa con: pip install reproject")
    exit(1)

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

# INPUT (immagini registrate dallo step precedente)
INPUT_LOCAL_REG = os.path.join(BASE_DIR, 'img_register', 'local')
INPUT_HUBBLE_REG = os.path.join(BASE_DIR, 'img_register', 'hubble')

# OUTPUT
OUTPUT_CROPPED = os.path.join(BASE_DIR, 'img_cropped')
OUTPUT_MOSAIC = os.path.join(BASE_DIR, 'mosaics')

# Parametri crop automatico
CROP_THRESHOLD = 0.01  # Soglia percentile per crop automatico
MIN_VALID_PIXELS = 100  # Minimo pixel validi per crop

# ============================================================================
# MENU INTERATTIVO
# ============================================================================

def list_available_sources():
    """Lista fonti disponibili con oggetti."""
    sources = {}
    
    # Local
    if os.path.exists(INPUT_LOCAL_REG):
        objects_local = []
        for item in os.listdir(INPUT_LOCAL_REG):
            item_path = os.path.join(INPUT_LOCAL_REG, item)
            if os.path.isdir(item_path):
                fits_count = len(glob.glob(os.path.join(item_path, 'reg_*.fits')))
                if fits_count > 0:
                    objects_local.append((item, fits_count))
        if objects_local:
            sources['local'] = objects_local
    
    # Hubble
    if os.path.exists(INPUT_HUBBLE_REG):
        objects_hubble = []
        for item in os.listdir(INPUT_HUBBLE_REG):
            item_path = os.path.join(INPUT_HUBBLE_REG, item)
            if os.path.isdir(item_path):
                fits_count = len(glob.glob(os.path.join(item_path, 'reg_*.fits')))
                if fits_count > 0:
                    objects_hubble.append((item, fits_count))
        if objects_hubble:
            sources['hubble'] = objects_hubble
    
    return sources


def interactive_menu():
    """Menu interattivo per selezionare fonte e oggetto."""
    print("\n" + "=" * 70)
    print("✂️  CROP + MOSAICO".center(70))
    print("=" * 70)
    
    sources = list_available_sources()
    
    if not sources:
        print("\n❌ Nessuna fonte trovata!")
        print(f"   Verifica che esistano immagini registrate in:")
        print(f"   - {INPUT_LOCAL_REG}")
        print(f"   - {INPUT_HUBBLE_REG}")
        print(f"\n   💡 Esegui prima: python scriptsintegrati/step1_wcsregister.py")
        return None, None
    
    # === STEP 1: Selezione Fonte ===
    print("\n📂 FONTI DISPONIBILI:")
    print("-" * 70)
    
    source_list = []
    idx = 1
    for source_name, objects in sources.items():
        total_imgs = sum(count for _, count in objects)
        print(f"   {idx}. {source_name:<15} ({len(objects)} oggetti, {total_imgs} immagini)")
        source_list.append(source_name)
        idx += 1
    
    print(f"   {idx}. TUTTE (processa tutte le fonti)")
    print("-" * 70)
    
    # Input fonte
    while True:
        try:
            choice = input(f"\n➤ Scegli fonte [1-{idx}]: ").strip()
            choice_idx = int(choice) - 1
            
            if choice_idx == idx - 1:
                selected_source = 'all'
                break
            elif 0 <= choice_idx < len(source_list):
                selected_source = source_list[choice_idx]
                break
            else:
                print(f"⚠️  Inserisci un numero tra 1 e {idx}")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Operazione annullata.")
            return None, None
    
    if selected_source == 'all':
        print(f"\n✓ Fonte selezionata: TUTTE")
        return 'all', None
    
    print(f"\n✓ Fonte selezionata: {selected_source}")
    
    # === STEP 2: Selezione Oggetto ===
    objects = sources[selected_source]
    
    print(f"\n🎯 OGGETTI DISPONIBILI ({selected_source}):")
    print("-" * 70)
    for i, (obj_name, img_count) in enumerate(objects, 1):
        print(f"   {i}. {obj_name:<20} ({img_count} immagini)")
    print(f"   {len(objects)+1}. TUTTI (tutti gli oggetti di {selected_source})")
    print("-" * 70)
    
    # Input oggetto
    while True:
        try:
            choice = input(f"\n➤ Scegli oggetto [1-{len(objects)+1}]: ").strip()
            obj_idx = int(choice) - 1
            
            if obj_idx == len(objects):
                selected_object = None
                print(f"\n✓ Oggetto: TUTTI")
                break
            elif 0 <= obj_idx < len(objects):
                selected_object = objects[obj_idx][0]
                print(f"\n✓ Oggetto: {selected_object}")
                break
            else:
                print(f"⚠️  Inserisci un numero tra 1 e {len(objects)+1}")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Operazione annullata.")
            return None, None
    
    # === RIEPILOGO ===
    print("\n" + "=" * 70)
    print("📋 RIEPILOGO SELEZIONE")
    print("=" * 70)
    print(f"   Fonte: {selected_source}")
    print(f"   Oggetto: {selected_object if selected_object else 'TUTTI'}")
    print("=" * 70)
    
    confirm = input("\n➤ Confermi e procedi? [S/n]: ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ Operazione annullata.")
        return None, None
    
    return selected_source, selected_object


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(source, object_name=None):
    """Setup logging per fonte/oggetto specifico."""
    if object_name:
        log_subdir = os.path.join(LOG_DIR, 'crop_mosaic', source, object_name)
        log_prefix = f"crop_mosaic_{source}_{object_name}"
    else:
        log_subdir = os.path.join(LOG_DIR, 'crop_mosaic', source)
        log_prefix = f"crop_mosaic_{source}_all"
    
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
# FUNZIONI CROP
# ============================================================================

def calculate_crop_bounds(data, threshold_percentile=CROP_THRESHOLD):
    """Calcola boundaries ottimali per crop automatico."""
    valid_mask = np.isfinite(data) & (data != 0)
    
    if not valid_mask.any():
        return None
    
    # Trova righe/colonne con dati validi
    rows_with_data = np.any(valid_mask, axis=1)
    cols_with_data = np.any(valid_mask, axis=0)
    
    # Trova boundaries
    row_indices = np.where(rows_with_data)[0]
    col_indices = np.where(cols_with_data)[0]
    
    if len(row_indices) == 0 or len(col_indices) == 0:
        return None
    
    y_min, y_max = row_indices[0], row_indices[-1]
    x_min, x_max = col_indices[0], col_indices[-1]
    
    # Aggiungi piccolo margine (5 pixel)
    margin = 5
    y_min = max(0, y_min - margin)
    y_max = min(data.shape[0] - 1, y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(data.shape[1] - 1, x_max + margin)
    
    return (y_min, y_max, x_min, x_max)


def crop_image(input_file, output_file, logger):
    """Croppa immagine rimuovendo bordi vuoti."""
    try:
        filename = os.path.basename(input_file)
        
        with fits.open(input_file) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()
            
            if data is None:
                logger.warning(f"Nessun dato in {filename}")
                return False
            
            original_shape = data.shape
            
            # Calcola crop bounds
            bounds = calculate_crop_bounds(data)
            
            if bounds is None:
                logger.warning(f"Impossibile calcolare crop per {filename}")
                return False
            
            y_min, y_max, x_min, x_max = bounds
            
            # Croppa
            cropped_data = data[y_min:y_max+1, x_min:x_max+1]
            
            valid_pixels = np.sum(np.isfinite(cropped_data) & (cropped_data != 0))
            
            if valid_pixels < MIN_VALID_PIXELS:
                logger.warning(f"Troppo pochi pixel validi dopo crop: {filename}")
                return False
            
            # Aggiorna WCS se presente
            try:
                wcs = WCS(header)
                if wcs.has_celestial:
                    # Aggiorna CRPIX per il crop
                    wcs.wcs.crpix[0] -= x_min
                    wcs.wcs.crpix[1] -= y_min
                    
                    # Aggiorna header
                    wcs_header = wcs.to_header()
                    for key in wcs_header:
                        header[key] = wcs_header[key]
            except:
                pass
            
            # Aggiorna NAXIS
            header['NAXIS1'] = cropped_data.shape[1]
            header['NAXIS2'] = cropped_data.shape[0]
            
            # Metadati crop
            header['CROPPED'] = True
            header['CROPDATE'] = datetime.now().isoformat()
            header['ORIGSHP1'] = original_shape[1]
            header['ORIGSHP2'] = original_shape[0]
            header['CROPX0'] = x_min
            header['CROPY0'] = y_min
            
            # Salva
            fits.PrimaryHDU(data=cropped_data, header=header).writeto(
                output_file, overwrite=True, output_verify='silentfix'
            )
            
            reduction = (1 - (cropped_data.size / data.size)) * 100
            
            logger.info(f"✓ {filename}: {original_shape} → {cropped_data.shape} (-{reduction:.1f}%)")
            
            return True
            
    except Exception as e:
        logger.error(f"Errore crop {os.path.basename(input_file)}: {e}")
        return False


def crop_images_for_object(source, object_name, logger):
    """Croppa tutte le immagini di un oggetto."""
    logger.info("=" * 80)
    logger.info(f"CROP IMMAGINI: {source}/{object_name}")
    logger.info("=" * 80)
    
    # Percorsi
    if source == 'local':
        input_dir = os.path.join(INPUT_LOCAL_REG, object_name)
    else:  # hubble
        input_dir = os.path.join(INPUT_HUBBLE_REG, object_name)
    
    output_dir = os.path.join(OUTPUT_CROPPED, source, object_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova immagini registrate
    fits_files = glob.glob(os.path.join(input_dir, 'reg_*.fits'))
    
    if not fits_files:
        logger.warning(f"Nessuna immagine registrata in {input_dir}")
        return 0, 0, []
    
    logger.info(f"Trovati {len(fits_files)} file")
    
    success_count = 0
    error_count = 0
    cropped_files = []
    
    print(f"\n✂️  Crop: {source}/{object_name} ({len(fits_files)} immagini)")
    
    for input_file in tqdm(fits_files, desc=f"  {object_name}", unit="img"):
        basename = os.path.basename(input_file)
        name = basename.replace('reg_', 'cropped_')
        output_file = os.path.join(output_dir, name)
        
        if crop_image(input_file, output_file, logger):
            success_count += 1
            cropped_files.append(output_file)
        else:
            error_count += 1
    
    logger.info(f"Croppate: {success_count}/{len(fits_files)}")
    
    return success_count, error_count, cropped_files


# ============================================================================
# FUNZIONI MOSAICO
# ============================================================================

def create_mosaic_from_cropped(cropped_files, source, object_name, logger):
    """Crea mosaico da immagini croppate (metodo accumulatori)."""
    if not cropped_files:
        logger.warning("Nessuna immagine croppata per mosaico")
        return None
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"CREAZIONE MOSAICO: {source}/{object_name}")
    logger.info("=" * 80)
    logger.info(f"Immagini da combinare: {len(cropped_files)}")
    
    try:
        print(f"\n🖼️  Creazione mosaico: {source}/{object_name}")
        
        # ============================================================
        # STEP 1: Determina dimensioni canvas comune
        # ============================================================
        
        print("  Analisi dimensioni canvas...")
        
        ra_min, ra_max = float('inf'), float('-inf')
        dec_min, dec_max = float('inf'), float('-inf')
        pixel_scales = []
        
        for filepath in cropped_files:
            try:
                with fits.open(filepath) as hdul:
                    header = hdul[0].header
                    wcs = WCS(header)
                    
                    if not wcs.has_celestial:
                        continue
                    
                    # Pixel scale
                    try:
                        cd = wcs.wcs.cd
                        scale = np.sqrt(cd[0,0]**2 + cd[0,1]**2) * 3600
                        pixel_scales.append(scale)
                    except:
                        scale = abs(wcs.wcs.cdelt[0]) * 3600
                        pixel_scales.append(scale)
                    
                    # Footprint
                    shape = hdul[0].data.shape
                    height, width = shape
                    corners_x = [0, width-1, width-1, 0]
                    corners_y = [0, 0, height-1, height-1]
                    
                    ra_corners, dec_corners = wcs.all_pix2world(corners_x, corners_y, 0)
                    
                    ra_min = min(ra_min, np.nanmin(ra_corners))
                    ra_max = max(ra_max, np.nanmax(ra_corners))
                    dec_min = min(dec_min, np.nanmin(dec_corners))
                    dec_max = max(dec_max, np.nanmax(dec_corners))
                    
            except Exception as e:
                logger.warning(f"Errore analisi {os.path.basename(filepath)}: {e}")
                continue
        
        if not pixel_scales:
            logger.error("Nessuna immagine valida per mosaico")
            return None
        
        best_scale = np.min(pixel_scales)
        best_scale_deg = best_scale / 3600.0
        
        logger.info(f"Risoluzione mosaico: {best_scale:.4f}\"/px")
        logger.info(f"Campo: RA={ra_max-ra_min:.4f}°, DEC={dec_max-dec_min:.4f}°")
        
        # ============================================================
        # STEP 2: Crea WCS mosaico
        # ============================================================
        
        ra_center = (ra_min + ra_max) / 2.0
        dec_center = (dec_min + dec_max) / 2.0
        
        mosaic_wcs = WCS(naxis=2)
        mosaic_wcs.wcs.crval = [ra_center, dec_center]
        mosaic_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        mosaic_wcs.wcs.cdelt = [-best_scale_deg, best_scale_deg]
        mosaic_wcs.wcs.crpix = [1, 1]
        mosaic_wcs.wcs.radesys = 'ICRS'
        mosaic_wcs.wcs.equinox = 2000.0
        
        # Calcola dimensioni
        x_corners, y_corners = mosaic_wcs.all_world2pix([ra_min, ra_max], [dec_min, dec_max], 0)
        
        x_min = int(np.floor(np.min(x_corners)))
        x_max = int(np.ceil(np.max(x_corners)))
        y_min = int(np.floor(np.min(y_corners)))
        y_max = int(np.ceil(np.max(y_corners)))
        
        mosaic_width = x_max - x_min + 1
        mosaic_height = y_max - y_min + 1
        
        mosaic_wcs.wcs.crpix = [-x_min + 1, -y_min + 1]
        
        logger.info(f"Dimensioni mosaico: {mosaic_width}×{mosaic_height}px")
        
        memory_gb = (mosaic_height * mosaic_width * 4 * 2) / (1024**3)
        logger.info(f"Memoria stimata: ~{memory_gb:.1f} GB")
        print(f"  ✓ Canvas: {mosaic_width}×{mosaic_height}px (~{memory_gb:.1f} GB)")
        
        # ============================================================
        # STEP 3: Accumula immagini (metodo robusto)
        # ============================================================
        
        print(f"  Combinazione immagini...")
        
        sum_array = np.zeros((mosaic_height, mosaic_width), dtype=np.float64)
        count_array = np.zeros((mosaic_height, mosaic_width), dtype=np.int32)
        
        valid_count = 0
        
        for filepath in tqdm(cropped_files, desc="  Combinazione", unit="img"):
            try:
                with fits.open(filepath) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                    wcs = WCS(header)
                    
                    if not wcs.has_celestial:
                        continue
                    
                    # Reproietta
                    reprojected, footprint = reproject_interp(
                        (data, wcs),
                        mosaic_wcs,
                        shape_out=(mosaic_height, mosaic_width),
                        order='bilinear'
                    )
                    
                    # Maschera validi
                    valid_mask = (
                        np.isfinite(reprojected) & 
                        (reprojected != 0) & 
                        (footprint > 0)
                    )
                    
                    n_valid = valid_mask.sum()
                    coverage = (n_valid / (mosaic_height * mosaic_width)) * 100
                    
                    if coverage < 0.01:
                        logger.warning(f"Coverage bassa ({coverage:.4f}%): {os.path.basename(filepath)}")
                        continue
                    
                    # Accumula
                    sum_array[valid_mask] += reprojected[valid_mask]
                    count_array[valid_mask] += 1
                    
                    valid_count += 1
                    logger.debug(f"✓ {os.path.basename(filepath)}: {n_valid:,} px ({coverage:.3f}%)")
                    
            except Exception as e:
                logger.error(f"Errore {os.path.basename(filepath)}: {e}")
                continue
            
            # Garbage collection periodico
            if (valid_count % 5) == 0:
                gc.collect()
        
        if valid_count == 0:
            logger.error("Nessuna immagine processata con successo")
            return None
        
        logger.info(f"Processate {valid_count}/{len(cropped_files)} immagini")
        
        # ============================================================
        # STEP 4: Calcola mosaico finale (media)
        # ============================================================
        
        print(f"  Calcolo finale...")
        
        with np.errstate(divide='ignore', invalid='ignore'):
            mosaic_data = np.divide(
                sum_array,
                count_array,
                out=np.full((mosaic_height, mosaic_width), np.nan, dtype=np.float32),
                where=count_array > 0
            )
        
        # Statistiche
        valid_pixels = np.isfinite(mosaic_data) & (mosaic_data != 0)
        coverage_total = (valid_pixels.sum() / mosaic_data.size) * 100
        
        logger.info("")
        logger.info("📊 STATISTICHE MOSAICO:")
        logger.info(f"  Dimensioni: {mosaic_width}×{mosaic_height}px")
        logger.info(f"  Risoluzione: {best_scale:.4f}\"/px")
        logger.info(f"  Coverage: {coverage_total:.1f}%")
        logger.info(f"  Immagini combinate: {valid_count}")
        
        if valid_pixels.sum() > 0:
            valid_data = mosaic_data[valid_pixels]
            logger.info(f"  Min: {np.nanmin(valid_data):.2e}")
            logger.info(f"  Max: {np.nanmax(valid_data):.2e}")
            logger.info(f"  Media: {np.nanmean(valid_data):.2e}")
            
            print(f"\n  📊 Statistiche:")
            print(f"     Coverage: {coverage_total:.1f}%")
            print(f"     Immagini: {valid_count}")
            print(f"     Range: {np.nanmin(valid_data):.2e} - {np.nanmax(valid_data):.2e}")
        
        # Converti NaN a 0
        mosaic_data[~valid_pixels] = 0.0
        
        # ============================================================
        # STEP 5: Salvataggio
        # ============================================================
        
        mosaic_header = mosaic_wcs.to_header()
        mosaic_header['OBJECT'] = object_name
        mosaic_header['SOURCE'] = source
        mosaic_header['NIMAGES'] = (valid_count, 'Number of images combined')
        mosaic_header['COMBMETH'] = ('mean_accumulator', 'Combination method')
        mosaic_header['PIXSCALE'] = (best_scale, 'Pixel scale (arcsec/px)')
        mosaic_header['COVERAGE'] = (coverage_total, 'Coverage percentage')
        mosaic_header['MOSDATE'] = datetime.now().isoformat()
        mosaic_header['CREATOR'] = 'step2_croppedmosaico.py'
        
        os.makedirs(OUTPUT_MOSAIC, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mosaic_filename = f"mosaic_{source}_{object_name}_{valid_count}img_{timestamp}.fits"
        mosaic_path = os.path.join(OUTPUT_MOSAIC, mosaic_filename)
        
        logger.info("")
        logger.info(f"💾 Salvataggio: {mosaic_filename}")
        
        fits.PrimaryHDU(data=mosaic_data, header=mosaic_header).writeto(
            mosaic_path, overwrite=True, output_verify='silentfix'
        )
        
        logger.info(f"✅ Mosaico salvato: {mosaic_path}")
        
        print(f"\n✅ Mosaico creato: {mosaic_filename}")
        print(f"   Path: {mosaic_path}")
        
        return mosaic_path
        
    except Exception as e:
        logger.error(f"❌ Errore creazione mosaico: {e}")
        import traceback
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return None


# ============================================================================
# MAIN
# ============================================================================

def process_single_object(source, object_name, logger):
    """Processa singolo oggetto (Crop + Mosaico)."""
    print("\n" + "=" * 70)
    print(f"📦 {source.upper()}/{object_name}")
    print("=" * 70)
    
    # STEP 1: Crop
    logger.info("")
    logger.info("STEP 1: CROP IMMAGINI")
    crop_success, crop_error, cropped_files = crop_images_for_object(source, object_name, logger)
    
    print(f"\n   ✓ Crop: {crop_success} successi, {crop_error} errori")
    
    if crop_success == 0:
        logger.warning(f"Nessuna immagine croppata per {object_name}, skip mosaico")
        return crop_success, crop_error, None
    
    # STEP 2: Mosaico
    logger.info("")
    logger.info("STEP 2: CREAZIONE MOSAICO")
    mosaic_path = create_mosaic_from_cropped(cropped_files, source, object_name, logger)
    
    if mosaic_path:
        print(f"   ✓ Mosaico salvato")
    else:
        print(f"   ⚠️  Errore creazione mosaico")
    
    return crop_success, crop_error, mosaic_path


def main():
    """Funzione principale."""
    print("=" * 70)
    print("✂️  CROP + MOSAICO".center(70))
    print("=" * 70)
    
    # Menu interattivo
    selected_source, selected_object = interactive_menu()
    
    if not selected_source:
        return
    
    # Setup logging
    logger = setup_logging(selected_source, selected_object)
    
    logger.info("=" * 80)
    logger.info(f"PIPELINE CROP+MOSAICO: {selected_source}" + 
               (f"/{selected_object}" if selected_object else " (TUTTI)"))
    logger.info("=" * 80)
    logger.info(f"Crop threshold: {CROP_THRESHOLD}")
    logger.info(f"Min valid pixels: {MIN_VALID_PIXELS}")
    logger.info("")
    
    total_crop_success = 0
    total_crop_error = 0
    mosaics_created = []
    
    # Determina cosa processare
    if selected_source == 'all':
        # TUTTE le fonti
        sources_data = list_available_sources()
        
        for source_name, objects in sources_data.items():
            for obj_name, _ in objects:
                crop_s, crop_e, mosaic_path = process_single_object(source_name, obj_name, logger)
                total_crop_success += crop_s
                total_crop_error += crop_e
                if mosaic_path:
                    mosaics_created.append(os.path.basename(mosaic_path))
    
    elif selected_object is None:
        # Tutti gli oggetti di una fonte
        sources_data = list_available_sources()
        objects = sources_data.get(selected_source, [])
        
        for obj_name, _ in objects:
            crop_s, crop_e, mosaic_path = process_single_object(selected_source, obj_name, logger)
            total_crop_success += crop_s
            total_crop_error += crop_e
            if mosaic_path:
                mosaics_created.append(os.path.basename(mosaic_path))
    
    else:
        # Singolo oggetto
        crop_s, crop_e, mosaic_path = process_single_object(selected_source, selected_object, logger)
        total_crop_success += crop_s
        total_crop_error += crop_e
        if mosaic_path:
            mosaics_created.append(os.path.basename(mosaic_path))
    
    # Riepilogo
    print("\n" + "=" * 70)
    print("📊 RIEPILOGO TOTALE")
    print("=" * 70)
    print(f"\n   STEP 1 (Crop):")
    print(f"      Successi: {total_crop_success}")
    print(f"      Errori: {total_crop_error}")
    print(f"\n   STEP 2 (Mosaico):")
    print(f"      Mosaici creati: {len(mosaics_created)}")
    
    if mosaics_created:
        print(f"\n✅ PIPELINE COMPLETATA!")
        print(f"\n   📁 Output:")
        print(f"      Immagini croppate: {OUTPUT_CROPPED}")
        print(f"      Mosaici: {OUTPUT_MOSAIC}")
        print(f"\n   🖼️  Mosaici creati:")
        for mosaic in mosaics_created:
            print(f"      - {mosaic}")
        print(f"\n   ➡️  Prossimo: python scriptsintegrati/step3_analizzapatch.py")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETATA")
    logger.info("=" * 80)
    logger.info(f"Crop: {total_crop_success} successi, {total_crop_error} errori")
    logger.info(f"Mosaici: {len(mosaics_created)} creati")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️ Tempo totale: {elapsed:.2f}s")