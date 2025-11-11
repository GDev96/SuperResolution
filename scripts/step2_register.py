"""
STEP 2: REGISTRAZIONE IMMAGINI (MIGLIORATO - MANTIENE RISOLUZIONE ORIGINALE)
Allinea le immagini di una fonte specifica (Hubble o Local) su un'unica griglia WCS.
MIGLIORIA: Ogni immagine mantiene la sua risoluzione nativa per massima qualità.
NUOVO: Menu interattivo per creazione mosaico opzionale.

Aggiunto: Menu interattivo per scegliere fonte e oggetto
"""

import os
import glob
import json
import time
import logging
import threading
from datetime import datetime
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

try:
    from reproject import reproject_interp
    from reproject.mosaicking import reproject_and_coadd
except ImportError:
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

# Input: cartelle con WCS (da Step 1)
INPUT_WCS_DIR = os.path.join(BASE_DIR, 'img_converted_wcs')

# Output: cartelle registrate
OUTPUT_REGISTERED_DIR = os.path.join(BASE_DIR, 'img_register')

# Output: mosaici
OUTPUT_MOSAIC_DIR = os.path.join(BASE_DIR, 'mosaics')

# Parametri Registrazione
NUM_THREADS = 7
REPROJECT_ORDER = 'bilinear'

log_lock = threading.Lock()

# ============================================================================
# MENU INTERATTIVO
# ============================================================================

def list_available_sources():
    """Lista le fonti disponibili (hubble, local)."""
    sources = []
    if os.path.exists(INPUT_WCS_DIR):
        for item in os.listdir(INPUT_WCS_DIR):
            item_path = os.path.join(INPUT_WCS_DIR, item)
            if os.path.isdir(item_path):
                sources.append(item)
    return sorted(sources)


def list_available_objects(source):
    """Lista gli oggetti disponibili per una fonte."""
    source_dir = os.path.join(INPUT_WCS_DIR, source)
    objects = []
    
    if os.path.exists(source_dir):
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            if os.path.isdir(item_path):
                # Conta file FITS nella cartella
                fits_count = len(glob.glob(os.path.join(item_path, '*.fits')) + 
                               glob.glob(os.path.join(item_path, '*.fit')))
                if fits_count > 0:
                    objects.append((item, fits_count))
    
    return sorted(objects)


def interactive_menu():
    """Menu interattivo per selezionare fonte e oggetto."""
    print("\n" + "=" * 70)
    print("🎯 SELEZIONE IMMAGINI DA REGISTRARE".center(70))
    print("=" * 70)
    
    # === STEP 1: Selezione Fonte ===
    sources = list_available_sources()
    
    if not sources:
        print("\n❌ Nessuna fonte trovata in data/img_converted_wcs/")
        print("   Esegui prima: python scripts/step1_convert_wcs.py")
        return None, None
    
    print("\n📂 FONTI DISPONIBILI:")
    print("-" * 70)
    for i, source in enumerate(sources, 1):
        # Conta oggetti per questa fonte
        objects = list_available_objects(source)
        obj_count = len(objects)
        total_images = sum(count for _, count in objects)
        print(f"   {i}. {source:<15} ({obj_count} oggetti, {total_images} immagini)")
    print("-" * 70)
    
    # Input fonte
    while True:
        try:
            choice = input(f"\n➤ Scegli fonte [1-{len(sources)}]: ").strip()
            source_idx = int(choice) - 1
            
            if 0 <= source_idx < len(sources):
                selected_source = sources[source_idx]
                break
            else:
                print(f"⚠️  Inserisci un numero tra 1 e {len(sources)}")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Operazione annullata.")
            return None, None
    
    print(f"\n✓ Fonte selezionata: {selected_source}")
    
    # === STEP 2: Selezione Oggetto ===
    objects = list_available_objects(selected_source)
    
    if not objects:
        print(f"\n❌ Nessun oggetto con immagini FITS in {selected_source}/")
        return None, None
    
    print(f"\n🎯 OGGETTI DISPONIBILI ({selected_source}):")
    print("-" * 70)
    for i, (obj_name, img_count) in enumerate(objects, 1):
        print(f"   {i}. {obj_name:<20} ({img_count} immagini)")
    print(f"   {len(objects)+1}. TUTTI (registra tutti gli oggetti)")
    print("-" * 70)
    
    # Input oggetto
    while True:
        try:
            choice = input(f"\n➤ Scegli oggetto [1-{len(objects)+1}]: ").strip()
            obj_idx = int(choice) - 1
            
            if obj_idx == len(objects):
                # Tutti gli oggetti
                selected_object = None
                print(f"\n✓ Registrerò TUTTI gli oggetti di {selected_source}")
                break
            elif 0 <= obj_idx < len(objects):
                selected_object = objects[obj_idx][0]
                img_count = objects[obj_idx][1]
                print(f"\n✓ Oggetto selezionato: {selected_object} ({img_count} immagini)")
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
    
    if selected_object:
        print(f"   Oggetto: {selected_object}")
        input_path = os.path.join(INPUT_WCS_DIR, selected_source, selected_object)
        output_path = os.path.join(OUTPUT_REGISTERED_DIR, selected_source, selected_object)
    else:
        print(f"   Oggetti: TUTTI ({len(objects)} oggetti)")
        input_path = os.path.join(INPUT_WCS_DIR, selected_source)
        output_path = os.path.join(OUTPUT_REGISTERED_DIR, selected_source)
    
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print("=" * 70)
    
    # Conferma
    confirm = input("\n➤ Confermi e procedi? [S/n]: ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ Operazione annullata.")
        return None, None
    
    return selected_source, selected_object


def ask_create_mosaic(num_images):
    """Chiede se creare il mosaico dopo la registrazione."""
    print("\n" + "=" * 70)
    print("🖼️  CREAZIONE MOSAICO".center(70))
    print("=" * 70)
    print(f"\n   Registrazione completata con successo!")
    print(f"   {num_images} immagini registrate disponibili.")
    print("\n   💡 Il mosaico combina tutte le immagini in un'unica immagine.")
    print("      • Usa la risoluzione migliore disponibile")
    print("      • Combina con metodo median (robusto)")
    print("      • Salva in: data/mosaics/")
    print("\n" + "=" * 70)
    
    while True:
        choice = input("\n➤ Vuoi creare il mosaico? [S/n]: ").strip().lower()
        if choice in ['s', 'si', 'sì', 'y', 'yes', '']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print("⚠️  Rispondi S (si) o N (no)")


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(source, object_name=None):
    """Setup logging per fonte/oggetto specifico."""
    if object_name:
        log_subdir = os.path.join(LOG_DIR, source, object_name)
        log_prefix = f"step2_register_{source}_{object_name}"
    else:
        log_subdir = os.path.join(LOG_DIR, source)
        log_prefix = f"step2_register_{source}_all"
    
    os.makedirs(log_subdir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_subdir, f'{log_prefix}_{timestamp}.log')
    
    # Clear existing handlers
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
# FUNZIONI ANALISI
# ============================================================================

def extract_wcs_info(filepath, logger):
    """Estrae info WCS da un file FITS."""
    try:
        with log_lock:
            logger.debug(f"📖 Lettura: {os.path.basename(filepath)}")
        
        with fits.open(filepath, mode='readonly') as hdul:
            header = hdul[0].header
            data = hdul[0].data
            
            if data is None:
                with log_lock:
                    logger.warning(f"⚠️  Nessun dato in {os.path.basename(filepath)}")
                return None
            
            wcs = WCS(header)
            if not wcs.has_celestial:
                with log_lock:
                    logger.warning(f"⚠️  WCS non valido in {os.path.basename(filepath)}")
                return None
            
            shape = data.shape
            center_ra, center_dec = wcs.wcs.crval
            
            # Pixel scale nativo
            try:
                cd = wcs.wcs.cd
                pixel_scale = np.sqrt(cd[0,0]**2 + cd[0,1]**2) * 3600  # arcsec/px
            except:
                pixel_scale = abs(wcs.wcs.cdelt[0]) * 3600
            
            with log_lock:
                logger.info(f"✓ {os.path.basename(filepath)}: {shape[1]}×{shape[0]}px, "
                          f"RA={center_ra:.4f}°, DEC={center_dec:.4f}°, scale={pixel_scale:.4f}\"/px")
            
            return {
                'file': filepath,
                'wcs': wcs,
                'shape': shape,
                'center_ra': center_ra,
                'center_dec': center_dec,
                'pixel_scale': pixel_scale
            }
    except Exception as e:
        with log_lock:
            logger.error(f"❌ Errore estraendo WCS da {os.path.basename(filepath)}: {e}")
        return None


def analyze_images(input_dir, source_name, logger):
    """Analizza tutte le immagini in una directory."""
    logger.info("=" * 80)
    logger.info(f"🔍 ANALISI IMMAGINI: {source_name}")
    logger.info("=" * 80)
    logger.info(f"Directory input: {input_dir}")
    
    fits_files = (glob.glob(os.path.join(input_dir, '*.fits')) + 
                  glob.glob(os.path.join(input_dir, '*.fit')))
    
    if not fits_files:
        logger.warning(f"⚠️  Nessun file FITS in {input_dir}")
        return []
    
    logger.info(f"📊 Trovati {len(fits_files)} file FITS")
    logger.info("")
    
    wcs_info_list = []
    for filepath in tqdm(fits_files, desc=f"  Analisi {source_name}", unit="file"):
        info = extract_wcs_info(filepath, logger)
        if info:
            wcs_info_list.append(info)
    
    logger.info("")
    logger.info(f"✅ {len(wcs_info_list)}/{len(fits_files)} immagini analizzate con successo")
    
    if len(wcs_info_list) != len(fits_files):
        failed = len(fits_files) - len(wcs_info_list)
        logger.warning(f"⚠️  {failed} immagini fallite")
    
    return wcs_info_list


def create_common_wcs_frame(wcs_info_list, logger):
    """Crea frame WCS comune che copre tutte le immagini."""
    if not wcs_info_list:
        return None
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🗺️  CREAZIONE FRAME WCS COMUNE")
    logger.info("=" * 80)
    
    # Trova boundaries
    ra_min, ra_max = float('inf'), float('-inf')
    dec_min, dec_max = float('inf'), float('-inf')
    pixel_scales = []
    
    logger.info(f"Analisi footprint di {len(wcs_info_list)} immagini...")
    
    for i, info in enumerate(wcs_info_list, 1):
        wcs = info['wcs']
        shape = info['shape']
        pixel_scales.append(info['pixel_scale'])
        
        # Corners dell'immagine
        height, width = shape
        corners_x = [0, width-1, width-1, 0]
        corners_y = [0, 0, height-1, height-1]
        
        try:
            ra_corners, dec_corners = wcs.all_pix2world(corners_x, corners_y, 0)
            
            img_ra_min, img_ra_max = np.min(ra_corners), np.max(ra_corners)
            img_dec_min, img_dec_max = np.min(dec_corners), np.max(dec_corners)
            
            logger.debug(f"  {i}. {os.path.basename(info['file'])}: "
                        f"RA=[{img_ra_min:.4f}°, {img_ra_max:.4f}°], "
                        f"DEC=[{img_dec_min:.4f}°, {img_dec_max:.4f}°]")
            
            ra_min = min(ra_min, img_ra_min)
            ra_max = max(ra_max, img_ra_max)
            dec_min = min(dec_min, img_dec_min)
            dec_max = max(dec_max, img_dec_max)
        except Exception as e:
            logger.warning(f"⚠️  Errore calcolo corners per {os.path.basename(info['file'])}: {e}")
            continue
    
    # Centro comune
    ra_center = (ra_min + ra_max) / 2.0
    dec_center = (dec_min + dec_max) / 2.0
    
    # Scala media (solo per riferimento)
    avg_scale = np.mean(pixel_scales)
    min_scale = np.min(pixel_scales)
    max_scale = np.max(pixel_scales)
    
    # Crea WCS comune
    common_wcs = WCS(naxis=2)
    common_wcs.wcs.crval = [ra_center, dec_center]
    common_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    common_wcs.wcs.cdelt = [avg_scale/3600, avg_scale/3600]  # gradi/px
    common_wcs.wcs.crpix = [1, 1]
    common_wcs.wcs.radesys = 'ICRS'
    common_wcs.wcs.equinox = 2000.0
    
    # Log info
    ra_span = (ra_max - ra_min) * 60  # arcmin
    dec_span = (dec_max - dec_min) * 60  # arcmin
    
    logger.info("")
    logger.info("📐 CAMPO COMUNE CALCOLATO:")
    logger.info(f"  RA:  {ra_min:.6f}° → {ra_max:.6f}° (span: {ra_span:.2f}')")
    logger.info(f"  DEC: {dec_min:.6f}° → {dec_max:.6f}° (span: {dec_span:.2f}')")
    logger.info(f"  Centro: RA={ra_center:.6f}°, DEC={dec_center:.6f}°")
    logger.info("")
    logger.info("📏 RISOLUZIONE NATIVA (preservata):")
    logger.info(f"  Media: {avg_scale:.4f}\"/px")
    logger.info(f"  Min:   {min_scale:.4f}\"/px")
    logger.info(f"  Max:   {max_scale:.4f}\"/px")
    logger.info(f"  Range: {max_scale - min_scale:.4f}\"/px")
    logger.info("")
    logger.info("✅ Frame WCS comune creato")
    
    return common_wcs


# ============================================================================
# REGISTRAZIONE CON RISOLUZIONE NATIVA
# ============================================================================

def reproject_image_native(img_info, common_wcs, output_dir, logger):
    """Reproietta immagine mantenendo risoluzione nativa."""
    filepath = img_info['file']
    filename = os.path.basename(filepath)
    
    try:
        with log_lock:
            logger.debug(f"🔄 Inizio reproiezione: {filename}")
        
        with fits.open(filepath, mode='readonly') as hdul:
            input_hdu = hdul[0]
            original_header = input_hdu.header.copy()
            original_wcs = WCS(original_header)
            data = input_hdu.data
            
            # Crea WCS target con risoluzione nativa
            native_scale = img_info['pixel_scale']  # arcsec/px
            native_scale_deg = native_scale / 3600.0
            
            target_wcs = WCS(naxis=2)
            target_wcs.wcs.crval = common_wcs.wcs.crval  # Stesso centro
            target_wcs.wcs.ctype = common_wcs.wcs.ctype
            target_wcs.wcs.cdelt = [-native_scale_deg, native_scale_deg]
            target_wcs.wcs.crpix = [1, 1]
            target_wcs.wcs.radesys = 'ICRS'
            target_wcs.wcs.equinox = 2000.0
            
            with log_lock:
                logger.debug(f"  WCS target creato con scala nativa: {native_scale:.4f}\"/px")
            
            # Calcola dimensioni canvas
            height, width = data.shape
            corners_x = [0, width-1, width-1, 0]
            corners_y = [0, 0, height-1, height-1]
            
            ra_corners, dec_corners = original_wcs.all_pix2world(corners_x, corners_y, 0)
            x_new, y_new = target_wcs.all_world2pix(ra_corners, dec_corners, 0)
            
            x_min, x_max = int(np.floor(np.min(x_new))), int(np.ceil(np.max(x_new)))
            y_min, y_max = int(np.floor(np.min(y_new))), int(np.ceil(np.max(y_new)))
            
            canvas_width = x_max - x_min + 1
            canvas_height = y_max - y_min + 1
            
            with log_lock:
                logger.debug(f"  Canvas calcolato: {canvas_width}×{canvas_height}px")
            
            # Aggiusta CRPIX per offset
            target_wcs.wcs.crpix = [-x_min + 1, -y_min + 1]
            
            with log_lock:
                logger.debug(f"  Inizio reproject_interp (order={REPROJECT_ORDER})...")
            
            # Reproiezione
            reprojected_data, footprint = reproject_interp(
                input_hdu,
                target_wcs,
                shape_out=(canvas_height, canvas_width),
                order=REPROJECT_ORDER
            )
            
            # Calcola copertura
            valid_pixels = np.sum(np.isfinite(reprojected_data))
            total_pixels = canvas_height * canvas_width
            coverage = (valid_pixels / total_pixels) * 100
            
            with log_lock:
                logger.debug(f"  Reproiezione completata. Coverage: {coverage:.1f}%")
            
            # Crea header output
            output_header = target_wcs.to_header()
            
            # Mantieni metadati importanti
            preserve_keys = ['DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 
                           'TELESCOP', 'OBJECT', 'TARGET', 'OBJCTRA', 'OBJCTDEC']
            for key in preserve_keys:
                if key in original_header:
                    output_header[key] = original_header[key]
            
            # Aggiungi metadati registrazione
            output_header['REGMTHD'] = 'reproject_interp_native'
            output_header['REGSRC'] = filename
            output_header['REGDATE'] = datetime.now().isoformat()
            output_header['REGCOVER'] = (coverage, 'Coverage percentage')
            output_header['REGVALID'] = (valid_pixels, 'Valid pixels')
            output_header['REGORD'] = REPROJECT_ORDER
            output_header['NATIVESC'] = (native_scale, 'Native pixel scale (arcsec/px)')
            output_header['ORIGSHP0'] = height
            output_header['ORIGSHP1'] = width
            output_header['NEWSHP0'] = canvas_height
            output_header['NEWSHP1'] = canvas_width
            
            # Salva
            output_filename = f"reg_{filename}"
            output_path = os.path.join(output_dir, output_filename)
            
            with log_lock:
                logger.debug(f"  Salvataggio: {output_filename}")
            
            fits.PrimaryHDU(data=reprojected_data, header=output_header).writeto(
                output_path,
                overwrite=True,
                output_verify='silentfix'
            )
            
            with log_lock:
                logger.info(f"✅ {filename} → {output_filename}")
                logger.info(f"   Original: {width}×{height}px @ {native_scale:.3f}\"/px")
                logger.info(f"   Registered: {canvas_width}×{canvas_height}px @ {native_scale:.3f}\"/px")
                logger.info(f"   Coverage: {coverage:.1f}% ({valid_pixels}/{total_pixels} px)")
                logger.info("")
            
            return True, filename, {
                'original_shape': (width, height),
                'registered_shape': (canvas_width, canvas_height),
                'pixel_scale': native_scale,
                'coverage': coverage,
                'valid_pixels': valid_pixels,
                'output_path': output_path
            }
            
    except Exception as e:
        with log_lock:
            logger.error(f"❌ {filename}: {e}")
            logger.debug(f"   Traceback:", exc_info=True)
        return False, filename, None


def register_images(wcs_info_list, common_wcs, output_dir, source_name, logger):
    """Registra immagini in parallelo."""
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"🔄 REGISTRAZIONE IMMAGINI: {source_name}")
    logger.info("=" * 80)
    logger.info(f"Numero immagini: {len(wcs_info_list)}")
    logger.info(f"Threads paralleli: {NUM_THREADS}")
    logger.info(f"Ordine interpolazione: {REPROJECT_ORDER}")
    logger.info(f"Directory output: {output_dir}")
    logger.info("")
    
    success_count = 0
    error_count = 0
    stats_list = []
    registered_files = []
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            executor.submit(reproject_image_native, img_info, common_wcs, output_dir, logger): img_info
            for img_info in wcs_info_list
        }
        
        with tqdm(total=len(futures), desc=f"  Registrazione {source_name}", unit="img") as pbar:
            for future in as_completed(futures):
                success, filename, stats = future.result()
                if success:
                    success_count += 1
                    if stats:
                        stats_list.append(stats)
                        registered_files.append(stats['output_path'])
                else:
                    error_count += 1
                pbar.update(1)
    
    # Statistiche finali
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 STATISTICHE REGISTRAZIONE")
    logger.info("=" * 80)
    logger.info(f"✅ Successi: {success_count}")
    logger.info(f"❌ Errori: {error_count}")
    logger.info(f"📈 Tasso successo: {success_count/(success_count+error_count)*100:.1f}%")
    
    if stats_list:
        coverages = [s['coverage'] for s in stats_list]
        pixel_scales = [s['pixel_scale'] for s in stats_list]
        
        logger.info("")
        logger.info("📏 RISOLUZIONE:")
        logger.info(f"  Media: {np.mean(pixel_scales):.4f}\"/px")
        logger.info(f"  Min: {np.min(pixel_scales):.4f}\"/px")
        logger.info(f"  Max: {np.max(pixel_scales):.4f}\"/px")
        
        logger.info("")
        logger.info("📐 COVERAGE:")
        logger.info(f"  Media: {np.mean(coverages):.1f}%")
        logger.info(f"  Min: {np.min(coverages):.1f}%")
        logger.info(f"  Max: {np.max(coverages):.1f}%")
    
    logger.info("")
    
    return success_count, error_count, registered_files


# ============================================================================
# CREAZIONE MOSAICO
# ============================================================================

def create_mosaic(registered_files, source_name, object_name, common_wcs, logger):
    """Crea mosaico dalle immagini registrate usando accumulatori (metodo robusto)."""
    if not registered_files:
        logger.warning("⚠️  Nessuna immagine registrata per creare mosaico")
        return None
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🖼️  CREAZIONE MOSAICO")
    logger.info("=" * 80)
    logger.info(f"Immagini da combinare: {len(registered_files)}")
    logger.info("")
    
    try:
        print(f"\n🖼️  Creazione mosaico da {len(registered_files)} immagini...")
        
        # ============================================================
        # STEP 1: ANALIZZA IMMAGINI E DETERMINA DIMENSIONI MOSAICO
        # ============================================================
        
        pixel_scales = []
        image_infos = []
        
        print("  Analisi immagini...")
        for filepath in tqdm(registered_files, desc="  Analisi", unit="img"):
            try:
                with fits.open(filepath) as hdul:
                    # Trova HDU con dati
                    data_hdu = None
                    for hdu in hdul:
                        if hdu.data is not None and len(hdu.data.shape) == 2:
                            data_hdu = hdu
                            break
                    
                    if data_hdu is None:
                        logger.warning(f"⚠️  Nessun HDU 2D in {os.path.basename(filepath)}")
                        continue
                    
                    header = data_hdu.header
                    
                    # Estrai pixel scale
                    native_scale = header.get('NATIVESC', None)
                    if native_scale:
                        pixel_scales.append(native_scale)
                    else:
                        wcs = WCS(header)
                        try:
                            cd = wcs.wcs.cd
                            scale = np.sqrt(cd[0,0]**2 + cd[0,1]**2) * 3600
                            pixel_scales.append(scale)
                        except:
                            pixel_scales.append(abs(wcs.wcs.cdelt[0]) * 3600)
                    
                    # Salva info per dopo
                    image_infos.append({
                        'path': filepath,
                        'wcs': WCS(header),
                        'shape': data_hdu.data.shape
                    })
                    
            except Exception as e:
                logger.warning(f"⚠️  Errore analisi {os.path.basename(filepath)}: {e}")
                continue
        
        if not image_infos:
            logger.error("❌ Nessuna immagine valida per il mosaico")
            return None
        
        # Usa risoluzione migliore
        best_scale = np.min(pixel_scales)
        logger.info(f"Risoluzione mosaico: {best_scale:.4f}\"/px (migliore disponibile)")
        print(f"  ✓ Risoluzione: {best_scale:.4f}\"/px")
        
        # ============================================================
        # STEP 2: CREA WCS MOSAICO E CALCOLA DIMENSIONI
        # ============================================================
        
        mosaic_wcs = WCS(naxis=2)
        mosaic_wcs.wcs.crval = common_wcs.wcs.crval
        mosaic_wcs.wcs.ctype = common_wcs.wcs.ctype
        mosaic_wcs.wcs.cdelt = [-best_scale/3600, best_scale/3600]
        mosaic_wcs.wcs.crpix = [1, 1]
        mosaic_wcs.wcs.radesys = 'ICRS'
        mosaic_wcs.wcs.equinox = 2000.0
        
        # Calcola boundaries totali
        logger.info("Calcolo footprint totale...")
        
        ra_min, ra_max = float('inf'), float('-inf')
        dec_min, dec_max = float('inf'), float('-inf')
        
        for info in image_infos:
            wcs = info['wcs']
            height, width = info['shape']
            
            corners_x = [0, width-1, width-1, 0]
            corners_y = [0, 0, height-1, height-1]
            
            try:
                ra_corners, dec_corners = wcs.all_pix2world(corners_x, corners_y, 0)
                ra_min = min(ra_min, np.min(ra_corners))
                ra_max = max(ra_max, np.max(ra_corners))
                dec_min = min(dec_min, np.min(dec_corners))
                dec_max = max(dec_max, np.max(dec_corners))
            except:
                continue
        
        # Converti boundaries in pixel del mosaico
        x_corners, y_corners = mosaic_wcs.all_world2pix([ra_min, ra_max], [dec_min, dec_max], 0)
        
        x_min = int(np.floor(np.min(x_corners)))
        x_max = int(np.ceil(np.max(x_corners)))
        y_min = int(np.floor(np.min(y_corners)))
        y_max = int(np.ceil(np.max(y_corners)))
        
        mosaic_width = x_max - x_min + 1
        mosaic_height = y_max - y_min + 1
        
        # Aggiusta CRPIX
        mosaic_wcs.wcs.crpix = [-x_min + 1, -y_min + 1]
        
        logger.info(f"Dimensioni mosaico: {mosaic_width}×{mosaic_height}px")
        logger.info(f"Campo: RA={ra_max-ra_min:.4f}°, DEC={dec_max-dec_min:.4f}°")
        print(f"  ✓ Dimensioni: {mosaic_width}×{mosaic_height}px")
        
        # Memoria stimata
        memory_gb = (mosaic_height * mosaic_width * 4 * 2) / (1024**3)
        logger.info(f"Memoria stimata: ~{memory_gb:.1f} GB")
        print(f"  ✓ Memoria: ~{memory_gb:.1f} GB")
        
        # ============================================================
        # STEP 3: COMBINA IMMAGINI CON ACCUMULATORI (METODO ROBUSTO)
        # ============================================================
        
        logger.info("")
        logger.info("Combinazione immagini con accumulatori...")
        print("\n  Combinazione immagini...")
        
        # Inizializza accumulatori
        sum_array = np.zeros((mosaic_height, mosaic_width), dtype=np.float64)
        count_array = np.zeros((mosaic_height, mosaic_width), dtype=np.int32)
        
        valid_count = 0
        
        for info in tqdm(image_infos, desc="  Combinazione", unit="img"):
            try:
                with fits.open(info['path']) as hdul:
                    # Trova HDU con dati
                    data_hdu = None
                    for hdu in hdul:
                        if hdu.data is not None and len(hdu.data.shape) == 2:
                            data_hdu = hdu
                            break
                    
                    if data_hdu is None:
                        continue
                    
                    data = data_hdu.data
                    wcs = info['wcs']
                    
                    # Reproietta su griglia mosaico
                    from reproject import reproject_interp
                    
                    reprojected, footprint = reproject_interp(
                        (data, wcs),
                        mosaic_wcs,
                        shape_out=(mosaic_height, mosaic_width),
                        order='bilinear'
                    )
                    
                    # Trova pixel validi
                    valid_mask = (
                        np.isfinite(reprojected) & 
                        (reprojected != 0) & 
                        (footprint > 0)
                    )
                    
                    n_valid = valid_mask.sum()
                    coverage = (n_valid / (mosaic_height * mosaic_width)) * 100
                    
                    if coverage < 0.01:
                        logger.warning(f"Coverage bassa ({coverage:.4f}%): {os.path.basename(info['path'])}")
                        continue
                    
                    # Accumula
                    sum_array[valid_mask] += reprojected[valid_mask]
                    count_array[valid_mask] += 1
                    
                    valid_count += 1
                    logger.debug(f"✓ {os.path.basename(info['path'])}: {n_valid:,} pixel ({coverage:.3f}%)")
                    
            except Exception as e:
                logger.error(f"Errore {os.path.basename(info['path'])}: {e}")
                continue
        
        if valid_count == 0:
            logger.error("Nessuna immagine processata con successo!")
            return None
        
        logger.info(f"Processate {valid_count}/{len(image_infos)} immagini")
        print(f"\n  ✓ Processate: {valid_count}/{len(image_infos)} immagini")
        
        # ============================================================
        # STEP 4: CALCOLA MOSAICO FINALE (MEDIA CON GESTIONE NaN)
        # ============================================================
        
        print(f"\n  Calcolo mosaico finale...")
        
        with np.errstate(divide='ignore', invalid='ignore'):
            mosaic_data = np.divide(
                sum_array,
                count_array,
                out=np.full((mosaic_height, mosaic_width), np.nan, dtype=np.float32),
                where=count_array > 0
            )
        
        # ============================================================
        # STEP 5: STATISTICHE
        # ============================================================
        
        valid_pixels = np.isfinite(mosaic_data) & (mosaic_data != 0)
        coverage_total = (valid_pixels.sum() / mosaic_data.size) * 100
        
        logger.info("")
        logger.info("📊 STATISTICHE MOSAICO:")
        logger.info(f"  Dimensioni: {mosaic_width}×{mosaic_height}px")
        logger.info(f"  Risoluzione: {best_scale:.4f}\"/px")
        logger.info(f"  Coverage: {coverage_total:.1f}%")
        
        if valid_pixels.sum() > 0:
            valid_data = mosaic_data[valid_pixels]
            logger.info(f"  Valore min: {np.nanmin(valid_data):.2e}")
            logger.info(f"  Valore max: {np.nanmax(valid_data):.2e}")
            logger.info(f"  Valore medio: {np.nanmean(valid_data):.2e}")
            
            print(f"\n  📊 Statistiche:")
            print(f"     Coverage: {coverage_total:.1f}%")
            print(f"     Min: {np.nanmin(valid_data):.2e}")
            print(f"     Max: {np.nanmax(valid_data):.2e}")
            print(f"     Media: {np.nanmean(valid_data):.2e}")
            
            # Distribuzione sovrapposizioni
            unique_counts = np.unique(count_array[count_array > 0])
            logger.info("")
            logger.info("  Distribuzione sovrapposizioni:")
            for count in sorted(unique_counts):
                n_pixels = (count_array == count).sum()
                percent = (n_pixels / valid_pixels.sum()) * 100
                logger.info(f"    {count} immagini: {n_pixels:,} px ({percent:.1f}%)")
        
        # ============================================================
        # STEP 6: NORMALIZZAZIONE (opzionale - preserva dinamica)
        # ============================================================
        
        # Converti NaN a 0 per compatibilità
        mosaic_data[~valid_pixels] = 0.0
        
        # ============================================================
        # STEP 7: SALVATAGGIO
        # ============================================================
        
        mosaic_header = mosaic_wcs.to_header()
        mosaic_header['OBJECT'] = object_name
        mosaic_header['SOURCE'] = source_name
        mosaic_header['NIMAGES'] = (valid_count, 'Number of images combined')
        mosaic_header['COMBMETH'] = ('mean_accumulator', 'Combination method')
        mosaic_header['PIXSCALE'] = (best_scale, 'Pixel scale (arcsec/px)')
        mosaic_header['COVERAGE'] = (coverage_total, 'Coverage percentage')
        mosaic_header['MOSDATE'] = datetime.now().isoformat()
        mosaic_header['CREATOR'] = 'step2_register.py'
        
        # Nome file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mosaic_filename = f"mosaic_{source_name}_{object_name}_{valid_count}img_{timestamp}.fits"
        
        os.makedirs(OUTPUT_MOSAIC_DIR, exist_ok=True)
        mosaic_path = os.path.join(OUTPUT_MOSAIC_DIR, mosaic_filename)
        
        logger.info("")
        logger.info(f"💾 Salvataggio: {mosaic_filename}")
        
        fits.PrimaryHDU(data=mosaic_data, header=mosaic_header).writeto(
            mosaic_path,
            overwrite=True,
            output_verify='silentfix'
        )
        
        logger.info(f"✅ Mosaico salvato: {mosaic_path}")
        
        print(f"\n✅ Mosaico creato: {mosaic_filename}")
        print(f"   Dimensioni: {mosaic_width}×{mosaic_height}px @ {best_scale:.3f}\"/px")
        print(f"   Coverage: {coverage_total:.1f}%")
        print(f"   Immagini: {valid_count}")
        print(f"   Path: {mosaic_path}")
        
        return mosaic_path
        
    except Exception as e:
        logger.error(f"❌ Errore creando mosaico: {e}")
        import traceback
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return None

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Funzione principale."""
    print("=" * 70)
    print("🔄 STEP 3: REGISTRAZIONE IMMAGINI (Risoluzione Nativa)".center(70))
    print("=" * 70)
    
    # Menu interattivo
    selected_source, selected_object = interactive_menu()
    
    if not selected_source:
        print("\n❌ Nessuna selezione effettuata.")
        return
    
    # Setup logging
    logger = setup_logging(selected_source, selected_object)
    logger.info("=" * 80)
    logger.info(f"STEP 3: REGISTRAZIONE - {selected_source}" + 
               (f"/{selected_object}" if selected_object else " (TUTTI)"))
    logger.info("=" * 80)
    logger.info(f"Data/Ora inizio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Fonte: {selected_source}")
    logger.info(f"Oggetto: {selected_object if selected_object else 'TUTTI'}")
    logger.info("")
    
    # Inizializza variabili per riepilogo finale
    mosaic_path = None
    mosaics_created = []
    
    # Determina percorsi
    if selected_object:
        # Singolo oggetto
        input_dir = os.path.join(INPUT_WCS_DIR, selected_source, selected_object)
        output_dir = os.path.join(OUTPUT_REGISTERED_DIR, selected_source, selected_object)
        
        logger.info(f"📂 PERCORSI:")
        logger.info(f"  Input:  {input_dir}")
        logger.info(f"  Output: {output_dir}")
        logger.info("")
        
        # Analizza immagini
        wcs_info_list = analyze_images(input_dir, f"{selected_source}/{selected_object}", logger)
        
        if not wcs_info_list:
            logger.error("❌ Nessuna immagine valida trovata!")
            return
        
        # Crea WCS comune
        common_wcs = create_common_wcs_frame(wcs_info_list, logger)
        
        if not common_wcs:
            logger.error("❌ Impossibile creare WCS comune!")
            return
        
        # Registra
        success, errors, registered_files = register_images(
            wcs_info_list, common_wcs, output_dir, 
            f"{selected_source}/{selected_object}", logger
        )
        
        # FIXME: Chiedi se creare mosaico
        if registered_files and ask_create_mosaic(len(registered_files)):
            mosaic_path = create_mosaic(
                registered_files, 
                selected_source, 
                selected_object, 
                common_wcs, 
                logger
            )
        else:
            print("\n⏭️  Creazione mosaico saltata.")
            logger.info("⏭️  Creazione mosaico saltata dall'utente.")
        
        # Riepilogo
        print("\n" + "=" * 70)
        print("📊 RIEPILOGO")
        print("=" * 70)
        print(f"   Fonte: {selected_source}")
        print(f"   Oggetto: {selected_object}")
        print(f"   Immagini registrate: {success}/{success+errors}")
        print(f"   Output: {output_dir}")
        if mosaic_path:
            print(f"   Mosaico: {os.path.basename(mosaic_path)}")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ REGISTRAZIONE COMPLETATA")
        logger.info("=" * 80)
        logger.info(f"Immagini registrate: {success}/{success+errors}")
        logger.info(f"Output salvato in: {output_dir}")
        if mosaic_path:
            logger.info(f"Mosaico salvato: {mosaic_path}")
        
    else:
        # TUTTI gli oggetti
        objects = list_available_objects(selected_source)
        total_success = 0
        total_errors = 0
        
        logger.info(f"📦 Registrazione di {len(objects)} oggetti...")
        logger.info("")
        
        print(f"\n🔄 Registrazione di {len(objects)} oggetti...")
        
        # Chiedi una sola volta per tutti
        create_mosaics = ask_create_mosaic(sum(count for _, count in objects))
        
        for obj_idx, (obj_name, img_count) in enumerate(objects, 1):
            print(f"\n{'─'*70}")
            print(f"📦 Oggetto {obj_idx}/{len(objects)}: {obj_name} ({img_count} immagini)")
            print(f"{'─'*70}")
            
            logger.info("=" * 80)
            logger.info(f"OGGETTO {obj_idx}/{len(objects)}: {obj_name}")
            logger.info("=" * 80)
            
            input_dir = os.path.join(INPUT_WCS_DIR, selected_source, obj_name)
            output_dir = os.path.join(OUTPUT_REGISTERED_DIR, selected_source, obj_name)
            
            logger.info(f"Input:  {input_dir}")
            logger.info(f"Output: {output_dir}")
            logger.info("")
            
            # Analizza
            wcs_info_list = analyze_images(input_dir, f"{selected_source}/{obj_name}", logger)
            
            if not wcs_info_list:
                logger.warning(f"⚠️  Nessuna immagine valida per {obj_name}")
                logger.info("")
                continue
            
            # WCS comune
            common_wcs = create_common_wcs_frame(wcs_info_list, logger)
            
            if not common_wcs:
                logger.warning(f"⚠️  Impossibile creare WCS comune per {obj_name}")
                logger.info("")
                continue
            
            # Registra
            success, errors, registered_files = register_images(
                wcs_info_list, common_wcs, output_dir,
                f"{selected_source}/{obj_name}", logger
            )
            
            total_success += success
            total_errors += errors
            
            # Crea mosaico se richiesto
            if create_mosaics and registered_files:
                mosaic_path_temp = create_mosaic(
                    registered_files, selected_source, obj_name, common_wcs, logger
                )
                if mosaic_path_temp:
                    mosaics_created.append(os.path.basename(mosaic_path_temp))
            
            logger.info(f"Oggetto {obj_name}: {success} OK, {errors} errori")
            logger.info("")
        
        # Riepilogo finale
        print("\n" + "=" * 70)
        print("📊 RIEPILOGO TOTALE")
        print("=" * 70)
        print(f"   Fonte: {selected_source}")
        print(f"   Oggetti processati: {len(objects)}")
        print(f"   Immagini registrate: {total_success}/{total_success+total_errors}")
        print(f"   Output: {OUTPUT_REGISTERED_DIR}/{selected_source}/")
        if mosaics_created:
            print(f"   Mosaici creati: {len(mosaics_created)}")
            print(f"   Salvati in: {OUTPUT_MOSAIC_DIR}/")
        
        logger.info("=" * 80)
        logger.info("✅ REGISTRAZIONE BATCH COMPLETATA")
        logger.info("=" * 80)
        logger.info(f"Oggetti processati: {len(objects)}")
        logger.info(f"Totale immagini registrate: {total_success}/{total_success+total_errors}")
        logger.info(f"Output in: {OUTPUT_REGISTERED_DIR}/{selected_source}/")
        if mosaics_created:
            logger.info(f"Mosaici creati: {len(mosaics_created)}")
            logger.info(f"Salvati in: {OUTPUT_MOSAIC_DIR}/")
    
    logger.info("")
    logger.info(f"Data/Ora fine: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    print(f"\n✅ STEP 3 COMPLETATO!")
    print(f"\n   📁 Immagini registrate: {OUTPUT_REGISTERED_DIR}/{selected_source}/")
    if mosaic_path or mosaics_created:
        print(f"   🖼️  Mosaici: {OUTPUT_MOSAIC_DIR}/")
    print(f"\n   ➡️  Prossimo passo: python scripts/step4_patch.py")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️  Tempo totale: {elapsed:.2f}s")