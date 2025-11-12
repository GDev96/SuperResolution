"""
STEP 1+2: CONVERSIONE WCS + REGISTRAZIONE (CON MENU INTERATTIVO)
Combina conversione coordinate → WCS e registrazione in un'unica pipeline.
Aggiunto: Menu interattivo per scegliere fonte e oggetto.
"""

import os
import sys
import glob
import time
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# INPUT
INPUT_LOCAL_RAW = os.path.join(BASE_DIR, 'local_raw')      # Immagini osservatorio
INPUT_HUBBLE = os.path.join(BASE_DIR, 'img_lights_1')      # Immagini Hubble

# OUTPUT STEP 1 (WCS)
OUTPUT_LOCAL_WCS = os.path.join(BASE_DIR, 'img_converted_wcs', 'local')
OUTPUT_HUBBLE_WCS = os.path.join(BASE_DIR, 'img_converted_wcs', 'hubble')

# OUTPUT STEP 2 (REGISTRATE)
OUTPUT_LOCAL_REG = os.path.join(BASE_DIR, 'img_register', 'local')
OUTPUT_HUBBLE_REG = os.path.join(BASE_DIR, 'img_register', 'hubble')

# Parametri registrazione
NUM_THREADS = 7
REPROJECT_ORDER = 'bilinear'

log_lock = threading.Lock()

# ============================================================================
# MENU INTERATTIVO
# ============================================================================

def list_available_sources():
    """Lista fonti disponibili con numero di oggetti."""
    sources = {}
    
    # Local
    if os.path.exists(INPUT_LOCAL_RAW):
        objects_local = []
        for item in os.listdir(INPUT_LOCAL_RAW):
            item_path = os.path.join(INPUT_LOCAL_RAW, item)
            if os.path.isdir(item_path):
                fits_count = len(glob.glob(os.path.join(item_path, '**', '*.fit*'), recursive=True))
                if fits_count > 0:
                    objects_local.append((item, fits_count))
        if objects_local:
            sources['local'] = objects_local
    
    # Hubble
    if os.path.exists(INPUT_HUBBLE):
        objects_hubble = []
        for item in os.listdir(INPUT_HUBBLE):
            item_path = os.path.join(INPUT_HUBBLE, item)
            if os.path.isdir(item_path):
                fits_count = len(glob.glob(os.path.join(item_path, '**', '*.fit*'), recursive=True))
                if fits_count > 0:
                    objects_hubble.append((item, fits_count))
        if objects_hubble:
            sources['hubble'] = objects_hubble
    
    return sources


def interactive_menu():
    """Menu interattivo per selezionare fonte e oggetto."""
    print("\n" + "=" * 70)
    print("🎯 CONVERSIONE WCS + REGISTRAZIONE".center(70))
    print("=" * 70)
    
    sources = list_available_sources()
    
    if not sources:
        print("\n❌ Nessuna fonte trovata!")
        print(f"   Verifica che esistano:")
        print(f"   - {INPUT_LOCAL_RAW}")
        print(f"   - {INPUT_HUBBLE}")
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
                # TUTTE
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
                # TUTTI
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
        log_subdir = os.path.join(LOG_DIR, 'wcs_register', source, object_name)
        log_prefix = f"wcs_register_{source}_{object_name}"
    else:
        log_subdir = os.path.join(LOG_DIR, 'wcs_register', source)
        log_prefix = f"wcs_register_{source}_all"
    
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
# STEP 1: CONVERSIONE WCS
# ============================================================================

def parse_coordinates(ra_str, dec_str):
    """Converte coordinate sessagesimali in decimali."""
    try:
        ra_str = ra_str.strip()
        dec_str = dec_str.strip()
        
        coord_str = f"{ra_str} {dec_str}"
        coord = SkyCoord(coord_str, unit=(u.hourangle, u.deg))
        
        return coord.ra.degree, coord.dec.degree
        
    except Exception:
        try:
            ra_parts = ra_str.split()
            dec_parts = dec_str.split()
            
            h, m, s = float(ra_parts[0]), float(ra_parts[1]), float(ra_parts[2])
            ra_deg = (h + m/60.0 + s/3600.0) * 15.0
            
            d, m, s = float(dec_parts[0]), float(dec_parts[1]), float(dec_parts[2])
            sign = 1 if d >= 0 else -1
            dec_deg = d + sign * (m/60.0 + s/3600.0)
            
            return ra_deg, dec_deg
            
        except Exception as e:
            raise ValueError(f"Impossibile parsare coordinate: RA='{ra_str}', DEC='{dec_str}': {e}")


def calculate_pixel_scale(header):
    """Calcola pixel scale da header."""
    xpixsz = header.get('XPIXSZ', None)
    focal = header.get('FOCALLEN', header.get('FOCAL', None))
    xbin = header.get('XBINNING', 1)
    
    if xpixsz and focal:
        pixel_size_mm = (xpixsz * xbin) / 1000.0
        pixel_scale_arcsec = 206.265 * pixel_size_mm / focal
        pixel_scale_deg = pixel_scale_arcsec / 3600.0
        return pixel_scale_deg
    
    return 1.5 / 3600.0


def create_wcs_from_header(header, data_shape):
    """Crea WCS da OBJCTRA/OBJCTDEC."""
    try:
        objctra = header.get('OBJCTRA', None)
        objctdec = header.get('OBJCTDEC', None)
        
        if not objctra or not objctdec:
            return None
        
        ra_deg, dec_deg = parse_coordinates(objctra, objctdec)
        pixel_scale = calculate_pixel_scale(header)
        
        wcs = WCS(naxis=2)
        height, width = data_shape
        wcs.wcs.crpix = [width / 2.0, height / 2.0]
        wcs.wcs.crval = [ra_deg, dec_deg]
        wcs.wcs.cdelt = [-pixel_scale, pixel_scale]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.radesys = 'ICRS'
        wcs.wcs.equinox = 2000.0
        
        return wcs
        
    except Exception:
        return None


def add_wcs_to_local_file(input_file, output_file, logger):
    """Aggiunge WCS a file local (con OBJCTRA/OBJCTDEC)."""
    try:
        filename = os.path.basename(input_file)
        
        with fits.open(input_file, mode='readonly') as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()
            
            if data is None:
                logger.warning(f"Nessun dato in {filename}")
                return False
            
            existing_wcs = WCS(header)
            if existing_wcs.has_celestial:
                logger.info(f"✓ {filename}: WCS già presente")
                hdul.writeto(output_file, overwrite=True)
                return True
            
            wcs = create_wcs_from_header(header, data.shape)
            
            if wcs is None:
                logger.warning(f"Impossibile creare WCS per {filename}")
                return False
            
            wcs_header = wcs.to_header()
            new_header = fits.Header()
            
            important_keys = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
                            'BZERO', 'BSCALE', 'DATE-OBS', 'EXPTIME', 'FILTER',
                            'INSTRUME', 'TELESCOP', 'XBINNING', 'YBINNING',
                            'XPIXSZ', 'YPIXSZ', 'GAIN', 'CCD-TEMP', 'FOCALLEN']
            
            for key in important_keys:
                if key in header:
                    new_header[key] = header[key]
            
            new_header.update(wcs_header)
            new_header['WCSADDED'] = True
            new_header['WCSSRC'] = 'OBJCTRA/OBJCTDEC conversion'
            new_header['WCSDATE'] = datetime.now().isoformat()
            
            fits.PrimaryHDU(data=data, header=new_header).writeto(
                output_file, overwrite=True, output_verify='silentfix'
            )
            
            logger.info(f"✓ {filename}: WCS aggiunto")
            return True
            
    except Exception as e:
        logger.error(f"Errore {os.path.basename(input_file)}: {e}")
        return False


def extract_hubble_wcs(input_file, output_file, logger):
    """Estrae WCS da file Hubble."""
    try:
        filename = os.path.basename(input_file)
        
        with fits.open(input_file) as hdul:
            sci_hdu = None
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    try:
                        wcs = WCS(hdu.header)
                        if wcs.has_celestial:
                            sci_hdu = hdu
                            break
                    except:
                        continue
            
            if sci_hdu is None:
                logger.warning(f"Nessun HDU con WCS in {filename}")
                return False
            
            sci_data = sci_hdu.data
            sci_header = sci_hdu.header.copy()
            
            if len(sci_data.shape) == 3:
                sci_data = sci_data[0]
            
            fits.PrimaryHDU(data=sci_data, header=sci_header).writeto(
                output_file, overwrite=True, output_verify='silentfix'
            )
            
            logger.info(f"✓ {filename}: WCS estratto")
            return True
            
    except Exception as e:
        logger.error(f"Errore {filename}: {e}")
        return False


def convert_wcs_for_object(source, object_name, logger):
    """Converte WCS per un singolo oggetto."""
    logger.info("=" * 80)
    logger.info(f"CONVERSIONE WCS: {source}/{object_name}")
    logger.info("=" * 80)
    
    # Determina percorsi
    if source == 'local':
        input_dir = os.path.join(INPUT_LOCAL_RAW, object_name)
        output_dir = os.path.join(OUTPUT_LOCAL_WCS, object_name)
        convert_func = add_wcs_to_local_file
    else:  # hubble
        input_dir = os.path.join(INPUT_HUBBLE, object_name)
        output_dir = os.path.join(OUTPUT_HUBBLE_WCS, object_name)
        convert_func = extract_hubble_wcs
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova file
    fits_files = glob.glob(os.path.join(input_dir, '**', '*.fit*'), recursive=True)
    
    if not fits_files:
        logger.warning(f"Nessun file in {input_dir}")
        return 0, 0
    
    logger.info(f"Trovati {len(fits_files)} file")
    
    success_count = 0
    error_count = 0
    
    print(f"\n🔄 Conversione WCS: {source}/{object_name} ({len(fits_files)} file)")
    
    for input_file in tqdm(fits_files, desc=f"  {object_name}", unit="file"):
        basename = os.path.basename(input_file)
        name, ext = os.path.splitext(basename)
        output_file = os.path.join(output_dir, f"{name}_wcs.fits")
        
        if convert_func(input_file, output_file, logger):
            success_count += 1
        else:
            error_count += 1
    
    logger.info(f"Convertiti: {success_count}/{len(fits_files)}")
    
    return success_count, error_count


# ============================================================================
# STEP 2: REGISTRAZIONE
# ============================================================================

def extract_wcs_info(filepath, logger):
    """Estrae info WCS da file."""
    try:
        with fits.open(filepath) as hdul:
            data_hdu = None
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    try:
                        wcs = WCS(hdu.header)
                        if wcs.has_celestial:
                            data_hdu = hdu
                            break
                    except:
                        continue
            
            if data_hdu is None:
                return None
            
            wcs = WCS(data_hdu.header)
            
            if len(data_hdu.data.shape) == 3:
                data = data_hdu.data[0]
            else:
                data = data_hdu.data
            
            shape = data.shape
            center_ra, center_dec = wcs.wcs.crval
            
            try:
                cd = wcs.wcs.cd
                pixel_scale = np.sqrt(cd[0,0]**2 + cd[0,1]**2) * 3600
            except:
                pixel_scale = abs(wcs.wcs.cdelt[0]) * 3600
            
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
            logger.error(f"Errore {os.path.basename(filepath)}: {e}")
        return None


def create_common_wcs_frame(wcs_info_list, logger):
    """Crea WCS comune."""
    logger.info("Creazione WCS comune...")
    
    ra_min, ra_max = float('inf'), float('-inf')
    dec_min, dec_max = float('inf'), float('-inf')
    pixel_scales = []
    
    for info in wcs_info_list:
        wcs = info['wcs']
        shape = info['shape']
        pixel_scales.append(info['pixel_scale'])
        
        height, width = shape
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
    
    ra_center = (ra_min + ra_max) / 2.0
    dec_center = (dec_min + dec_max) / 2.0
    avg_scale = np.mean(pixel_scales)
    
    common_wcs = WCS(naxis=2)
    common_wcs.wcs.crval = [ra_center, dec_center]
    common_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    common_wcs.wcs.cdelt = [avg_scale/3600, avg_scale/3600]
    common_wcs.wcs.crpix = [1, 1]
    common_wcs.wcs.radesys = 'ICRS'
    common_wcs.wcs.equinox = 2000.0
    
    logger.info(f"WCS comune: RA={ra_center:.4f}°, DEC={dec_center:.4f}°, scale={avg_scale:.4f}\"/px")
    
    return common_wcs


def reproject_image_native(img_info, common_wcs, output_dir, logger):
    """Reproietta immagine mantenendo risoluzione nativa."""
    filepath = img_info['file']
    filename = os.path.basename(filepath)
    
    try:
        with fits.open(filepath) as hdul:
            data_hdu = None
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    try:
                        WCS(hdu.header)
                        data_hdu = hdu
                        break
                    except:
                        continue
            
            if data_hdu is None:
                return False, filename, None
            
            data = data_hdu.data
            header = data_hdu.header.copy()
            wcs = WCS(header)
            
            if len(data.shape) == 3:
                data = data[0]
            
            native_scale = img_info['pixel_scale']
            native_scale_deg = native_scale / 3600.0
            
            # WCS target con risoluzione nativa
            target_wcs = WCS(naxis=2)
            target_wcs.wcs.crval = common_wcs.wcs.crval
            target_wcs.wcs.ctype = common_wcs.wcs.ctype
            target_wcs.wcs.cdelt = [-native_scale_deg, native_scale_deg]
            target_wcs.wcs.crpix = [1, 1]
            target_wcs.wcs.radesys = 'ICRS'
            target_wcs.wcs.equinox = 2000.0
            
            # Calcola dimensioni
            height, width = data.shape
            corners_x = [0, width-1, width-1, 0]
            corners_y = [0, 0, height-1, height-1]
            
            ra_corners, dec_corners = wcs.all_pix2world(corners_x, corners_y, 0)
            x_new, y_new = target_wcs.all_world2pix(ra_corners, dec_corners, 0)
            
            x_min, x_max = int(np.floor(np.min(x_new))), int(np.ceil(np.max(x_new)))
            y_min, y_max = int(np.floor(np.min(y_new))), int(np.ceil(np.max(y_new)))
            
            canvas_width = x_max - x_min + 1
            canvas_height = y_max - y_min + 1
            
            target_wcs.wcs.crpix = [-x_min + 1, -y_min + 1]
            
            # Reproiezione
            reprojected_data, footprint = reproject_interp(
                (data, wcs),
                target_wcs,
                shape_out=(canvas_height, canvas_width),
                order=REPROJECT_ORDER
            )
            
            valid_pixels = np.sum(np.isfinite(reprojected_data))
            coverage = (valid_pixels / (canvas_height * canvas_width)) * 100
            
            # Header output
            output_header = target_wcs.to_header()
            
            for key in ['DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'TELESCOP']:
                if key in header:
                    output_header[key] = header[key]
            
            output_header['REGMTHD'] = 'reproject_interp_native'
            output_header['REGSRC'] = filename
            output_header['REGDATE'] = datetime.now().isoformat()
            output_header['REGCOVER'] = (coverage, 'Coverage percentage')
            output_header['NATIVESC'] = (native_scale, 'Native pixel scale (arcsec/px)')
            
            # Salva
            output_filename = f"reg_{filename}"
            output_path = os.path.join(output_dir, output_filename)
            
            fits.PrimaryHDU(data=reprojected_data, header=output_header).writeto(
                output_path, overwrite=True, output_verify='silentfix'
            )
            
            with log_lock:
                logger.info(f"✓ {filename}: {canvas_width}×{canvas_height}px @ {native_scale:.3f}\"/px, coverage={coverage:.1f}%")
            
            return True, filename, {'output_path': output_path, 'coverage': coverage}
            
    except Exception as e:
        with log_lock:
            logger.error(f"❌ {filename}: {e}")
        return False, filename, None


def register_images_for_object(source, object_name, logger):
    """Registra immagini per un singolo oggetto."""
    logger.info("=" * 80)
    logger.info(f"REGISTRAZIONE: {source}/{object_name}")
    logger.info("=" * 80)
    
    # Percorsi
    if source == 'local':
        input_dir = os.path.join(OUTPUT_LOCAL_WCS, object_name)
        output_dir = os.path.join(OUTPUT_LOCAL_REG, object_name)
    else:  # hubble
        input_dir = os.path.join(OUTPUT_HUBBLE_WCS, object_name)
        output_dir = os.path.join(OUTPUT_HUBBLE_REG, object_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova file con WCS
    fits_files = glob.glob(os.path.join(input_dir, '*_wcs.fits'))
    
    if not fits_files:
        logger.warning(f"Nessun file WCS in {input_dir}")
        return 0, 0
    
    logger.info(f"Trovati {len(fits_files)} file con WCS")
    
    # Analizza immagini
    wcs_info_list = []
    print(f"\n🔍 Analisi WCS: {source}/{object_name}")
    
    for filepath in tqdm(fits_files, desc=f"  Analisi", unit="file"):
        info = extract_wcs_info(filepath, logger)
        if info:
            wcs_info_list.append(info)
    
    if not wcs_info_list:
        logger.error("Nessuna immagine con WCS valido")
        return 0, 0
    
    logger.info(f"Immagini valide: {len(wcs_info_list)}/{len(fits_files)}")
    
    # WCS comune
    common_wcs = create_common_wcs_frame(wcs_info_list, logger)
    
    if not common_wcs:
        logger.error("Impossibile creare WCS comune")
        return 0, 0
    
    # Registrazione
    print(f"\n🔄 Registrazione: {source}/{object_name}")
    
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            executor.submit(reproject_image_native, info, common_wcs, output_dir, logger): info
            for info in wcs_info_list
        }
        
        with tqdm(total=len(wcs_info_list), desc=f"  {object_name}", unit="img") as pbar:
            for future in as_completed(futures):
                success, filename, stats = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                pbar.update(1)
    
    logger.info(f"Registrate: {success_count}/{len(wcs_info_list)}")
    
    return success_count, error_count


# ============================================================================
# MAIN
# ============================================================================

def process_single_object(source, object_name, logger):
    """Processa singolo oggetto (WCS + Registrazione)."""
    print("\n" + "=" * 70)
    print(f"📦 {source.upper()}/{object_name}")
    print("=" * 70)
    
    # STEP 1: Conversione WCS
    logger.info("")
    logger.info("STEP 1: CONVERSIONE WCS")
    wcs_success, wcs_error = convert_wcs_for_object(source, object_name, logger)
    
    print(f"\n   ✓ WCS: {wcs_success} successi, {wcs_error} errori")
    
    if wcs_success == 0:
        logger.warning(f"Nessun file convertito per {object_name}, skip registrazione")
        return 0, 0, 0, 0
    
    # STEP 2: Registrazione
    logger.info("")
    logger.info("STEP 2: REGISTRAZIONE")
    reg_success, reg_error = register_images_for_object(source, object_name, logger)
    
    print(f"   ✓ Registrazione: {reg_success} successi, {reg_error} errori")
    
    return wcs_success, wcs_error, reg_success, reg_error


def main():
    """Funzione principale."""
    print("=" * 70)
    print("🚀 CONVERSIONE WCS + REGISTRAZIONE".center(70))
    print("=" * 70)
    
    # Menu interattivo
    selected_source, selected_object = interactive_menu()
    
    if not selected_source:
        return
    
    # Setup logging
    logger = setup_logging(selected_source, selected_object)
    
    logger.info("=" * 80)
    logger.info(f"PIPELINE WCS+REGISTRAZIONE: {selected_source}" + 
               (f"/{selected_object}" if selected_object else " (TUTTI)"))
    logger.info("=" * 80)
    logger.info(f"Threads: {NUM_THREADS}")
    logger.info(f"Interpolazione: {REPROJECT_ORDER}")
    logger.info("")
    
    total_wcs_success = 0
    total_wcs_error = 0
    total_reg_success = 0
    total_reg_error = 0
    
    # Determina cosa processare
    if selected_source == 'all':
        # TUTTE le fonti
        sources_data = list_available_sources()
        
        for source_name, objects in sources_data.items():
            for obj_name, _ in objects:
                wcs_s, wcs_e, reg_s, reg_e = process_single_object(source_name, obj_name, logger)
                total_wcs_success += wcs_s
                total_wcs_error += wcs_e
                total_reg_success += reg_s
                total_reg_error += reg_e
    
    elif selected_object is None:
        # Tutti gli oggetti di una fonte
        sources_data = list_available_sources()
        objects = sources_data.get(selected_source, [])
        
        for obj_name, _ in objects:
            wcs_s, wcs_e, reg_s, reg_e = process_single_object(selected_source, obj_name, logger)
            total_wcs_success += wcs_s
            total_wcs_error += wcs_e
            total_reg_success += reg_s
            total_reg_error += reg_e
    
    else:
        # Singolo oggetto
        wcs_s, wcs_e, reg_s, reg_e = process_single_object(selected_source, selected_object, logger)
        total_wcs_success += wcs_s
        total_wcs_error += wcs_e
        total_reg_success += reg_s
        total_reg_error += reg_e
    
    # Riepilogo
    print("\n" + "=" * 70)
    print("📊 RIEPILOGO TOTALE")
    print("=" * 70)
    print(f"\n   STEP 1 (Conversione WCS):")
    print(f"      Successi: {total_wcs_success}")
    print(f"      Errori: {total_wcs_error}")
    print(f"\n   STEP 2 (Registrazione):")
    print(f"      Successi: {total_reg_success}")
    print(f"      Errori: {total_reg_error}")
    
    if total_reg_success > 0:
        print(f"\n✅ PIPELINE COMPLETATA!")
        print(f"\n   📁 Output:")
        if selected_source in ['local', 'all']:
            print(f"      Local: {OUTPUT_LOCAL_REG}")
        if selected_source in ['hubble', 'all']:
            print(f"      Hubble: {OUTPUT_HUBBLE_REG}")
        print(f"\n   ➡️  Prossimo: python scriptale/step2_croppedmosaico.py")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETATA")
    logger.info("=" * 80)
    logger.info(f"WCS: {total_wcs_success} successi, {total_wcs_error} errori")
    logger.info(f"Registrazione: {total_reg_success} successi, {total_reg_error} errori")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️ Tempo totale: {elapsed:.2f}s")