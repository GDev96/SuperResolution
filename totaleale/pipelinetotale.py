"""
PIPELINE COMPLETA UNIFICATA CON MENU INTERATTIVO
Combina tutti gli step in un unico file con menu per scegliere cosa eseguire.
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
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)
warnings.filterwarnings('ignore')

# Prova a importare reproject
try:
    from reproject import reproject_interp
    REPROJECT_AVAILABLE = True
except ImportError:
    REPROJECT_AVAILABLE = False

# ============================================================================
# CONFIGURAZIONE GLOBALE
# ============================================================================

BASE_DIR = r'F:\Super Revolt Gaia\parte 2(patch)\data'
LOG_DIR = r'F:\Super Revolt Gaia\logs'

# INPUT STEP 1
INPUT_OSSERVATORIO = os.path.join(BASE_DIR, 'local_raw')
INPUT_LITH = os.path.join(BASE_DIR, 'img_lights_1')

# OUTPUT STEP 1
OUTPUT_OSSERVATORIO_WCS = os.path.join(BASE_DIR, 'osservatorio_con_wcs')
OUTPUT_LITH_WCS = os.path.join(BASE_DIR, 'lith_con_wcs')

# INPUT STEP 2
INPUT_HUBBLE = OUTPUT_LITH_WCS
INPUT_OBSERVATORY = OUTPUT_OSSERVATORIO_WCS

# OUTPUT STEP 2
OUTPUT_HUBBLE = os.path.join(BASE_DIR, '3_registered_native', 'hubble')
OUTPUT_OBSERVATORY = os.path.join(BASE_DIR, '3_registered_native', 'observatory')

# INPUT STEP 3
INPUT_DIRS_CROPPED = {
    'hubble': Path(BASE_DIR) / '3_registered_native' / 'hubble',
    'observatory': Path(BASE_DIR) / '3_registered_native' / 'observatory'
}

# OUTPUT STEP 3
OUTPUT_DIR_BASE_CROPPED = Path(BASE_DIR) / '4_cropped'
OUTPUT_DIRS_CROPPED = {
    'hubble': OUTPUT_DIR_BASE_CROPPED / 'hubble',
    'observatory': OUTPUT_DIR_BASE_CROPPED / 'observatory'
}

MOSAIC_OUTPUT_DIR = Path(BASE_DIR) / '5_mosaics'
MOSAIC_OUTPUT_FILE = MOSAIC_OUTPUT_DIR / 'final_mosaic.fits'

# STEP 5
HUBBLE_DIR_ANALYZE = Path(BASE_DIR) / '4_cropped' / 'hubble'
OBS_DIR_ANALYZE = Path(BASE_DIR) / '4_cropped' / 'observatory'
OUTPUT_DIR_ANALYZE = Path(BASE_DIR) / 'analisi_overlap'

# STEP 6
INPUT_CROPPED_HUBBLE = Path(BASE_DIR) / '4_cropped' / 'hubble'
INPUT_CROPPED_OBSERVATORY = Path(BASE_DIR) / '4_cropped' / 'observatory'
INPUT_REGISTERED_HUBBLE = Path(BASE_DIR) / '3_registered_native' / 'hubble'
INPUT_REGISTERED_OBSERVATORY = Path(BASE_DIR) / '3_registered_native' / 'observatory'

# Parametri
NUM_THREADS = 7
REPROJECT_ORDER = 'bilinear'
TARGET_PATCH_ARCMIN = 1.0
PATCH_OVERLAP_PERCENT = 10
TARGET_FOV_ARCMIN = 0.85
OVERLAP_PERCENT = 25
MIN_VALID_PERCENT = 50
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

log_lock = threading.Lock()

# ============================================================================
# SETUP LOGGING
# ============================================================================

def setup_logging():
    """Configura logging per l'intera pipeline."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'pipeline_unified_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    if REPROJECT_AVAILABLE:
        logger.info(f"Reproject: disponibile")
    return logger

# ============================================================================
# STEP 1: FUNZIONI CONVERSIONE WCS
# ============================================================================

def parse_coordinates(ra_str, dec_str):
    """Converte coordinate da formato sessagesimale a decimale."""
    try:
        ra_str = ra_str.strip()
        dec_str = dec_str.strip()
        coord_str = f"{ra_str} {dec_str}"
        coord = SkyCoord(coord_str, unit=(u.hourangle, u.deg))
        return coord.ra.degree, coord.dec.degree
    except Exception as e:
        try:
            ra_parts = ra_str.split()
            dec_parts = dec_str.split()
            h, m, s = float(ra_parts[0]), float(ra_parts[1]), float(ra_parts[2])
            ra_deg = (h + m/60.0 + s/3600.0) * 15.0
            d, m, s = float(dec_parts[0]), float(dec_parts[1]), float(dec_parts[2])
            sign = 1 if d >= 0 else -1
            dec_deg = d + sign * (m/60.0 + s/3600.0)
            return ra_deg, dec_deg
        except Exception as e2:
            raise ValueError(f"Impossibile parsare coordinate: RA='{ra_str}', DEC='{dec_str}': {e2}")


def calculate_pixel_scale(header):
    """Calcola pixel scale da informazioni nel header."""
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
    """Crea WCS completo da informazioni nel header."""
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
    except Exception as e:
        return None


def add_wcs_to_file(input_file, output_file, logger):
    """Aggiunge WCS a un file FITS che ha OBJCTRA/OBJCTDEC."""
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
            important_keys = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
                            'BZERO', 'BSCALE', 'DATE-OBS', 'EXPTIME', 'FILTER',
                            'INSTRUME', 'TELESCOP', 'XBINNING', 'YBINNING',
                            'XPIXSZ', 'YPIXSZ', 'GAIN', 'CCD-TEMP', 'FOCALLEN']
            
            new_header = fits.Header()
            for key in important_keys:
                if key in header:
                    new_header[key] = header[key]
            
            new_header.update(wcs_header)
            new_header['WCSADDED'] = True
            new_header['WCSSRC'] = 'OBJCTRA/OBJCTDEC conversion'
            new_header['WCSDATE'] = datetime.now().isoformat()
            
            primary_hdu = fits.PrimaryHDU(data=data, header=new_header)
            primary_hdu.writeto(output_file, overwrite=True, output_verify='silentfix')
            
            logger.info(f"✓ {filename}: WCS aggiunto")
            return True
            
    except Exception as e:
        logger.error(f"Errore {os.path.basename(input_file)}: {e}")
        return False


def extract_lith_data(filename, logger):
    """Estrae dati e WCS da file LITH/HST."""
    try:
        with fits.open(filename) as hdul:
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
                logger.warning(f"Nessun HDU con WCS in {os.path.basename(filename)}")
                return None, None, None
            
            sci_data = sci_hdu.data
            sci_header = sci_hdu.header.copy()
            wcs = WCS(sci_header)
            
            if len(sci_data.shape) == 3:
                sci_data = sci_data[0]
            
            if not wcs.has_celestial:
                logger.warning(f"WCS non valido in {os.path.basename(filename)}")
                return None, None, None
            
            shape = sci_data.shape
            ra, dec = wcs.wcs.crval
            
            try:
                cd = wcs.wcs.cd
                pixel_scale = np.sqrt(cd[0,0]**2 + cd[0,1]**2) * 3600
            except:
                pixel_scale = 0.04
            
            info = {
                'shape': shape,
                'ra': ra,
                'dec': dec,
                'pixel_scale': pixel_scale
            }
            
            logger.info(f"✓ {os.path.basename(filename)}: {shape[1]}x{shape[0]}px, RA={ra:.4f}°, DEC={dec:.4f}°")
            
            return sci_data, sci_header, info
            
    except Exception as e:
        logger.error(f"✗ {os.path.basename(filename)}: {e}")
        return None, None, None


def process_osservatorio_folder(input_dir, output_dir, logger):
    """Processa osservatorio convertendo coordinate in WCS."""
    fits_files = glob.glob(os.path.join(input_dir, '**', '*.fit'), recursive=True) + \
                 glob.glob(os.path.join(input_dir, '**', '*.fits'), recursive=True)
    
    if not fits_files:
        logger.warning(f"Nessun file in {input_dir}")
        return 0, 0, None
    
    logger.info(f"Trovati {len(fits_files)} file osservatorio")
    
    prepared_count = 0
    failed_count = 0
    ra_list = []
    dec_list = []
    scale_list = []
    
    with tqdm(total=len(fits_files), desc="  Osservatorio", unit="file") as pbar:
        for input_file in fits_files:
            basename = os.path.basename(input_file)
            name, ext = os.path.splitext(basename)
            output_file = os.path.join(output_dir, f"{name}_wcs.fits")
            
            success = add_wcs_to_file(input_file, output_file, logger)
            
            if success:
                try:
                    with fits.open(output_file) as hdul:
                        wcs = WCS(hdul[0].header)
                        if wcs.has_celestial:
                            ra, dec = wcs.wcs.crval
                            pixel_scale = abs(wcs.wcs.cdelt[0]) * 3600
                            ra_list.append(ra)
                            dec_list.append(dec)
                            scale_list.append(pixel_scale)
                    prepared_count += 1
                except:
                    failed_count += 1
            else:
                failed_count += 1
            
            pbar.update(1)
    
    stats = None
    if ra_list:
        stats = {
            'ra_range': (min(ra_list), max(ra_list)),
            'dec_range': (min(dec_list), max(dec_list)),
            'avg_scale': np.mean(scale_list)
        }
    
    return prepared_count, failed_count, stats


def process_lith_folder(input_dir, output_dir, logger):
    """Processa LITH/HST."""
    fits_files = glob.glob(os.path.join(input_dir, '**', '*.fit'), recursive=True) + \
                 glob.glob(os.path.join(input_dir, '**', '*.fits'), recursive=True)
    
    if not fits_files:
        return 0, 0, None
    
    logger.info(f"Trovati {len(fits_files)} file LITH")
    
    prepared_count = 0
    failed_count = 0
    ra_list = []
    dec_list = []
    scale_list = []
    
    with tqdm(total=len(fits_files), desc="  LITH", unit="file") as pbar:
        for input_file in fits_files:
            data, header, info = extract_lith_data(input_file, logger)
            
            if data is not None:
                basename = os.path.basename(input_file)
                name, ext = os.path.splitext(basename)
                output_file = os.path.join(output_dir, f"{name}_wcs.fits")
                
                try:
                    primary_hdu = fits.PrimaryHDU(data=data, header=header)
                    primary_hdu.header['ORIGINAL'] = basename
                    primary_hdu.header['PREPDATE'] = datetime.now().isoformat()
                    primary_hdu.header['SOURCE'] = 'lith'
                    
                    primary_hdu.writeto(output_file, overwrite=True, output_verify='silentfix')
                    
                    prepared_count += 1
                    ra_list.append(info['ra'])
                    dec_list.append(info['dec'])
                    scale_list.append(info['pixel_scale'])
                except Exception as e:
                    logger.error(f"Errore {basename}: {e}")
                    failed_count += 1
            else:
                failed_count += 1
            
            pbar.update(1)
    
    stats = None
    if ra_list:
        stats = {
            'ra_range': (min(ra_list), max(ra_list)),
            'dec_range': (min(dec_list), max(dec_list)),
            'avg_scale': np.mean(scale_list)
        }
    
    return prepared_count, failed_count, stats

# ============================================================================
# STEP 2: FUNZIONI REGISTRAZIONE
# ============================================================================

def extract_wcs_info(filepath, logger):
    """Estrae info WCS da file FITS."""
    try:
        with fits.open(filepath) as hdul:
            data_hdu = None
            hdu_idx = 0
            
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    try:
                        wcs = WCS(hdu.header)
                        if wcs.has_celestial:
                            data_hdu = hdu
                            hdu_idx = i
                            break
                    except:
                        continue
            
            if data_hdu is None:
                with log_lock:
                    logger.warning(f"Nessun HDU con WCS valido trovato in {os.path.basename(filepath)}")
                return None
            
            wcs = WCS(data_hdu.header)

            if len(data_hdu.data.shape) == 3:
                data = data_hdu.data[0]
            else:
                data = data_hdu.data
            
            ny, nx = data.shape
            center = wcs.pixel_to_world(nx/2, ny/2)
            
            try:
                if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
                    cd = wcs.wcs.cd
                    pixel_scale_deg = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
                else:
                    pixel_scale_deg = abs(wcs.wcs.cdelt[0])
                pixel_scale_arcsec = pixel_scale_deg * 3600.0
            except:
                pixel_scale_arcsec = 0.0
            
            return {
                'file': filepath,
                'hdu_index': hdu_idx,
                'wcs': wcs,
                'shape': data.shape,
                'center_ra': center.ra.deg,
                'center_dec': center.dec.deg,
                'pixel_scale': pixel_scale_arcsec,
            }
            
    except Exception as e:
        with log_lock:
            logger.error(f"Errore WCS {os.path.basename(filepath)}: {e}")
        return None


def analyze_images(input_dir, source_name, logger):
    """Analizza tutte le immagini in una directory."""
    files = glob.glob(os.path.join(input_dir, '*.fits')) + \
            glob.glob(os.path.join(input_dir, '*.fit')) + \
            glob.glob(os.path.join(input_dir, '*_wcs.fits'))
    
    files = list(set(files))
    
    if not files:
        with log_lock:
            logger.warning(f"Nessun file in {input_dir}")
        return []
    
    print(f"\n📂 {source_name}: {len(files)} file")
    
    wcs_info_list = []
    
    with tqdm(total=len(files), desc=f"  Analisi {source_name}", unit="file") as pbar:
        for filepath in files:
            info = extract_wcs_info(filepath, logger)
            if info:
                wcs_info_list.append(info)
                with log_lock:
                    logger.info(f"✓ {os.path.basename(filepath)}: "
                               f"RA={info['center_ra']:.4f}°, DEC={info['center_dec']:.4f}°, "
                               f"scale={info['pixel_scale']:.4f}\"/px (NATIVA)")
            else:
                with log_lock:
                    logger.warning(f"✗ {os.path.basename(filepath)}: WCS non valido")
            pbar.update(1)
    
    print(f"   ✓ {len(wcs_info_list)}/{len(files)} con WCS valido")
    
    return wcs_info_list


def create_common_wcs_frame(wcs_info_list, logger):
    """Crea un WCS di riferimento comune."""
    with log_lock:
        logger.info("=" * 60)
        logger.info("CREAZIONE FRAME WCS COMUNE (riferimento)")
        logger.info("=" * 60)
    
    if not wcs_info_list:
        with log_lock:
            logger.error("Nessuna immagine fornita per creare WCS comune.")
        return None
    
    ra_min = float('inf')
    ra_max = float('-inf')
    dec_min = float('inf')
    dec_max = float('-inf')
    
    for info in wcs_info_list:
        wcs = info['wcs']
        shape = info['shape']
        ny, nx = shape
        
        corners_pix = np.array([
            [0, 0], [nx, 0], [0, ny], [nx, ny]
        ])
        
        corners_world = wcs.pixel_to_world(corners_pix[:, 0], corners_pix[:, 1])
        
        for coord in corners_world:
            ra = coord.ra.deg
            dec = coord.dec.deg
            ra_min = min(ra_min, ra)
            ra_max = max(ra_max, ra)
            dec_min = min(dec_min, dec)
            dec_max = max(dec_max, dec)
    
    ra_center = (ra_min + ra_max) / 2.0
    dec_center = (dec_min + dec_max) / 2.0
    
    ra_span = ra_max - ra_min
    dec_span = dec_max - dec_min
    
    margin_factor = 1.05
    ra_span *= margin_factor
    dec_span *= margin_factor
    
    ref_pixel_scale_deg = 0.04 / 3600.0
    
    reference_wcs = WCS(naxis=2)
    reference_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    reference_wcs.wcs.crval = [ra_center, dec_center]
    reference_wcs.wcs.crpix = [1, 1]
    reference_wcs.wcs.cdelt = [-ref_pixel_scale_deg, ref_pixel_scale_deg]
    reference_wcs.wcs.radesys = 'ICRS'
    reference_wcs.wcs.equinox = 2000.0
    
    with log_lock:
        logger.info(f"WCS Comune creato:")
        logger.info(f"  Centro: RA={ra_center:.4f}°, DEC={dec_center:.4f}°")
        logger.info(f"  Span: RA={ra_span:.4f}°, DEC={dec_span:.4f}°")
    
    print(f"\n✓ WCS comune creato:")
    print(f"   Centro: RA={ra_center:.4f}°, DEC={dec_center:.4f}°")
    print(f"   Span: RA={ra_span:.4f}°, DEC={dec_span:.4f}°")
    
    return reference_wcs


def reproject_image_native(wcs_info, common_wcs, output_dir, logger):
    """Riproietta un'immagine mantenendo la sua risoluzione NATIVA."""
    try:
        filepath = wcs_info['file']
        hdu_index = wcs_info['hdu_index']
        native_pixel_scale = wcs_info['pixel_scale']
        filename = os.path.basename(filepath)
        
        with fits.open(filepath) as hdul:
            hdu = hdul[hdu_index]
            data = hdu.data
            header = hdu.header.copy()
            wcs_orig = WCS(header)
            
            if len(data.shape) == 3:
                data = data[0]
            
            original_shape = data.shape
            
            native_pixel_scale_deg = native_pixel_scale / 3600.0
            
            ny_orig, nx_orig = original_shape
            
            try:
                if hasattr(wcs_orig.wcs, 'cd') and wcs_orig.wcs.cd is not None:
                    cd = wcs_orig.wcs.cd
                    pixel_scale_ra = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
                    pixel_scale_dec = np.sqrt(cd[1,0]**2 + cd[1,1]**2)
                else:
                    pixel_scale_ra = abs(wcs_orig.wcs.cdelt[0])
                    pixel_scale_dec = abs(wcs_orig.wcs.cdelt[1])
            except:
                pixel_scale_ra = native_pixel_scale_deg
                pixel_scale_dec = native_pixel_scale_deg
            
            ra_span = pixel_scale_ra * nx_orig
            dec_span = pixel_scale_dec * ny_orig
            
            nx_out = int(np.ceil(abs(ra_span) / native_pixel_scale_deg))
            ny_out = int(np.ceil(abs(dec_span) / native_pixel_scale_deg))
            
            shape_out = (ny_out, nx_out)
            
            target_wcs = WCS(naxis=2)
            target_wcs.wcs.ctype = common_wcs.wcs.ctype
            target_wcs.wcs.crval = wcs_orig.wcs.crval
            target_wcs.wcs.crpix = [nx_out / 2.0, ny_out / 2.0]
            target_wcs.wcs.cdelt = [-native_pixel_scale_deg, native_pixel_scale_deg]
            target_wcs.wcs.radesys = common_wcs.wcs.radesys
            target_wcs.wcs.equinox = common_wcs.wcs.equinox
            
            reprojected_data, footprint = reproject_interp(
                (data, wcs_orig),
                target_wcs,
                shape_out=shape_out,
                order=REPROJECT_ORDER
            )
            
            valid_pixels = np.sum(footprint > 0)
            total_pixels = footprint.size
            coverage = (valid_pixels / total_pixels * 100) if total_pixels > 0 else 0
            
            new_header = target_wcs.to_header()
            
            for key in ['DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'TELESCOP']:
                if key in header:
                    new_header[key] = header[key]
            
            new_header['ORIGINAL'] = filename
            new_header['REGDATE'] = datetime.now().isoformat()
            new_header['REGCOV'] = (coverage, "Percentuale copertura valida")
            new_header['REGVALID'] = (valid_pixels, "Numero di pixel validi")
            new_header['REGORD'] = (str(REPROJECT_ORDER), "Ordine interpolazione")
            new_header['NATIVESC'] = (native_pixel_scale, "Risoluzione nativa (arcsec/px)")
            new_header['ORIGSHP0'] = (original_shape[0], "Shape originale (altezza)")
            new_header['ORIGSHP1'] = (original_shape[1], "Shape originale (larghezza)")
            
            output_filename = f"reg_{os.path.splitext(filename)[0]}.fits"
            output_path = os.path.join(output_dir, output_filename)
            
            fits.PrimaryHDU(
                data=reprojected_data.astype(np.float32),
                header=new_header
            ).writeto(output_path, overwrite=True)
            
            with log_lock:
                logger.info(f"✓ {filename}: coverage={coverage:.1f}%, "
                           f"shape={shape_out}, native_scale={native_pixel_scale:.4f}\"/px")
            
            return {
                'status': 'success',
                'file': filename,
                'coverage': coverage,
                'valid_pixels': valid_pixels,
                'output_path': output_path,
                'native_scale': native_pixel_scale,
                'output_shape': shape_out
            }
            
    except Exception as e:
        with log_lock:
            logger.error(f"Errore {os.path.basename(filepath)}: {e}")
        return {'status': 'error', 'file': os.path.basename(filepath), 'reason': str(e)}


def register_images(wcs_info_list, common_wcs, output_dir, source_name, logger):
    """Registra tutte le immagini con multithreading."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🔄 Registrazione {source_name}: {len(wcs_info_list)} immagini")
    
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            executor.submit(reproject_image_native, info, common_wcs, output_dir, logger): info
            for info in wcs_info_list
        }
        
        with tqdm(total=len(wcs_info_list), desc=f"  {source_name}") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    
                    if result['status'] == 'success':
                        success_count += 1
                    else:
                        error_count += 1
                        
                except Exception as exc:
                    error_count += 1
                    with log_lock:
                        logger.error(f"Exception: {exc}")
                
                pbar.update(1)
    
    print(f"   ✓ Successo: {success_count}")
    print(f"   ✗ Errori: {error_count}")
    
    with log_lock:
        logger.info(f"{source_name}: {success_count} successo, {error_count} errori")
    
    return success_count, error_count

# ============================================================================
# STEP 3: FUNZIONI RITAGLIO E MOSAICO
# ============================================================================

def find_smallest_dimensions(all_files):
    """Trova le dimensioni dell'immagine più piccola."""
    print("\n🔍 Ricerca dimensioni minime...")
    
    min_height = float('inf')
    min_width = float('inf')
    smallest_file = None
    
    for filepath in tqdm(all_files, desc="Scansione", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                height, width = hdul[0].data.shape
                
                if height < min_height or width < min_width:
                    if height < min_height:
                        min_height = height
                    if width < min_width:
                        min_width = width
                    smallest_file = filepath
                    
        except Exception as e:
            print(f"\n⚠️ ATTENZIONE: Impossibile leggere {filepath}: {e}")
            continue
    
    print(f"\n✅ Dimensioni minime trovate: {min_width} x {min_height} pixel")
    print(f"   File più piccolo: {Path(smallest_file).name}")
    
    return min_height, min_width


def crop_image(input_path, output_path, target_height, target_width):
    """Ritaglia un'immagine FITS alle dimensioni target."""
    try:
        with fits.open(input_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            
            current_height, current_width = data.shape
            
            y_offset = (current_height - target_height) // 2
            x_offset = (current_width - target_width) // 2
            
            cropped_data = data[
                y_offset:y_offset + target_height,
                x_offset:x_offset + target_width
            ]
            
            if 'CRPIX1' in header:
                header['CRPIX1'] -= x_offset
            if 'CRPIX2' in header:
                header['CRPIX2'] -= y_offset
                
            header['NAXIS1'] = target_width
            header['NAXIS2'] = target_height
            
            header['HISTORY'] = 'Cropped by pipeline'
            header['CROPX'] = (x_offset, 'X offset for cropping')
            header['CROPY'] = (y_offset, 'Y offset for cropping')
            header['ORIGW'] = (current_width, 'Original width')
            header['ORIGH'] = (current_height, 'Original height')
            
            fits.PrimaryHDU(data=cropped_data, header=header).writeto(
                output_path, overwrite=True
            )
            
            return True
            
    except Exception as e:
        print(f"\n❌ ERRORE nel ritaglio di {input_path.name}: {e}")
        return False


def crop_all_images():
    """Esegue il ritaglio di tutte le immagini."""
    print("\n" + "✂️ "*35)
    print("RITAGLIO IMMAGINI REGISTRATE".center(70))
    print("✂️ "*35)
    
    for output_dir in OUTPUT_DIRS_CROPPED.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Cartelle di input:")
    for name, path in INPUT_DIRS_CROPPED.items():
        print(f"   - {name}: {path}")
    
    print(f"\n📂 Cartelle di output:")
    for name, path in OUTPUT_DIRS_CROPPED.items():
        print(f"   - {name}: {path}")
    
    all_files = []
    file_mapping = {}
    
    for category, input_dir in INPUT_DIRS_CROPPED.items():
        files = glob.glob(str(input_dir / '*.fits'))
        files.extend(glob.glob(str(input_dir / '*.fit')))
        
        for f in files:
            all_files.append(f)
            file_mapping[f] = category
        
        print(f"\n   {category}: {len(files)} file trovati")
    
    if not all_files:
        print(f"\n❌ ERRORE: Nessun file FITS trovato.")
        return False
    
    print(f"\n✅ Totale: {len(all_files)} file da ritagliare")
    
    min_height, min_width = find_smallest_dimensions(all_files)
    
    print(f"\n🔍 Le immagini verranno ritagliate a: {min_width} x {min_height} pixel")
    
    print("\n✂️ Ritaglio in corso...\n")
    
    success_count = 0
    failed_count = 0
    
    for filepath in tqdm(all_files, desc="Ritaglio", unit="file"):
        category = file_mapping[filepath]
        filename = Path(filepath).name
        output_path = OUTPUT_DIRS_CROPPED[category] / filename
        
        if crop_image(Path(filepath), output_path, min_height, min_width):
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n{'='*70}")
    print("✅ RITAGLIO COMPLETATO!")
    print(f"{'='*70}")
    print(f"   Immagini ritagliate con successo: {success_count}")
    if failed_count > 0:
        print(f"   ⚠️ Immagini con errori: {failed_count}")
    print(f"\n   Dimensioni finali: {min_width} x {min_height} pixel")
    
    return success_count > 0


def create_mosaic():
    """Crea il mosaico."""
    print("\n" + "🖼️ "*35)
    print("CREAZIONE MOSAICO".center(70))
    print("🖼️ "*35)
    
    MOSAIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    REGISTERED_DIRS = [
        OUTPUT_DIRS_CROPPED['hubble'],
        OUTPUT_DIRS_CROPPED['observatory']
    ]
    
    print(f"\n📂 Cartelle di input:")
    for d in REGISTERED_DIRS:
        print(f"   - {d}")
    print(f"\n📂 File di output:")
    print(f"   - {MOSAIC_OUTPUT_FILE}")
    
    all_files = []
    for d in REGISTERED_DIRS:
        all_files.extend(glob.glob(str(d / '*.fits')))
        all_files.extend(glob.glob(str(d / '*.fit')))
        
    if not all_files:
        print(f"\n❌ ERRORE: Nessun file FITS trovato.")
        return False
        
    print(f"\n✅ Trovati {len(all_files)} file FITS da combinare.")
    
    try:
        with fits.open(all_files[0]) as hdul:
            template_header = hdul[0].header
            shape = hdul[0].data.shape
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile leggere il primo file: {e}")
        return False
        
    print(f"   Dimensioni mosaico: {shape[1]} x {shape[0]} pixel")
    
    total_flux = np.zeros(shape, dtype=np.float64)
    n_pixels = np.zeros(shape, dtype=np.int32)
    
    print("\n🔄 Combinazione immagini in corso...")
    
    for filepath in tqdm(all_files, desc="Combinazione", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                img_data = hdul[0].data
                
                if img_data.shape != shape:
                    print(f"\n⚠️ {filepath} ha dimensioni diverse. Saltato.")
                    continue
                    
                valid_mask = ~np.isnan(img_data)
                img_data_no_nan = np.nan_to_num(img_data, nan=0.0, copy=False)
                total_flux += img_data_no_nan
                n_pixels[valid_mask] += 1
                
        except Exception as e:
            print(f"\n⚠️ Errore nel leggere {filepath}: {e}")
            
    print("\n🧮 Calcolo della media finale...")
    
    mosaic_data = np.full(shape, np.nan, dtype=np.float32)
    valid_stack = n_pixels > 0
    mosaic_data[valid_stack] = (total_flux[valid_stack] / n_pixels[valid_stack]).astype(np.float32)
    
    print(f"\n💾 Salvataggio mosaico...")
    
    template_header['HISTORY'] = 'Mosaico creato dalla pipeline'
    template_header['NCOMBINE'] = (len(all_files), 'Numero di file combinati')
    
    try:
        fits.PrimaryHDU(data=mosaic_data, header=template_header).writeto(MOSAIC_OUTPUT_FILE, overwrite=True)
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile salvare il file: {e}")
        return False

    print(f"\n{'='*70}")
    print("✅ MOSAICO COMPLETATO!")
    print(f"{'='*70}")
    print(f"   File salvato in: {MOSAIC_OUTPUT_FILE}")
    
    return True

# ============================================================================

class ImageAnalyzer:
    """Analizza immagine FITS con WCS"""
    
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.data = None
        self.header = None
        self.wcs = None
        self.info = {}
        
    def load(self):
        """Carica e analizza immagine"""
        try:
            with fits.open(self.filepath) as hdul:
                data_hdu = None
                for i, hdu in enumerate(hdul):
                    if hdu.data is not None and len(hdu.data.shape) >= 2:
                        data_hdu = hdu
                        break
                
                if data_hdu is None:
                    return False
                
                self.data = data_hdu.data
                self.header = data_hdu.header
                
                if len(self.data.shape) == 3:
                    self.data = self.data[0]
                
                ny, nx = self.data.shape
                
                self.info = {
                    'filename': self.filepath.name,
                    'shape': (ny, nx),
                    'dtype': str(self.data.dtype),
                    'size_mb': round(self.data.nbytes / (1024**2), 2),
                }
                
                valid_mask = np.isfinite(self.data)
                valid_data = self.data[valid_mask]
                
                if len(valid_data) > 0:
                    self.info['stats'] = {
                        'valid_pixels': int(valid_mask.sum()),
                        'coverage_percent': round(100 * valid_mask.sum() / self.data.size, 2),
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'mean': float(np.mean(valid_data)),
                        'median': float(np.median(valid_data)),
                    }
                
                try:
                    self.wcs = WCS(self.header)
                    if self.wcs.has_celestial:
                        self._analyze_wcs()
                    else:
                        self.wcs = None
                        return False
                except:
                    self.wcs = None
                    return False
                
                return True
                
        except Exception as e:
            print(f"   ✗ Errore caricamento {self.filepath.name}: {e}")
            return False
    
    def _analyze_wcs(self):
        """Analizza WCS e calcola FOV"""
        ny, nx = self.data.shape
        center = self.wcs.pixel_to_world(nx/2, ny/2)
        pixel_scale = self._get_pixel_scale()
        
        corners = self.wcs.pixel_to_world([0, nx, nx, 0], [0, 0, ny, ny])
        ra_vals = [c.ra.deg for c in corners]
        dec_vals = [c.dec.deg for c in corners]
        
        ra_span = max(ra_vals) - min(ra_vals)
        dec_span = max(dec_vals) - min(dec_vals)
        
        self.info['wcs'] = {
            'center_ra': float(center.ra.deg),
            'center_dec': float(center.dec.deg),
            'pixel_scale_arcsec': pixel_scale,
            'pixel_scale_arcmin': pixel_scale / 60.0,
            'fov_ra_deg': ra_span,
            'fov_dec_deg': dec_span,
            'fov_ra_arcmin': ra_span * 60,
            'fov_dec_arcmin': dec_span * 60,
            'ra_range': [min(ra_vals), max(ra_vals)],
            'dec_range': [min(dec_vals), max(dec_vals)],
        }
    
    def _get_pixel_scale(self):
        """Calcola pixel scale in arcsec"""
        try:
            if hasattr(self.wcs.wcs, 'cd') and self.wcs.wcs.cd is not None:
                cd = self.wcs.wcs.cd
                pixel_scale_deg = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
            elif hasattr(self.wcs.wcs, 'cdelt'):
                pixel_scale_deg = abs(self.wcs.wcs.cdelt[0])
            else:
                p1 = self.wcs.pixel_to_world(0, 0)
                p2 = self.wcs.pixel_to_world(1, 0)
                pixel_scale_deg = p1.separation(p2).deg
            
            return pixel_scale_deg * 3600
        except:
            return None
    
    def calculate_patch_size(self, target_arcmin):
        """Calcola dimensione patch ottimale in pixel."""
        if 'wcs' not in self.info or self.info['wcs']['pixel_scale_arcsec'] is None:
            return None
        
        pixel_scale_arcsec = self.info['wcs']['pixel_scale_arcsec']
        target_arcsec = target_arcmin * 60
        
        patch_size_px = int(target_arcsec / pixel_scale_arcsec)
        patch_size_px = ((patch_size_px + 7) // 8) * 8
        
        actual_arcsec = patch_size_px * pixel_scale_arcsec
        actual_arcmin = actual_arcsec / 60
        
        ny, nx = self.data.shape
        
        n_x = nx // patch_size_px
        n_y = ny // patch_size_px
        total_no_overlap = n_x * n_y
        
        step = int(patch_size_px * (1 - PATCH_OVERLAP_PERCENT/100))
        n_x_overlap = max(1, (nx - patch_size_px) // step + 1) if step > 0 else 1
        n_y_overlap = max(1, (ny - patch_size_px) // step + 1) if step > 0 else 1
        total_with_overlap = n_x_overlap * n_y_overlap
        
        return {
            'target_arcmin': target_arcmin,
            'patch_size_px': patch_size_px,
            'actual_size_arcmin': round(actual_arcmin, 4),
            'actual_size_arcsec': round(actual_arcsec, 2),
            'patches_no_overlap': {'x': n_x, 'y': n_y, 'total': total_no_overlap},
            'patches_with_overlap': {'x': n_x_overlap, 'y': n_y_overlap, 'total': total_with_overlap},
            'step_size': step,
        }


class OverlapAnalyzer:
    """Analizza overlap tra due immagini con WCS"""
    
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.overlap_info = None
    
    def calculate_overlap(self):
        """Calcola overlap tra le due immagini"""
        if self.img1.wcs is None or self.img2.wcs is None:
            return None
        
        ny1, nx1 = self.img1.data.shape
        ny2, nx2 = self.img2.data.shape
        
        corners1 = self.img1.wcs.pixel_to_world([0, nx1, nx1, 0], [0, 0, ny1, ny1])
        ra1 = [c.ra.deg for c in corners1]
        dec1 = [c.dec.deg for c in corners1]
        
        corners2 = self.img2.wcs.pixel_to_world([0, nx2, nx2, 0], [0, 0, ny2, ny2])
        ra2 = [c.ra.deg for c in corners2]
        dec2 = [c.dec.deg for c in corners2]
        
        ra_min = max(min(ra1), min(ra2))
        ra_max = min(max(ra1), max(ra2))
        dec_min = max(min(dec1), min(dec2))
        dec_max = min(max(dec1), max(dec2))
        
        if ra_min >= ra_max or dec_min >= dec_max:
            return None
        
        overlap_ra = ra_max - ra_min
        overlap_dec = dec_max - dec_min
        overlap_area_deg2 = overlap_ra * overlap_dec
        overlap_area_arcmin2 = overlap_area_deg2 * 3600
        
        area1_deg2 = (max(ra1) - min(ra1)) * (max(dec1) - min(dec1))
        area2_deg2 = (max(ra2) - min(ra2)) * (max(dec2) - min(dec2))
        
        fraction_img1 = overlap_area_deg2 / area1_deg2 if area1_deg2 > 0 else 0
        fraction_img2 = overlap_area_deg2 / area2_deg2 if area2_deg2 > 0 else 0
        
        self.overlap_info = {
            'overlap_exists': True,
            'overlap_area_deg2': overlap_area_deg2,
            'overlap_area_arcmin2': overlap_area_arcmin2,
            'overlap_ra_deg': overlap_ra,
            'overlap_dec_deg': overlap_dec,
            'overlap_ra_arcmin': overlap_ra * 60,
            'overlap_dec_arcmin': overlap_dec * 60,
            'fraction_img1': fraction_img1,
            'fraction_img2': fraction_img2,
            'ra_range': [ra_min, ra_max],
            'dec_range': [dec_min, dec_max],
        }
        
        return self.overlap_info
    
    def visualize(self, output_path=None):
        """Crea visualizzazione overlap"""
        if self.overlap_info is None:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ny1, nx1 = self.img1.data.shape
        corners1 = self.img1.wcs.pixel_to_world([0, nx1, nx1, 0, 0], [0, 0, ny1, ny1, 0])
        ra1 = [c.ra.deg for c in corners1]
        dec1 = [c.dec.deg for c in corners1]
        ax.plot(ra1, dec1, 'b-', linewidth=2, label=f'Img1')
        
        ny2, nx2 = self.img2.data.shape
        corners2 = self.img2.wcs.pixel_to_world([0, nx2, nx2, 0, 0], [0, 0, ny2, ny2, 0])
        ra2 = [c.ra.deg for c in corners2]
        dec2 = [c.dec.deg for c in corners2]
        ax.plot(ra2, dec2, 'r-', linewidth=2, label=f'Img2')
        
        ov = self.overlap_info
        if ov['overlap_exists']:
            ra_ov = [ov['ra_range'][0], ov['ra_range'][1], ov['ra_range'][1], ov['ra_range'][0], ov['ra_range'][0]]
            dec_ov = [ov['dec_range'][0], ov['dec_range'][0], ov['dec_range'][1], ov['dec_range'][1], ov['dec_range'][0]]
            ax.fill(ra_ov, dec_ov, alpha=0.3, color='green', label='Overlap')
        
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('DEC (deg)')
        ax.set_title('Overlap Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        plt.close()


def load_images_analyze(directory, label):
    """Carica tutte le immagini FITS da una directory"""
    print(f"\n📂 Caricamento {label}...")
    
    if not directory.exists():
        print(f"   ✗ Directory non trovata!")
        return []
    
    files = glob.glob(str(directory / '*.fits'))
    files.extend(glob.glob(str(directory / '*.fit')))
    
    if not files:
        print(f"   ✗ Nessun file FITS trovato")
        return []
    
    print(f"   Trovati {len(files)} file FITS")
    
    images = []
    for filepath in files:
        img = ImageAnalyzer(filepath)
        if img.load():
            images.append(img)
    
    print(f"   ✓ Caricate {len(images)} immagini con WCS valido")
    return images


def analyze_all_pairs(hubble_imgs, obs_imgs):
    """Analizza overlap per tutte le coppie possibili"""
    print(f"\n🔍 Analisi overlap...")
    
    results = []
    
    for h_img in hubble_imgs:
        for o_img in obs_imgs:
            analyzer = OverlapAnalyzer(h_img, o_img)
            overlap = analyzer.calculate_overlap()
            
            if overlap and overlap['overlap_exists']:
                results.append({
                    'hubble': h_img,
                    'observatory': o_img,
                    'overlap': overlap,
                    'analyzer': analyzer
                })
    
    results.sort(key=lambda x: x['overlap']['overlap_area_arcmin2'], reverse=True)
    
    return results

# ============================================================================
# STEP 6: FUNZIONI PATCHES
# ============================================================================

def extract_patches_from_image(filepath, output_dir, source_type, logger):
    """Estrae patches da una singola immagine."""
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
                return []
            
            data = data_hdu.data
            header = data_hdu.header
            wcs = WCS(header)
            
            if len(data.shape) == 3:
                data = data[0]
            
            try:
                if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
                    cd = wcs.wcs.cd
                    pixel_scale_deg = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
                else:
                    pixel_scale_deg = abs(wcs.wcs.cdelt[0])
                pixel_scale_arcsec = pixel_scale_deg * 3600.0
            except:
                return []
            
            target_arcsec = TARGET_FOV_ARCMIN * 60
            patch_size = int(target_arcsec / pixel_scale_arcsec)
            patch_size = ((patch_size + 7) // 8) * 8
            
            # ⚠️ CORREZIONE CRITICA: Calcolo step con controllo sicurezza
            step = int(patch_size * (1 - OVERLAP_PERCENT / 100))
            if step <= 0:
                step = 1  # Previene loop infiniti
                logger.warning(f"Step calcolato ≤0 per {os.path.basename(filepath)}, corretto a 1")
            
            # Log info per debug
            logger.debug(f"  {os.path.basename(filepath)}: patch_size={patch_size}, step={step}, overlap={OVERLAP_PERCENT}%")
            
            ny, nx = data.shape
            patches = []
            patch_idx = 0
            
            # Contatore sicurezza per evitare loop infiniti
            max_iterations = 10000
            iteration_count = 0
            
            y = 0
            while y + patch_size <= ny:
                x = 0
                while x + patch_size <= nx:
                    # Controllo sicurezza
                    iteration_count += 1
                    if iteration_count > max_iterations:
                        logger.error(f"ERRORE: Troppe iterazioni in {os.path.basename(filepath)}! Loop infinito rilevato.")
                        logger.error(f"  Parametri: patch_size={patch_size}, step={step}, nx={nx}, ny={ny}")
                        return patches  # Ritorna le patches estratte finora
                    
                    patch_data = data[y:y+patch_size, x:x+patch_size]
                    
                    valid_mask = np.isfinite(patch_data)
                    valid_percent = 100 * valid_mask.sum() / patch_data.size
                    
                    if valid_percent >= MIN_VALID_PERCENT:
                        center_x = x + patch_size / 2
                        center_y = y + patch_size / 2
                        center_coord = wcs.pixel_to_world(center_x, center_y)
                        
                        basename = Path(filepath).stem
                        output_filename = f"{basename}_patch_{patch_idx:04d}.fits"
                        output_path = output_dir / output_filename
                        
                        patch_wcs = wcs.deepcopy()
                        patch_wcs.wcs.crpix[0] -= x
                        patch_wcs.wcs.crpix[1] -= y
                        
                        patch_header = patch_wcs.to_header()
                        for key in ['DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'TELESCOP']:
                            if key in header:
                                patch_header[key] = header[key]
                        
                        patch_header['ORIGINAL'] = os.path.basename(filepath)
                        patch_header['PATCHIDX'] = patch_idx
                        patch_header['PATCHX'] = x
                        patch_header['PATCHY'] = y
                        patch_header['PATCHVAL'] = valid_percent
                        patch_header['PIXSCALE'] = pixel_scale_arcsec
                        patch_header['PATCHSZ'] = patch_size
                        patch_header['STEPSIZE'] = step
                        
                        fits.PrimaryHDU(data=patch_data.astype(np.float32), header=patch_header).writeto(
                            output_path, overwrite=True
                        )
                        
                        patches.append({
                            'filename': output_filename,
                            'source_image': os.path.basename(filepath),
                            'patch_index': patch_idx,
                            'x': x,
                            'y': y,
                            'patch_size_px': patch_size,
                            'center_ra': center_coord.ra.deg,
                            'center_dec': center_coord.dec.deg,
                            'pixel_scale_arcsec': pixel_scale_arcsec,
                            'valid_percent': valid_percent,
                            'source_type': source_type
                        })
                        
                        patch_idx += 1
                    
                    x += step  # ⚠️ Ora step è garantito essere > 0
                y += step      # ⚠️ Ora step è garantito essere > 0
            
            logger.info(f"✓ {os.path.basename(filepath)}: {len(patches)} patches estratte ({iteration_count} iterazioni)")
            return patches
            
    except Exception as e:
        logger.error(f"Errore: {e}")
        return []

def extract_all_patches(input_dir, output_dir, source_type, logger):
    """Estrae patches da tutte le immagini in una directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(input_dir.glob('*.fits')) + list(input_dir.glob('*.fit'))
    
    if not files:
        return []
    
    print(f"\n🔍 Estrazione patches da {source_type.upper()}: {len(files)} immagini")
    
    all_patches = []
    
    for filepath in tqdm(files, desc=f"  {source_type}", unit="img"):
        patches = extract_patches_from_image(filepath, output_dir, source_type, logger)
        all_patches.extend(patches)
    
    print(f"   ✓ {len(all_patches)} patches estratte")
    
    metadata_file = output_dir / f'{source_type}_patches_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_patches': len(all_patches),
            'target_fov_arcmin': TARGET_FOV_ARCMIN,
            'patches': all_patches
        }, f, indent=2)
    
    return all_patches


def create_dataset_split(patches, output_dir, logger):
    """Crea split train/val/test."""
    if not patches:
        return None
    
    np.random.seed(42)
    indices = np.random.permutation(len(patches))
    
    n_train = int(len(patches) * TRAIN_RATIO)
    n_val = int(len(patches) * VAL_RATIO)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    splits = {
        'train': [patches[i] for i in train_indices],
        'val': [patches[i] for i in val_indices],
        'test': [patches[i] for i in test_indices],
    }
    
    split_file = output_dir / 'dataset_split.json'
    with open(split_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    logger.info(f"Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    
    return splits


def create_patch_pairs(hubble_patches, obs_patches, output_dir, logger, threshold_arcmin=2.0):
    """Crea coppie di patches."""
    print(f"\n🔗 Creazione coppie patches...")
    
    pairs = []
    
    for h_patch in tqdm(hubble_patches, desc="  Pairing"):
        h_coord = SkyCoord(ra=h_patch['center_ra']*u.deg, dec=h_patch['center_dec']*u.deg)
        
        best_dist = float('inf')
        best_obs = None
        
        for o_patch in obs_patches:
            o_coord = SkyCoord(ra=o_patch['center_ra']*u.deg, dec=o_patch['center_dec']*u.deg)
            sep = h_coord.separation(o_coord).arcmin
            
            if sep < best_dist:
                best_dist = sep
                best_obs = o_patch
        
        if best_dist < threshold_arcmin and best_obs is not None:
            pairs.append({
                'hubble_patch': h_patch['filename'],
                'observatory_patch': best_obs['filename'],
                'separation_arcmin': best_dist
            })
    
    pairs_file = output_dir / 'patch_pairs.json'
    with open(pairs_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_pairs': len(pairs),
            'threshold_arcmin': threshold_arcmin,
            'pairs': pairs
        }, f, indent=2)
    
    print(f"   ✓ {len(pairs)} coppie create")
    
    return pairs

# ============================================================================
# FUNZIONI MAIN PER OGNI STEP
# ============================================================================

def main_step1(logger):
    """Step 1: WCS"""
    logger.info("STEP 1: CONVERSIONE WCS")
    
    print("=" * 70)
    print(f"🔭 STEP 1: CONVERSIONE COORDINATE → WCS".center(70))
    print("=" * 70)
    
    os.makedirs(OUTPUT_OSSERVATORIO_WCS, exist_ok=True)
    os.makedirs(OUTPUT_LITH_WCS, exist_ok=True)
    
    print("\n📡 OSSERVATORIO")
    prep_oss, fail_oss, stats_oss = process_osservatorio_folder(
        INPUT_OSSERVATORIO,
        OUTPUT_OSSERVATORIO_WCS,
        logger
    )
    
    print(f"\n   ✓ Processati: {prep_oss}")
    print(f"   ✗ Falliti: {fail_oss}")
    
    print("\n🛰️  LITH/HST")
    prep_lith, fail_lith, stats_lith = process_lith_folder(
        INPUT_LITH,
        OUTPUT_LITH_WCS,
        logger
    )
    
    print(f"\n   ✓ Processati: {prep_lith}")
    print(f"   ✗ Falliti: {fail_lith}")
    
    total_prep = prep_oss + prep_lith
    total_fail = fail_oss + fail_lith
    
    print("\n" + "=" * 70)
    print("📊 RIEPILOGO")
    print("=" * 70)
    print(f"   TOTALE: {total_prep} preparati, {total_fail} falliti")
    
    if total_prep > 0:
        print(f"\n✅ COMPLETATO!")
    else:
        print(f"\n⚠️ Nessun file processato.")
    
    return total_prep > 0


def main_step2(logger):
    """Step 2: Registrazione"""
    if not REPROJECT_AVAILABLE:
        print("\n⚠️ Libreria reproject non disponibile")
        return False
    
    logger.info("STEP 2: REGISTRAZIONE")
    
    print("\n" + "🔭"*35)
    print(f"STEP 2: REGISTRAZIONE".center(70))
    print("🔭"*35)
    
    print(f"\n{'='*70}")
    print("ANALISI IMMAGINI")
    print(f"{'='*70}")
    
    hubble_info = analyze_images(INPUT_HUBBLE, "HUBBLE", logger)
    obs_info = analyze_images(INPUT_OBSERVATORY, "OBSERVATORY", logger)
    
    all_info = hubble_info + obs_info
    if not all_info:
        print(f"\n❌ Nessuna immagine con WCS valido.")
        return False
    
    print(f"\n{'='*70}")
    print("CREAZIONE FRAME WCS COMUNE")
    print(f"{'='*70}")
    
    common_wcs = create_common_wcs_frame(all_info, logger)
    
    if common_wcs is None:
        print(f"\n❌ Impossibile creare WCS comune!")
        return False
    
    print(f"\n{'='*70}")
    print("REGISTRAZIONE")
    print(f"{'='*70}")
    
    total_success = 0
    total_error = 0
    
    if hubble_info:
        s, e = register_images(hubble_info, common_wcs, OUTPUT_HUBBLE, "Hubble", logger)
        total_success += s
        total_error += e
    
    if obs_info:
        s, e = register_images(obs_info, common_wcs, OUTPUT_OBSERVATORY, "Observatory", logger)
        total_success += s
        total_error += e
    
    print(f"\n{'='*70}")
    print("📊 RIEPILOGO")
    print(f"{'='*70}")
    print(f"\n   Totale registrate: {total_success}")
    print(f"   Totale errori: {total_error}")
    
    if total_success > 0:
        print(f"\n✅ REGISTRAZIONE COMPLETATA!")
    
    return total_success > 0


def main_step3():
    """Step 3: Cropped + Mosaico"""
    print("\n" + "✂️ "*35)
    print("STEP 3: RITAGLIO + MOSAICO".center(70))
    print("✂️ "*35)
    
    if not crop_all_images():
        print("\n❌ Errore durante il ritaglio.")
        return False
    
    print("\n\n")
    
    if not create_mosaic():
        print("\n❌ Errore durante il mosaico.")
        return False
    
    print("\n✅ STEP 3 COMPLETATO!")
    return True


def main_step5():
    """Step 5: Analisi"""
    print("\n" + "🔭"*35)
    print(f"STEP 5: ANALISI OVERLAP".center(70))
    print("🔭"*35)
    
    OUTPUT_DIR_ANALYZE.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("CARICAMENTO IMMAGINI")
    print(f"{'='*70}")
    
    hubble_imgs = load_images_analyze(HUBBLE_DIR_ANALYZE, "HUBBLE")
    obs_imgs = load_images_analyze(OBS_DIR_ANALYZE, "OBSERVATORY")
    
    if not hubble_imgs or not obs_imgs:
        print(f"\n❌ Impossibile caricare le immagini")
        return False
    
    overlap_results = analyze_all_pairs(hubble_imgs, obs_imgs)
    
    print(f"\n{'='*70}")
    print("RISULTATI OVERLAP")
    print(f"{'='*70}")
    
    if overlap_results:
        print(f"\n✅ Trovate {len(overlap_results)} coppie con overlap!")
        
        print(f"\n🏆 TOP 5:")
        for i, result in enumerate(overlap_results[:5], 1):
            print(f"\n   {i}. Overlap area: {result['overlap']['overlap_area_arcmin2']:.2f} arcmin²")
        
        best = overlap_results[0]
        viz_path = OUTPUT_DIR_ANALYZE / f'overlap_best_match.png'
        best['analyzer'].visualize(viz_path)
        print(f"\n📊 Visualizzazione salvata: {viz_path}")
    else:
        print(f"\n⚠️ Nessun overlap trovato")
    
    print(f"\n✅ ANALISI COMPLETATA")
    
    return True


def main_step6(logger):
    """Step 6: Patches"""
    print("\n" + "🎯"*35)
    print("STEP 6: ESTRAZIONE PATCHES".center(70))
    print("🎯"*35)
    
    print(f"\n📋 OPZIONI:")
    print(f"\n1️⃣ IMMAGINI CROPPED (CONSIGLIATO)")
    print(f"2️⃣ IMMAGINI REGISTERED")
    
    while True:
        choice = input("\n👉 Scegli [1/2, default=1]: ").strip()
        
        if choice == '' or choice == '1':
            input_hubble = INPUT_CROPPED_HUBBLE
            input_obs = INPUT_CROPPED_OBSERVATORY
            output_dir = Path(BASE_DIR) / '6_patches_from_cropped'
            print(f"\n✅ Selezionato: CROPPED")
            break
        elif choice == '2':
            input_hubble = INPUT_REGISTERED_HUBBLE
            input_obs = INPUT_REGISTERED_OBSERVATORY
            output_dir = Path(BASE_DIR) / '6_patches_from_registered'
            print(f"\n✅ Selezionato: REGISTERED")
            break
        else:
            print(f"❌ Scelta non valida.")
    
    output_hubble_patches = output_dir / 'hubble_native'
    output_obs_patches = output_dir / 'observatory_native'
    
    output_hubble_patches.mkdir(parents=True, exist_ok=True)
    output_obs_patches.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("ESTRAZIONE PATCHES")
    print(f"{'='*70}")
    
    hubble_patches = []
    obs_patches = []
    
    if input_hubble.exists():
        hubble_patches = extract_all_patches(input_hubble, output_hubble_patches, 'hubble', logger)
    
    if input_obs.exists():
        obs_patches = extract_all_patches(input_obs, output_obs_patches, 'observatory', logger)
    
    print(f"\n{'='*70}")
    print("📊 RIEPILOGO")
    print(f"{'='*70}")
    print(f"\n   Hubble patches: {len(hubble_patches)}")
    print(f"   Observatory patches: {len(obs_patches)}")
    print(f"   TOTALE: {len(hubble_patches) + len(obs_patches)}")
    
    if hubble_patches and obs_patches:
        pairs = create_patch_pairs(hubble_patches, obs_patches, output_dir, logger)
    
    print(f"\n{'='*70}")
    print("SPLIT DATASET")
    print(f"{'='*70}")
    
    if hubble_patches:
        print(f"\n📊 Hubble:")
        hubble_splits = create_dataset_split(hubble_patches, output_hubble_patches, logger)
        if hubble_splits:
            print(f"   Train: {len(hubble_splits['train'])}")
            print(f"   Val: {len(hubble_splits['val'])}")
            print(f"   Test: {len(hubble_splits['test'])}")
    
    if obs_patches:
        print(f"\n📊 Observatory:")
        obs_splits = create_dataset_split(obs_patches, output_obs_patches, logger)
        if obs_splits:
            print(f"   Train: {len(obs_splits['train'])}")
            print(f"   Val: {len(obs_splits['val'])}")
            print(f"   Test: {len(obs_splits['test'])}")
    
    print(f"\n✅ ESTRAZIONE COMPLETATA")
    
    return True

# ============================================================================
# MENU PRINCIPALE
# ============================================================================

def show_menu():
    """Mostra il menu principale."""
    print("\n" + "="*70)
    print("🚀 PIPELINE COMPLETA - MENU".center(70))
    print("="*70)
    print("\n📋 OPERAZIONI:")
    print("\n1️⃣  WCS - Conversione coordinate → WCS")
    print("2️⃣  REGISTRO - Registrazione immagini")
    print("3️⃣  CROPPED - Ritaglio + Mosaico")
    print("4️⃣  ANALIZZA - Analisi overlap")
    print("5️⃣  PATCH - Estrazione patches")
    print("6️⃣  TUTTI - Esegui tutti gli step (circa 30 min in tutto)")
    print("0️⃣  ESCI")
    print("\n" + "="*70)


def main():
    """Funzione principale con menu."""
    logger = setup_logging()
    
    print("\n" + "🌟"*35)
    print("PIPELINE ASTRONOMICA COMPLETA".center(70))
    print("🌟"*35)
    
    while True:
        show_menu()
        
        choice = input("\n👉 Scegli [0-6]: ").strip()
        
        if choice == '0':
            print("\n👋 Arrivederci!")
            break
        
        elif choice == '1':
            start = time.time()
            success = main_step1(logger)
            elapsed = time.time() - start
            print(f"\n⏱️ Tempo: {elapsed:.1f} secondi")
            print("\n✅ Step 1 completato!" if success else "\n❌ Step 1 fallito!")
        
        elif choice == '2':
            start = time.time()
            success = main_step2(logger)
            elapsed = time.time() - start
            print(f"\n⏱️ Tempo: {elapsed:.1f} secondi")
            print("\n✅ Step 2 completato!" if success else "\n❌ Step 2 fallito!")
        
        elif choice == '3':
            start = time.time()
            success = main_step3()
            elapsed = time.time() - start
            print(f"\n⏱️ Tempo: {elapsed:.1f} secondi")
            print("\n✅ Step 3 completato!" if success else "\n❌ Step 3 fallito!")
        
        elif choice == '4':
            start = time.time()
            success = main_step5()
            elapsed = time.time() - start
            print(f"\n⏱️ Tempo: {elapsed:.1f} secondi")
            print("\n✅ Step 5 completato!" if success else "\n❌ Step 5 fallito!")
        
        elif choice == '5':
            start = time.time()
            success = main_step6(logger)
            elapsed = time.time() - start
            print(f"\n⏱️ Tempo: {elapsed:.1f} secondi")
            print("\n✅ Step 6 completato!" if success else "\n❌ Step 6 fallito!")
        
        elif choice == '6':
            print("\n🚀 ESECUZIONE COMPLETA")
            print("="*70)
            
            total_start = time.time()
            
            print("\n▶️ STEP 1: WCS")
            success1 = main_step1(logger)
            
            if not success1:
                print("\n❌ Pipeline interrotta")
                continue
            
            print("\n▶️ STEP 2: REGISTRO")
            success2 = main_step2(logger)
            
            if not success2:
                print("\n❌ Pipeline interrotta")
                continue
            
            print("\n▶️ STEP 3: CROPPED")
            success3 = main_step3()
            
            if not success3:
                print("\n❌ Pipeline interrotta")
                continue
            
            print("\n▶️ STEP 5: ANALIZZA")
            success5 = main_step5()
            
            print("\n▶️ STEP 6: PATCH")
            success6 = main_step6(logger)
            
            total_time = time.time() - total_start
            
            print("\n" + "="*70)
            print("📊 RIEPILOGO PIPELINE")
            print("="*70)
            print(f"\n   Step 1: {'✅' if success1 else '❌'}")
            print(f"   Step 2: {'✅' if success2 else '❌'}")
            print(f"   Step 3: {'✅' if success3 else '❌'}")
            print(f"   Step 5: {'✅' if success5 else '❌'}")
            print(f"   Step 6: {'✅' if success6 else '❌'}")
            print(f"\n   ⏱️ Tempo totale: {total_time:.1f}s ({total_time/60:.1f} min)")
            
            if all([success1, success2, success3, success6]):
                print("\n✅ PIPELINE COMPLETATA!")
            else:
                print("\n⚠️ Pipeline con alcuni errori")
        
        else:
            print("\n❌ Scelta non valida.")
        
        input("\n⏸️ Premi INVIO per continuare...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()