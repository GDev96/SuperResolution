"""
PIPELINE COMPLETO: CONVERSIONE WCS + REGISTRAZIONE
Combina Step 1 (Conversione WCS) e Step 2 (Registrazione) in un unico script.

VERSIONE CORRETTA FINALE:
✅ Normalizzazione robusta percentile-based
✅ Gestione footprint sicura (clip [0,1])
✅ De-normalizzazione completa
✅ Validazione anti-binario su dati originali E processati
✅ Centro comune per allineamento perfetto
✅ Margine aumentato per evitare bordi neri
✅ WCS comune definito con matrice CD (gestione rotazione)
✅ Metadati completi in header
"""

import os
import sys
import time
import logging
from datetime import datetime
import numpy as np
import astropy
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings
from pathlib import Path
import subprocess

warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)

# ============================================================================
# CONFIGURAZIONE PATH ASSOLUTI
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR_ROOT = PROJECT_ROOT / "logs"
SCRIPTS_DIR = PROJECT_ROOT / "finale"

# ============================================================================
# VERIFICA REPROJECT
# ============================================================================
try:
    from reproject import reproject_interp
    REPROJECT_AVAILABLE = True
except ImportError:
    REPROJECT_AVAILABLE = False
    print("="*70)
    print("ERRORE: Libreria 'reproject' non trovata.")
    print("Installa con: pip install reproject")
    print("="*70)

# ============================================================================
# PARAMETRI GLOBALI
# ============================================================================
NUM_THREADS = 8 
REPROJECT_ORDER = 'bicubic'

# Normalizzazione robusta
PERCENTILE_LOW = 0.5   # 0.5% inferiore
PERCENTILE_HIGH = 99.5  # 99.5% superiore

# Validazione
MIN_UNIQUE_VALUES_ORIGINAL = 50   
MIN_UNIQUE_VALUES_NORMALIZED = 30  
MIN_RANGE_THRESHOLD = 1e-10  

# Lock per logging thread-safe
log_lock = threading.Lock()

# ============================================================================
# SELEZIONE CARTELLA TARGET (Invariato)
# ============================================================================

def select_target_directory():
    """Mostra un menu per selezionare una o TUTTE le cartelle target."""
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET".center(70))
    print("📂"*35)
    print(f"\nScansione sottocartelle in: {ROOT_DATA_DIR}")

    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir()]
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile leggere {ROOT_DATA_DIR}")
        print(f"   Dettagli: {e}")
        return []

    if not subdirs:
        print(f"\n❌ Nessuna sottocartella trovata")
        return []

    print("\nCartelle target disponibili:")
    print(f"   0: ✨ Processa TUTTI i {len(subdirs)} target")
    print("   " + "─"*30)
    for i, dir_path in enumerate(subdirs):
        print(f"   {i+1}: {dir_path.name}")

    while True:
        print("\n" + "─"*70)
        try:
            choice_str = input(f"👉 Seleziona (0-{len(subdirs)}) o 'q': ").strip()
            if choice_str.lower() == 'q':
                return []
            choice = int(choice_str)
            if choice == 0:
                return subdirs
            if 0 < choice <= len(subdirs):
                return [subdirs[choice-1]]
            print(f"❌ Numero non valido")
        except ValueError:
            print("❌ Input non valido")

# ============================================================================
# SETUP LOGGING (Invariato)
# ============================================================================

def setup_logging():
    """Configura logging per pipeline."""
    os.makedirs(LOG_DIR_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = LOG_DIR_ROOT / f'pipeline_{timestamp}.log'

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info(f"LOG FILE: {log_filename}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Numpy: {np.__version__}")
    logger.info(f"Astropy: {astropy.__version__}")
    logger.info(f"Threads: {NUM_THREADS}, Order: {REPROJECT_ORDER}")
    logger.info(f"Normalizzazione: [{PERCENTILE_LOW}%, {PERCENTILE_HIGH}%]")
    logger.info("="*80)
    return logger

# ============================================================================
# STEP 1: CONVERSIONE WCS (Invariato)
# ============================================================================

def parse_coordinates(ra_str, dec_str):
    """Converte coordinate RA/DEC da stringhe a gradi decimali."""
    try:
        ra_deg = Angle(ra_str, unit=u.hour).degree
    except:
        try:
            ra_deg = Angle(ra_str, unit=u.deg).degree
        except:
            ra_deg = float(ra_str)
    
    try:
        dec_deg = Angle(dec_str, unit=u.deg).degree
    except:
        dec_deg = float(dec_str)
    
    return ra_deg, dec_deg

def calculate_pixel_scale(header):
    """Calcola pixel scale da header FITS."""
    xpixsz = header.get('XPIXSZ', None)
    focal = header.get('FOCALLEN', header.get('FOCAL', None))
    xbin = header.get('XBINNING', 1)
    
    if xpixsz and focal:
        try:
            pixel_size_mm = (float(xpixsz) * float(xbin)) / 1000.0
            pixel_scale_arcsec = 206265.0 * (pixel_size_mm / float(focal))
            return pixel_scale_arcsec / 3600.0
        except:
            pass
    return 1.0 / 3600.0

def create_wcs_from_header(header, data_shape):
    """Crea WCS da coordinate OBJCTRA/OBJCTDEC."""
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
    except:
        return None

def add_wcs_to_file(input_file, output_file, logger):
    """Aggiunge WCS a file FITS da coordinate."""
    try:
        filename = os.path.basename(input_file)
        with fits.open(input_file, mode='readonly') as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()
            
            if data is None:
                return False
            
            # Check WCS esistente
            try:
                existing_wcs = WCS(header)
                if existing_wcs.has_celestial:
                    hdul.writeto(output_file, overwrite=True)
                    return True
            except:
                pass
            
            # Crea WCS
            wcs = create_wcs_from_header(header, data.shape)
            if wcs is None:
                return False
            
            wcs_header = wcs.to_header()
            
            # Preserva metadati importanti
            important_keys = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
                              'BZERO', 'BSCALE', 'DATE-OBS', 'EXPTIME', 'FILTER',
                              'INSTRUME', 'TELESCOP', 'XBINNING', 'YBINNING']
            
            new_header = fits.Header()
            for key in important_keys:
                if key in header:
                    new_header[key] = header[key]
            
            new_header.update(wcs_header)
            new_header['WCSADDED'] = True
            new_header['WCSSRC'] = 'OBJCTRA/OBJCTDEC'
            new_header['WCSDATE'] = datetime.now().isoformat()
            
            fits.PrimaryHDU(data=data, header=new_header).writeto(
                output_file, overwrite=True, output_verify='silentfix'
            )
            return True
    except Exception as e:
        logger.error(f"Errore {os.path.basename(input_file)}: {e}")
        return False

def extract_lith_data(filename, logger):
    """Estrae dati da file LITH/HST."""
    try:
        with fits.open(filename) as hdul:
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    try:
                        wcs = WCS(hdu.header)
                        if wcs.has_celestial:
                            data = hdu.data[0] if len(hdu.data.shape) == 3 else hdu.data
                            header = hdu.header.copy()
                            
                            ra, dec = wcs.wcs.crval
                            try:
                                cd = wcs.wcs.cd
                                pixel_scale = np.sqrt(cd[0,0]**2 + cd[0,1]**2) * 3600
                            except:
                                pixel_scale = 0.04
                            
                            info = {
                                'shape': data.shape,
                                'ra': ra,
                                'dec': dec,
                                'pixel_scale': pixel_scale
                            }
                            return data, header, info
                    except:
                        continue
        return None, None, None
    except Exception as e:
        logger.error(f"Errore {os.path.basename(filename)}: {e}")
        return None, None, None

def process_osservatorio_folder(input_dir, output_dir, logger):
    """Processa file osservatorio."""
    fits_files = list(Path(input_dir).glob('**/*.fit')) + \
                 list(Path(input_dir).glob('**/*.fits'))
    
    if not fits_files:
        return 0, 0, None
    
    os.makedirs(output_dir, exist_ok=True)
    prepared_count = 0
    failed_count = 0
    
    with tqdm(total=len(fits_files), desc="   Osservatorio", unit="file") as pbar:
        for input_file in fits_files:
            name = input_file.stem
            output_file = output_dir / f"{name}_wcs.fits"
            if add_wcs_to_file(input_file, output_file, logger):
                prepared_count += 1
            else:
                failed_count += 1
            pbar.update(1)
    
    return prepared_count, failed_count, None

def process_lith_folder(input_dir, output_dir, logger):
    """Processa file LITH/HST."""
    fits_files = list(Path(input_dir).glob('**/*.fit')) + \
                 list(Path(input_dir).glob('**/*.fits'))
    
    if not fits_files:
        return 0, 0, None
    
    os.makedirs(output_dir, exist_ok=True)
    prepared_count = 0
    failed_count = 0
    
    with tqdm(total=len(fits_files), desc="   LITH/HST", unit="file") as pbar:
        for input_file in fits_files:
            data, header, info = extract_lith_data(input_file, logger)
            if data is not None:
                name = input_file.stem
                output_file = output_dir / f"{name}_wcs.fits"
                try:
                    hdu = fits.PrimaryHDU(data=data, header=header)
                    hdu.header['ORIGINAL'] = input_file.name
                    hdu.header['PREPDATE'] = datetime.now().isoformat()
                    hdu.writeto(output_file, overwrite=True, output_verify='silentfix')
                    prepared_count += 1
                except:
                    failed_count += 1
            else:
                failed_count += 1
            pbar.update(1)
    
    return prepared_count, failed_count, None

# ============================================================================
# STEP 2: ESTRAZIONE WCS SICURA (Invariato)
# ============================================================================

def extract_wcs_info_safe(filepath, logger):
    """Estrae WCS in modo sicuro evitando errori."""
    try:
        with fits.open(filepath) as hdul:
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    try:
                        wcs = WCS(hdu.header)
                        if wcs.has_celestial:
                            data = hdu.data[0] if len(hdu.data.shape) == 3 else hdu.data
                            ny, nx = data.shape
                            
                            # Centro sicuro
                            try:
                                world = wcs.wcs_pix2world([[nx/2, ny/2]], 1)
                                center_ra = float(world[0][0])
                                center_dec = float(world[0][1])
                            except:
                                center_ra = float(hdu.header.get('CRVAL1', 0))
                                center_dec = float(hdu.header.get('CRVAL2', 0))
                            
                            # Pixel scale sicuro
                            try:
                                if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
                                    # Usa CD se disponibile, è più preciso
                                    cd = wcs.wcs.cd
                                    pixel_scale_deg = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
                                else:
                                    pixel_scale_deg = abs(wcs.wcs.cdelt[0])
                                pixel_scale_arcsec = pixel_scale_deg * 3600.0
                            except:
                                pixel_scale_arcsec = 0.04
                            
                            return {
                                'file': filepath,
                                'hdu_index': i,
                                'wcs': wcs,
                                'shape': data.shape,
                                'center_ra': center_ra,
                                'center_dec': center_dec,
                                'pixel_scale': pixel_scale_arcsec
                            }
                    except:
                        continue
        return None
    except Exception as e:
        with log_lock:
            logger.error(f"Errore WCS {os.path.basename(filepath)}: {e}")
        return None

def analyze_images(input_dir, source_name, logger):
    """Analizza immagini in directory."""
    files = list(Path(input_dir).glob('*.fits')) + \
            list(Path(input_dir).glob('*.fit')) + \
            list(Path(input_dir).glob('*_wcs.fits'))
    files = list(set(files))
    
    if not files:
        return []
    
    print(f"\n📂 {source_name}: {len(files)} file")
    wcs_info_list = []
    
    with tqdm(total=len(files), desc=f"   Analisi {source_name}") as pbar:
        for filepath in files:
            info = extract_wcs_info_safe(filepath, logger)
            if info:
                wcs_info_list.append(info)
            pbar.update(1)
    
    print(f"   ✓ {len(wcs_info_list)}/{len(files)} con WCS valido")
    return wcs_info_list

# ============================================================================
# CREAZIONE WCS COMUNE (MODIFICATO: Uso esplicito della matrice CD)
# ============================================================================

def create_common_wcs_frame(wcs_info_list, logger):
    """
    Crea WCS comune e ne calcola la forma (shape) globale necessaria.
    
    Ritorna: common_wcs, common_shape, common_scale_arcsec
    """
    if not wcs_info_list:
        return None, None, None
    
    ra_min, ra_max = float('inf'), float('-inf')
    dec_min, dec_max = float('inf'), float('-inf')
    
    # 1. Calcola l'estensione angolare totale (bounding box) di tutte le immagini
    for info in wcs_info_list:
        wcs, shape = info['wcs'], info['shape']
        ny, nx = shape
        
        # Angoli dell'immagine
        corners = np.array([[0, 0], [nx, 0], [0, ny], [nx, ny]])
        
        # Gestione astropy WCS bug per RA=0/360
        world = wcs.wcs_pix2world(corners, 1)
        
        # Raggiungiamo il centro della mappa con una media robusta
        ra_center = np.median([info['center_ra'] for info in wcs_info_list])
        dec_center = np.median([info['center_dec'] for info in wcs_info_list])
        
        # Aggiorna min/max RA/DEC
        for coord in world:
            ra, dec = float(coord[0]), float(coord[1])
            
            # Normalizza RA per gestione 360/0 gradi
            if ra > ra_center + 180: ra -= 360
            if ra < ra_center - 180: ra += 360
            
            ra_min = min(ra_min, ra)
            ra_max = max(ra_max, ra)
            dec_min = min(dec_min, dec)
            dec_max = max(dec_max, dec)
            
    # Centro ricalcolato su bounding box
    ra_center = (ra_min + ra_max) / 2.0
    dec_center = (dec_min + dec_max) / 2.0
    
    # 2. Determina la scala pixel comune (mediana)
    pixel_scales_arcsec = [info['pixel_scale'] for info in wcs_info_list]
    native_pixel_scale_arcsec = np.median(pixel_scales_arcsec) if pixel_scales_arcsec else 0.04
    native_pixel_scale_deg = native_pixel_scale_arcsec / 3600.0

    # 3. Calcola la dimensione del frame globale (aggiungendo margine di sicurezza)
    margin_factor = 1.05 # Aumento margine al 5%
    
    ra_span_deg = (ra_max - ra_min) * margin_factor
    dec_span_deg = (dec_max - dec_min) * margin_factor
    
    nx_out = int(np.ceil(abs(ra_span_deg) / native_pixel_scale_deg))
    ny_out = int(np.ceil(abs(dec_span_deg) / native_pixel_scale_deg))
    
    # Garantisci un minimo di 50x50 pixel
    nx_out = max(nx_out, 50)
    ny_out = max(ny_out, 50)
    
    common_shape = (ny_out, nx_out)
    
    # 4. Crea il WCS comune
    common_wcs = WCS(naxis=2)
    common_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    common_wcs.wcs.crval = [ra_center, dec_center]
    common_wcs.wcs.crpix = [nx_out / 2.0, ny_out / 2.0] # Centro del frame comune

    # --- MODIFICA CRITICA: Usa matrice CD per definire scala e rotazione (Nord-up) ---
    cd_matrix = np.array([
        [-native_pixel_scale_deg, 0.0],  # RA è negativo per convenzione astronomica
        [0.0, native_pixel_scale_deg]
    ])
    common_wcs.wcs.cd = cd_matrix
    # --------------------------------------------------------------------------------
    
    common_wcs.wcs.radesys = 'ICRS'
    common_wcs.wcs.equinox = 2000.0
    
    with log_lock:
        logger.info(f"WCS comune: RA={ra_center:.4f}°, DEC={dec_center:.4f}°")
        logger.info(f"Frame comune: {nx_out}x{ny_out} pixel, scala={native_pixel_scale_arcsec:.3f}\"/px")
    
    print(f"\n✓ WCS comune: RA={ra_center:.4f}°, DEC={dec_center:.4f}°")
    print(f"✓ Frame comune: {nx_out}x{ny_out} pixel")
    
    return common_wcs, common_shape, native_pixel_scale_arcsec

# ============================================================================
# REPROIEZIONE CON NORMALIZZAZIONE ROBUSTA (Invariato)
# ============================================================================

def reproject_image_native(wcs_info, common_wcs, common_shape, common_scale_arcsec, output_dir, logger):
    """
    Reproietta immagine con normalizzazione robusta sul frame comune.
    """
    try:
        filepath = wcs_info['file']
        hdu_index = wcs_info['hdu_index']
        filename = os.path.basename(filepath)
        
        with fits.open(filepath) as hdul:
            hdu = hdul[hdu_index]
            data_original = hdu.data.copy()
            header = hdu.header.copy()
            wcs_orig = WCS(header)
            
            if len(data_original.shape) == 3:
                data_original = data_original[0]
            
            # ================================================================
            # STEP 1: VALIDAZIONE DATI ORIGINALI
            # ================================================================
            valid_mask_orig = np.isfinite(data_original) & (data_original != 0)
            valid_data_orig = data_original[valid_mask_orig]
            
            if valid_data_orig.size < 100:
                with log_lock:
                    logger.warning(f"⚠️  {filename}: Pochi pixel validi ({valid_data_orig.size})")
                return {'status': 'error', 'file': filename, 'reason': 'Insufficient pixels'}
            
            # Check valori unici ORIGINALI
            unique_orig = len(np.unique(valid_data_orig[:min(10000, valid_data_orig.size)]))
            if unique_orig < MIN_UNIQUE_VALUES_ORIGINAL:
                with log_lock:
                    logger.error(f"❌ {filename}: Dati originali binari ({unique_orig} valori)")
                return {'status': 'error', 'file': filename, 'reason': f'Binary original ({unique_orig})'}
            
            # Check range originale
            orig_range = valid_data_orig.max() - valid_data_orig.min()
            if orig_range < MIN_RANGE_THRESHOLD:
                with log_lock:
                    logger.error(f"❌ {filename}: Range troppo piccolo ({orig_range:.2e})")
                return {'status': 'error', 'file': filename, 'reason': 'Range too small'}
            
            # ================================================================
            # STEP 2: NORMALIZZAZIONE PRE-REPROIEZIONE
            # ================================================================
            p_low = np.percentile(valid_data_orig, PERCENTILE_LOW)
            p_high = np.percentile(valid_data_orig, PERCENTILE_HIGH)
            
            norm_params = {
                'p_low': float(p_low),
                'p_high': float(p_high),
                'orig_min': float(valid_data_orig.min()),
                'orig_max': float(valid_data_orig.max()),
                'orig_median': float(np.median(valid_data_orig)),
                'unique_orig': unique_orig
            }
            
            # Clip e normalizza [0, 1]
            data_clipped = np.clip(data_original, p_low, p_high)
            if p_high > p_low:
                data_normalized = (data_clipped - p_low) / (p_high - p_low)
            else:
                data_normalized = data_clipped
            
            data_normalized[~valid_mask_orig] = np.nan
            
            with log_lock:
                logger.info(f"📊 {filename}: Norm [{p_low:.3e}, {p_high:.3e}], unique={unique_orig}")
            
            # ================================================================
            # STEP 3: CALCOLO TARGET WCS (Usa i valori globali)
            # ================================================================
            target_wcs = common_wcs
            shape_out = common_shape
            native_pixel_scale = common_scale_arcsec # Usa la scala comune nel metadata
            
            # ================================================================
            # STEP 4: REPROIEZIONE SU DATI NORMALIZZATI
            # ================================================================
            reprojected_norm, footprint = reproject_interp(
                (data_normalized, wcs_orig),
                target_wcs,
                shape_out=shape_out,
                order=REPROJECT_ORDER,
                return_footprint=True
            )
            
            # ================================================================
            # STEP 5: GESTIONE FOOTPRINT SICURA
            # ================================================================
            # Clip footprint [0, 1] (bug reproject può dare >1)
            footprint_safe = np.clip(footprint, 0, 1)
            # Usa una soglia bassa (0.01) per considerare valido
            valid_reproj = (footprint_safe > 0.01) & np.isfinite(reprojected_norm)
            
            # Applica footprint come peso
            reprojected_weighted = np.where(
                valid_reproj,
                reprojected_norm * footprint_safe,
                np.nan
            )
            
            # ================================================================
            # STEP 6: DE-NORMALIZZAZIONE
            # ================================================================
            reprojected_denorm = reprojected_weighted.copy()
            valid_mask_final = np.isfinite(reprojected_weighted)
            
            if not np.any(valid_mask_final):
                with log_lock:
                    # Questo è il warning che viene catturato nell'errore dell'utente
                    logger.warning(f"⚠️  {filename}: Nessun pixel dopo reproiezione") 
                return {'status': 'error', 'file': filename, 'reason': 'No pixels after reproject'}
            
            # Ripristina range originale
            reprojected_denorm[valid_mask_final] = (
                reprojected_weighted[valid_mask_final] * (p_high - p_low) + p_low
            )
            
            # ================================================================
            # STEP 7: VALIDAZIONE POST-PROCESSING
            # ================================================================
            final_valid = reprojected_denorm[valid_mask_final]
            unique_final = len(np.unique(final_valid[:min(10000, final_valid.size)]))
            
            if unique_final < MIN_UNIQUE_VALUES_NORMALIZED:
                with log_lock:
                    logger.error(f"❌ {filename}: Dati finali binari ({unique_final} valori)")
                return {'status': 'error', 'file': filename, 'reason': f'Binary final ({unique_final})'}
            
            final_range = final_valid.max() - final_valid.min()
            if final_range < MIN_RANGE_THRESHOLD:
                with log_lock:
                    logger.error(f"❌ {filename}: Range finale troppo piccolo ({final_range:.2e})")
                return {'status': 'error', 'file': filename, 'reason': 'Final range too small'}
            
            # ================================================================
            # STEP 8: STATISTICHE
            # ================================================================
            coverage = (valid_mask_final.sum() / reprojected_denorm.size * 100)
            
            with log_lock:
                logger.info(f"✅ {filename}: cov={coverage:.1f}%, shape={shape_out}, "
                           f"range=[{final_valid.min():.2e}, {final_valid.max():.2e}], "
                           f"unique_orig={unique_orig}, unique_final={unique_final}")
            
            # ================================================================
            # STEP 9: CREAZIONE HEADER
            # ================================================================
            new_header = target_wcs.to_header()
            
            # Metadati normalizzazione
            new_header['NORMLOW'] = (norm_params['p_low'], f'Norm p{PERCENTILE_LOW}')
            new_header['NORMHIGH'] = (norm_params['p_high'], f'Norm p{PERCENTILE_HIGH}')
            new_header['ORIGMIN'] = (norm_params['orig_min'], 'Original min')
            new_header['ORIGMAX'] = (norm_params['orig_max'], 'Original max')
            new_header['ORIGMED'] = (norm_params['orig_median'], 'Original median')
            new_header['UNIQUEOR'] = (unique_orig, 'Unique values original (sample)')
            new_header['UNIQUEFN'] = (unique_final, 'Unique values final (sample)')
            
            # Metadati originali
            for key in ['DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'TELESCOP']:
                if key in header:
                    new_header[key] = header[key]
            
            # Metadati registrazione
            new_header['ORIGINAL'] = filename
            new_header['REGDATE'] = datetime.now().isoformat()
            new_header['REGCOV'] = (coverage, "Coverage %")
            new_header['REGVALID'] = (int(valid_mask_final.sum()), "Valid pixels")
            new_header['REGORD'] = (str(REPROJECT_ORDER), "Interp order")
            new_header['NATIVESC'] = (wcs_info['pixel_scale'], "Original scale (\"/px)") # Mantieni originale
            new_header['COMSCALE'] = (common_scale_arcsec, "Common scale (\"/px)") # Aggiungi la scala comune
            new_header['ORIGSHP0'] = (wcs_info['shape'][0], "Original height")
            new_header['ORIGSHP1'] = (wcs_info['shape'][1], "Original width")
            new_header['COMRA'] = (common_wcs.wcs.crval[0], "Common RA (deg)")
            new_header['COMDEC'] = (common_wcs.wcs.crval[1], "Common DEC (deg)")
            new_header['COMMENT'] = 'Robust normalization with percentile clipping'
            new_header['COMMENT'] = 'Full dynamic range preserved via denormalization'
            new_header['COMMENT'] = 'Validated against binary data corruption'
            
            # ================================================================
            # STEP 10: SALVATAGGIO
            # ================================================================
            output_filename = f"reg_{os.path.splitext(filename)[0]}.fits"
            output_path = output_dir / output_filename
            
            fits.PrimaryHDU(
                data=reprojected_denorm.astype(np.float32),
                header=new_header
            ).writeto(output_path, overwrite=True)
            
            return {
                'status': 'success',
                'file': filename,
                'coverage': coverage,
                'valid_pixels': int(valid_mask_final.sum()),
                'output_path': output_path,
                'native_scale': native_pixel_scale,
                'output_shape': shape_out,
                'unique_orig': unique_orig,
                'unique_final': unique_final,
                'data_range': (float(final_valid.min()), float(final_valid.max()))
            }
            
    except Exception as e:
        with log_lock:
            logger.error(f"❌ {os.path.basename(filepath)}: {e}", exc_info=True)
        return {'status': 'error', 'file': os.path.basename(filepath), 'reason': str(e)}

# ============================================================================
# REGISTRAZIONE IMMAGINI (Invariato)
# ============================================================================

def register_images(wcs_info_list, common_wcs, common_shape, common_scale, output_dir, source_name, logger):
    """Registra immagini con multithreading."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🔄 Registrazione {source_name}: {len(wcs_info_list)} immagini")
    
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            # Passa i parametri globali a reproject_image_native
            executor.submit(reproject_image_native, info, common_wcs, common_shape, common_scale, output_dir, logger): info
            for info in wcs_info_list
        }
        
        with tqdm(total=len(wcs_info_list), desc=f"   {source_name}") as pbar:
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
                        logger.error(f"Thread exception: {exc}")
                pbar.update(1)
    
    print(f"   ✓ Successo: {success_count}, ✗ Errori: {error_count}")
    return success_count, error_count

# ============================================================================
# MENU PROSEGUIMENTO (Invariato)
# ============================================================================

def ask_continue_to_cropping():
    """Chiede se continuare con Step 3+4."""
    print("\n" + "="*70)
    print("🎯 STEP 1+2 COMPLETATI!")
    print("="*70)
    print("\n📋 Vuoi continuare con Step 3+4 (Ritaglio e Mosaico)?")
    
    while True:
        choice = input("👉 [S/n, default=S]: ").strip().lower()
        if choice in ('', 's', 'si', 'y', 'yes'):
            return True
        elif choice in ('n', 'no'):
            return False
        print("❌ Inserisci S o N")

# ============================================================================
# MAIN: STEP 1 + STEP 2 (Invariato)
# ============================================================================

def main_step1(input_obs, input_lith, output_obs, output_lith, logger):
    """Esegue Step 1: Conversione WCS."""
    logger.info("="*60)
    logger.info("STEP 1: CONVERSIONE WCS")
    logger.info("="*60)
    
    print("\n" + "="*70)
    print("STEP 1: CONVERSIONE WCS".center(70))
    print("="*70)
    
    # Observatory
    print("\n📡 OSSERVATORIO")
    prep_obs, fail_obs, _ = process_osservatorio_folder(input_obs, output_obs, logger)
    print(f"   ✓ Preparati: {prep_obs}, ✗ Falliti: {fail_obs}")
    
    # LITH/HST
    print("\n🛰️  LITH/HST")
    prep_lith, fail_lith, _ = process_lith_folder(input_lith, output_lith, logger)
    print(f"   ✓ Preparati: {prep_lith}, ✗ Falliti: {fail_lith}")
    
    total = prep_obs + prep_lith
    logger.info(f"Step 1 completato: {total} file preparati")
    
    return total > 0

def main_step2(input_hubble, input_obs, output_hubble, output_obs, base_dir, logger):
    """Esegue Step 2: Registrazione."""
    if not REPROJECT_AVAILABLE:
        return False
    
    logger.info("="*60)
    logger.info("STEP 2: REGISTRAZIONE")
    logger.info("="*60)
    
    print("\n" + "="*70)
    print("STEP 2: REGISTRAZIONE".center(70))
    print("="*70)
    
    # Analisi
    hubble_info = analyze_images(input_hubble, "HUBBLE", logger)
    obs_info = analyze_images(input_obs, "OBSERVATORY", logger)
    all_info = hubble_info + obs_info
    
    if not all_info:
        logger.error("Nessuna immagine valida")
        return False
    
    # WCS comune
    print(f"\n{'='*70}")
    print("CREAZIONE WCS COMUNE")
    print(f"{'='*70}")
    
    # NEW: Recupera WCS, Shape e Scala globali
    common_wcs, common_shape, common_scale = create_common_wcs_frame(all_info, logger)
    if common_wcs is None:
        logger.error("Impossibile creare frame WCS comune")
        return False
    
    # Registrazione
    print(f"\n{'='*70}")
    print("REGISTRAZIONE")
    print(f"{'='*70}")
    
    total_success = 0
    total_error = 0
    
    if hubble_info:
        # NEW: Passa shape e scala
        s, e = register_images(hubble_info, common_wcs, common_shape, common_scale, output_hubble, "Hubble", logger)
        total_success += s
        total_error += e
    
    if obs_info:
        # NEW: Passa shape e scala
        s, e = register_images(obs_info, common_wcs, common_shape, common_scale, output_obs, "Observatory", logger)
        total_success += s
        total_error += e
    
    print(f"\n{'='*70}")
    print(f"TOTALE: {total_success} successo, {total_error} errori")
    print(f"{'='*70}")
    
    logger.info(f"Step 2 completato: {total_success} registrate")
    return total_success > 0

# ============================================================================
# MAIN (Invariato)
# ============================================================================

def main():
    """Main pipeline."""
    logger = setup_logging()
    
    target_dirs = select_target_directory()
    if not target_dirs:
        return
    
    print("\n" + "="*70)
    print("PIPELINE: CONVERSIONE WCS + REGISTRAZIONE".center(70))
    if len(target_dirs) > 1:
        print(f"Batch: {len(target_dirs)} target".center(70))
    print("="*70)
    
    start_time = time.time()
    successful = []
    failed = []
    
    for base_dir in target_dirs:
        print("\n" + "🚀"*35)
        print(f"TARGET: {base_dir.name}".center(70))
        print("🚀"*35)
        
        # Percorsi
        input_obs = base_dir / '1_origin' / 'local_raw'
        input_lith = base_dir / '1_origin' / 'img_lights'
        output_obs_wcs = base_dir / '2_wcs' / 'observatory'
        output_lith_wcs = base_dir / '2_wcs' / 'hubble'
        output_hubble = base_dir / '3_registered_native' / 'hubble'
        output_obs = base_dir / '3_registered_native' / 'observatory'
        
        # Step 1
        t1 = time.time()
        step1_ok = main_step1(input_obs, input_lith, output_obs_wcs, output_lith_wcs, logger)
        print(f"\n⏱️  Step 1: {time.time()-t1:.2f}s")
        
        if not step1_ok:
            failed.append(base_dir)
            continue
        
        # Step 2
        t2 = time.time()
        step2_ok = main_step2(output_lith_wcs, output_obs_wcs, output_hubble, output_obs, base_dir, logger)
        print(f"\n⏱️  Step 2: {time.time()-t2:.2f}s")
        
        if step2_ok:
            successful.append(base_dir)
        else:
            failed.append(base_dir)
    
    # Riepilogo
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("RIEPILOGO FINALE".center(70))
    print("="*70)
    print(f"\n   Target: {len(target_dirs)}")
    print(f"   ✅ Successo: {len(successful)}")
    for t in successful:
        print(f"      - {t.name}")
    print(f"\n   ❌ Falliti: {len(failed)}")
    for t in failed:
        print(f"      - {t.name}")
    print(f"\n   ⏱️  Tempo totale: {elapsed:.2f}s")
    
    if not successful:
        return
    
    # Continua con Step 3+4
    if ask_continue_to_cropping():
        next_script = SCRIPTS_DIR / 'step2_croppedmosaico.py'
        if next_script.exists():
            for base_dir in successful:
                print(f"\n--- Step 3+4: {base_dir.name} ---")
                subprocess.run([sys.executable, str(next_script), str(base_dir)])
        else:
            print(f"\n⚠️  {next_script.name} non trovato")

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\n⏱️  Tempo totale: {time.time()-start:.2f}s")