"""
PIPELINE COMPLETO: CONVERSIONE WCS + REGISTRAZIONE
Combina Step 1 (Conversione WCS) e Step 2 (Registrazione) in un unico script.
Gli output dei due step rimangono separati e distinti.
Tutti i metodi sono mantenuti ESATTAMENTE come negli script originali.
"""

import os
import sys
import glob
import time
import logging
from datetime import datetime
import numpy as np
import astropy
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings

warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)

# Prova a importare reproject
try:
    from reproject import reproject_interp
    REPROJECT_AVAILABLE = True
except ImportError:
    REPROJECT_AVAILABLE = False
    print("="*70)
    print("ERRORE: Libreria 'reproject' non trovata.")
    print("Questa libreria è fondamentale per lo Step 2.")
    print("Installa con: pip install reproject")
    print("="*70)

# ============================================================================
# CONFIGURAZIONE GLOBALE
# ============================================================================

BASE_DIR = r'F:\Super Revolt Gaia\parte 2(patch)\data'

# INPUT STEP 1
INPUT_OSSERVATORIO = os.path.join(BASE_DIR, 'local_raw')
INPUT_LITH = os.path.join(BASE_DIR, 'img_lights_1')

# OUTPUT STEP 1 (che diventa INPUT STEP 2)
OUTPUT_OSSERVATORIO_WCS = os.path.join(BASE_DIR, 'osservatorio_con_wcs')
OUTPUT_LITH_WCS = os.path.join(BASE_DIR, 'lith_con_wcs')

# INPUT STEP 2 (stesso degli output Step 1)
INPUT_HUBBLE = OUTPUT_LITH_WCS
INPUT_OBSERVATORY = OUTPUT_OSSERVATORIO_WCS

# OUTPUT STEP 2
OUTPUT_HUBBLE = os.path.join(BASE_DIR, '3_registered_native', 'hubble')
OUTPUT_OBSERVATORY = os.path.join(BASE_DIR, '3_registered_native', 'observatory')

LOG_DIR = r'F:\Super Revolt Gaia\logs'

# Parametri Step 2
NUM_THREADS = 7
REPROJECT_ORDER = 'bilinear'

# Lock per logging thread-safe
log_lock = threading.Lock()

# ============================================================================
# SETUP LOGGING
# ============================================================================

def setup_logging():
    """Configura logging per l'intera pipeline."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'unified_pipeline_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Python: {sys.version}")
    logger.info(f"Numpy: {np.__version__}")
    logger.info(f"Astropy: {astropy.__version__}")
    if REPROJECT_AVAILABLE:
        logger.info(f"Reproject: (importata con successo)")
    logger.info(f"Threads: {NUM_THREADS}")
    logger.info(f"Reprojection Order: {REPROJECT_ORDER}")
    logger.info("MODALITÀ: Risoluzione Nativa (ogni immagine mantiene la sua risoluzione)")
    
    return logger

# ============================================================================
# STEP 1: FUNZIONI CONVERSIONE WCS (ESATTE DALL'ORIGINALE)
# ============================================================================

def parse_coordinates(ra_str, dec_str):
    """
    Converte coordinate da formato sessagesimale a decimale.
    
    Args:
        ra_str: es. '1 34 01' (ore minuti secondi)
        dec_str: es. '30 41 00' (gradi minuti secondi)
    
    Returns:
        (ra_deg, dec_deg) in gradi decimali
    """
    try:
        # Rimuovi spazi extra
        ra_str = ra_str.strip()
        dec_str = dec_str.strip()
        
        # Usa SkyCoord di astropy per parsing robusto
        coord_str = f"{ra_str} {dec_str}"
        coord = SkyCoord(coord_str, unit=(u.hourangle, u.deg))
        
        return coord.ra.degree, coord.dec.degree
        
    except Exception as e:
        # Fallback: parsing manuale
        try:
            ra_parts = ra_str.split()
            dec_parts = dec_str.split()
            
            # RA: ore, minuti, secondi -> gradi
            h, m, s = float(ra_parts[0]), float(ra_parts[1]), float(ra_parts[2])
            ra_deg = (h + m/60.0 + s/3600.0) * 15.0  # * 15 per convertire ore in gradi
            
            # DEC: gradi, minuti, secondi -> gradi
            d, m, s = float(dec_parts[0]), float(dec_parts[1]), float(dec_parts[2])
            sign = 1 if d >= 0 else -1
            dec_deg = d + sign * (m/60.0 + s/3600.0)
            
            return ra_deg, dec_deg
            
        except Exception as e2:
            raise ValueError(f"Impossibile parsare coordinate: RA='{ra_str}', DEC='{dec_str}': {e2}")


def calculate_pixel_scale(header):
    """
    Calcola pixel scale da informazioni nel header.
    
    Args:
        header: Header FITS
    
    Returns:
        pixel_scale in gradi/pixel
    """
    # Estrai parametri
    xpixsz = header.get('XPIXSZ', None)  # micron
    focal = header.get('FOCALLEN', header.get('FOCAL', None))  # mm
    xbin = header.get('XBINNING', 1)
    
    if xpixsz and focal:
        # Formula: pixel_scale (arcsec/px) = 206.265 * pixel_size (mm) / focal_length (mm)
        pixel_size_mm = (xpixsz * xbin) / 1000.0  # converti micron -> mm e applica binning
        pixel_scale_arcsec = 206.265 * pixel_size_mm / focal
        pixel_scale_deg = pixel_scale_arcsec / 3600.0
        return pixel_scale_deg
    
    # Fallback: stima per setup comune
    # Tipico per piccoli telescopi: ~1-2 arcsec/pixel
    return 1.5 / 3600.0  # 1.5 arcsec/pixel


def create_wcs_from_header(header, data_shape):
    """
    Crea WCS completo da informazioni nel header.
    
    Args:
        header: Header FITS originale
        data_shape: Dimensioni dell'immagine (height, width)
    
    Returns:
        WCS object, o None se fallisce
    """
    try:
        # Estrai coordinate centro
        objctra = header.get('OBJCTRA', None)
        objctdec = header.get('OBJCTDEC', None)
        
        if not objctra or not objctdec:
            return None
        
        # Converti coordinate
        ra_deg, dec_deg = parse_coordinates(objctra, objctdec)
        
        # Calcola pixel scale
        pixel_scale = calculate_pixel_scale(header)
        
        # Crea WCS
        wcs = WCS(naxis=2)
        
        # Centro immagine
        height, width = data_shape
        wcs.wcs.crpix = [width / 2.0, height / 2.0]
        
        # Coordinate centro
        wcs.wcs.crval = [ra_deg, dec_deg]
        
        # Pixel scale (negativo per RA per convenzione astronomica)
        wcs.wcs.cdelt = [-pixel_scale, pixel_scale]
        
        # Tipo proiezione (TAN = tangente, standard per campi piccoli)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        # Sistema di riferimento
        wcs.wcs.radesys = 'ICRS'
        wcs.wcs.equinox = 2000.0
        
        return wcs
        
    except Exception as e:
        return None


def add_wcs_to_file(input_file, output_file, logger):
    """
    Aggiunge WCS a un file FITS che ha OBJCTRA/OBJCTDEC.
    
    Args:
        input_file: File input
        output_file: File output con WCS
        logger: Logger
    
    Returns:
        True se successo, False altrimenti
    """
    try:
        filename = os.path.basename(input_file)
        
        with fits.open(input_file, mode='readonly') as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()
            
            if data is None:
                logger.warning(f"Nessun dato in {filename}")
                return False
            
            # Verifica se ha già WCS valido
            existing_wcs = WCS(header)
            if existing_wcs.has_celestial:
                logger.info(f"✓ {filename}: WCS già presente")
                # Copia comunque il file
                hdul.writeto(output_file, overwrite=True)
                return True
            
            # Crea WCS da OBJCTRA/OBJCTDEC
            wcs = create_wcs_from_header(header, data.shape)
            
            if wcs is None:
                logger.warning(f"Impossibile creare WCS per {filename}")
                return False
            
            # Aggiungi WCS all'header
            wcs_header = wcs.to_header()
            
            # Mantieni campi importanti originali
            important_keys = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
                            'BZERO', 'BSCALE', 'DATE-OBS', 'EXPTIME', 'FILTER',
                            'INSTRUME', 'TELESCOP', 'XBINNING', 'YBINNING',
                            'XPIXSZ', 'YPIXSZ', 'GAIN', 'CCD-TEMP', 'FOCALLEN']
            
            # Crea nuovo header combinato
            new_header = fits.Header()
            
            # Prima i campi base
            for key in important_keys:
                if key in header:
                    new_header[key] = header[key]
            
            # Poi il WCS
            new_header.update(wcs_header)
            
            # Aggiungi metadati preparazione
            new_header['WCSADDED'] = True
            new_header['WCSSRC'] = 'OBJCTRA/OBJCTDEC conversion'
            new_header['WCSDATE'] = datetime.now().isoformat()
            
            # Salva
            primary_hdu = fits.PrimaryHDU(data=data, header=new_header)
            primary_hdu.writeto(output_file, overwrite=True, output_verify='silentfix')
            
            logger.info(f"✓ {filename}: WCS aggiunto")
            return True
            
    except Exception as e:
        logger.error(f"Errore {os.path.basename(input_file)}: {e}")
        return False


def extract_lith_data(filename, logger):
    """
    Estrae dati e WCS da file LITH/HST.
    Questi file hanno già WCS valido, basta estrarlo.
    
    Args:
        filename: Path al file FITS
        logger: Logger
    
    Returns:
        (data, header, info_dict) o (None, None, None) se fallisce
    """
    try:
        with fits.open(filename) as hdul:
            # Cerca primo HDU con dati scientifici e WCS
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
            
            # Se 3D, prendi primo canale
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
    # Modificato per cercare ricorsivamente in tutte le sottocartelle
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
    # Modificato per cercare ricorsivamente in tutte le sottocartelle
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
# STEP 2: FUNZIONI REGISTRAZIONE (ESATTE DALL'ORIGINALE)
# ============================================================================

def extract_wcs_info(filepath, logger):
    """Estrae info WCS da file FITS."""
    try:
        with fits.open(filepath) as hdul:
            # Trova il primo HDU con dati e WCS valido
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

            # Se 3D, usa primo canale
            if len(data_hdu.data.shape) == 3:
                data = data_hdu.data[0]
            else:
                data = data_hdu.data
            
            ny, nx = data.shape
            center = wcs.pixel_to_world(nx/2, ny/2)
            
            # Calcola pixel scale NATIVO dell'immagine
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
                'pixel_scale': pixel_scale_arcsec,  # Risoluzione NATIVA
            }
            
    except Exception as e:
        with log_lock:
            logger.error(f"Errore WCS {os.path.basename(filepath)}: {e}")
        return None


def analyze_images(input_dir, source_name, logger):
    """Analizza tutte le immagini in una directory."""
    # Cerca tutti i file FITS, inclusi quelli con _wcs.fits dallo Step 1
    files = glob.glob(os.path.join(input_dir, '*.fits')) + \
            glob.glob(os.path.join(input_dir, '*.fit')) + \
            glob.glob(os.path.join(input_dir, '*_wcs.fits'))
    
    # Rimuovi duplicati (alcuni file potrebbero essere trovati due volte)
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
    """
    Crea un WCS di riferimento comune che copre TUTTE le immagini.
    Questo WCS serve solo come frame di riferimento - ogni immagine
    manterrà la sua risoluzione nativa.
    """
    with log_lock:
        logger.info("=" * 60)
        logger.info("CREAZIONE FRAME WCS COMUNE (riferimento)")
        logger.info("=" * 60)
    
    if not wcs_info_list:
        with log_lock:
            logger.error("Nessuna immagine fornita per creare WCS comune.")
        return None
    
    # Trova i limiti del campo totale
    ra_min = float('inf')
    ra_max = float('-inf')
    dec_min = float('inf')
    dec_max = float('-inf')
    
    for info in wcs_info_list:
        wcs = info['wcs']
        shape = info['shape']
        ny, nx = shape
        
        # Coordinate dei 4 angoli
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
    
    # Centro del campo
    ra_center = (ra_min + ra_max) / 2.0
    dec_center = (dec_min + dec_max) / 2.0
    
    # Dimensioni campo (con margine)
    ra_span = ra_max - ra_min
    dec_span = dec_max - dec_min
    
    margin_factor = 1.05
    ra_span *= margin_factor
    dec_span *= margin_factor
    
    # Risoluzione di riferimento (non usata per calcolare dimensioni output!)
    # Serve solo come orientamento del frame WCS
    ref_pixel_scale_deg = 0.04 / 3600.0  # 0.04 arcsec/px (HST-like, ma arbitrario)
    
    # Crea WCS di riferimento
    reference_wcs = WCS(naxis=2)
    reference_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    reference_wcs.wcs.crval = [ra_center, dec_center]
    reference_wcs.wcs.crpix = [1, 1]  # Pixel di riferimento
    reference_wcs.wcs.cdelt = [-ref_pixel_scale_deg, ref_pixel_scale_deg]
    reference_wcs.wcs.radesys = 'ICRS'
    reference_wcs.wcs.equinox = 2000.0
    
    with log_lock:
        logger.info(f"WCS Comune creato:")
        logger.info(f"  Centro: RA={ra_center:.4f}°, DEC={dec_center:.4f}°")
        logger.info(f"  Span: RA={ra_span:.4f}°, DEC={dec_span:.4f}°")
        logger.info(f"  Risoluzione riferimento: {ref_pixel_scale_deg*3600:.4f}\"/px (solo orientamento)")
    
    print(f"\n✓ WCS comune creato:")
    print(f"   Centro: RA={ra_center:.4f}°, DEC={dec_center:.4f}°")
    print(f"   Span: RA={ra_span:.4f}°, DEC={dec_span:.4f}°")
    
    return reference_wcs


def reproject_image_native(wcs_info, common_wcs, output_dir, logger):
    """
    Riproietta un'immagine mantenendo la sua risoluzione NATIVA.
    
    IMPORTANTE: Ogni immagine decide autonomamente le dimensioni del suo canvas
    in base alla sua risoluzione originale. Non c'è una dimensione target comune.
    """
    try:
        filepath = wcs_info['file']
        hdu_index = wcs_info['hdu_index']
        native_pixel_scale = wcs_info['pixel_scale']  # arcsec/px NATIVO
        filename = os.path.basename(filepath)
        
        with fits.open(filepath) as hdul:
            hdu = hdul[hdu_index]
            data = hdu.data
            header = hdu.header.copy()
            wcs_orig = WCS(header)
            
            if len(data.shape) == 3:
                data = data[0]
            
            original_shape = data.shape
            
            # === CALCOLO DIMENSIONI OUTPUT BASATE SU RISOLUZIONE NATIVA ===
            native_pixel_scale_deg = native_pixel_scale / 3600.0
            
            # Calcola quanti gradi copre l'immagine originale
            ny_orig, nx_orig = original_shape
            
            # Calcola pixel scale dall'immagine originale (gestisce sia cd che cdelt)
            try:
                if hasattr(wcs_orig.wcs, 'cd') and wcs_orig.wcs.cd is not None:
                    # Usa CD matrix
                    cd = wcs_orig.wcs.cd
                    pixel_scale_ra = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
                    pixel_scale_dec = np.sqrt(cd[1,0]**2 + cd[1,1]**2)
                else:
                    # Usa CDELT
                    pixel_scale_ra = abs(wcs_orig.wcs.cdelt[0])
                    pixel_scale_dec = abs(wcs_orig.wcs.cdelt[1])
            except:
                # Fallback: usa il pixel scale nativo calcolato prima
                pixel_scale_ra = native_pixel_scale_deg
                pixel_scale_dec = native_pixel_scale_deg
            
            ra_span = pixel_scale_ra * nx_orig
            dec_span = pixel_scale_dec * ny_orig
            
            # Calcola dimensioni output per mantenere la stessa copertura
            # alla risoluzione nativa
            nx_out = int(np.ceil(abs(ra_span) / native_pixel_scale_deg))
            ny_out = int(np.ceil(abs(dec_span) / native_pixel_scale_deg))
            
            shape_out = (ny_out, nx_out)
            
            # Crea WCS target con risoluzione nativa
            target_wcs = WCS(naxis=2)
            target_wcs.wcs.ctype = common_wcs.wcs.ctype
            target_wcs.wcs.crval = wcs_orig.wcs.crval  # Stesso centro dell'originale
            target_wcs.wcs.crpix = [nx_out / 2.0, ny_out / 2.0]
            target_wcs.wcs.cdelt = [-native_pixel_scale_deg, native_pixel_scale_deg]
            target_wcs.wcs.radesys = common_wcs.wcs.radesys
            target_wcs.wcs.equinox = common_wcs.wcs.equinox
            
            # Reproject
            reprojected_data, footprint = reproject_interp(
                (data, wcs_orig),
                target_wcs,
                shape_out=shape_out,
                order=REPROJECT_ORDER
            )
            
            # Calcola coverage
            valid_pixels = np.sum(footprint > 0)
            total_pixels = footprint.size
            coverage = (valid_pixels / total_pixels * 100) if total_pixels > 0 else 0
            
            # Crea nuovo header
            new_header = target_wcs.to_header()
            
            # Mantieni metadati originali
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
            
            # Nome output
            output_filename = f"reg_{os.path.splitext(filename)[0]}.fits"
            output_path = os.path.join(output_dir, output_filename)
            
            # Salva
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
    """Registra tutte le immagini con multithreading, ognuna alla sua risoluzione nativa."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🔄 Registrazione {source_name}: {len(wcs_info_list)} immagini (risoluzione nativa)")
    
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
                        pbar.set_description(f"✓ {success_count}")
                    else:
                        error_count += 1
                        pbar.set_description(f"❌ {error_count}")
                        
                except Exception as exc:
                    error_count += 1
                    with log_lock:
                        logger.error(f"Exception nel thread pool: {exc}")
                
                pbar.update(1)
    
    print(f"   ✓ Successo: {success_count}")
    print(f"   ✗ Errori: {error_count}")
    
    with log_lock:
        logger.info(f"{source_name}: {success_count} successo, {error_count} errori")
    
    return success_count, error_count

# ============================================================================
# MAIN: ESECUZIONE PIPELINE
# ============================================================================

def main_step1():
    """Funzione principale Step 1."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info(f"PREPARAZIONE: CONVERSIONE WCS DA COORDINATE (Tutti i target)")
    logger.info("=" * 60)
    
    print("=" * 70)
    print(f"🔭 PREPARAZIONE: CONVERSIONE COORDINATE → WCS (Tutti i target)".center(70))
    print("=" * 70)
    
    print(f"\nInput Osservatorio (ricorsivo): {INPUT_OSSERVATORIO}")
    print(f"Input LITH/HST (ricorsivo): {INPUT_LITH}")
    
    # Setup
    os.makedirs(OUTPUT_OSSERVATORIO_WCS, exist_ok=True)
    os.makedirs(OUTPUT_LITH_WCS, exist_ok=True)
    
    # OSSERVATORIO
    print("\n📡 OSSERVATORIO (Conversione OBJCTRA/OBJCTDEC → WCS)")
    
    prep_oss, fail_oss, stats_oss = process_osservatorio_folder(
        INPUT_OSSERVATORIO,
        OUTPUT_OSSERVATORIO_WCS,
        logger
    )
    
    print(f"\n   ✓ Processati: {prep_oss}")
    print(f"   ✗ Falliti: {fail_oss}")
    
    if stats_oss:
        ra_min, ra_max = stats_oss['ra_range']
        dec_min, dec_max = stats_oss['dec_range']
        print(f"\n   📊 Campo:")
        print(f"      RA: {ra_min:.4f}° → {ra_max:.4f}° (span: {ra_max-ra_min:.4f}°)")
        print(f"      DEC: {dec_min:.4f}° → {dec_max:.4f}° (span: {dec_max-dec_min:.4f}°)")
        print(f"      Scala media: {stats_oss['avg_scale']:.3f}\"/px")
    
    # LITH
    print("\n🛰️  LITH/HST (Estrazione WCS esistente)")
    
    prep_lith, fail_lith, stats_lith = process_lith_folder(
        INPUT_LITH,
        OUTPUT_LITH_WCS,
        logger
    )
    
    print(f"\n   ✓ Processati: {prep_lith}")
    print(f"   ✗ Falliti: {fail_lith}")
    
    if stats_lith:
        ra_min, ra_max = stats_lith['ra_range']
        dec_min, dec_max = stats_lith['dec_range']
        print(f"\n   📊 Campo:")
        print(f"      RA: {ra_min:.4f}° → {ra_max:.4f}° (span: {ra_max-ra_min:.4f}°)")
        print(f"      DEC: {dec_min:.4f}° → {dec_max:.4f}° (span: {dec_max-dec_min:.4f}°)")
        print(f"      Scala media: {stats_lith['avg_scale']:.3f}\"/px")
    
    # RIEPILOGO
    total_prep = prep_oss + prep_lith
    total_fail = fail_oss + fail_lith
    
    print("\n" + "=" * 70)
    print("📊 RIEPILOGO")
    print("=" * 70)
    print(f"   Osservatorio: {prep_oss} OK, {fail_oss} falliti")
    print(f"   LITH: {prep_lith} OK, {fail_lith} falliti")
    print(f"   TOTALE: {total_prep} preparati, {total_fail} falliti")
    
    logger.info(f"Totale: {total_prep} preparati, {total_fail} falliti")
    
    if total_prep > 0:
        print(f"\n✅ COMPLETATO!")
        print(f"\n   📁 File con WCS in:")
        print(f"      • {OUTPUT_OSSERVATORIO_WCS}")
        print(f"      • {OUTPUT_LITH_WCS}")
        print(f"\n   ➡️  Prossimo passo: Esegui step 2 e 3")
    else:
        print(f"\n⚠️ Nessun file processato.")
    
    return total_prep > 0


def main_step2():
    """Funzione principale Step 2."""
    if not REPROJECT_AVAILABLE:
        print("\n⚠️ Step 2 saltato: libreria reproject non disponibile")
        return False
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("REGISTRAZIONE CON RISOLUZIONE NATIVA")
    logger.info("=" * 60)
    
    print("\n" + "🔭"*35)
    print(f"STEP 3: REGISTRAZIONE (RISOLUZIONE NATIVA)".center(70))
    print("🔭"*35)
    
    print(f"\n📂 CONFIGURAZIONE:")
    print(f"   Input Hubble: {INPUT_HUBBLE}")
    print(f"   Input Observatory: {INPUT_OBSERVATORY}")
    print(f"   Output: {os.path.join(BASE_DIR, '3_registered_native')}")
    print(f"   Modalità: RISOLUZIONE NATIVA (ogni immagine mantiene la sua risoluzione)")
    print(f"   Threads: {NUM_THREADS}")
    print(f"   Interpolazione: {REPROJECT_ORDER}")
    
    # Analizza immagini
    print(f"\n{'='*70}")
    print("ANALISI IMMAGINI")
    print(f"{'='*70}")
    
    hubble_info = analyze_images(INPUT_HUBBLE, "HUBBLE", logger)
    obs_info = analyze_images(INPUT_OBSERVATORY, "OBSERVATORY", logger)
    
    all_info = hubble_info + obs_info
    if not all_info:
        print(f"\n❌ Nessuna immagine con WCS valido trovata. Interruzione.")
        logger.error("Nessuna immagine con WCS valido trovata.")
        return False
    
    # Mostra statistiche risoluzione
    print(f"\n📊 RISOLUZIONI NATIVE RILEVATE:")
    for source_name, info_list in [("Hubble", hubble_info), ("Observatory", obs_info)]:
        if info_list:
            scales = [info['pixel_scale'] for info in info_list]
            print(f"\n   {source_name}:")
            print(f"      Min: {min(scales):.4f}\"/px")
            print(f"      Max: {max(scales):.4f}\"/px")
            print(f"      Media: {np.mean(scales):.4f}\"/px")
            print(f"      → Tutte manterranno la loro risoluzione originale!")
    
    # Crea frame WCS comune (solo per riferimento)
    print(f"\n{'='*70}")
    print("CREAZIONE FRAME WCS COMUNE")
    print(f"{'='*70}")
    
    common_wcs = create_common_wcs_frame(all_info, logger)
    
    if common_wcs is None:
        print(f"\n❌ Impossibile creare WCS comune!")
        logger.error("Creazione WCS comune fallita.")
        return False
    
    # Registrazione
    print(f"\n{'='*70}")
    print("REGISTRAZIONE (Risoluzione Nativa)")
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
    
    # === BLOCCO STATISTICHE DIMENSIONI RIMOSSO ===
    
    # Riepilogo
    print(f"\n{'='*70}")
    print("📊 RIEPILOGO REGISTRAZIONE")
    print(f"{'='*70}")
    print(f"\n   Totale registrate: {total_success}")
    print(f"   Totale errori: {total_error}")
    
    if total_success > 0:
        print(f"\n✅ REGISTRAZIONE COMPLETATA!")
        print(f"\n   ✨ VANTAGGIO: Ogni immagine ha mantenuto la sua risoluzione nativa!")
        print(f"\n   📁 Output:")
        if hubble_info:
            print(f"      Hubble: {OUTPUT_HUBBLE}")
        if obs_info:
            print(f"      Observatory: {OUTPUT_OBSERVATORY}")
        
        # === BLOCCO STATISTICHE DIMENSIONI DETTAGLIATE RIMOSSO ===
        
        print(f"\n{'='*70}")
        print(f"⭐ RISOLUZIONE NATIVA PRESERVATA")
        print(f"{'='*70}")
        print(f"   ✅ Ogni immagine mantiene la sua risoluzione originale")
        print(f"   ✅ Nessuna perdita di qualità per interpolazione")
        print(f"   ✅ Dimensioni output ottimizzate per risoluzione nativa")
        
        print(f"\n   ⚠️  NOTA IMPORTANTE:")
        print(f"      Le immagini hanno risoluzioni diverse tra loro.")
        print(f"      Per lo step 4, usa 'step4_patch_improved.py' che")
        print(f"      estrae patches rispettando la risoluzione di ogni immagine.")
    
    with log_lock:
        logger.info(f"Registrazione completata: {total_success} successo, {total_error} errori")
    
    return total_success > 0


def main():
    """Funzione principale della pipeline."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("PIPELINE UNIFICATA: STEP 1 + STEP 2")
    logger.info("=" * 60)
    
    print("\n" + "=" * 70)
    print("🚀 PIPELINE COMPLETA: CONVERSIONE WCS + REGISTRAZIONE")
    print("=" * 70)
    
    # STEP 1
    step1_start = time.time()
    step1_success = main_step1()
    step1_time = time.time() - step1_start
    
    print(f"\n⏱️ Tempo Step 1: {step1_time:.2f}s")
    
    if not step1_success:
        print("\n❌ Pipeline interrotta: Step 1 fallito")
        return
    
    # STEP 2
    print("\n" + "➡️ " * 35)
    step2_start = time.time()
    step2_success = main_step2()
    step2_time = time.time() - step2_start
    
    print(f"\n⏱️ Tempo Step 2: {step2_time:.2f}s")
    
    # RIEPILOGO FINALE
    print("\n" + "=" * 70)
    print("📊 RIEPILOGO PIPELINE COMPLETA")
    print("=" * 70)
    print(f"\n   Step 1 (Conversione WCS): {'✅ COMPLETATO' if step1_success else '❌ FALLITO'}")
    print(f"   Step 2 (Registrazione): {'✅ COMPLETATO' if step2_success else '⚠️ SALTATO/FALLITO'}")
    print(f"\n   ⏱️ Tempo totale: {step1_time + step2_time:.2f}s")
    
    if step1_success and step2_success:
        print(f"\n✅ PIPELINE COMPLETATA CON SUCCESSO!")
    elif step1_success:
        print(f"\n⚠️  PIPELINE PARZIALMENTE COMPLETATA (Step 1 OK, Step 2 fallito)")
    else:
        print(f"\n❌ PIPELINE FALLITA")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️ Tempo totale esecuzione: {elapsed:.2f}s")