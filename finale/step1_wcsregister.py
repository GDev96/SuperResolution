"""
PIPELINE COMPLETO: CONVERSIONE WCS + REGISTRAZIONE
Combina Step 1 (Conversione WCS) e Step 2 (Registrazione) in un unico script.
VERSIONE CORRETTA con extract_wcs_info_safe e allineamento centro comune.
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
    print("Questa libreria è fondamentale per lo Step 2.")
    print("Installa con: pip install reproject")
    print("="*70)

NUM_THREADS = 1
REPROJECT_ORDER = 'bilinear'
log_lock = threading.Lock()

# ============================================================================
# SELEZIONE CARTELLA TARGET
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
        print(f"\n❌ ERRORE: Impossibile leggere la cartella {ROOT_DATA_DIR}")
        print(f"   Dettagli: {e}")
        return []

    if not subdirs:
        print(f"\n❌ ERRORE: Nessuna sottocartella trovata in {ROOT_DATA_DIR}")
        print("   Assicurati di aver creato cartelle come 'M33', 'M42', ecc.")
        return []

    print("\nCartelle target disponibili:")
    print(f"   0: ✨ Processa TUTTI i {len(subdirs)} target")
    print("   " + "─"*30)
    for i, dir_path in enumerate(subdirs):
        print(f"   {i+1}: {dir_path.name}")

    while True:
        print("\n" + "─"*70)
        try:
            choice_str = input(f"👉 Seleziona un numero (0-{len(subdirs)}) o 'q' per uscire: ").strip()

            if choice_str.lower() == 'q':
                print("👋 Uscita.")
                return []

            choice = int(choice_str)

            if choice == 0:
                print(f"\n✅ Selezionati TUTTI i {len(subdirs)} target.")
                return subdirs
            
            choice_idx = choice - 1
            if 0 <= choice_idx < len(subdirs):
                selected_dir = subdirs[choice_idx]
                print(f"\n✅ Cartella selezionata: {selected_dir.name}")
                print(f"   Percorso completo: {selected_dir}")
                return [selected_dir]
            else:
                print(f"❌ Scelta non valida. Inserisci un numero tra 0 e {len(subdirs)}.")
        except ValueError:
            print("❌ Input non valido. Inserisci un numero.")
        except Exception as e:
            print(f"❌ Errore: {e}")
            return []

# ============================================================================
# SETUP LOGGING
# ============================================================================

def setup_logging():
    """Configura logging per l'intera pipeline."""
    os.makedirs(LOG_DIR_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = LOG_DIR_ROOT / f'unified_pipeline_{timestamp}.log'

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info(f"LOG FILE: {log_filename}")
    logger.info("=" * 80)
    logger.info(f"Python: {sys.version}")
    logger.info(f"Numpy: {np.__version__}")
    logger.info(f"Astropy: {astropy.__version__}")
    if REPROJECT_AVAILABLE:
        logger.info(f"Reproject: (importata con successo)")
    logger.info(f"Threads: {NUM_THREADS}")
    logger.info(f"Reprojection Order: {REPROJECT_ORDER}")
    logger.info("MODALITÀ: Risoluzione Nativa con Centro Comune")

    return logger

# ============================================================================
# STEP 1: FUNZIONI CONVERSIONE WCS
# ============================================================================

def parse_coordinates(ra_str, dec_str):
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
    try:
        filename = os.path.basename(input_file)
        with fits.open(input_file, mode='readonly') as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()
            if data is None:
                logger.warning(f"Nessun dato in {filename}")
                return False
            try:
                existing_wcs = WCS(header)
                if existing_wcs.has_celestial:
                    logger.info(f"✓ {filename}: WCS già presente")
                    hdul.writeto(output_file, overwrite=True)
                    return True
            except Exception:
                pass
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
            info = { 'shape': shape, 'ra': ra, 'dec': dec, 'pixel_scale': pixel_scale }
            logger.info(f"✓ {os.path.basename(filename)}: {shape[1]}x{shape[0]}px, RA={ra:.4f}°, DEC={dec:.4f}°")
            return sci_data, sci_header, info
    except Exception as e:
        logger.error(f"✗ {os.path.basename(filename)}: {e}")
        return None, None, None

def process_osservatorio_folder(input_dir, output_dir, logger):
    fits_files = list(Path(input_dir).glob('**/*.fit')) + list(Path(input_dir).glob('**/*.fits'))
    if not fits_files:
        logger.warning(f"Nessun file in {input_dir}")
        return 0, 0, None
    logger.info(f"Trovati {len(fits_files)} file osservatorio")
    prepared_count = 0
    failed_count = 0
    ra_list, dec_list, scale_list = [], [], []
    with tqdm(total=len(fits_files), desc="   Osservatorio", unit="file") as pbar:
        for input_file in fits_files:
            basename = input_file.name
            name, ext = os.path.splitext(basename)
            output_file = output_dir / f"{name}_wcs.fits"
            success = add_wcs_to_file(input_file, output_file, logger)
            if success:
                try:
                    with fits.open(output_file) as hdul:
                        wcs = WCS(hdul[0].header)
                        if wcs.has_celestial:
                            ra, dec = wcs.wcs.crval
                            pixel_scale = abs(wcs.wcs.cdelt[0]) * 3600
                            ra_list.append(ra); dec_list.append(dec); scale_list.append(pixel_scale)
                    prepared_count += 1
                except:
                    failed_count += 1
            else:
                failed_count += 1
            pbar.update(1)
    stats = None
    if ra_list:
        stats = { 'ra_range': (min(ra_list), max(ra_list)), 'dec_range': (min(dec_list), max(dec_list)), 'avg_scale': np.mean(scale_list) }
    return prepared_count, failed_count, stats

def process_lith_folder(input_dir, output_dir, logger):
    fits_files = list(Path(input_dir).glob('**/*.fit')) + list(Path(input_dir).glob('**/*.fits'))
    if not fits_files:
        return 0, 0, None
    logger.info(f"Trovati {len(fits_files)} file LITH")
    prepared_count = 0
    failed_count = 0
    ra_list, dec_list, scale_list = [], [], []
    with tqdm(total=len(fits_files), desc="   LITH", unit="file") as pbar:
        for input_file in fits_files:
            data, header, info = extract_lith_data(input_file, logger)
            if data is not None:
                basename = input_file.name
                name, ext = os.path.splitext(basename)
                output_file = output_dir / f"{name}_wcs.fits"
                try:
                    primary_hdu = fits.PrimaryHDU(data=data, header=header)
                    primary_hdu.header['ORIGINAL'] = basename
                    primary_hdu.header['PREPDATE'] = datetime.now().isoformat()
                    primary_hdu.header['SOURCE'] = 'lith'
                    primary_hdu.writeto(output_file, overwrite=True, output_verify='silentfix')
                    prepared_count += 1
                    ra_list.append(info['ra']); dec_list.append(info['dec']); scale_list.append(info['pixel_scale'])
                except Exception as e:
                    logger.error(f"Errore {basename}: {e}")
                    failed_count += 1
            else:
                failed_count += 1
            pbar.update(1)
    stats = None
    if ra_list:
        stats = { 'ra_range': (min(ra_list), max(ra_list)), 'dec_range': (min(dec_list), max(dec_list)), 'avg_scale': np.mean(scale_list) }
    return prepared_count, failed_count, stats

# ============================================================================
# STEP 2: ESTRAZIONE WCS SICURA (DA CODICE FUNZIONANTE)
# ============================================================================

def extract_wcs_info_safe(filepath, logger):
    """
    Estrae informazioni WCS in modo sicuro evitando errori di unità.
    VERSIONE CORRETTA dal codice funzionante.
    """
    try:
        with fits.open(filepath) as hdul:
            # Trova HDU con dati validi
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
            header = data_hdu.header
            
            # Gestisci dati 3D
            if len(data_hdu.data.shape) == 3:
                data = data_hdu.data[0]
            else:
                data = data_hdu.data
            
            ny, nx = data.shape
            
            # ✅ METODO SICURO: Usa wcs_pix2world invece di pixel_to_world
            try:
                # Calcola centro dai pixel centrali
                center_x = nx / 2.0
                center_y = ny / 2.0
                
                # Converti usando wcs_pix2world (più robusto)
                world_coords = wcs.wcs_pix2world([[center_x, center_y]], 1)
                center_ra = float(world_coords[0][0])
                center_dec = float(world_coords[0][1])
                
            except Exception as e:
                logger.debug(f"Fallback to header values: {e}")
                # Fallback ai valori dell'header
                center_ra = float(header.get('CRVAL1', 0))
                center_dec = float(header.get('CRVAL2', 0))
            
            # ✅ CALCOLO PIXEL SCALE SICURO
            try:
                # Usa CD matrix se disponibile
                if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
                    cd = wcs.wcs.cd
                    pixel_scale_deg = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
                elif 'CD1_1' in header and 'CD2_2' in header:
                    cd1_1 = float(header['CD1_1'])
                    cd2_2 = float(header['CD2_2'])
                    pixel_scale_deg = np.sqrt(cd1_1**2 + cd2_2**2)
                elif 'CDELT1' in header and 'CDELT2' in header:
                    cdelt1 = abs(float(header['CDELT1']))
                    cdelt2 = abs(float(header['CDELT2']))
                    pixel_scale_deg = np.sqrt(cdelt1**2 + cdelt2**2)
                else:
                    pixel_scale_deg = abs(wcs.wcs.cdelt[0])
                    
                pixel_scale_arcsec = pixel_scale_deg * 3600.0
                
            except Exception as e:
                logger.debug(f"Pixel scale fallback: {e}")
                pixel_scale_arcsec = 0.04  # Default per HST
                pixel_scale_deg = pixel_scale_arcsec / 3600.0
            
            return {
                'file': filepath,
                'hdu_index': hdu_idx,
                'wcs': wcs,
                'shape': data.shape,
                'center_ra': center_ra,
                'center_dec': center_dec,
                'pixel_scale': pixel_scale_arcsec,
            }
            
    except Exception as e:
        with log_lock:
            logger.error(f"Errore WCS {os.path.basename(filepath)}: {e}")
        return None

# ============================================================================
# ANALISI IMMAGINI
# ============================================================================

def analyze_images(input_dir, source_name, logger):
    """Analizza immagini usando extract_wcs_info_safe."""
    files = list(Path(input_dir).glob('*.fits')) + list(Path(input_dir).glob('*.fit')) + list(Path(input_dir).glob('*_wcs.fits'))
    files = list(set(files))
    if not files:
        with log_lock:
            logger.warning(f"Nessun file in {input_dir}")
        return []
    
    print(f"\n📂 {source_name}: {len(files)} file")
    wcs_info_list = []
    
    with tqdm(total=len(files), desc=f"   Analisi {source_name}", unit="file") as pbar:
        for filepath in files:
            info = extract_wcs_info_safe(filepath, logger)
            if info:
                wcs_info_list.append(info)
                with log_lock:
                    logger.info(f"✓ {os.path.basename(filepath)}: RA={info['center_ra']:.4f}°, DEC={info['center_dec']:.4f}°, scale={info['pixel_scale']:.4f}\"/px")
            else:
                with log_lock:
                    logger.warning(f"✗ {os.path.basename(filepath)}: WCS non valido")
            pbar.update(1)
    
    print(f"   ✓ {len(wcs_info_list)}/{len(files)} con WCS valido")
    return wcs_info_list

# ============================================================================
# CREAZIONE WCS COMUNE
# ============================================================================

def create_common_wcs_frame(wcs_info_list, logger):
    """Crea WCS comune per il frame di riferimento."""
    with log_lock:
        logger.info("=" * 60)
        logger.info("CREAZIONE FRAME WCS COMUNE (riferimento)")
        logger.info("=" * 60)
    
    if not wcs_info_list:
        with log_lock:
            logger.error("Nessuna immagine fornita per creare WCS comune.")
        return None
    
    # Calcola bounds usando le coordinate dei centri e gli span
    ra_min, ra_max = float('inf'), float('-inf')
    dec_min, dec_max = float('inf'), float('-inf')
    
    for info in wcs_info_list:
        wcs, shape = info['wcs'], info['shape']
        ny, nx = shape
        
        # Calcola i 4 angoli dell'immagine
        corners_pix = np.array([[0, 0], [nx, 0], [0, ny], [nx, ny]])
        corners_world = wcs.wcs_pix2world(corners_pix, 1)
        
        for coord in corners_world:
            ra, dec = float(coord[0]), float(coord[1])
            ra_min = min(ra_min, ra)
            ra_max = max(ra_max, ra)
            dec_min = min(dec_min, dec)
            dec_max = max(dec_max, dec)
    
    # Centro del frame comune
    ra_center = (ra_min + ra_max) / 2.0
    dec_center = (dec_min + dec_max) / 2.0
    
    ra_span = (ra_max - ra_min) * 1.05  # +5% margine
    dec_span = (dec_max - dec_min) * 1.05
    
    # Crea WCS di riferimento
    reference_wcs = WCS(naxis=2)
    reference_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    reference_wcs.wcs.crval = [ra_center, dec_center]
    reference_wcs.wcs.crpix = [1, 1]
    reference_wcs.wcs.cdelt = [-(0.04 / 3600.0), 0.04 / 3600.0]  # Riferimento HST
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

# ============================================================================
# REPROIEZIONE CON CENTRO COMUNE (CORRETTO)
# ============================================================================

def reproject_image_native(wcs_info, common_wcs, output_dir, logger):
    """
    Reproietta un'immagine mantenendo la risoluzione nativa.
    ✅ CORRETTO: Usa centro COMUNE per allineamento.
    """
    try:
        filepath = wcs_info['file']
        hdu_index = wcs_info['hdu_index']
        native_pixel_scale = wcs_info['pixel_scale']
        filename = os.path.basename(filepath)
        
        with fits.open(filepath) as hdul:
            hdu = hdul[hdu_index]
            data, header = hdu.data, hdu.header.copy()
            wcs_orig = WCS(header)
            
            # Gestisci dati 3D
            if len(data.shape) == 3:
                data = data[0]
            
            original_shape = data.shape
            native_pixel_scale_deg = native_pixel_scale / 3600.0
            ny_orig, nx_orig = original_shape
            
            # ✅ CALCOLO SPAN BASATO SUL WCS ORIGINALE
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
            
            # Calcola span in gradi (con margine)
            margin_factor = 1.1  # +10% margine
            ra_span = pixel_scale_ra * nx_orig * margin_factor
            dec_span = pixel_scale_dec * ny_orig * margin_factor
            
            # Calcola dimensioni output mantenendo risoluzione nativa
            nx_out = int(np.ceil(abs(ra_span) / native_pixel_scale_deg))
            ny_out = int(np.ceil(abs(dec_span) / native_pixel_scale_deg))
            shape_out = (ny_out, nx_out)
            
            # ✅ CORREZIONE CRUCIALE: Usa centro COMUNE per TUTTE le immagini
            target_wcs = WCS(naxis=2)
            target_wcs.wcs.ctype = common_wcs.wcs.ctype
            target_wcs.wcs.crval = common_wcs.wcs.crval  # ✅ Centro COMUNE (non wcs_orig)
            target_wcs.wcs.crpix = [nx_out / 2.0, ny_out / 2.0]
            target_wcs.wcs.cdelt = [-native_pixel_scale_deg, native_pixel_scale_deg]
            target_wcs.wcs.radesys = common_wcs.wcs.radesys
            target_wcs.wcs.equinox = common_wcs.wcs.equinox
            
            # Reproiezione con reproject_interp
            reprojected_data, footprint = reproject_interp(
                (data, wcs_orig), target_wcs, shape_out=shape_out, order=REPROJECT_ORDER
            )
            
            # Calcola copertura
            valid_pixels = np.sum(footprint > 0)
            total_pixels = footprint.size
            coverage = (valid_pixels / total_pixels * 100) if total_pixels > 0 else 0
            
            # ✅ MANTIENI NaN (non convertire in zero)
            # reprojected_data mantiene i NaN per bordi corretti
            
            # Crea header con metadati completi
            new_header = target_wcs.to_header()
            
            # Copia metadati importanti
            for key in ['DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'TELESCOP']:
                if key in header:
                    new_header[key] = header[key]
            
            # Aggiungi metadati registrazione
            new_header['ORIGINAL'] = filename
            new_header['REGDATE'] = datetime.now().isoformat()
            new_header['REGCOV'] = (coverage, "Percentuale copertura valida")
            new_header['REGVALID'] = (valid_pixels, "Numero di pixel validi")
            new_header['REGORD'] = (str(REPROJECT_ORDER), "Ordine interpolazione")
            new_header['NATIVESC'] = (native_pixel_scale, "Risoluzione nativa (arcsec/px)")
            new_header['ORIGSHP0'] = (original_shape[0], "Shape originale (altezza)")
            new_header['ORIGSHP1'] = (original_shape[1], "Shape originale (larghezza)")
            new_header['COMRA'] = (common_wcs.wcs.crval[0], "Centro RA comune (deg)")
            new_header['COMDEC'] = (common_wcs.wcs.crval[1], "Centro DEC comune (deg)")
            new_header['COMMENT'] = 'Registered using common WCS center'
            
            # Salva file
            output_filename = f"reg_{os.path.splitext(filename)[0]}.fits"
            output_path = output_dir / output_filename
            
            fits.PrimaryHDU(
                data=reprojected_data.astype(np.float32),
                header=new_header
            ).writeto(output_path, overwrite=True)
            
            with log_lock:
                logger.info(f"✓ {filename}: coverage={coverage:.1f}%, shape={shape_out}, native_scale={native_pixel_scale:.4f}\"/px")
            
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

# ============================================================================
# REGISTRAZIONE IMMAGINI
# ============================================================================

def register_images(wcs_info_list, common_wcs, output_dir, source_name, logger):
    """Registra immagini usando multithreading."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🔄 Registrazione {source_name}: {len(wcs_info_list)} immagini (risoluzione nativa)")
    
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            executor.submit(reproject_image_native, info, common_wcs, output_dir, logger): info
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
                        logger.error(f"Exception nel thread pool: {exc}")
                pbar.update(1)
    
    print(f"   ✓ Successo: {success_count}")
    print(f"   ✗ Errori: {error_count}")
    
    with log_lock:
        logger.info(f"{source_name}: {success_count} successo, {error_count} errori")
    
    return success_count, error_count

# ============================================================================
# MENU DI PROSEGUIMENTO
# ============================================================================

def ask_continue_to_cropping():
    """Chiede all'utente se vuole proseguire con Step 3+4."""
    print("\n" + "="*70)
    print("🎯 STEP 1 E 2 (WCS e Registrazione) COMPLETATI!")
    print("="*70)
    print("\n📋 OPZIONI:")
    print("   1️⃣  Continua con Step 3+4 (Ritaglio e Mosaico)")
    print("   2️⃣  Termina qui")
    
    while True:
        print("\n" + "─"*70)
        choice = input("👉 Vuoi continuare con 'step2_croppedmosaico.py'? [S/n, default=S]: ").strip().lower()
        if choice in ('', 's', 'si', 'y', 'yes'):
            print("\n✅ Avvio Step 3+4 (step2_croppedmosaico.py)...")
            return True
        elif choice in ('n', 'no'):
            print("\n✅ Pipeline interrotta")
            print("   Per eseguire Step 3+4 in seguito, esegui 'step2_croppedmosaico.py'")
            return False
        else:
            print("❌ Scelta non valida. Inserisci S per Sì o N per No.")

# ============================================================================
# MAIN: ESECUZIONE PIPELINE
# ============================================================================

def main_step1(INPUT_OSSERVATORIO, INPUT_LITH, OUTPUT_OSSERVATORIO_WCS, OUTPUT_LITH_WCS, logger):
    """Funzione principale Step 1."""
    logger.info("=" * 60)
    logger.info(f"PREPARAZIONE: CONVERSIONE WCS DA COORDINATE")
    logger.info("=" * 60)
    
    print("=" * 70)
    print(f"🔭 PREPARAZIONE: CONVERSIONE COORDINATE → WCS".center(70))
    print("=" * 70)
    print(f"\nInput Osservatorio (ricorsivo): {INPUT_OSSERVATORIO}")
    print(f"Input LITH/HST (ricorsivo): {INPUT_LITH}")
    
    os.makedirs(OUTPUT_OSSERVATORIO_WCS, exist_ok=True)
    os.makedirs(OUTPUT_LITH_WCS, exist_ok=True)
    
    # OSSERVATORIO
    print("\n📡 OSSERVATORIO (Conversione OBJCTRA/OBJCTDEC → WCS)")
    prep_oss, fail_oss, stats_oss = process_osservatorio_folder(INPUT_OSSERVATORIO, OUTPUT_OSSERVATORIO_WCS, logger)
    print(f"\n   ✓ Processati: {prep_oss}")
    print(f"   ✗ Falliti: {fail_oss}")
    
    if stats_oss:
        ra_min, ra_max = stats_oss['ra_range']
        dec_min, dec_max = stats_oss['dec_range']
        print(f"\n   📊 Campo: RA: {ra_min:.4f}° → {ra_max:.4f}° | DEC: {dec_min:.4f}° → {dec_max:.4f}° | Scala media: {stats_oss['avg_scale']:.3f}\"/px")

    # LITH
    print("\n🛰️  LITH/HST (Estrazione WCS esistente)")
    prep_lith, fail_lith, stats_lith = process_lith_folder(INPUT_LITH, OUTPUT_LITH_WCS, logger)
    print(f"\n   ✓ Processati: {prep_lith}")
    print(f"   ✗ Falliti: {fail_lith}")
    
    if stats_lith:
        ra_min, ra_max = stats_lith['ra_range']
        dec_min, dec_max = stats_lith['dec_range']
        print(f"\n   📊 Campo: RA: {ra_min:.4f}° → {ra_max:.4f}° | DEC: {dec_min:.4f}° → {dec_max:.4f}° | Scala media: {stats_lith['avg_scale']:.3f}\"/px")

    total_prep = prep_oss + prep_lith
    logger.info(f"Totale: {total_prep} preparati, {fail_lith + fail_oss} falliti")
    
    if total_prep > 0:
        print(f"\n✅ COMPLETATO! File con WCS in:")
        print(f"       • {OUTPUT_OSSERVATORIO_WCS}")
        print(f"       • {OUTPUT_LITH_WCS}")
    else:
        print(f"\n⚠️ Nessun file processato.")
    
    return total_prep > 0

def main_step2(INPUT_HUBBLE, INPUT_OBSERVATORY, OUTPUT_HUBBLE, OUTPUT_OBSERVATORY, BASE_DIR, logger):
    """Funzione principale Step 2."""
    if not REPROJECT_AVAILABLE:
        print("\n⚠️ Step 2 saltato: libreria reproject non disponibile")
        return False
    
    logger.info("=" * 60)
    logger.info("REGISTRAZIONE CON RISOLUZIONE NATIVA E CENTRO COMUNE")
    logger.info("=" * 60)
    
    print("\n" + "🔭"*35)
    print(f"STEP 2: REGISTRAZIONE (CENTRO COMUNE)".center(70))
    print("🔭"*35)
    print(f"\n📂 CONFIGURAZIONE:")
    print(f"   Input Hubble: {INPUT_HUBBLE}")
    print(f"   Input Observatory: {INPUT_OBSERVATORY}")
    print(f"   Output: {BASE_DIR / '3_registered_native'}")

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

    print(f"\n📊 RISOLUZIONI NATIVE RILEVATE:")
    for source_name, info_list in [("Hubble", hubble_info), ("Observatory", obs_info)]:
        if info_list:
            scales = [info['pixel_scale'] for info in info_list]
            print(f"\n   {source_name}:")
            print(f"       Min: {min(scales):.4f}\"/px | Max: {max(scales):.4f}\"/px | Media: {np.mean(scales):.4f}\"/px")

    print(f"\n{'='*70}")
    print("CREAZIONE FRAME WCS COMUNE")
    print(f"{'='*70}")
    
    common_wcs = create_common_wcs_frame(all_info, logger)
    if common_wcs is None:
        print(f"\n❌ Impossibile creare WCS comune!")
        logger.error("Creazione WCS comune fallita.")
        return False

    print(f"\n{'='*70}")
    print("REGISTRAZIONE (Risoluzione Nativa + Centro Comune)")
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
    print("📊 RIEPILOGO REGISTRAZIONE")
    print(f"{'='*70}")
    print(f"\n   Totale registrate: {total_success}")
    print(f"   Totale errori: {total_error}")
    
    if total_success > 0:
        print(f"\n✅ REGISTRAZIONE COMPLETATA!")
        print(f"\n   ✨ VANTAGGIO:")
        print(f"      • Ogni immagine mantiene risoluzione nativa")
        print(f"      • Tutte le immagini usano centro comune")
        print(f"      • Stelle perfettamente allineate")
    
    with log_lock:
        logger.info(f"Registrazione completata: {total_success} successo, {total_error} errori")
    
    return total_success > 0

def main():
    """Funzione principale della pipeline."""
    
    logger = setup_logging()
    
    # Seleziona uno o più target
    target_dirs = select_target_directory()
    if not target_dirs:
        print("Nessun target selezionato. Uscita.")
        return

    logger.info("=" * 60)
    logger.info("PIPELINE UNIFICATA: STEP 1 + STEP 2 (VERSIONE CORRETTA)")
    logger.info(f"TARGET SELEZIONATI: {[d.name for d in target_dirs]}")
    logger.info("=" * 60)

    print("\n" + "=" * 70)
    print("🚀 PIPELINE COMPLETA: CONVERSIONE WCS + REGISTRAZIONE")
    if len(target_dirs) > 1:
        print(f"Modalità Batch: {len(target_dirs)} target".center(70))
    print("=" * 70)

    start_time_total = time.time()
    successful_targets = []
    failed_targets = []

    for BASE_DIR in target_dirs:
        print("\n" + "🚀"*35)
        print(f"INIZIO ELABORAZIONE TARGET: {BASE_DIR.name}".center(70))
        print("🚀"*35)
        logger.info(f"INIZIO TARGET: {BASE_DIR.name}")

        # Definizione percorsi per QUESTO target
        INPUT_OSSERVATORIO = BASE_DIR / '1_origin' / 'local_raw'
        INPUT_LITH = BASE_DIR / '1_origin' / 'img_lights'
        OUTPUT_OSSERVATORIO_WCS = BASE_DIR / '2_wcs' / 'observatory'
        OUTPUT_LITH_WCS = BASE_DIR / '2_wcs' / 'hubble'
        INPUT_HUBBLE = OUTPUT_LITH_WCS
        INPUT_OBSERVATORY = OUTPUT_OSSERVATORIO_WCS
        OUTPUT_HUBBLE = BASE_DIR / '3_registered_native' / 'hubble'
        OUTPUT_OBSERVATORY = BASE_DIR / '3_registered_native' / 'observatory'

        # STEP 1
        step1_start = time.time()
        step1_success = main_step1(INPUT_OSSERVATORIO, INPUT_LITH, OUTPUT_OSSERVATORIO_WCS, OUTPUT_LITH_WCS, logger)
        step1_time = time.time() - step1_start
        print(f"\n⏱️  Tempo Step 1 (WCS): {step1_time:.2f}s")

        if not step1_success:
            print(f"\n❌ Pipeline interrotta per {BASE_DIR.name}: Step 1 fallito")
            logger.error(f"Step 1 fallito per {BASE_DIR.name}")
            failed_targets.append(BASE_DIR)
            continue

        # STEP 2
        step2_start = time.time()
        step2_success = main_step2(INPUT_HUBBLE, INPUT_OBSERVATORY, OUTPUT_HUBBLE, OUTPUT_OBSERVATORY, BASE_DIR, logger)
        step2_time = time.time() - step2_start
        print(f"\n⏱️  Tempo Step 2 (Registro): {step2_time:.2f}s")
        
        if step2_success:
            successful_targets.append(BASE_DIR)
            logger.info(f"TARGET COMPLETATO: {BASE_DIR.name}")
        else:
            failed_targets.append(BASE_DIR)
            logger.error(f"Step 2 fallito per {BASE_DIR.name}")
            
        print(f"⏱️  Tempo totale per {BASE_DIR.name}: {step1_time + step2_time:.2f}s")

    # RIEPILOGO FINALE
    elapsed_total = time.time() - start_time_total
    print("\n" + "=" * 70)
    print("📊 RIEPILOGO PIPELINE COMPLETA (BATCH)")
    print("=" * 70)
    print(f"   Target totali selezionati: {len(target_dirs)}")
    print(f"   ✅ Completati con successo: {len(successful_targets)}")
    for target in successful_targets:
        print(f"      - {target.name}")
    print(f"\n   ❌ Falliti: {len(failed_targets)}")
    for target in failed_targets:
        print(f"      - {target.name}")
    print(f"\n   ⏱️  Tempo totale batch: {elapsed_total:.2f}s")

    if not successful_targets:
        print("\n❌ Nessun target completato con successo.")
        return

    if ask_continue_to_cropping():
        try:
            next_script = SCRIPTS_DIR / 'step2_croppedmosaico.py'
            
            if next_script.exists():
                print(f"\n🚀 Avvio Step 3+4 in loop per {len(successful_targets)} target...")
                for BASE_DIR in successful_targets:
                    print(f"\n--- Avvio per {BASE_DIR.name} ---")
                    subprocess.run([sys.executable, str(next_script), str(BASE_DIR)])
                    print(f"--- Completato {BASE_DIR.name} ---")
            else:
                print(f"\n⚠️  Script {next_script.name} non trovato nella directory {SCRIPTS_DIR}")
                print(f"   Eseguilo manualmente quando pronto")
        except Exception as e:
            print(f"\n⚠️  Impossibile avviare automaticamente {next_script.name}: {e}")
            print(f"   Eseguilo manualmente: python {next_script.name}")
    else:
        print("\n👋 Arrivederci!")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️  Tempo totale esecuzione script: {elapsed:.2f}s")