"""
PIPELINE COMPLETO: CONVERSIONE WCS + REGISTRAZIONE
Combina Step 1 (Conversione WCS) e Step 2 (Registrazione) in un unico script.
Gli output dei due step rimangono separati e distinti.
Tutti i metodi sono mantenuti ESATTAMENTE come negli script originali.

MODIFICATO:
- Aggiunto menu per avviare step2_croppedmosaico.py (Step 3+4)
- Aggiunto menu iniziale per selezionare la cartella del target.
- AGGIUNTO: Opzione 'Processa TUTTI' nel menu di selezione.
- AGGIUNTO: Loop principale in main() per processare uno o più target.
- Tutti i percorsi sono ora relativi alla cartella del target selezionata.
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
# NUOVA FUNZIONE: SELEZIONE CARTELLA TARGET (MODIFICATA)
# ============================================================================

def select_target_directory():
    """
    Mostra un menu per selezionare una o TUTTE le cartelle target.
    Restituisce:
    - Elenco di Path (con uno o più elementi)
    - Elenco vuoto (se 'q' o errore)
    """
    ROOT_DATA_DIR = Path(r'F:\Super Revolt Gaia\SuperResolution\data')

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
    # === MODIFICA: Aggiunta opzione 0 ===
    print(f"   0: ✨ Processa TUTTI i {len(subdirs)} target")
    print("   " + "─"*30)
    for i, dir_path in enumerate(subdirs):
        print(f"   {i+1}: {dir_path.name}")
    # === FINE MODIFICA ===

    while True:
        print("\n" + "─"*70)
        try:
            choice_str = input(f"👉 Seleziona un numero (0-{len(subdirs)}) o 'q' per uscire: ").strip()

            if choice_str.lower() == 'q':
                print("👋 Uscita.")
                return [] # Restituisce lista vuota

            choice = int(choice_str)

            # === MODIFICA: Gestione opzione 0 ===
            if choice == 0:
                print(f"\n✅ Selezionati TUTTI i {len(subdirs)} target.")
                return subdirs # Restituisce la lista completa
            # === FINE MODIFICA ===
            
            choice_idx = choice - 1
            if 0 <= choice_idx < len(subdirs):
                selected_dir = subdirs[choice_idx]
                print(f"\n✅ Cartella selezionata: {selected_dir.name}")
                print(f"   Percorso completo: {selected_dir}")
                return [selected_dir]  # Restituisce lista con un elemento
            else:
                print(f"❌ Scelta non valida. Inserisci un numero tra 0 e {len(subdirs)}.")
        except ValueError:
            print("❌ Input non valido. Inserisci un numero.")
        except Exception as e:
            print(f"❌ Errore: {e}")
            return []

# ============================================================================
# CONFIGURAZIONE GLOBALE
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent  # C:\...\SuperResolution\finale
PROJECT_ROOT = SCRIPT_DIR.parent               # C:\...\SuperResolution
LOG_DIR_ROOT = PROJECT_ROOT / 'logs'           # C:\...\SuperResolution\logs

NUM_THREADS = 1
REPROJECT_ORDER = 'bilinear'
log_lock = threading.Lock()

# ============================================================================
# SETUP LOGGING
# ============================================================================

def setup_logging():
    """Configura logging per l'intera pipeline."""
    os.makedirs(LOG_DIR_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = LOG_DIR_ROOT / f'unified_pipeline_{timestamp}.log'

    # ✅ NUOVO: Rimuovi handler esistenti per evitare duplicati
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
    
    # ✅ NUOVO: Log della configurazione path (debug)
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
    logger.info("MODALITÀ: Risoluzione Nativa (ogni immagine mantiene la sua risoluzione)")

    return logger

# ============================================================================
# STEP 1: FUNZIONI CONVERSIONE WCS (Invariate)
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
            important_keys = ['SIMPLE', 'BITPIX', 'NAIS', 'NAXIS1', 'NAXIS2',
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
# STEP 2: FUNZIONI REGISTRAZIONE (Invariate)
# ============================================================================

def extract_wcs_info(filepath, logger):
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
                'file': filepath, 'hdu_index': hdu_idx, 'wcs': wcs, 'shape': data.shape,
                'center_ra': center.ra.deg, 'center_dec': center.dec.deg,
                'pixel_scale': pixel_scale_arcsec,
            }
    except Exception as e:
        with log_lock:
            logger.error(f"Errore WCS {os.path.basename(filepath)}: {e}")
        return None

def analyze_images(input_dir, source_name, logger):
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
            info = extract_wcs_info(filepath, logger)
            if info:
                wcs_info_list.append(info)
                with log_lock:
                    logger.info(f"✓ {os.path.basename(filepath)}: RA={info['center_ra']:.4f}°, DEC={info['center_dec']:.4f}°, scale={info['pixel_scale']:.4f}\"/px (NATIVA)")
            else:
                with log_lock:
                    logger.warning(f"✗ {os.path.basename(filepath)}: WCS non valido")
            pbar.update(1)
    print(f"   ✓ {len(wcs_info_list)}/{len(files)} con WCS valido")
    return wcs_info_list

def create_common_wcs_frame(wcs_info_list, logger):
    with log_lock:
        logger.info("=" * 60); logger.info("CREAZIONE FRAME WCS COMUNE (riferimento)"); logger.info("=" * 60)
    if not wcs_info_list:
        with log_lock: logger.error("Nessuna immagine fornita per creare WCS comune."); return None
    ra_min, ra_max, dec_min, dec_max = float('inf'), float('-inf'), float('inf'), float('-inf')
    for info in wcs_info_list:
        wcs, shape = info['wcs'], info['shape']
        ny, nx = shape
        corners_pix = np.array([[0, 0], [nx, 0], [0, ny], [nx, ny]])
        corners_world = wcs.pixel_to_world(corners_pix[:, 0], corners_pix[:, 1])
        for coord in corners_world:
            ra, dec = coord.ra.deg, coord.dec.deg
            ra_min = min(ra_min, ra); ra_max = max(ra_max, ra); dec_min = min(dec_min, dec); dec_max = max(dec_max, dec)
    ra_center, dec_center = (ra_min + ra_max) / 2.0, (dec_min + dec_max) / 2.0
    ra_span, dec_span = (ra_max - ra_min) * 1.05, (dec_max - dec_min) * 1.05
    ref_pixel_scale_deg = 0.04 / 3600.0
    reference_wcs = WCS(naxis=2)
    reference_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    reference_wcs.wcs.crval = [ra_center, dec_center]
    reference_wcs.wcs.crpix = [1, 1]
    reference_wcs.wcs.cdelt = [-ref_pixel_scale_deg, ref_pixel_scale_deg]
    reference_wcs.wcs.radesys = 'ICRS'
    reference_wcs.wcs.equinox = 2000.0
    with log_lock:
        logger.info(f"WCS Comune creato: Centro: RA={ra_center:.4f}°, DEC={dec_center:.4f}°, Span: RA={ra_span:.4f}°, DEC={dec_span:.4f}°")
    print(f"\n✓ WCS comune creato: Centro: RA={ra_center:.4f}°, DEC={dec_center:.4f}°, Span: RA={ra_span:.4f}°, DEC={dec_span:.4f}°")
    return reference_wcs

def reproject_image_native(wcs_info, common_wcs, output_dir, logger):
    try:
        filepath = wcs_info['file']
        hdu_index = wcs_info['hdu_index']
        native_pixel_scale = wcs_info['pixel_scale']
        filename = os.path.basename(filepath)
        with fits.open(filepath) as hdul:
            hdu = hdul[hdu_index]
            data, header = hdu.data, hdu.header.copy()
            wcs_orig = WCS(header)
            if len(data.shape) == 3: data = data[0]
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
            ra_span, dec_span = pixel_scale_ra * nx_orig, pixel_scale_dec * ny_orig
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
                (data, wcs_orig), target_wcs, shape_out=shape_out, order=REPROJECT_ORDER
            )
            valid_pixels = np.sum(footprint > 0)
            total_pixels = footprint.size
            coverage = (valid_pixels / total_pixels * 100) if total_pixels > 0 else 0
            new_header = target_wcs.to_header()
            for key in ['DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'TELESCOP']:
                if key in header: new_header[key] = header[key]
            new_header['ORIGINAL'] = filename
            new_header['REGDATE'] = datetime.now().isoformat()
            new_header['REGCOV'] = (coverage, "Percentuale copertura valida")
            new_header['REGVALID'] = (valid_pixels, "Numero di pixel validi")
            new_header['REGORD'] = (str(REPROJECT_ORDER), "Ordine interpolazione")
            new_header['NATIVESC'] = (native_pixel_scale, "Risoluzione nativa (arcsec/px)")
            new_header['ORIGSHP0'] = (original_shape[0], "Shape originale (altezza)")
            new_header['ORIGSHP1'] = (original_shape[1], "Shape originale (larghezza)")
            output_filename = f"reg_{os.path.splitext(filename)[0]}.fits"
            output_path = output_dir / output_filename
            fits.PrimaryHDU(data=reprojected_data.astype(np.float32), header=new_header).writeto(output_path, overwrite=True)
            with log_lock:
                logger.info(f"✓ {filename}: coverage={coverage:.1f}%, shape={shape_out}, native_scale={native_pixel_scale:.4f}\"/px")
            return { 'status': 'success', 'file': filename, 'coverage': coverage, 'valid_pixels': valid_pixels,
                     'output_path': output_path, 'native_scale': native_pixel_scale, 'output_shape': shape_out }
    except Exception as e:
        with log_lock:
            logger.error(f"Errore {os.path.basename(filepath)}: {e}")
        return {'status': 'error', 'file': os.path.basename(filepath), 'reason': str(e)}

def register_images(wcs_info_list, common_wcs, output_dir, source_name, logger):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🔄 Registrazione {source_name}: {len(wcs_info_list)} immagini (risoluzione nativa)")
    success_count, error_count = 0, 0
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = { executor.submit(reproject_image_native, info, common_wcs, output_dir, logger): info for info in wcs_info_list }
        with tqdm(total=len(wcs_info_list), desc=f"   {source_name}") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result['status'] == 'success': success_count += 1
                    else: error_count += 1
                except Exception as exc:
                    error_count += 1
                    with log_lock: logger.error(f"Exception nel thread pool: {exc}")
                pbar.update(1)
    print(f"   ✓ Successo: {success_count}\n   ✗ Errori: {error_count}")
    with log_lock: logger.info(f"{source_name}: {success_count} successo, {error_count} errori")
    return success_count, error_count

# ============================================================================
# MENU DI PROSEGUIMENTO (STEP 2 -> STEP 3+4)
# ============================================================================

def ask_continue_to_cropping():
    """Chiede all'utente se vuole proseguire con Step 3+4 (Ritaglio e Mosaico)."""
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
# MAIN: ESECUZIONE PIPELINE (MODIFICATO PER LOOP)
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
    print(f"\n   ✓ Processati: {prep_oss}\n   ✗ Falliti: {fail_oss}")
    if stats_oss:
        ra_min, ra_max = stats_oss['ra_range']; dec_min, dec_max = stats_oss['dec_range']
        print(f"\n   📊 Campo: RA: {ra_min:.4f}° → {ra_max:.4f}° | DEC: {dec_min:.4f}° → {dec_max:.4f}° | Scala media: {stats_oss['avg_scale']:.3f}\"/px")

    # LITH
    print("\n🛰️  LITH/HST (Estrazione WCS esistente)")
    prep_lith, fail_lith, stats_lith = process_lith_folder(INPUT_LITH, OUTPUT_LITH_WCS, logger)
    print(f"\n   ✓ Processati: {prep_lith}\n   ✗ Falliti: {fail_lith}")
    if stats_lith:
        ra_min, ra_max = stats_lith['ra_range']; dec_min, dec_max = stats_lith['dec_range']
        print(f"\n   📊 Campo: RA: {ra_min:.4f}° → {ra_max:.4f}° | DEC: {dec_min:.4f}° → {dec_max:.4f}° | Scala media: {stats_lith['avg_scale']:.3f}\"/px")

    total_prep = prep_oss + prep_lith
    logger.info(f"Totale: {total_prep} preparati, {fail_lith + fail_oss} falliti")
    if total_prep > 0:
        print(f"\n✅ COMPLETATO! File con WCS in:\n       • {OUTPUT_OSSERVATORIO_WCS}\n       • {OUTPUT_LITH_WCS}")
    else:
        print(f"\n⚠️ Nessun file processato.")
    return total_prep > 0

def main_step2(INPUT_HUBBLE, INPUT_OBSERVATORY, OUTPUT_HUBBLE, OUTPUT_OBSERVATORY, BASE_DIR, logger):
    """Funzione principale Step 2."""
    if not REPROJECT_AVAILABLE:
        print("\n⚠️ Step 2 saltato: libreria reproject non disponibile")
        return False
    logger.info("=" * 60); logger.info("REGISTRAZIONE CON RISOLUZIONE NATIVA"); logger.info("=" * 60)
    print("\n" + "🔭"*35)
    print(f"STEP 3: REGISTRAZIONE (RISOLUZIONE NATIVA)".center(70))
    print("🔭"*35)
    print(f"\n📂 CONFIGURAZIONE:\n   Input Hubble: {INPUT_HUBBLE}\n   Input Observatory: {INPUT_OBSERVATORY}\n   Output: {BASE_DIR / '3_registered_native'}")

    print(f"\n{'='*70}\nANALISI IMMAGINI\n{'='*70}")
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
            print(f"\n   {source_name}:\n       Min: {min(scales):.4f}\"/px | Max: {max(scales):.4f}\"/px | Media: {np.mean(scales):.4f}\"/px")

    print(f"\n{'='*70}\nCREAZIONE FRAME WCS COMUNE\n{'='*70}")
    common_wcs = create_common_wcs_frame(all_info, logger)
    if common_wcs is None:
        print(f"\n❌ Impossibile creare WCS comune!"); logger.error("Creazione WCS comune fallita."); return False

    print(f"\n{'='*70}\nREGISTRAZIONE (Risoluzione Nativa)\n{'='*70}")
    total_success, total_error = 0, 0
    if hubble_info:
        s, e = register_images(hubble_info, common_wcs, OUTPUT_HUBBLE, "Hubble", logger); total_success += s; total_error += e
    if obs_info:
        s, e = register_images(obs_info, common_wcs, OUTPUT_OBSERVATORY, "Observatory", logger); total_success += s; total_error += e

    print(f"\n{'='*70}\n📊 RIEPILOGO REGISTRAZIONE\n{'='*70}")
    print(f"\n   Totale registrate: {total_success}\n   Totale errori: {total_error}")
    if total_success > 0:
        print(f"\n✅ REGISTRAZIONE COMPLETATA!")
        print(f"\n   ✨ VANTAGGIO: Ogni immagine ha mantenuto la sua risoluzione nativa!")
    with log_lock: logger.info(f"Registrazione completata: {total_success} successo, {total_error} errori")
    return total_success > 0

def main():
    """Funzione principale della pipeline."""
    
    # --- MODIFICA: Log e selezione avvengono prima del loop ---
    logger = setup_logging()
    
    # Seleziona uno o più target
    target_dirs = select_target_directory()
    if not target_dirs:
        print("Nessun target selezionato. Uscita.")
        return
    # --- FINE MODIFICA ---

    logger.info("=" * 60)
    logger.info("PIPELINE UNIFICATA: STEP 1 + STEP 2")
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

    # --- MODIFICA: Loop principale sui target selezionati ---
    for BASE_DIR in target_dirs:
        print("\n" + "🚀"*35)
        print(f"INIZIO ELABORAZIONE TARGET: {BASE_DIR.name}".center(70))
        print("🚀"*35)
        logger.info(f"INIZIO TARGET: {BASE_DIR.name}")

        # Definizione percorsi per QUESTO target
        INPUT_OSSERVATORIO = BASE_DIR / '1_originarie' / 'local_raw'
        INPUT_LITH = BASE_DIR / '1_originarie' / 'img_lights'
        OUTPUT_OSSERVATORIO_WCS = BASE_DIR / '2_wcs' / 'osservatorio'
        OUTPUT_LITH_WCS = BASE_DIR / '2_wcs' / 'hubble'
        INPUT_HUBBLE = OUTPUT_LITH_WCS
        INPUT_OBSERVATORY = OUTPUT_OSSERVATORIO_WCS
        OUTPUT_HUBBLE = BASE_DIR / '3_registered_native' / 'hubble'
        OUTPUT_OBSERVATORY = BASE_DIR / '3_registered_native' / 'observatory'

        # STEP 1
        step1_start = time.time()
        step1_success = main_step1(INPUT_OSSERVATORIO, INPUT_LITH, OUTPUT_OSSERVATORIO_WCS, OUTPUT_LITH_WCS, logger)
        step1_time = time.time() - step1_start
        print(f"\n⏱️ Tempo Step 1 (WCS): {step1_time:.2f}s")

        if not step1_success:
            print(f"\n❌ Pipeline interrotta per {BASE_DIR.name}: Step 1 fallito")
            logger.error(f"Step 1 fallito per {BASE_DIR.name}")
            failed_targets.append(BASE_DIR)
            continue # Salta al prossimo target

        # STEP 2
        step2_start = time.time()
        step2_success = main_step2(INPUT_HUBBLE, INPUT_OBSERVATORY, OUTPUT_HUBBLE, OUTPUT_OBSERVATORY, BASE_DIR, logger)
        step2_time = time.time() - step2_start
        print(f"\n⏱️ Tempo Step 2 (Registro): {step2_time:.2f}s")
        
        if step2_success:
            successful_targets.append(BASE_DIR)
            logger.info(f"TARGET COMPLETATO: {BASE_DIR.name}")
        else:
            failed_targets.append(BASE_DIR)
            logger.error(f"Step 2 fallito per {BASE_DIR.name}")
            
        print(f"⏱️ Tempo totale per {BASE_DIR.name}: {step1_time + step2_time:.2f}s")

    # --- FINE MODIFICA: Fine loop ---

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
    print(f"\n   ⏱️ Tempo totale batch: {elapsed_total:.2f}s")

    if not successful_targets:
        print("\n❌ Nessun target completato con successo.")
        return

    # === MODIFICA: Chiede di continuare e avvia lo script successivo in loop ===
    if ask_continue_to_cropping():
        try:
            script_dir = Path(__file__).parent
            next_script = script_dir / 'step2_croppedmosaico.py'
            
            if next_script.exists():
                print(f"\n🚀 Avvio Step 3+4 in loop per {len(successful_targets)} target...")
                for BASE_DIR in successful_targets:
                    print(f"\n--- Avvio per {BASE_DIR.name} ---")
                    # Passa il BASE_DIR come argomento
                    subprocess.run([sys.executable, str(next_script), str(BASE_DIR)])
                    print(f"--- Completato {BASE_DIR.name} ---")
            else:
                print(f"\n⚠️  Script {next_script.name} non trovato nella directory {script_dir}")
                print(f"   Eseguilo manualmente quando pronto")
        except Exception as e:
            print(f"\n⚠️  Impossibile avviare automaticamente {next_script.name}: {e}")
            print(f"   Eseguilo manualmente: python {next_script.name}")
    else:
        print("\n👋 Arrivederci!")
    # === FINE MODIFICA ===


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️ Tempo totale esecuzione script: {elapsed:.2f}s")