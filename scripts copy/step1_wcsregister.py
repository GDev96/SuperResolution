"""
PIPELINE COMPLETO: CONVERSIONE WCS + REGISTRAZIONE
Combina Step 1 (Conversione WCS) e Step 2 (Registrazione) in un unico script.
FIX ALLINEAMENTO: Frame WCS globale unificato per tutte le immagini.
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
# CONFIGURAZIONE PATH DINAMICI
# ============================================================================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR_ROOT = PROJECT_ROOT / "logs"
SCRIPTS_DIR = CURRENT_SCRIPT_DIR

# ============================================================================
# PARAMETRI NORMALIZZAZIONE E VALIDAZIONE
# ============================================================================
MIN_UNIQUE_VALUES_ORIGINAL = 100
MIN_UNIQUE_VALUES_NORMALIZED = 50
MIN_RANGE_THRESHOLD = 0.01  # Range minimo 1% (dati normalizzati [0,1])
PERCENTILE_LOW = 1.0
PERCENTILE_HIGH = 99.0

# ============================================================================
# CONFIGURAZIONE PLATE SOLVING SIRIL
# ============================================================================
SIRIL_CLI_PATH = Path("C:/Program Files/Siril/bin/siril-cli.exe")
PLATE_SOLVE_TIMEOUT = 300

print(f"📂 Project Root: {PROJECT_ROOT}")
print(f"📂 Data Dir: {ROOT_DATA_DIR}")

# ============================================================================
# IMPORT REPROJECT
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

NUM_THREADS = 7
REPROJECT_ORDER = 'bilinear'
log_lock = threading.Lock()

# ============================================================================
# FUNZIONI MENU
# ============================================================================

def select_target_directory():
    """Mostra menu per selezionare target."""
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET".center(70))
    print("📂"*35)
    
    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        return []

    if not subdirs:
        print(f"\n❌ Nessuna cartella trovata in {ROOT_DATA_DIR}")
        return []

    print("\nCartelle disponibili:")
    print(f"   0: ✨ TUTTI i {len(subdirs)} target")
    print("   " + "─"*30)
    for i, d in enumerate(subdirs):
        print(f"   {i+1}: {d.name}")

    while True:
        try:
            choice = input(f"\n👉 Seleziona (0-{len(subdirs)}) o 'q': ").strip()
            if choice.lower() == 'q':
                return []
            
            choice = int(choice)
            if choice == 0:
                return subdirs
            if 0 < choice <= len(subdirs):
                return [subdirs[choice-1]]
            print(f"❌ Numero non valido")
        except:
            print("❌ Input non valido")

def ask_continue_to_cropping():
    """Chiede se continuare con step2."""
    print("\n" + "="*70)
    print("🎯 STEP 1-2 COMPLETATI!")
    print("="*70)
    choice = input("\n👉 Continuare con step2_croppedmosaico.py? [S/n]: ").strip().lower()
    return choice in ('', 's', 'si', 'y', 'yes')

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    """Configura logging."""
    os.makedirs(LOG_DIR_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR_ROOT / f'pipeline_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Python: {sys.version}")
    logger.info(f"Numpy: {np.__version__}")
    logger.info(f"Astropy: {astropy.__version__}")
    return logger

# ============================================================================
# STEP 1: CONVERSIONE WCS
# ============================================================================

def parse_coordinates(ra_str, dec_str):
    """Converte coordinate sessagesimali a decimali."""
    try:
        coord = SkyCoord(f"{ra_str} {dec_str}", unit=(u.hourangle, u.deg))
        return coord.ra.degree, coord.dec.degree
    except:
        # Fallback manuale
        ra_parts = ra_str.split()
        dec_parts = dec_str.split()
        h, m, s = map(float, ra_parts)
        ra_deg = (h + m/60 + s/3600) * 15
        d, m, s = map(float, dec_parts)
        sign = 1 if d >= 0 else -1
        dec_deg = d + sign * (m/60 + s/3600)
        return ra_deg, dec_deg

def calculate_pixel_scale(header):
    """Calcola pixel scale da header."""
    xpixsz = header.get('XPIXSZ')
    focal = header.get('FOCALLEN', header.get('FOCAL'))
    xbin = header.get('XBINNING', 1)
    
    if xpixsz and focal:
        pixel_size_mm = (xpixsz * xbin) / 1000.0
        pixel_scale_arcsec = 206.265 * pixel_size_mm / focal
        return pixel_scale_arcsec / 3600.0
    return 1.5 / 3600.0

def create_wcs_from_header(header, data_shape):
    """Crea WCS da OBJCTRA/OBJCTDEC."""
    try:
        objctra = header.get('OBJCTRA')
        objctdec = header.get('OBJCTDEC')
        if not objctra or not objctdec:
            return None
        
        ra_deg, dec_deg = parse_coordinates(objctra, objctdec)
        pixel_scale = calculate_pixel_scale(header)
        
        wcs = WCS(naxis=2)
        height, width = data_shape
        wcs.wcs.crpix = [width/2, height/2]
        wcs.wcs.crval = [ra_deg, dec_deg]
        wcs.wcs.cdelt = [-pixel_scale, pixel_scale]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.radesys = 'ICRS'
        wcs.wcs.equinox = 2000.0
        return wcs
    except:
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

def solve_file_with_siril(input_file, output_dir, logger):
    """
    Esegue plate solving su un singolo file con Siril CLI.
    
    Args:
        input_file: Path al file FITS da risolvere
        output_dir: Directory output
        logger: Logger
    
    Returns:
        dict con status e percorso file risolto
    """
    try:
        filename = input_file.name
        name_no_ext = input_file.stem
        output_file = output_dir / f"{name_no_ext}_wcs.fits"
        
        # Se esiste già, skippa
        if output_file.exists():
            with log_lock:
                logger.info(f"⏭️  {filename}: WCS già presente, skip")
            return {'status': 'solved', 'file': filename, 'output': output_file}
        
        # Comando Siril per plate solving
        # Sintassi: siril-cli -s "platesolve <input> -out=<output> -catalogue=nomad"
        cmd = [
            str(SIRIL_CLI_PATH),
            "-s",
            f'platesolve "{input_file}" -out="{output_file}" -catalogue=nomad -force'
        ]
        
        with log_lock:
            logger.debug(f"🔍 {filename}: Comando Siril: {' '.join(cmd)}")
        
        # Esegui Siril
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=PLATE_SOLVE_TIMEOUT,
            cwd=input_file.parent  # Working directory
        )
        
        # Verifica successo
        if result.returncode == 0 and output_file.exists():
            # Verifica WCS valido
            try:
                with fits.open(output_file) as hdul:
                    wcs = WCS(hdul[0].header)
                    if wcs.has_celestial:
                        with log_lock:
                            logger.info(f"✅ {filename}: Plate solving completato")
                        return {'status': 'success', 'file': filename, 'output': output_file}
            except Exception as e:
                with log_lock:
                    logger.error(f"❌ {filename}: WCS non valido dopo solve: {e}")
                if output_file.exists():
                    output_file.unlink()
                return {'status': 'error', 'file': filename, 'reason': 'Invalid WCS'}
        
        # Fallimento
        with log_lock:
            logger.error(f"❌ {filename}: Plate solving fallito")
            logger.debug(f"   stdout: {result.stdout[:200]}")
            logger.debug(f"   stderr: {result.stderr[:200]}")
        
        return {'status': 'error', 'file': filename, 'reason': 'Siril failed'}
        
    except subprocess.TimeoutExpired:
        with log_lock:
            logger.error(f"⏱️  {filename}: Timeout ({PLATE_SOLVE_TIMEOUT}s)")
        return {'status': 'error', 'file': filename, 'reason': 'Timeout'}
    
    except Exception as e:
        with log_lock:
            logger.error(f"❌ {filename}: {e}")
        return {'status': 'error', 'file': filename, 'reason': str(e)}


def process_osservatorio_folder(input_dir, output_dir, logger):
    """
    Processa osservatorio con PLATE SOLVING (Siril CLI).
    FIX: Aggiunto plate solving reale invece di solo conversione coordinate.
    """
    fits_files = list(Path(input_dir).glob('**/*.fit')) + list(Path(input_dir).glob('**/*.fits'))
    
    if not fits_files:
        logger.warning(f"Nessun file in {input_dir}")
        return 0, 0, None
    
    # Verifica Siril installato
    if not SIRIL_CLI_PATH.exists():
        logger.error(f"❌ Siril CLI non trovato: {SIRIL_CLI_PATH}")
        logger.error("   Installa Siril da https://siril.org/download/")
        print(f"\n❌ Siril non trovato: {SIRIL_CLI_PATH}")
        print("   Plate solving disabilitato. Installa Siril per abilitarlo.")
        return 0, len(fits_files), None
    
    logger.info(f"Trovati {len(fits_files)} file osservatorio → Plate Solving con Siril")
    
    prepared_count = 0
    failed_count = 0
    ra_list = []
    dec_list = []
    scale_list = []
    
    # Plate solving con multithreading
    with ThreadPoolExecutor(max_workers=min(NUM_THREADS, 2)) as executor:  # Max 2 thread per Siril
        futures = {
            executor.submit(solve_file_with_siril, input_file, output_dir, logger): input_file.name
            for input_file in fits_files
        }
        
        with tqdm(total=len(fits_files), desc="  Plate Solving", unit="file") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    
                    if result['status'] in ['success', 'solved']:
                        # Estrai info WCS
                        try:
                            with fits.open(result['output']) as hdul:
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
                        
                except Exception as exc:
                    failed_count += 1
                    filename = futures[future]
                    logger.error(f"❌ {filename}: Exception: {exc}")
                
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
    """Processa HST (estrazione WCS)."""
    fits_files = list(Path(input_dir).glob('**/*.fit')) + list(Path(input_dir).glob('**/*.fits'))
    
    if not fits_files:
        return 0, 0, None
    
    logger.info(f"Trovati {len(fits_files)} file HST")
    
    prepared_count = 0
    failed_count = 0
    ra_list = []
    dec_list = []
    scale_list = []
    
    with tqdm(total=len(fits_files), desc="  HST", unit="file") as pbar:
        for f in fits_files:
            try:
                with fits.open(f) as hdul:
                    for hdu in hdul:
                        if hdu.data is not None and len(hdu.data.shape) >= 2:
                            wcs = WCS(hdu.header)
                            if wcs.has_celestial:
                                data = hdu.data[0] if len(hdu.data.shape) == 3 else hdu.data
                                
                                output_file = output_dir / f"{f.stem}_wcs.fits"
                                
                                new_hdu = fits.PrimaryHDU(data=data, header=hdu.header.copy())
                                new_hdu.header['ORIGINAL'] = f.name
                                new_hdu.header['PREPDATE'] = datetime.now().isoformat()
                                new_hdu.writeto(output_file, overwrite=True, output_verify='silentfix')
                                
                                ra, dec = wcs.wcs.crval
                                scale = np.sqrt(wcs.wcs.cd[0,0]**2 + wcs.wcs.cd[0,1]**2) * 3600 if hasattr(wcs.wcs, 'cd') else 0.04
                                ra_list.append(ra)
                                dec_list.append(dec)
                                scale_list.append(scale)
                                prepared_count += 1
                                break
            except:
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
# STEP 2: REGISTRAZIONE CON FRAME WCS GLOBALE UNIFICATO
# ============================================================================

def extract_wcs_info(filepath, logger):
    """Estrae WCS da file FITS."""
    try:
        with fits.open(filepath) as hdul:
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    wcs = WCS(hdu.header)
                    if wcs.has_celestial:
                        data = hdu.data[0] if len(hdu.data.shape) == 3 else hdu.data
                        center = wcs.pixel_to_world(data.shape[1]/2, data.shape[0]/2)
                        
                        # Calcola pixel scale
                        if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
                            scale_deg = np.sqrt(wcs.wcs.cd[0,0]**2 + wcs.wcs.cd[0,1]**2)
                        else:
                            scale_deg = abs(wcs.wcs.cdelt[0])
                        
                        return {
                            'file': filepath,
                            'hdu_index': i,
                            'wcs': wcs,
                            'shape': data.shape,
                            'center_ra': center.ra.deg,
                            'center_dec': center.dec.deg,
                            'pixel_scale': scale_deg * 3600
                        }
        return None
    except Exception as e:
        logger.error(f"❌ {filepath.name}: {e}")
        return None

def analyze_images(input_dir, source_name, logger):
    """Analizza immagini in directory."""
    files = list(set(list(Path(input_dir).glob('*.fits')) + 
                     list(Path(input_dir).glob('*.fit')) + 
                     list(Path(input_dir).glob('*_wcs.fits'))))
    
    if not files:
        return []
    
    print(f"\n📂 {source_name}: {len(files)} file")
    wcs_info = []
    
    with tqdm(total=len(files), desc=f"  Analisi {source_name}") as pbar:
        for f in files:
            info = extract_wcs_info(f, logger)
            if info:
                wcs_info.append(info)
                with log_lock:
                    logger.info(f"✓ {os.path.basename(f)}: "
                               f"RA={info['center_ra']:.4f}°, DEC={info['center_dec']:.4f}°, "
                               f"scale={info['pixel_scale']:.4f}\"/px (NATIVA)")
            else:
                with log_lock:
                    logger.warning(f"✗ {os.path.basename(f)}: WCS non valido")
            pbar.update(1)
    
    print(f"   ✓ {len(wcs_info)}/{len(files)} con WCS valido")
    
    return wcs_info


def register_to_unified_frame(wcs_info_list, global_wcs, global_shape, ra_center, dec_center, output_dir, source_name, logger):
    """
    Registra immagini nel frame WCS globale unificato.
    FIX: Tutte le immagini vengono reproiettate nello STESSO frame.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    success, error = 0, 0
    
    def reproject_single(info):
        """Reproietta singola immagine nel frame globale."""
        try:
            filepath = info['file']
            filename = filepath.name
            
            with fits.open(filepath) as hdul:
                hdu = hdul[info['hdu_index']]
                data = hdu.data.copy()
                header = hdu.header.copy()
                wcs_orig = WCS(header)
                
                if len(data.shape) == 3:
                    data = data[0]
                
                # Validazione
                valid = np.isfinite(data) & (data != 0)
                if valid.sum() < 100:
                    logger.error(f"❌ {filename}: dati insufficienti")
                    return {'status': 'error'}
                
                # Normalizzazione
                p_low = np.percentile(data[valid], PERCENTILE_LOW)
                p_high = np.percentile(data[valid], PERCENTILE_HIGH)
                data_norm = np.clip((data - p_low) / (p_high - p_low), 0, 1)
                data_norm[~valid] = np.nan
                
                # Reproiezione nel FRAME GLOBALE
                reprojected, footprint = reproject_interp(
                    (data_norm, wcs_orig),
                    global_wcs,
                    shape_out=global_shape,
                    order=REPROJECT_ORDER
                )
                
                # Denormalizzazione
                valid_out = (footprint > 0.01) & np.isfinite(reprojected)
                if not valid_out.any():
                    logger.error(f"❌ {filename}: nessun pixel valido")
                    return {'status': 'error'}
                
                reprojected_denorm = np.where(valid_out, 
                    reprojected * footprint * (p_high - p_low) + p_low, 
                    np.nan)
                
                # Salvataggio
                new_header = global_wcs.to_header()
                for key in ['DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'TELESCOP']:
                    if key in header:
                        new_header[key] = header[key]
                
                new_header['ORIGINAL'] = filename
                new_header['ALIGNED'] = (True, 'Aligned to global frame')
                new_header['GLOBCENR'] = (ra_center, 'Global center RA')
                new_header['GLOBCEND'] = (dec_center, 'Global center DEC')
                new_header['NORMLOW'] = (p_low, 'Norm low')
                new_header['NORMHIGH'] = (p_high, 'Norm high')
                
                output_path = output_dir / f"reg_{filepath.stem}.fits"
                fits.PrimaryHDU(data=reprojected_denorm.astype(np.float32), 
                               header=new_header).writeto(output_path, overwrite=True)
                
                logger.info(f"✅ {filename}: registrato in frame globale")
                return {'status': 'success'}
                
        except Exception as e:
            logger.error(f"❌ {filename}: {e}")
            return {'status': 'error'}
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(reproject_single, info): info for info in wcs_info_list}
        
        with tqdm(total=len(wcs_info_list), desc=f"  {source_name}") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result['status'] == 'success':
                    success += 1
                else:
                    error += 1
                pbar.update(1)
    
    print(f"   ✓ Successo: {success}, ✗ Errori: {error}")
    return success, error

# ============================================================================
# MAIN
# ============================================================================

def main_step1(input_obs, input_lith, output_obs, output_lith, logger):
    """Step 1: Conversione WCS."""
    print("\n" + "="*70)
    print("STEP 1: CONVERSIONE WCS".center(70))
    print("="*70)
    
    os.makedirs(output_obs, exist_ok=True)
    os.makedirs(output_lith, exist_ok=True)
    
    print("\n📡 OSSERVATORIO (Plate Solving)")
    prep_obs, fail_obs, _ = process_osservatorio_folder(input_obs, output_obs, logger)
    print(f"   ✓ Processati: {prep_obs}, ✗ Falliti: {fail_obs}")
    
    print("\n🛰️  HST (Estrazione WCS)")
    prep_lith, fail_lith, _ = process_lith_folder(input_lith, output_lith, logger)
    print(f"   ✓ Processati: {prep_lith}, ✗ Falliti: {fail_lith}")
    
    return (prep_obs + prep_lith) > 0

def main_step2(input_hubble, input_obs, output_hubble, output_obs, target_name, logger):
    """Step 2: Registrazione con frame globale unificato."""
    if not REPROJECT_AVAILABLE:
        print("\n⚠️ Step 2 saltato: reproject non disponibile")
        return False
    
    print("\n" + "="*70)
    print("STEP 2: REGISTRAZIONE (FRAME GLOBALE UNIFICATO)".center(70))
    print("="*70)
    
    # Analisi
    hubble_info = analyze_images(input_hubble, "HUBBLE", logger)
    obs_info = analyze_images(input_obs, "OBSERVATORY", logger)
    all_info = hubble_info + obs_info
    
    if not all_info:
        logger.error("Nessuna immagine con WCS valido")
        return False
    
    # Calcola frame WCS globale unificato
    print(f"\n🌐 CALCOLO FRAME WCS GLOBALE ({target_name})")
    
    all_ra, all_dec = [], []
    for info in all_info:
        wcs = info['wcs']
        ny, nx = info['shape']
        corners = np.array([[0,0], [nx,0], [0,ny], [nx,ny]])
        coords = wcs.pixel_to_world(corners[:,0], corners[:,1])
        all_ra.extend([c.ra.deg for c in coords])
        all_dec.extend([c.dec.deg for c in coords])
    
    ra_center = (min(all_ra) + max(all_ra)) / 2
    dec_center = (min(all_dec) + max(all_dec)) / 2
    ra_span = (max(all_ra) - min(all_ra)) * 1.05
    dec_span = (max(all_dec) - min(all_dec)) * 1.05
    
    # Scala target (usa risoluzione migliore disponibile)
    if hubble_info:
        target_scale_arcsec = np.mean([i['pixel_scale'] for i in hubble_info])
    else:
        target_scale_arcsec = np.mean([i['pixel_scale'] for i in obs_info])
    
    target_scale_deg = target_scale_arcsec / 3600
    
    nx_global = int(np.ceil(ra_span / target_scale_deg))
    ny_global = int(np.ceil(dec_span / target_scale_deg))
    
    # WCS globale
    global_wcs = WCS(naxis=2)
    global_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    global_wcs.wcs.crval = [ra_center, dec_center]
    global_wcs.wcs.crpix = [nx_global/2, ny_global/2]
    global_wcs.wcs.cd = np.array([[-target_scale_deg, 0], [0, target_scale_deg]])
    global_wcs.wcs.radesys = 'ICRS'
    global_wcs.wcs.equinox = 2000.0
    
    logger.info(f"Frame globale: centro=({ra_center:.4f}, {dec_center:.4f}), "
                f"canvas={nx_global}x{ny_global}, scala={target_scale_arcsec:.4f}\"/px")
    
    print(f"   Centro: RA={ra_center:.4f}°, DEC={dec_center:.4f}°")
    print(f"   Canvas: {nx_global}x{ny_global} px")
    print(f"   Scala: {target_scale_arcsec:.4f}\"/px")
    
    # Registrazione
    total_success, total_error = 0, 0
    
    if hubble_info:
        s, e = register_to_unified_frame(hubble_info, global_wcs, (ny_global, nx_global),
                                        ra_center, dec_center, output_hubble, "HUBBLE", logger)
        total_success += s
        total_error += e
    
    if obs_info:
        s, e = register_to_unified_frame(obs_info, global_wcs, (ny_global, nx_global),
                                        ra_center, dec_center, output_obs, "OBSERVATORY", logger)
        total_success += s
        total_error += e
    
    print(f"\n📊 RIEPILOGO: {total_success} successo, {total_error} errori")
    return total_success > 0

def main():
    """Pipeline principale."""
    logger = setup_logging()
    
    targets = select_target_directory()
    if not targets:
        return
    
    print("\n" + "="*70)
    print("🚀 PIPELINE: WCS + REGISTRAZIONE (FRAME GLOBALE)".center(70))
    print("="*70)
    
    successful = []
    failed = []
    
    for target_dir in targets:
        print(f"\n{'🚀'*35}")
        print(f"TARGET: {target_dir.name}".center(70))
        print(f"{'🚀'*35}")
        
        # Path
        input_obs = target_dir / '1_originarie' / 'local_raw'
        input_lith = target_dir / '1_originarie' / 'img_lights'
        output_obs_wcs = target_dir / '2_wcs' / 'osservatorio'
        output_lith_wcs = target_dir / '2_wcs' / 'hubble'
        output_hubble = target_dir / '3_registered_native' / 'hubble'
        output_obs = target_dir / '3_registered_native' / 'observatory'
        
        # Step 1
        if not main_step1(input_obs, input_lith, output_obs_wcs, output_lith_wcs, logger):
            logger.error(f"Step 1 fallito: {target_dir.name}")
            failed.append(target_dir)
            continue
        
        # Step 2
        if main_step2(output_lith_wcs, output_obs_wcs, output_hubble, output_obs, target_dir.name, logger):
            successful.append(target_dir)
        else:
            failed.append(target_dir)
    
    # Riepilogo
    print("\n" + "="*70)
    print("📊 RIEPILOGO FINALE".center(70))
    print("="*70)
    print(f"✅ Completati: {len(successful)}")
    for t in successful:
        print(f"   - {t.name}")
    print(f"\n❌ Falliti: {len(failed)}")
    for t in failed:
        print(f"   - {t.name}")
    
    if successful and ask_continue_to_cropping():
        next_script = SCRIPTS_DIR / 'step2_croppedmosaico.py'
        if next_script.exists():
            for target in successful:
                subprocess.run([sys.executable, str(next_script), str(target)])