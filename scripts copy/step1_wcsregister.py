"""
PIPELINE COMPLETO: CONVERSIONE WCS + REGISTRAZIONE
Combina Step 1 (Conversione WCS) e Step 2 (Registrazione) in un unico script.
FIX ALLINEAMENTO: Frame WCS globale unificato per tutte le immagini.
FIX MEMORIA: Processing seriale con garbage collection.
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
import gc

# ✅ FIX PATH + IMPORT CORRETTO
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cleanRAM import pulisci_ram

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

ASTROMETRYNET_SOLVE_FIELD = Path("C:/cygwin64/bin/solve-field.exe")  # Se installato
USE_ASTROMETRYNET = False  # Cambia a True se vuoi usare Astrometry.net

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

# ============================================================================
# PARAMETRI PROCESSING
# ============================================================================
NUM_THREADS = 1  # SEMPRE 1 per evitare OOM
REPROJECT_ORDER = 'bilinear'
MAX_CANVAS_DIMENSION = 10000  # Limite canvas per evitare OOM
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
    choice = input("\n👉 Continuare con step2_croppedmosaico.py? [S/n]: ")

    # Aggiunto controllo per evitare problemi con input vuoto
    if choice == '':
        choice = 's'

    return choice.strip().lower() in ('s', 'si', 'y', 'yes')

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    """Configura logging con file e console."""
    os.makedirs(LOG_DIR_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR_ROOT / f'pipeline_{timestamp}.log'
    
    # Rimuovi handler esistenti
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configura logging
    logging.basicConfig(
        level=logging.DEBUG,  # Cambiato da INFO a DEBUG per più dettagli
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Forza ricreazione
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info(f"LOG FILE: {log_file}")
    logger.info("="*70)
    logger.info(f"Python: {sys.version}")
    logger.info(f"Numpy: {np.__version__}")
    logger.info(f"Astropy: {astropy.__version__}")
    
    return logger

# ============================================================================
# STEP 1: CONVERSIONE WCS
# ============================================================================

def solve_file_with_siril(input_file, output_dir, logger):
    """Plate solving con Siril."""
    try:
        output_file = output_dir / f"{input_file.stem}_wcs.fits"
        
        if output_file.exists():
            logger.info(f"⏭️  {input_file.name}: già risolto")
            return {'status': 'solved', 'file': input_file.name, 'output': output_file}
        
        siril_script = f"""
            cd "{input_file.parent}"
            setcatalogue nomad
            platesolve "{input_file.name}" -out="{output_file.name}" -force
            close
            """
        
        script_path = input_file.parent / f"solve_{input_file.stem}.ssf"
        with open(script_path, 'w') as f:
            f.write(siril_script)
        
        cmd = [str(SIRIL_CLI_PATH), "-s", str(script_path)]
        
        logger.debug(f"🔍 {input_file.name}: Eseguo Siril...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=PLATE_SOLVE_TIMEOUT,
            cwd=input_file.parent
        )
        
        try:
            script_path.unlink()
        except:
            pass
        
        if result.returncode == 0 and output_file.exists():
            try:
                with fits.open(output_file) as hdul:
                    wcs = WCS(hdul[0].header)
                    if wcs.has_celestial:
                        logger.info(f"✅ {input_file.name}: plate solving OK")
                        return {'status': 'success', 'file': input_file.name, 'output': output_file}
            except Exception as e:
                logger.error(f"❌ {input_file.name}: WCS non valido: {e}")
                if output_file.exists():
                    output_file.unlink()
                return {'status': 'error', 'file': input_file.name, 'reason': 'Invalid WCS'}
        
        logger.error(f"❌ {input_file.name}: plate solving fallito")
        if result.stdout:
            logger.debug(f"   STDOUT: {result.stdout[:300]}")
        if result.stderr:
            logger.debug(f"   STDERR: {result.stderr[:300]}")
        
        return {'status': 'error', 'file': input_file.name, 'reason': 'Siril failed'}
        
    except subprocess.TimeoutExpired:
        logger.error(f"⏱️  {input_file.name}: timeout ({PLATE_SOLVE_TIMEOUT}s)")
        return {'status': 'error', 'file': input_file.name, 'reason': 'Timeout'}
    
    except Exception as e:
        logger.error(f"❌ {input_file.name}: {e}")
        return {'status': 'error', 'file': input_file.name, 'reason': str(e)}

def solve_file_with_astrometry(input_file, output_dir, logger):
    """Plate solving con Astrometry.net."""
    try:
        output_file = output_dir / f"{input_file.stem}_wcs.fits"
        
        if output_file.exists():
            return {'status': 'solved', 'file': input_file.name, 'output': output_file}
        
        cmd = [
            str(ASTROMETRYNET_SOLVE_FIELD),
            str(input_file),
            "--overwrite",
            "--no-plots",
            "--new-fits", str(output_file),
            "--scale-units", "arcsecperpix",
            "--scale-low", "0.5",
            "--scale-high", "3.0"
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=PLATE_SOLVE_TIMEOUT)
        
        if result.returncode == 0 and output_file.exists():
            logger.info(f"✅ {input_file.name}: plate solving OK (Astrometry.net)")
            return {'status': 'success', 'file': input_file.name, 'output': output_file}
        
        logger.error(f"❌ {input_file.name}: Astrometry.net fallito")
        return {'status': 'error', 'file': input_file.name}
        
    except Exception as e:
        logger.error(f"❌ {input_file.name}: {e}")
        return {'status': 'error', 'file': input_file.name}

def process_osservatorio_folder(input_dir, output_dir, logger):
    """
    Processa local_raw: estrae WCS da OBJCTRA/OBJCTDEC.
    FIX: Processing seriale.
    """
    fits_files = list(Path(input_dir).glob('**/*.fit')) + list(Path(input_dir).glob('**/*.fits'))
    
    if not fits_files:
        return 0, 0, None
    
    logger.info(f"Observatory (local_raw): {len(fits_files)} file")
    
    prep, fail = 0, 0
    ra_list, dec_list, scale_list = [], [], []
    created_from_metadata = 0
    platesolve_success = 0
    
    # Verifica disponibilità plate solving
    plate_solve_available = False
    solve_func = None
    solver_name = None
    
    if USE_ASTROMETRYNET and ASTROMETRYNET_SOLVE_FIELD.exists():
        plate_solve_available = True
        solve_func = solve_file_with_astrometry
        solver_name = "Astrometry.net"
    elif SIRIL_CLI_PATH.exists():
        plate_solve_available = True
        solve_func = solve_file_with_siril
        solver_name = "Siril"
    
    def process_single_file(filepath):
        """Processa singolo file: crea WCS da metadati o plate solving."""
        nonlocal created_from_metadata, platesolve_success
        
        output_file = output_dir / f"{filepath.stem}_wcs.fits"
        
        if output_file.exists():
            logger.debug(f"⏭️  {filepath.name}: già elaborato")
            return {'status': 'exists', 'file': filepath.name, 'output': output_file}
        
        # METODO 1: CREA WCS DA OBJCTRA/OBJCTDEC
        try:
            with fits.open(filepath) as hdul:
                header = hdul[0].header
                data = hdul[0].data
                
                if data is None:
                    raise ValueError("Nessun dato nel file")
                
                objctra = header.get('OBJCTRA')
                objctdec = header.get('OBJCTDEC')
                xpixsz = header.get('XPIXSZ')
                focallen = header.get('FOCALLEN', header.get('FOCAL'))
                xbin = header.get('XBINNING', 1)
                
                if not (objctra and objctdec):
                    raise ValueError(f"Metadati WCS mancanti")
                
                # Conversione coordinate
                try:
                    coord = SkyCoord(f"{objctra} {objctdec}", unit=(u.hourangle, u.deg))
                    ra_deg = coord.ra.degree
                    dec_deg = coord.dec.degree
                except:
                    ra_parts = str(objctra).split()
                    dec_parts = str(objctdec).split()
                    
                    if len(ra_parts) != 3 or len(dec_parts) != 3:
                        raise ValueError(f"Formato coordinate non valido")
                    
                    h, m, s = map(float, ra_parts)
                    ra_deg = (h + m/60.0 + s/3600.0) * 15.0
                    
                    d, arcm, arcs = map(float, dec_parts)
                    sign = 1 if d >= 0 else -1
                    dec_deg = d + sign * (abs(arcm)/60.0 + abs(arcs)/3600.0)
                
                logger.debug(f"{filepath.name}: RA={ra_deg:.6f}°, DEC={dec_deg:.6f}°")
                
                # Calcolo pixel scale
                if xpixsz and focallen and focallen > 0:
                    pixel_size_mm = (xpixsz * xbin) / 1000.0
                    pixel_scale_arcsec = 206.265 * pixel_size_mm / focallen
                    
                    # FIX: Verifica realismo
                    if pixel_scale_arcsec < 0.1:
                        estimated_base_focal = focallen * 0.63
                        pixel_scale_arcsec = 206.265 * pixel_size_mm / estimated_base_focal
                        logger.warning(f"⚠️ {filepath.name}: FOCALLEN={focallen}mm corretto → scala={pixel_scale_arcsec:.4f}\"/px")
                    
                    logger.debug(f"{filepath.name}: Pixel scale: {pixel_scale_arcsec:.4f}\"/px")
                    
                elif focallen and focallen > 0:
                    pixel_size_mm = (12.0 * xbin) / 1000.0
                    pixel_scale_arcsec = 206.265 * pixel_size_mm / focallen
                    
                    if pixel_scale_arcsec < 0.1:
                        estimated_base_focal = focallen * 0.63
                        pixel_scale_arcsec = 206.265 * pixel_size_mm / estimated_base_focal
                    
                    logger.warning(f"⚠️ {filepath.name}: XPIXSZ mancante → {pixel_scale_arcsec:.4f}\"/px")
                else:
                    pixel_scale_arcsec = 1.5
                    logger.warning(f"⚠️ {filepath.name}: FOCALLEN=0 → scala default 1.5\"/px")
                
                pixel_scale_deg = pixel_scale_arcsec / 3600.0
                
                # Creazione WCS
                wcs = WCS(naxis=2)
                height, width = data.shape
                wcs.wcs.crpix = [width / 2.0, height / 2.0]
                wcs.wcs.crval = [ra_deg, dec_deg]
                wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]
                wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
                wcs.wcs.radesys = 'ICRS'
                wcs.wcs.equinox = 2000.0
                
                # Salvataggio
                new_header = wcs.to_header()
                
                preserve_keys = [
                    'DATE-OBS', 'EXPOSURE', 'EXPTIME', 'FILTER', 
                    'INSTRUME', 'OBSERVER', 'TELESCOP',
                    'CCD-TEMP', 'GAIN', 'RDNOISE', 'PEDESTAL',
                    'XBINNING', 'YBINNING', 'XPIXSZ', 'YPIXSZ',
                    'FOCALLEN', 'APERTURE', 'SAT_LEVE',
                    'SITELAT', 'SITELONG', 'FOCUSPOS',
                    'OBJCTRA', 'OBJCTDEC'
                ]
                
                for key in preserve_keys:
                    if key in header:
                        try:
                            new_header[key] = header[key]
                        except:
                            pass
                
                new_header['ORIGINAL'] = filepath.name
                new_header['WCSMETHO'] = ('OBJCTRA/OBJCTDEC', 'WCS creation method')
                new_header['PREPDATE'] = datetime.now().isoformat()
                new_header['PIXSCALE'] = (pixel_scale_arcsec, 'Pixel scale (arcsec/px)')
                new_header['WCSCRA'] = (ra_deg, 'WCS center RA (deg)')
                new_header['WCSCDEC'] = (dec_deg, 'WCS center DEC (deg)')
                
                fits.PrimaryHDU(data=data, header=new_header).writeto(
                    output_file, 
                    overwrite=True, 
                    output_verify='silentfix'
                )
                
                created_from_metadata += 1
                
                logger.info(f"✅ {filepath.name}: WCS creato (RA={ra_deg:.4f}°, DEC={dec_deg:.4f}°, scala={pixel_scale_arcsec:.2f}\"/px)")
                
                return {
                    'status': 'success',
                    'method': 'metadata',
                    'file': filepath.name,
                    'output': output_file,
                    'ra': ra_deg,
                    'dec': dec_deg,
                    'scale': pixel_scale_arcsec
                }
                
        except Exception as e:
            logger.debug(f"⚠️ {filepath.name}: metadati falliti: {e}")
            
            # FALLBACK: PLATE SOLVING
            if plate_solve_available:
                logger.debug(f"   → Tentativo plate solving con {solver_name}...")
                
                result = solve_func(filepath, output_dir, logger)
                
                if result['status'] == 'success':
                    platesolve_success += 1
                    
                    try:
                        with fits.open(result['output']) as hdul:
                            wcs = WCS(hdul[0].header)
                            ra, dec = wcs.wcs.crval
                            
                            if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
                                scale = np.sqrt(wcs.wcs.cd[0,0]**2 + wcs.wcs.cd[0,1]**2) * 3600
                            else:
                                scale = abs(wcs.wcs.cdelt[0]) * 3600
                            
                            return {
                                'status': 'success',
                                'method': 'platesolve',
                                'file': filepath.name,
                                'output': result['output'],
                                'ra': ra,
                                'dec': dec,
                                'scale': scale
                            }
                    except:
                        pass
                    
                    return result
                else:
                    logger.error(f"❌ {filepath.name}: tutti i metodi falliti")
                    return {'status': 'error', 'file': filepath.name}
            else:
                logger.error(f"❌ {filepath.name}: plate solving non disponibile")
                return {'status': 'error', 'file': filepath.name}
    
    # ✅ PROCESSING SERIALE (NO ThreadPoolExecutor)
    print(f"\n🔄 Processing {len(fits_files)} file (SERIALE)...")
    
    with tqdm(total=len(fits_files), desc="  Observatory", unit="file") as pbar:
        for filepath in fits_files:
            result = process_single_file(filepath)
            
            if result['status'] in ['success', 'exists']:
                if 'ra' in result and 'dec' in result and 'scale' in result:
                    ra_list.append(result['ra'])
                    dec_list.append(result['dec'])
                    scale_list.append(result['scale'])
                else:
                    try:
                        with fits.open(result['output']) as hdul:
                            wcs = WCS(hdul[0].header)
                            if wcs.has_celestial:
                                ra_list.append(wcs.wcs.crval[0])
                                dec_list.append(wcs.wcs.crval[1])
                                scale = hdul[0].header.get('PIXSCALE', 1.5)
                                scale_list.append(scale)
                    except:
                        pass
                
                prep += 1
            else:
                fail += 1
            
            pbar.update(1)
            
            # Garbage collection ogni 5 file
            if (pbar.n % 5) == 0:
                gc.collect()
    
    print(f"   📊 Metodi usati:")
    print(f"      ✓ WCS da metadati: {created_from_metadata}")
    print(f"      ✓ Plate solving: {platesolve_success}")
    print(f"   ✓ Processati: {prep}, ✗ Falliti: {fail}")
    
    stats = None
    if ra_list:
        stats = {
            'ra_range': (min(ra_list), max(ra_list)),
            'dec_range': (min(dec_list), max(dec_list)),
            'avg_scale': np.mean(scale_list)
        }
        
        logger.info(f"   Range RA: [{stats['ra_range'][0]:.4f}, {stats['ra_range'][1]:.4f}]°")
        logger.info(f"   Range DEC: [{stats['dec_range'][0]:.4f}, {stats['dec_range'][1]:.4f}]°")
        logger.info(f"   Scala media: {stats['avg_scale']:.2f}\"/px")
        
        print(f"   📍 Range RA: [{stats['ra_range'][0]:.4f}, {stats['ra_range'][1]:.4f}]°")
        print(f"   📍 Range DEC: [{stats['dec_range'][0]:.4f}, {stats['dec_range'][1]:.4f}]°")
        print(f"   📏 Scala media: {stats['avg_scale']:.2f}\"/px")
    
    return prep, fail, stats

def process_lith_folder(input_dir, output_dir, logger):
    """Processa HST (estrazione WCS)."""
    fits_files = list(Path(input_dir).glob('**/*.fit')) + list(Path(input_dir).glob('**/*.fits'))
    
    if not fits_files:
        return 0, 0, None
    
    logger.info(f"HST: {len(fits_files)} file")
    
    prep, fail = 0, 0
    ra_list, dec_list, scale_list = [], [], []
    
    with tqdm(total=len(fits_files), desc="  HST") as pbar:
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
                                prep += 1
                                break
            except:
                fail += 1
            pbar.update(1)
    
    stats = None
    if ra_list:
        stats = {
            'ra_range': (min(ra_list), max(ra_list)),
            'dec_range': (min(dec_list), max(dec_list)),
            'avg_scale': np.mean(scale_list)
        }
    
    return prep, fail, stats

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
            pbar.update(1)
    
    print(f"   ✓ {len(wcs_info)}/{len(files)} con WCS valido")
    return wcs_info

def register_to_unified_frame(wcs_info_list, global_wcs, global_shape, ra_center, dec_center, output_dir, source_name, logger):
    """
    Registra immagini nel frame WCS globale unificato.
    FIX MEMORIA: Processing SERIALE con garbage collection esplicito.
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
                    return {'status': 'error', 'file': filename}
                
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
                    logger.warning(f"⚠️ {filename}: nessun pixel valido dopo reproiezione")
                    return {'status': 'error', 'file': filename}
                
                reprojected_denorm = np.where(valid_out, 
                    reprojected * footprint * (p_high - p_low) + p_low, 
                    np.nan)
                
                # Salvataggio
                new_header = global_wcs.to_header()
                for key in ['DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'TELESCOP']:
                    if key in header:
                        try:
                            new_header[key] = header[key]
                        except:
                            pass
                
                new_header['ORIGINAL'] = filename
                new_header['ALIGNED'] = (True, 'Aligned to global frame')
                new_header['GLOBCENR'] = (ra_center, 'Global center RA')
                new_header['GLOBCEND'] = (dec_center, 'Global center DEC')
                new_header['NORMLOW'] = (p_low, 'Norm low percentile')
                new_header['NORMHIGH'] = (p_high, 'Norm high percentile')
                
                output_path = output_dir / f"reg_{filepath.stem}.fits"
                fits.PrimaryHDU(data=reprojected_denorm.astype(np.float32), 
                               header=new_header).writeto(output_path, overwrite=True)
                
                logger.info(f"✅ {filename}: registrato in frame globale")
                return {'status': 'success', 'file': filename}
                
        except MemoryError as e:
            logger.error(f"❌ {filename}: OUT OF MEMORY - {e}")
            return {'status': 'error', 'file': filename, 'reason': 'OOM'}
            
        except Exception as e:
            logger.error(f"❌ {filename}: {e}")
            return {'status': 'error', 'file': filename, 'reason': str(e)}
        
        finally:
            # ✅ CLEANUP MEMORIA ESPLICITO
            try:
                del data, data_norm, reprojected, footprint, reprojected_denorm
            except:
                pass
            gc.collect()
    
    # ✅ PROCESSING SERIALE (NO ThreadPoolExecutor)
    print(f"\n🔄 Registrazione {source_name}: {len(wcs_info_list)} immagini")
    print(f"   ⚙️  Modalità: SERIALE (1 immagine alla volta)")
    
    with tqdm(total=len(wcs_info_list), desc=f"  {source_name}", unit="img") as pbar:
        for idx, info in enumerate(wcs_info_list):
            # Stima memoria
            mem_mb = (global_shape[0] * global_shape[1] * 4 * 3) / (1024**2)
            pbar.set_postfix({
                'img': f"{idx+1}/{len(wcs_info_list)}", 
                'mem': f"~{mem_mb:.0f}MB"
            })
            
            result = reproject_single(info)
            
            if result['status'] == 'success':
                success += 1
            else:
                error += 1
            
            pbar.update(1)
            
            # ✅ GARBAGE COLLECTION ogni 3 immagini
            if (idx + 1) % 3 == 0:
                gc.collect()
                pbar.set_postfix({'gc': '♻️ cleanup'})
    
    print(f"   ✓ Successo: {success}, ✗ Errori: {error}")
    logger.info(f"{source_name}: {success} successi, {error} errori")
    
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
    
    # ✅ FIX: Usa funzione corretta
    print("\n🧹 Pulizia RAM post-Step1...")
    pulisci_ram(verbose=True, aggressive=True)

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
    
    # ✅ LIMITE CANVAS GLOBALE
    if nx_global > MAX_CANVAS_DIMENSION or ny_global > MAX_CANVAS_DIMENSION:
        scale_factor = max(nx_global, ny_global) / MAX_CANVAS_DIMENSION
        target_scale_arcsec *= scale_factor
        target_scale_deg = target_scale_arcsec / 3600
        
        nx_global = int(np.ceil(ra_span / target_scale_deg))
        ny_global = int(np.ceil(dec_span / target_scale_deg))
        
        logger.warning(f"⚠️ Canvas ridimensionato: {nx_global}x{ny_global} (scala {target_scale_arcsec:.4f}\"/px)")
        print(f"\n   ⚠️ Canvas ridotto a {nx_global}x{ny_global} px")
    
    # Calcola memoria
    mem_per_image_gb = (nx_global * ny_global * 4 * 3) / (1024**3)
    logger.info(f"💾 Memoria stimata: {mem_per_image_gb:.2f} GB per immagine")
    print(f"   💾 Memoria: ~{mem_per_image_gb:.2f} GB/immagine")
    
    if mem_per_image_gb > 2.0:
        logger.warning(f"⚠️ Memoria elevata!")
        print(f"   ⚠️ ATTENZIONE: Memoria elevata ({mem_per_image_gb:.1f} GB)")
    
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

        print("\n🧹 Pulizia RAM post-Hubble...")
        pulisci_ram(verbose=True, aggressive=True)
    
    if obs_info:
        s, e = register_to_unified_frame(obs_info, global_wcs, (ny_global, nx_global),
                                        ra_center, dec_center, output_obs, "OBSERVATORY", logger)
        total_success += s
        total_error += e

        # ✅ FIX: Usa funzione corretta
        print("\n🧹 Pulizia RAM post-Observatory...")
        pulisci_ram(verbose=True, aggressive=True)

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

if __name__ == "__main__":
    main()