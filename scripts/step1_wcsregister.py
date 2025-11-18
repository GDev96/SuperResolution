"""
PIPELINE UNIFICATO: PLATE SOLVING (SIRIL) + WCS EXTRACTION + REGISTRAZIONE GLOBALE

CORREZIONE CHIAVE:
- Eliminata la dipendenza da Astrometry.net API.
- Plate Solving per file locali (`local_raw`) eseguito tramite Siril CLI (richiede Siril 1.4.0+).
- Mantenuto il FIX CRITICO per il calcolo del WCS comune (margine 15%) che risolveva
  l'errore "Nessun pixel dopo reproiezione".
"""

import os
import sys
import time
import logging
import shutil
from datetime import datetime
import numpy as np
import astropy
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Angle
import astropy.units as u
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import warnings
import subprocess

try:
    from reproject import reproject_interp
    REPROJECT_AVAILABLE = True
except ImportError:
    REPROJECT_AVAILABLE = False
    print("="*70)
    print("ERRORE: Libreria 'reproject' non trovata.")
    print("Installa con: pip install reproject")
    print("="*70)

warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)

# ============================================================================
# CONFIGURAZIONE GLOBALE & SIRIL
# ============================================================================
NUM_THREADS = 8 
REPROJECT_ORDER = 'bicubic'
# Se Siril non è nel tuo PATH, imposta il percorso completo qui
# Esempio Windows: SIRIL_EXEC_PATH = "C:\\Program Files\\Siril\\siril.exe"
SIRIL_EXEC_PATH = "C:\\Program Files\\Siril\\bin\\siril.exe" 
SIRIL_MIN_VERSION = "1.4.0" # Richiesto per il comando 'wcs' avanzato
log_lock = threading.Lock()

# PARAMETRI DI VALIDAZIONE (Mantenuti)
PERCENTILE_LOW = 0.5   
PERCENTILE_HIGH = 99.5 
MIN_UNIQUE_VALUES_ORIGINAL = 50   
MIN_UNIQUE_VALUES_NORMALIZED = 30  
MIN_RANGE_THRESHOLD = 1e-10  

# ============================================================================
# CONFIGURAZIONE PATH
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR_ROOT = PROJECT_ROOT / "logs"
SCRIPTS_DIR = PROJECT_ROOT / "finale"

# ============================================================================
# SETUP LOGGING & SELEZIONE TARGET (Mantenute)
# ============================================================================

def setup_logging():
    """Configura logging per pipeline."""
    os.makedirs(LOG_DIR_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = LOG_DIR_ROOT / f'pipeline_siril_{timestamp}.log'

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
    logger.info(f"MODALITÀ: Plate Solving Siril ({SIRIL_EXEC_PATH}) + WCS Extraction")
    logger.info(f"LOG FILE: {log_filename}")
    logger.info(f"Astropy: {astropy.__version__}")
    logger.info(f"Threads: {NUM_THREADS}")
    logger.info("="*80)
    return logger

def select_target_directory():
    # ... (Funzione identica a quella precedente per selezionare la directory) ...
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET".center(70))
    print("📂"*35)
    print(f"\nScansione sottocartelle in: {ROOT_DATA_DIR}")

    subdirs_with_data = []
    
    try:
        for d in ROOT_DATA_DIR.iterdir():
            if d.is_dir():
                fits_files_local = list((d / '1_origin' / 'local_raw').glob('**/*.fit*'))
                fits_files_lith = list((d / '1_origin' / 'img_lights').glob('**/*.fit*'))
                
                summary = []
                if fits_files_lith:
                     summary.append(f"{len(fits_files_lith)} file FITS in img_lights (HST)")
                if fits_files_local:
                    summary.append(f"{len(fits_files_local)} file FITS in local_raw (Local)")
                
                if summary:
                    subdirs_with_data.append({'path': d, 'summary': ", ".join(summary)})
                    
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile leggere {ROOT_DATA_DIR}")
        print(f"   Dettagli: {e}")
        return []

    if not subdirs_with_data:
        print(f"\n❌ Nessuna sottocartella contenente dati FITS trovata.")
        return []

    print("\nTarget disponibili:")
    print(f"   0: ✨ Processa TUTTI i {len(subdirs_with_data)} target")
    print("   " + "─"*30)
    for i, item in enumerate(subdirs_with_data):
        print(f"   {i+1}: {item['path'].name.ljust(15)} ({item['summary']})")

    while True:
        print("\n" + "─"*70)
        try:
            choice_str = input(f"👉 Seleziona (0-{len(subdirs_with_data)}) o 'q': ").strip()
            if choice_str.lower() == 'q':
                return []
            choice = int(choice_str)
            if choice == 0:
                return [item['path'] for item in subdirs_with_data]
            if 0 < choice <= len(subdirs_with_data):
                return [subdirs_with_data[choice-1]['path']]
            print(f"❌ Numero non valido")
        except ValueError:
            print("❌ Input non valido")

# ============================================================================
# SIRIL PLATE SOLVING FUNCTION (SOSTITUZIONE API)
# ============================================================================

def solve_file_with_siril(input_file, output_dir, logger):
    """
    Esegue il plate solving su un file FITS usando Siril CLI.
    """
    filename = input_file.name
    output_file = output_dir / f"{input_file.stem}_solved.fits"
    
    # 1. Check se già risolto
    if output_file.exists():
        try:
            with fits.open(output_file) as hdul:
                if WCS(hdul[0].header).has_celestial:
                    return {'status': 'solved', 'file': filename, 'output': output_file}
        except:
            pass # Riprova se file corrotto
            
    # 2. Copia il file originale nella cartella di output
    try:
        shutil.copy2(input_file, output_file)
    except Exception as e:
        with log_lock: logger.error(f"❌ {filename}: Errore copia file: {e}")
        return {'status': 'error', 'file': filename, 'reason': f"Errore copia file: {e}"}

    # 3. Crea uno script temporaneo per Siril
    temp_script_path = output_dir / f"siril_solve_{input_file.stem}.txt"
    # Carica la copia, esegui il plate solving WCS, e sovrascrivi
    script_content = f"""
    requires {SIRIL_MIN_VERSION}
    # Imposta la directory di lavoro su quella di output
    setproc dir {output_dir.resolve()}
    
    # Carica il file copiato, lo risolve, e lo salva con WCS
    load {output_file.name} 
    wcs # Comando di plate solving di Siril (richiede connessione internet la prima volta per indici)
    save {output_file.name} 
    """
    
    with open(temp_script_path, 'w') as f:
        f.write(script_content)
        
    # 4. Esegui Siril
    try:
        command = [
            SIRIL_EXEC_PATH,
            "-s", str(temp_script_path.resolve()),
        ]
        
        with log_lock: logger.info(f"⚙️ Esecuzione Siril su {filename}...")

        # Esegui Siril
        result = subprocess.run(
            command, 
            check=False, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',  # FIX: Forza UTF-8 invece di CP1252
            errors='replace',   # Ignora caratteri invalidi
            timeout=300
        )        

        if result.returncode != 0:
            with log_lock:
                logger.error(f"❌ {filename}: Siril fallito (Codice {result.returncode}). Controlla log Siril: {result.stderr.strip()}")
            return {'status': 'error', 'file': filename, 'reason': f"Siril failed: {result.stderr.strip()}"}

        # 5. Check WCS
        with fits.open(output_file) as hdul:
            if WCS(hdul[0].header).has_celestial:
                with log_lock: logger.info(f"✅ {filename}: Plate Solved con Siril.")
                return {'status': 'success', 'file': filename, 'output': output_file}
            else:
                with log_lock: logger.error(f"❌ {filename}: Siril eseguito ma WCS non trovato nell'header.")
                return {'status': 'error', 'file': filename, 'reason': 'Siril executed but WCS not found.'}

    except FileNotFoundError:
        with log_lock: logger.error(f"❌ {filename}: Eseguibile Siril non trovato. Controlla il path: {SIRIL_EXEC_PATH}")
        return {'status': 'error', 'file': filename, 'reason': f"Eseguibile Siril non trovato: {SIRIL_EXEC_PATH}"}
        
    except Exception as e:
        with log_lock: logger.error(f"❌ {filename}: Errore Siril: {e}")
        return {'status': 'error', 'file': filename, 'reason': str(e)}
        
    finally:
        # Cleanup temporary script
        if temp_script_path.exists():
            os.remove(temp_script_path)


def prepare_images_with_platesolving(logger, input_dir_obs, output_dir_prep, input_dir_lith, output_dir_lith_prep):
    """
    Gestisce sia i file locali (con plate solving SIRIL) che i file HST/LITH (con estrazione WCS).
    """
    os.makedirs(output_dir_prep, exist_ok=True)
    os.makedirs(output_dir_lith_prep, exist_ok=True)

    # 1. Processing HST/LITH files (WCS Extraction - NO PLATE SOLVING)
    print("\n🛰️  LITH/HST (Img_lights) - Estrazione WCS esistente...")
    prep_lith, fail_lith, _ = process_lith_folder(input_dir_lith, output_dir_lith_prep, logger)
    print(f"   ✓ Preparati: {prep_lith}, ✗ Falliti: {fail_lith}")

    # 2. Processing Local files (Plate Solving SIRIL)
    local_fits_files = list(Path(input_dir_obs).glob('**/*.fit')) + list(Path(input_dir_obs).glob('**/*.fits'))
    if not local_fits_files:
        print("\n📡 OSSERVATORIO (Local_raw) - Nessun file trovato per Plate Solving.")
        total_prepared_paths = list(output_dir_lith_prep.glob('*_wcs.fits'))
        return total_prepared_paths, []
    
    print("\n📡 OSSERVATORIO (Local_raw) - Plate Solving con Siril CLI...")
    
    solved_files_info = []
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            executor.submit(solve_file_with_siril, input_file, output_dir_prep, logger): input_file.name
            for input_file in local_fits_files
        }
        
        with tqdm(total=len(local_fits_files), desc="   Plate Solving (Siril)", unit="file") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result['status'] in ['success', 'solved']:
                    solved_files_info.append(result['output'])
                pbar.update(1)
                
    
    # Unisci tutti i file preparati (solved + lith/hst)
    total_prepared_paths = solved_files_info + list(output_dir_lith_prep.glob('*_wcs.fits'))

    print(f"   ✓ Risolti: {len(solved_files_info)}/{len(local_fits_files)} file locali.")
    print(f"   ✓ Totale file WCS pronti: {len(total_prepared_paths)}.")

    return total_prepared_paths, []

# ============================================================================
# UTILITY WCS & REGISTRAZIONE (Mantenute, vedi script precedente per il corpo completo)
# ============================================================================
def parse_coordinates(ra_str, dec_str):
    try: ra_deg = Angle(ra_str, unit=u.hour).degree
    except:
        try: ra_deg = Angle(ra_str, unit=u.deg).degree
        except: ra_deg = float(ra_str)
    try: dec_deg = Angle(dec_str, unit=u.deg).degree
    except: dec_deg = float(dec_str)
    return ra_deg, dec_deg

def extract_lith_data(filename, logger):
    """Estrae dati e WCS da file LITH/HST (Multi-HDU)."""
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
                            except: pixel_scale = 0.04
                            return data, header, {'shape': data.shape, 'ra': ra, 'dec': dec, 'pixel_scale': pixel_scale}
                    except: continue
        return None, None, None
    except Exception as e:
        logger.error(f"Errore {os.path.basename(filename)}: {e}")
        return None, None, None

def process_lith_folder(input_dir, output_dir, logger):
    """Processa file LITH/HST (Estrazione WCS)."""
    fits_files = list(Path(input_dir).glob('**/*.fit')) + list(Path(input_dir).glob('**/*.fits'))
    if not fits_files: return 0, 0, None
    os.makedirs(output_dir, exist_ok=True)
    prepared_count = 0; failed_count = 0
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
                except: failed_count += 1
            else: failed_count += 1
            pbar.update(1)
    return prepared_count, failed_count, None

def extract_wcs_info_safe(filepath, logger):
    try:
        with fits.open(filepath) as hdul:
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    try:
                        wcs = WCS(hdu.header)
                        if wcs.has_celestial:
                            data = hdu.data[0] if len(hdu.data.shape) == 3 else hdu.data
                            ny, nx = data.shape
                            world = wcs.wcs_pix2world([[nx/2, ny/2]], 1)
                            center_ra = float(world[0][0])
                            center_dec = float(world[0][1])
                            
                            if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
                                cd = wcs.wcs.cd
                                pixel_scale_deg = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
                            else: pixel_scale_deg = abs(wcs.wcs.cdelt[0])
                            pixel_scale_arcsec = pixel_scale_deg * 3600.0
                            
                            return {'file': filepath, 'hdu_index': i, 'wcs': wcs, 'shape': data.shape, 'center_ra': center_ra, 'center_dec': center_dec, 'pixel_scale': pixel_scale_arcsec}
                    except: continue
        return None
    except Exception as e:
        with log_lock: logger.error(f"Errore WCS {os.path.basename(filepath)}: {e}")
        return None

def analyze_images(files_list, source_name, logger):
    if not files_list: return []
    print(f"\n📂 Analisi WCS {source_name}: {len(files_list)} file")
    wcs_info_list = []
    with tqdm(total=len(files_list), desc=f"   Analisi {source_name}") as pbar:
        for filepath in files_list:
            info = extract_wcs_info_safe(filepath, logger)
            if info: wcs_info_list.append(info)
            pbar.update(1)
    print(f"   ✓ {len(wcs_info_list)}/{len(files_list)} con WCS valido")
    return wcs_info_list

def create_common_wcs_frame(wcs_info_list, logger):
    """
    Crea frame WCS comune con MARGINE ADATTIVO per evitare pixel persi.
    """
    if not wcs_info_list: 
        return None, None, None
    
    # FASE 1: RACCOLTA BOUNDS DA TUTTI I CORNER
    ra_min, ra_max = float('inf'), float('-inf')
    dec_min, dec_max = float('inf'), float('-inf')
    center_ra_approx = np.median([info['center_ra'] for info in wcs_info_list])
    
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSTICA BOUNDS:")
    
    for idx, info in enumerate(wcs_info_list):
        wcs, shape = info['wcs'], info['shape']
        ny, nx = shape
        
        # Corner in pixel space
        corners = np.array([[0, 0], [nx, 0], [0, ny], [nx, ny]])
        
        # Trasforma in world coordinates
        world = wcs.wcs_pix2world(corners, 1)
        
        logger.info(f"  Immagine {idx+1}: {info['file'].name}")
        logger.info(f"    Centro: RA={info['center_ra']:.4f}, DEC={info['center_dec']:.4f}")
        logger.info(f"    Shape: {nx}x{ny}, Scale: {info['pixel_scale']:.4f}\"/px")
        
        # Gestione wrap-around RA
        for coord in world:
            ra, dec = float(coord[0]), float(coord[1])
            
            # Normalizza RA rispetto al centro approssimativo
            if ra > center_ra_approx + 180: 
                ra -= 360
            if ra < center_ra_approx - 180: 
                ra += 360
            
            ra_min = min(ra_min, ra)
            ra_max = max(ra_max, ra)
            dec_min = min(dec_min, dec)
            dec_max = max(dec_max, dec)
        
        logger.info(f"    Corner RA range: [{world[:,0].min():.4f}, {world[:,0].max():.4f}]")
        logger.info(f"    Corner DEC range: [{world[:,1].min():.4f}, {world[:,1].max():.4f}]")
    
    # FASE 2: CALCOLO CENTRO E SPAN
    ra_center = (ra_min + ra_max) / 2.0
    dec_center = (dec_min + dec_max) / 2.0
    
    ra_span_deg = ra_max - ra_min
    dec_span_deg = dec_max - dec_min
    
    logger.info(f"\nBounds globali (pre-margine):")
    logger.info(f"  RA: [{ra_min:.4f}, {ra_max:.4f}] → span={ra_span_deg:.4f}°")
    logger.info(f"  DEC: [{dec_min:.4f}, {dec_max:.4f}] → span={dec_span_deg:.4f}°")
    logger.info(f"  Centro: RA={ra_center:.4f}°, DEC={dec_center:.4f}°")
    
    # FASE 3: PIXEL SCALE
    pixel_scales_arcsec = [info['pixel_scale'] for info in wcs_info_list]
    native_pixel_scale_arcsec = np.median(pixel_scales_arcsec) if pixel_scales_arcsec else 0.04
    native_pixel_scale_deg = native_pixel_scale_arcsec / 3600.0
    
    # FASE 4: MARGINE ADATTIVO
    # Se ci sono poche immagini o sono già sovrapposte, usa margine maggiore
    num_images = len(wcs_info_list)
    
    if num_images == 1:
        # Singola immagine: aggiungi 20% per sicurezza
        margin_factor = 1.2
    elif ra_span_deg < native_pixel_scale_deg * 100 and dec_span_deg < native_pixel_scale_deg * 100:
        # Immagini molto vicine/sovrapposte: margine 50%
        margin_factor = 1.5
    elif ra_span_deg < native_pixel_scale_deg * 500:
        # Campo piccolo: margine 30%
        margin_factor = 1.3
    else:
        # Campo grande: margine 15%
        margin_factor = 1.15
    
    logger.info(f"\nMargine applicato: {margin_factor:.2f}x ({(margin_factor-1)*100:.0f}%)")
    
    ra_span_deg *= margin_factor
    dec_span_deg *= margin_factor
    
    # FASE 5: CALCOLO DIMENSIONI CANVAS
    nx_out = int(np.ceil(abs(ra_span_deg) / native_pixel_scale_deg))
    ny_out = int(np.ceil(abs(dec_span_deg) / native_pixel_scale_deg))
    
    # Dimensioni minime
    nx_out = max(nx_out, 100)
    ny_out = max(ny_out, 100)
    
    # Limita dimensioni massime
    MAX_DIMENSION = 15000
    if nx_out > MAX_DIMENSION or ny_out > MAX_DIMENSION:
        logger.warning(f"Canvas ({nx_out}x{ny_out}) troppo grande! Ridimensiono...")
        scale_factor = max(nx_out, ny_out) / MAX_DIMENSION
        native_pixel_scale_deg *= scale_factor
        native_pixel_scale_arcsec = native_pixel_scale_deg * 3600.0
        nx_out = int(np.ceil(ra_span_deg / native_pixel_scale_deg))
        ny_out = int(np.ceil(dec_span_deg / native_pixel_scale_deg))
        logger.warning(f"Nuove dimensioni: {nx_out}x{ny_out}, scala={native_pixel_scale_arcsec:.4f}\"/px")
    
    common_shape = (ny_out, nx_out)
    
    # FASE 6: CREA WCS COMUNE
    common_wcs = WCS(naxis=2)
    common_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    common_wcs.wcs.crval = [ra_center, dec_center]
    common_wcs.wcs.crpix = [nx_out / 2.0, ny_out / 2.0]
    
    # CD matrix
    cd_matrix = np.array([
        [-native_pixel_scale_deg, 0.0],
        [0.0, native_pixel_scale_deg]
    ])
    common_wcs.wcs.cd = cd_matrix
    common_wcs.wcs.radesys = 'ICRS'
    common_wcs.wcs.equinox = 2000.0
    
    # FASE 7: VERIFICA COVERAGE
    logger.info("\nVERIFICA COVERAGE PRE-REPROIEZIONE:")
    all_pixels_covered = True
    
    for idx, info in enumerate(wcs_info_list):
        wcs_orig = info['wcs']
        shape_orig = info['shape']
        ny, nx = shape_orig
        
        # Test corners
        corners_pix = np.array([[0, 0], [nx, 0], [0, ny], [nx, ny]])
        corners_world = wcs_orig.wcs_pix2world(corners_pix, 1)
        
        # Proietta nel nuovo frame
        corners_new_pix = common_wcs.wcs_world2pix(corners_world, 1)
        
        # Verifica se dentro bounds
        x_min, x_max = corners_new_pix[:, 0].min(), corners_new_pix[:, 0].max()
        y_min, y_max = corners_new_pix[:, 1].min(), corners_new_pix[:, 1].max()
        
        inside = (x_min >= -10 and x_max <= nx_out + 10 and 
                  y_min >= -10 and y_max <= ny_out + 10)
        
        status = "✓ OK" if inside else "✗ FUORI BOUNDS"
        logger.info(f"  Img {idx+1}: X=[{x_min:.0f}, {x_max:.0f}], Y=[{y_min:.0f}, {y_max:.0f}] {status}")
        
        if not inside:
            all_pixels_covered = False
            logger.warning(f"  ⚠️ Immagine {idx+1} esce dai bounds! Aumenta margine.")
    
    # FASE 8: RIEPILOGO
    size_mb = (nx_out * ny_out * 4) / (1024**2)
    
    logger.info("\n" + "="*60)
    logger.info("WCS COMUNE FINALE:")
    logger.info(f"  Centro: RA={ra_center:.4f}°, DEC={dec_center:.4f}°")
    logger.info(f"  Estensione: RA={ra_span_deg:.4f}°, DEC={dec_span_deg:.4f}°")
    logger.info(f"  Canvas: {nx_out}x{ny_out} pixel")
    logger.info(f"  Scala: {native_pixel_scale_arcsec:.4f}\"/px")
    logger.info(f"  Memoria: {size_mb:.1f} MB/immagine")
    logger.info(f"  Coverage check: {'✓ PASS' if all_pixels_covered else '✗ FAIL (possibili pixel persi)'}")
    logger.info("="*60)
    
    print(f"\n✓ WCS comune: RA={ra_center:.4f}°, DEC={dec_center:.4f}°")
    print(f"✓ Frame comune: {nx_out}x{ny_out} pixel, scala={native_pixel_scale_arcsec:.4f}\"/px")
    print(f"✓ Coverage: {'OK' if all_pixels_covered else 'WARNING - controlla log'}")
    
    return common_wcs, common_shape, native_pixel_scale_arcsec


# SOSTITUISCI la funzione reproject_image_native completa (linea ~547):

def reproject_image_native(wcs_info, common_wcs, common_shape, common_scale_arcsec, output_dir, logger):
    """
    Reproiezione con fix per WCS HST/HUBBLE.
    FIX CRITICO: Usa reproject_adaptive invece di reproject_interp per gestire WCS complessi.
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
            
            # ...existing code per len(shape)==3...
            if len(data_original.shape) == 3:
                data_original = data_original[0]
            
            # FIX 1: Normalizza RADESYS tra WCS input e output
            if hasattr(wcs_orig.wcs, 'radesys') and wcs_orig.wcs.radesys:
                if wcs_orig.wcs.radesys != common_wcs.wcs.radesys:
                    with log_lock:
                        logger.warning(f"{filename}: RADESYS mismatch (input={wcs_orig.wcs.radesys}, common={common_wcs.wcs.radesys})")
                        logger.info(f"  Forzando conversione a {common_wcs.wcs.radesys}")
                    
                    # Forza conversione WCS
                    wcs_orig = wcs_orig.deepcopy()
                    wcs_orig.wcs.radesys = common_wcs.wcs.radesys
                    wcs_orig.wcs.equinox = common_wcs.wcs.equinox
            
            # ...existing validation code...
            valid_mask_orig = np.isfinite(data_original) & (data_original != 0)
            valid_data_orig = data_original[valid_mask_orig]
            if valid_data_orig.size < 100 or (valid_data_orig.max() - valid_data_orig.min()) < MIN_RANGE_THRESHOLD:
                with log_lock:
                    logger.warning(f"⚠️  {filename}: Dati originali insufficienti.")
                return {'status': 'error', 'file': filename, 'reason': 'Insufficient data range/pixels'}
            
            unique_orig = len(np.unique(valid_data_orig[:min(10000, valid_data_orig.size)]))
            if unique_orig < MIN_UNIQUE_VALUES_ORIGINAL:
                with log_lock:
                    logger.error(f"❌ {filename}: Dati originali binari ({unique_orig} valori)")
                return {'status': 'error', 'file': filename, 'reason': f'Binary original ({unique_orig})'}
            
            # Normalization
            p_low = np.percentile(valid_data_orig, PERCENTILE_LOW)
            p_high = np.percentile(valid_data_orig, PERCENTILE_HIGH)
            data_clipped = np.clip(data_original, p_low, p_high)
            data_normalized = (data_clipped - p_low) / (p_high - p_low) if p_high > p_low else data_clipped
            data_normalized[~valid_mask_orig] = np.nan
            
            # FIX 2: Usa reproject_adaptive per WCS complessi (più robusto)
            with log_lock:
                logger.info(f"🔄 {filename}: Avvio reproiezione (metodo: adaptive)...")
            
            try:
                from reproject import reproject_adaptive
                reprojected_norm, footprint = reproject_adaptive(
                    (data_normalized, wcs_orig), 
                    common_wcs, 
                    shape_out=common_shape,
                    return_footprint=True,
                    kernel='gaussian',
                    sample_region_width=4
                )
            except ImportError:
                # Fallback a interp con ordine ridotto
                with log_lock:
                    logger.warning(f"{filename}: reproject_adaptive non disponibile, uso interp order=1")
                reprojected_norm, footprint = reproject_interp(
                    (data_normalized, wcs_orig), 
                    common_wcs, 
                    shape_out=common_shape,
                    order=1,  # RIDOTTO da 'bicubic' a 1 (lineare)
                    return_footprint=True
                )
            
            # DEBUG OUTPUT
            with log_lock:
                logger.info(f"DEBUG {filename}:")
                logger.info(f"  WCS orig: RADESYS={wcs_orig.wcs.radesys}, EQUINOX={wcs_orig.wcs.equinox}")
                logger.info(f"  WCS common: RADESYS={common_wcs.wcs.radesys}, EQUINOX={common_wcs.wcs.equinox}")
                logger.info(f"  Input shape: {data_normalized.shape}")
                logger.info(f"  Output shape: {reprojected_norm.shape}")
                logger.info(f"  Input finite: {np.isfinite(data_normalized).sum()}/{data_normalized.size}")
                logger.info(f"  Output finite: {np.isfinite(reprojected_norm).sum()}/{reprojected_norm.size}")
                logger.info(f"  Footprint >0: {(footprint > 0).sum()}/{footprint.size}")
                logger.info(f"  Footprint max: {footprint.max():.4f}")
            
            # ...resto del codice esistente (footprint_safe, denorm, save)...
            footprint_safe = np.clip(footprint, 0, 1)
            valid_reproj = (footprint_safe > 0.01) & np.isfinite(reprojected_norm)
            reprojected_weighted = np.where(valid_reproj, reprojected_norm * footprint_safe, np.nan)
            
            reprojected_denorm = reprojected_weighted.copy()
            valid_mask_final = np.isfinite(reprojected_weighted)
            if not np.any(valid_mask_final):
                with log_lock:
                    logger.error(f"❌ {filename}: NESSUN PIXEL DOPO REPROIEZIONE.")
                return {'status': 'error', 'file': filename, 'reason': 'No pixels after reproject'}
            
            reprojected_denorm[valid_mask_final] = (reprojected_weighted[valid_mask_final] * (p_high - p_low) + p_low)
            
            final_valid = reprojected_denorm[valid_mask_final]
            unique_final = len(np.unique(final_valid[:min(10000, final_valid.size)]))
            if unique_final < MIN_UNIQUE_VALUES_NORMALIZED or (final_valid.max() - final_valid.min()) < MIN_RANGE_THRESHOLD:
                with log_lock:
                    logger.error(f"❌ {filename}: Dati finali corrotti/binari ({unique_final} valori)")
                return {'status': 'error', 'file': filename, 'reason': 'Binary/corrupted final data'}
            
            coverage = (valid_mask_final.sum() / reprojected_denorm.size * 100)
            new_header = common_wcs.to_header()
            new_header['REGDATE'] = datetime.now().isoformat()
            new_header['REGCOV'] = (coverage, "Coverage %")
            
            output_filename = f"reg_{os.path.splitext(filename)[0]}.fits"
            output_path = output_dir / output_filename
            fits.PrimaryHDU(data=reprojected_denorm.astype(np.float32), header=new_header).writeto(output_path, overwrite=True)
            
            with log_lock:
                logger.info(f"✅ {filename}: cov={coverage:.1f}%, shape={common_shape}")
            return {'status': 'success', 'file': filename, 'coverage': coverage}
            
    except Exception as e:
        with log_lock:
            logger.error(f"❌ {os.path.basename(filepath)}: {e}", exc_info=True)
        return {'status': 'error', 'file': os.path.basename(filepath), 'reason': str(e)}

def register_images(wcs_info_list, common_wcs, common_shape, common_scale, output_dir, source_name, logger):
    """Registra immagini con multithreading."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🔄 Registrazione {source_name}: {len(wcs_info_list)} immagini")
    
    success_count = 0; error_count = 0
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            executor.submit(reproject_image_native, info, common_wcs, common_shape, common_scale, output_dir, logger): info
            for info in wcs_info_list
        }
        with tqdm(total=len(wcs_info_list), desc=f"   {source_name}") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result['status'] == 'success': success_count += 1
                    else: error_count += 1
                except Exception as exc:
                    error_count += 1
                    with log_lock: logger.error(f"Thread exception: {exc}")
                pbar.update(1)
    
    print(f"   ✓ Successo: {success_count}, ✗ Errori: {error_count}")
    return success_count, error_count

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main_step1(input_obs, input_lith, output_obs_wcs, output_lith_wcs, logger):
    """Esegue Plate Solving (Siril) e WCS Extraction."""
    logger.info("="*60)
    logger.info("STEP 1: PLATE SOLVING (SIRIL) + WCS EXTRACTION")
    logger.info("="*60)
    
    print("\n" + "="*70)
    print("STEP 1: PLATE SOLVING (SIRIL) + WCS EXTRACTION".center(70))
    print("="*70)
    
    prepared_file_paths, _ = prepare_images_with_platesolving(
        logger, 
        input_obs, 
        output_obs_wcs, 
        input_lith,
        output_lith_wcs
    )
    
    total = len(prepared_file_paths)
    logger.info(f"Step 1 completato: {total} file con WCS pronti.")
    
    # Ritorna i percorsi separati per il raggruppamento nello Step 2
    local_wcs_files = [p for p in prepared_file_paths if p.parent == output_obs_wcs]
    lith_wcs_files = [p for p in prepared_file_paths if p.parent == output_lith_wcs]
    
    return total > 0, local_wcs_files, lith_wcs_files

def main_step2(local_wcs_files, lith_wcs_files, output_hubble, output_obs, logger):
    """Esegue Step 2: Registrazione."""
    if not REPROJECT_AVAILABLE: return False
    
    logger.info("="*60)
    logger.info("STEP 2: REGISTRAZIONE GLOBALE")
    logger.info("="*60)
    
    print("\n" + "="*70)
    print("STEP 2: REGISTRAZIONE GLOBALE".center(70))
    print("="*70)
    
    hubble_info = analyze_images(lith_wcs_files, "HUBBLE (WCS)", logger)
    obs_info = analyze_images(local_wcs_files, "OBSERVATORY (WCS)", logger)
    all_info = hubble_info + obs_info
    
    if not all_info:
        logger.error("Nessuna immagine con WCS valido trovata per la registrazione.")
        return False
    
    print(f"\n{'='*70}")
    print("CREAZIONE WCS COMUNE")
    print(f"{'='*70}")
    
    common_wcs, common_shape, common_scale = create_common_wcs_frame(all_info, logger)
    if common_wcs is None:
        logger.error("Impossibile creare frame WCS comune")
        return False
    
    print(f"\n{'='*70}")
    print("REGISTRAZIONE")
    print(f"{'='*70}")
    
    total_success = 0; total_error = 0
    
    if hubble_info:
        s, e = register_images(hubble_info, common_wcs, common_shape, common_scale, output_hubble, "Hubble", logger)
        total_success += s; total_error += e
    
    if obs_info:
        s, e = register_images(obs_info, common_wcs, common_shape, common_scale, output_obs, "Observatory", logger)
        total_success += s; total_error += e
    
    print(f"\n{'='*70}")
    print(f"TOTALE REGISTRAZIONE: {total_success} successo, {total_error} errori")
    print(f"{'='*70}")
    
    logger.info(f"Step 2 completato: {total_success} registrate")
    return total_success > 0

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

def main():
    """Main pipeline."""
    logger = setup_logging()
    
    target_dirs = select_target_directory()
    if not target_dirs: return
    
    print("\n" + "="*70)
    print("PIPELINE: WCS SOLVING (SIRIL) + REGISTRATION (FULL)".center(70))
    print("="*70)
    
    start_time = time.time()
    successful = []; failed = []
    
    for base_dir in target_dirs:
        print("\n" + "🚀"*35)
        print(f"TARGET: {base_dir.name}".center(70))
        print("🚀"*35)
        
        # Percorsi
        input_obs = base_dir / '1_origin' / 'local_raw'
        input_lith = base_dir / '1_origin' / 'img_lights'
        # Output Plate Solved (Local)
        output_obs_wcs = base_dir / '2_platesolved' / 'observatory'
        # Output WCS Extracted (HST/LITH)
        output_lith_wcs = base_dir / '2_platesolved' / 'hubble'
        # Output Registrazione
        output_hubble = base_dir / '3_registered_native' / 'hubble'
        output_obs = base_dir / '3_registered_native' / 'observatory'

        # Step 1
        t1 = time.time()
        step1_ok, local_wcs_files, lith_wcs_files = main_step1(input_obs, input_lith, output_obs_wcs, output_lith_wcs, logger)
        print(f"\n⏱️  Step 1: {time.time()-t1:.2f}s")
        
        if not step1_ok:
            failed.append(base_dir)
            continue
        
        # Step 2
        t2 = time.time()
        step2_ok = main_step2(local_wcs_files, lith_wcs_files, output_hubble, output_obs, logger)
        print(f"\n⏱️  Step 2: {time.time()-t2:.2f}s")
        
        if step2_ok: successful.append(base_dir)
        else: failed.append(base_dir)
    
    # Riepilogo finale
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("RIEPILOGO FINALE".center(70))
    print("="*70)
    print(f"\n   Target: {len(target_dirs)}")
    print(f"   ✅ Successo: {len(successful)}")
    for t in successful: print(f"      - {t.name}")
    print(f"\n   ❌ Falliti: {len(failed)}")
    for t in failed: print(f"      - {t.name}")
    print(f"\n   ⏱️  Tempo totale: {elapsed:.2f}s")
    
    if not successful: return
    
    # Continua con Step 3+4
    if ask_continue_to_cropping():
        next_script = SCRIPTS_DIR / 'step2_croppedmosaico.py'
        if next_script.exists():
            for base_dir in successful:
                print(f"\n--- Step 3+4: {base_dir.name} ---")
                subprocess.run([sys.executable, str(next_script), str(base_dir)])
        else:
            print(f"\n⚠️  {next_script.name} non trovato. Non è possibile proseguire con il mosaico.")

if __name__ == "__main__":
    main()