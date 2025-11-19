"""
PIPELINE COMPLETO: CONVERSIONE WCS + REGISTRAZIONE
Combina Step 1 (Conversione WCS) e Step 2 (Registrazione) in un unico script.
Gli output dei due step rimangono separati e distinti.
Tutti i metodi sono mantenuti ESATTAMENTE come negli script originali.
MODIFICATO: Integra gestione path dinamici e menu selezione da v1.
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
# CONFIGURAZIONE PATH DINAMICI (UNIVERSALI)
# ============================================================================
# 1. Ottiene la directory corrente dello script (es. .../SuperResolution/scripts)
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent

# 2. Risale alla root del progetto (es. .../SuperResolution)
#    Assumiamo che lo script sia dentro "scripts/", quindi il genitore è la root.
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent

# 3. Dove cercare i dati (M33, ecc.)
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# 4. Dove salvare i log
LOG_DIR_ROOT = ROOT_DATA_DIR / "logs"

# 5. Cartella script (uguale alla directory corrente)
SCRIPTS_DIR = CURRENT_SCRIPT_DIR

# ============================================================================
# PARAMETRI NORMALIZZAZIONE E VALIDAZIONE
# ============================================================================
MIN_UNIQUE_VALUES_ORIGINAL = 100      # Minimo valori unici nell'immagine originale
MIN_UNIQUE_VALUES_NORMALIZED = 50     # Minimo valori unici dopo reproiezione
MIN_RANGE_THRESHOLD = 0.01            # Range minimo dati (max-min)
PERCENTILE_LOW = 1.0                  # Percentile basso per clipping
PERCENTILE_HIGH = 99.0                # Percentile alto per clipping

# ============================================================================

# ============================================================================
# CONFIGURAZIONE PLATE SOLVING SIRIL
# ============================================================================
SIRIL_CLI_PATH = Path("C:/Program Files/Siril/bin/siril-cli.exe")
PLATE_SOLVE_TIMEOUT = 300  # secondi per file
# ============================================================================

print(f"📂 Project Root rilevata: {PROJECT_ROOT}")
print(f"📂 Data Dir rilevata:    {ROOT_DATA_DIR}")
# ============================================================================
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

# Parametri Step 2
NUM_THREADS = 7
REPROJECT_ORDER = 'bilinear'

# Lock per logging thread-safe
log_lock = threading.Lock()

# ============================================================================
# FUNZIONI MENU E SELEZIONE (DA V1)
# ============================================================================

def select_target_directory():
    """
    Mostra un menu per selezionare una o TUTTE le cartelle target.
    """
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET".center(70))
    print("📂"*35)
    print(f"\nScansione sottocartelle in: {ROOT_DATA_DIR}")

    try:
        # Filtra solo le cartelle che contengono dati (es. M33) ed esclude 'logs' e 'splits'
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile leggere la cartella {ROOT_DATA_DIR}")
        print(f"   Dettagli: {e}")
        if not ROOT_DATA_DIR.exists():
             print(f"   ⚠️ La cartella non esiste. Creala e inserisci le cartelle target (es. M33).")
        return []

    if not subdirs:
        print(f"\n❌ ERRORE: Nessuna sottocartella trovata in {ROOT_DATA_DIR}")
        print("   Assicurati di avere la struttura: data/M33/1_originarie/...")
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
# SETUP LOGGING
# ============================================================================

def setup_logging():
    """Configura logging per l'intera pipeline."""
    os.makedirs(LOG_DIR_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = LOG_DIR_ROOT / f'unified_pipeline_{timestamp}.log'
    
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
    """Processa LITH/HST."""
    # Modificato per cercare ricorsivamente in tutte le sottocartelle
    fits_files = list(Path(input_dir).glob('**/*.fit')) + list(Path(input_dir).glob('**/*.fits'))
    
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
    files = list(Path(input_dir).glob('*.fits')) + \
            list(Path(input_dir).glob('*.fit')) + \
            list(Path(input_dir).glob('*_wcs.fits'))
    
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
    FIX: Gestione robusta normalizzazione per prevenire binarizzazione.
    """
    try:
        filepath = wcs_info['file']
        hdu_index = wcs_info['hdu_index']
        native_pixel_scale = wcs_info['pixel_scale']
        filename = os.path.basename(filepath)
        
        with fits.open(filepath) as hdul:
            hdu = hdul[hdu_index]
            data_original = hdu.data.copy()
            header = hdu.header.copy()
            wcs_orig = WCS(header)
            
            # Gestione 3D
            if len(data_original.shape) == 3:
                data_original = data_original[0]
            
            original_shape = data_original.shape
            
            # === VALIDAZIONE DATI ORIGINALI ===
            valid_mask_orig = np.isfinite(data_original) & (data_original != 0)
            valid_data_orig = data_original[valid_mask_orig]
            
            if valid_data_orig.size < 100:
                with log_lock:
                    logger.error(f"❌ {filename}: Dati insufficienti ({valid_data_orig.size} px)")
                return {'status': 'error', 'file': filename, 'reason': 'Insufficient data'}
            
            # Verifica range ORIGINALE (prima della normalizzazione)
            data_min = valid_data_orig.min()
            data_max = valid_data_orig.max()
            data_range = data_max - data_min
            
            if data_range < MIN_RANGE_THRESHOLD:
                with log_lock:
                    logger.error(f"❌ {filename}: Range dati originali troppo piccolo ({data_range:.2e})")
                return {'status': 'error', 'file': filename, 'reason': f'Range too small: {data_range:.2e}'}
            
            # Verifica valori unici ORIGINALI
            unique_orig = len(np.unique(valid_data_orig[:min(10000, valid_data_orig.size)]))
            if unique_orig < MIN_UNIQUE_VALUES_ORIGINAL:
                with log_lock:
                    logger.error(f"❌ {filename}: Dati originali binari ({unique_orig} valori unici)")
                return {'status': 'error', 'file': filename, 'reason': f'Binary original data ({unique_orig} unique)'}
            
            with log_lock:
                logger.info(f"📊 {filename}: range=[{data_min:.2e}, {data_max:.2e}], unique={unique_orig}")
            
            # === NORMALIZZAZIONE ROBUSTA ===
            # Usa percentili per clipping outliers
            p_low = np.percentile(valid_data_orig, PERCENTILE_LOW)
            p_high = np.percentile(valid_data_orig, PERCENTILE_HIGH)
            
            with log_lock:
                logger.info(f"   Clipping: p{PERCENTILE_LOW}={p_low:.2e}, p{PERCENTILE_HIGH}={p_high:.2e}")
            
            # Clipping
            data_clipped = np.clip(data_original, p_low, p_high)
            
            # Normalizzazione [0, 1]
            if p_high > p_low:
                data_normalized = (data_clipped - p_low) / (p_high - p_low)
            else:
                with log_lock:
                    logger.warning(f"⚠️  {filename}: p_high == p_low, skip normalizzazione")
                data_normalized = data_clipped.copy()
            
            # Maschera invalidi
            data_normalized[~valid_mask_orig] = np.nan
            
            # === CALCOLO DIMENSIONI OUTPUT ===
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
            
            # === WCS TARGET ===
            target_wcs = WCS(naxis=2)
            target_wcs.wcs.ctype = common_wcs.wcs.ctype
            target_wcs.wcs.crval = wcs_orig.wcs.crval
            target_wcs.wcs.crpix = [nx_out / 2.0, ny_out / 2.0]
            target_wcs.wcs.cdelt = [-native_pixel_scale_deg, native_pixel_scale_deg]
            target_wcs.wcs.radesys = common_wcs.wcs.radesys
            target_wcs.wcs.equinox = common_wcs.wcs.equinox
            
            # === REPROIEZIONE ===
            with log_lock:
                logger.info(f"🔄 {filename}: Reproiezione {original_shape} → {shape_out}")
            
            reprojected_norm, footprint = reproject_interp(
                (data_normalized, wcs_orig),
                target_wcs,
                shape_out=shape_out,
                order=REPROJECT_ORDER
            )
            
            # === VALIDAZIONE OUTPUT ===
            footprint_safe = np.clip(footprint, 0, 1)
            valid_reproj = (footprint_safe > 0.01) & np.isfinite(reprojected_norm)
            
            if not np.any(valid_reproj):
                with log_lock:
                    logger.error(f"❌ {filename}: Nessun pixel valido dopo reproiezione")
                return {'status': 'error', 'file': filename, 'reason': 'No valid pixels after reproject'}
            
            # === DENORMALIZZAZIONE ===
            reprojected_weighted = np.where(valid_reproj, reprojected_norm * footprint_safe, np.nan)
            
            # CRUCIALE: Denormalizza usando gli stessi parametri
            reprojected_denorm = reprojected_weighted.copy()
            valid_mask_final = np.isfinite(reprojected_weighted)
            
            if p_high > p_low:
                reprojected_denorm[valid_mask_final] = (
                    reprojected_weighted[valid_mask_final] * (p_high - p_low) + p_low
                )
            else:
                reprojected_denorm = reprojected_weighted
            
            # === VERIFICA FINALE BINARIZZAZIONE ===
            final_valid = reprojected_denorm[valid_mask_final]
            unique_final = len(np.unique(final_valid[:min(10000, final_valid.size)]))
            final_range = final_valid.max() - final_valid.min()
            
            with log_lock:
                logger.info(f"   Output: unique={unique_final}, range=[{final_valid.min():.2e}, {final_valid.max():.2e}]")
            
            if unique_final < MIN_UNIQUE_VALUES_NORMALIZED:
                with log_lock:
                    logger.error(f"❌ {filename}: Output binarizzato ({unique_final} valori unici)")
                return {'status': 'error', 'file': filename, 'reason': f'Binarized output ({unique_final} unique)'}
                
            # === SALVATAGGIO ===
            coverage = (valid_mask_final.sum() / reprojected_denorm.size * 100)
            
            new_header = target_wcs.to_header()
            
            for key in ['DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'TELESCOP']:
                if key in header:
                    new_header[key] = header[key]
            
            new_header['ORIGINAL'] = filename
            new_header['REGDATE'] = datetime.now().isoformat()
            new_header['REGCOV'] = (coverage, "Coverage %")
            new_header['NATIVESC'] = (native_pixel_scale, "Native scale (arcsec/px)")
            new_header['NORMLOW'] = (p_low, "Normalization low clip")
            new_header['NORMHIGH'] = (p_high, "Normalization high clip")
            new_header['DATAOK'] = (True, "Data validation passed")
            
            output_filename = f"reg_{os.path.splitext(filename)[0]}.fits"
            output_path = output_dir / output_filename
            
            fits.PrimaryHDU(
                data=reprojected_denorm.astype(np.float32),
                header=new_header
            ).writeto(output_path, overwrite=True)
            
            with log_lock:
                logger.info(f"✅ {filename}: cov={coverage:.1f}%, shape={shape_out}")
            
            return {
                'status': 'success',
                'file': filename,
                'coverage': coverage,
                'output_path': output_path,
                'native_scale': native_pixel_scale,
                'output_shape': shape_out
            }
            
    except Exception as e:
        with log_lock:
            logger.error(f"❌ {os.path.basename(filepath)}: {e}", exc_info=True)
        return {'status': 'error', 'file': os.path.basename(filepath), 'reason': str(e)}

def register_images(wcs_info_list, common_wcs, output_dir, source_name, logger):
    """
    Registra tutte le immagini con multithreading.
    FIX: Calcola WCS target comune per tutte le immagini prima della reproiezione.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🔄 Registrazione {source_name}: {len(wcs_info_list)} immagini (risoluzione nativa)")
    
    # === FIX: CALCOLA BOUNDS COMUNI PER TUTTE LE IMMAGINI ===
    with log_lock:
        logger.info(f"\n{'='*60}")
        logger.info(f"CALCOLO BOUNDS COMUNI PER {source_name}")
        logger.info(f"{'='*60}")
    
    ra_min_global = float('inf')
    ra_max_global = float('-inf')
    dec_min_global = float('inf')
    dec_max_global = float('-inf')
    
    # Trova estensione totale di tutte le immagini
    for info in wcs_info_list:
        wcs_orig = info['wcs']
        shape = info['shape']
        ny, nx = shape
        
        # Coordinate dei 4 angoli
        corners_pix = np.array([
            [0, 0], [nx, 0], [0, ny], [nx, ny]
        ])
        corners_world = wcs_orig.pixel_to_world(corners_pix[:, 0], corners_pix[:, 1])
        
        for coord in corners_world:
            ra = coord.ra.deg
            dec = coord.dec.deg
            ra_min_global = min(ra_min_global, ra)
            ra_max_global = max(ra_max_global, ra)
            dec_min_global = min(dec_min_global, dec)
            dec_max_global = max(dec_max_global, dec)
    
    # Centro comune
    ra_center_global = (ra_min_global + ra_max_global) / 2.0
    dec_center_global = (dec_min_global + dec_max_global) / 2.0
    
    with log_lock:
        logger.info(f"Bounds globali {source_name}:")
        logger.info(f"  RA: [{ra_min_global:.6f}, {ra_max_global:.6f}]")
        logger.info(f"  DEC: [{dec_min_global:.6f}, {dec_max_global:.6f}]")
        logger.info(f"  Centro comune: RA={ra_center_global:.6f}°, DEC={dec_center_global:.6f}°")
    
    # === CREA WCS TARGET COMUNE (non per-immagine!) ===
    # Usa risoluzione media del gruppo
    scales = [info['pixel_scale'] for info in wcs_info_list]
    avg_scale_arcsec = np.mean(scales)
    avg_scale_deg = avg_scale_arcsec / 3600.0
    
    # WCS comune per TUTTE le immagini del gruppo
    shared_target_wcs = WCS(naxis=2)
    shared_target_wcs.wcs.ctype = common_wcs.wcs.ctype
    shared_target_wcs.wcs.crval = [ra_center_global, dec_center_global]  # ← Centro comune!
    shared_target_wcs.wcs.cdelt = [-avg_scale_deg, avg_scale_deg]
    shared_target_wcs.wcs.radesys = common_wcs.wcs.radesys
    shared_target_wcs.wcs.equinox = common_wcs.wcs.equinox
    
    with log_lock:
        logger.info(f"WCS target comune creato:")
        logger.info(f"  Centro: RA={ra_center_global:.6f}°, DEC={dec_center_global:.6f}°")
        logger.info(f"  Scala: {avg_scale_arcsec:.4f}\"/px")
    
    # === REPROIEZIONE CON WCS CONDIVISO ===
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Passa shared_target_wcs invece di common_wcs
        futures = {
            executor.submit(
                reproject_image_native_aligned,  # ← Nuova funzione
                info, 
                shared_target_wcs,  # ← WCS condiviso!
                ra_center_global,
                dec_center_global,
                output_dir, 
                logger
            ): info
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


def reproject_image_native_aligned(wcs_info, shared_target_wcs, ra_center_global, dec_center_global, output_dir, logger):
    """
    Reproietta immagine ALLINEATA al frame WCS condiviso del gruppo.
    """
    try:
        filepath = wcs_info['file']
        hdu_index = wcs_info['hdu_index']
        native_pixel_scale = wcs_info['pixel_scale']
        filename = os.path.basename(filepath)
        
        with fits.open(filepath) as hdul:
            hdu = hdul[hdu_index]
            data_original = hdu.data.copy()
            header = hdu.header.copy()
            wcs_orig = WCS(header)
            
            # Gestione 3D
            if len(data_original.shape) == 3:
                data_original = data_original[0]
            
            original_shape = data_original.shape
            
            # === VALIDAZIONE (come prima) ===
            valid_mask_orig = np.isfinite(data_original) & (data_original != 0)
            valid_data_orig = data_original[valid_mask_orig]
            
            if valid_data_orig.size < 100:
                with log_lock:
                    logger.error(f"❌ {filename}: Dati insufficienti ({valid_data_orig.size} px)")
                return {'status': 'error', 'file': filename, 'reason': 'Insufficient data'}
            
            data_min = valid_data_orig.min()
            data_max = valid_data_orig.max()
            data_range = data_max - data_min
            
            if data_range < MIN_RANGE_THRESHOLD:
                with log_lock:
                    logger.error(f"❌ {filename}: Range troppo piccolo ({data_range:.2e})")
                return {'status': 'error', 'file': filename, 'reason': f'Range too small'}
            
            unique_orig = len(np.unique(valid_data_orig[:min(10000, valid_data_orig.size)]))
            if unique_orig < MIN_UNIQUE_VALUES_ORIGINAL:
                with log_lock:
                    logger.error(f"❌ {filename}: Dati binari ({unique_orig} valori)")
                return {'status': 'error', 'file': filename, 'reason': f'Binary data'}
            
            with log_lock:
                logger.info(f"📊 {filename}: range=[{data_min:.2e}, {data_max:.2e}], unique={unique_orig}")
            
            # === NORMALIZZAZIONE ===
            p_low = np.percentile(valid_data_orig, PERCENTILE_LOW)
            p_high = np.percentile(valid_data_orig, PERCENTILE_HIGH)
            
            with log_lock:
                logger.info(f"   Clipping: p{PERCENTILE_LOW}={p_low:.2e}, p{PERCENTILE_HIGH}={p_high:.2e}")
            
            data_clipped = np.clip(data_original, p_low, p_high)
            
            if p_high > p_low:
                data_normalized = (data_clipped - p_low) / (p_high - p_low)
            else:
                data_normalized = data_clipped.copy()
            
            data_normalized[~valid_mask_orig] = np.nan
            
            # === CALCOLO DIMENSIONI OUTPUT (basato su estensione originale) ===
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
            
            # === CRUCIALE: USA SHARED_TARGET_WCS (non creare nuovo per ogni immagine) ===
            # Calcola CRPIX per centrare questa immagine nel frame condiviso
            
            # Centro immagine originale nel frame globale
            center_orig = wcs_orig.wcs.crval  # [RA, DEC] centro originale
            
            # Calcola offset dal centro globale (in gradi)
            delta_ra = center_orig[0] - ra_center_global
            delta_dec = center_orig[1] - dec_center_global
            
            # Converti offset in pixel nel sistema target
            delta_ra_px = delta_ra / native_pixel_scale_deg
            delta_dec_px = delta_dec / native_pixel_scale_deg
            
            # CRPIX del frame output (centro immagine nel suo frame locale)
            crpix_x = nx_out / 2.0 - delta_ra_px
            crpix_y = ny_out / 2.0 + delta_dec_px  # + perché DEC cresce verso l'alto
            
            # Crea WCS target PER QUESTA IMMAGINE (ma allineato al frame globale)
            target_wcs = WCS(naxis=2)
            target_wcs.wcs.ctype = shared_target_wcs.wcs.ctype
            target_wcs.wcs.crval = [ra_center_global, dec_center_global]  # ← Centro GLOBALE
            target_wcs.wcs.crpix = [crpix_x, crpix_y]  # ← CRPIX corretto per allineamento
            target_wcs.wcs.cdelt = [-native_pixel_scale_deg, native_pixel_scale_deg]
            target_wcs.wcs.radesys = shared_target_wcs.wcs.radesys
            target_wcs.wcs.equinox = shared_target_wcs.wcs.equinox
            
            with log_lock:
                logger.info(f"🔄 {filename}: Reproiezione {original_shape} → {shape_out}")
                logger.info(f"   Centro orig: RA={center_orig[0]:.6f}, DEC={center_orig[1]:.6f}")
                logger.info(f"   Offset da centro globale: ΔRA={delta_ra:.6f}°, ΔDEC={delta_dec:.6f}")
                logger.info(f"   CRPIX target: [{crpix_x:.2f}, {crpix_y:.2f}]")
            
            # === REPROIEZIONE ===
            reprojected_norm, footprint = reproject_interp(
                (data_normalized, wcs_orig),
                target_wcs,
                shape_out=shape_out,
                order=REPROJECT_ORDER
            )
            
            # === VALIDAZIONE E DENORMALIZZAZIONE (come prima) ===
            footprint_safe = np.clip(footprint, 0, 1)
            valid_reproj = (footprint_safe > 0.01) & np.isfinite(reprojected_norm)
            
            if not np.any(valid_reproj):
                with log_lock:
                    logger.error(f"❌ {filename}: Nessun pixel valido")
                return {'status': 'error', 'file': filename, 'reason': 'No valid pixels'}
            
            reprojected_weighted = np.where(valid_reproj, reprojected_norm * footprint_safe, np.nan)
            reprojected_denorm = reprojected_weighted.copy()
            valid_mask_final = np.isfinite(reprojected_weighted)
            
            if p_high > p_low:
                reprojected_denorm[valid_mask_final] = (
                    reprojected_weighted[valid_mask_final] * (p_high - p_low) + p_low
                )
            
            # Verifica finale
            final_valid = reprojected_denorm[valid_mask_final]
            unique_final = len(np.unique(final_valid[:min(10000, final_valid.size)]))
            
            with log_lock:
                logger.info(f"   Output: unique={unique_final}, range=[{final_valid.min():.2e}, {final_valid.max():.2e}]")
            
            if unique_final < MIN_UNIQUE_VALUES_NORMALIZED:
                with log_lock:
                    logger.error(f"❌ {filename}: Output binarizzato")
                return {'status': 'error', 'file': filename, 'reason': 'Binarized output'}
            
            # === SALVATAGGIO ===
            coverage = (valid_mask_final.sum() / reprojected_denorm.size * 100)
            
            new_header = target_wcs.to_header()
            
            for key in ['DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'TELESCOP']:
                if key in header:
                    new_header[key] = header[key]
            
            new_header['ORIGINAL'] = filename
            new_header['REGDATE'] = datetime.now().isoformat()
            new_header['REGCOV'] = (coverage, "Coverage %")
            new_header['NATIVESC'] = (native_pixel_scale, "Native scale (arcsec/px)")
            new_header['NORMLOW'] = (p_low, "Normalization low clip")
            new_header['NORMHIGH'] = (p_high, "Normalization high clip")
            new_header['ALIGNED'] = (True, "Aligned to global frame")
            new_header['GLOBCENR'] = (ra_center_global, "Global frame center RA")
            new_header['GLOBCEND'] = (dec_center_global, "Global frame center DEC")
            
            output_filename = f"reg_{os.path.splitext(filename)[0]}.fits"
            output_path = output_dir / output_filename
            
            fits.PrimaryHDU(
                data=reprojected_denorm.astype(np.float32),
                header=new_header
            ).writeto(output_path, overwrite=True)
            
            with log_lock:
                logger.info(f"✅ {filename}: cov={coverage:.1f}%, shape={shape_out}")
            
            return {
                'status': 'success',
                'file': filename,
                'coverage': coverage,
                'output_path': output_path,
                'native_scale': native_pixel_scale,
                'output_shape': shape_out
            }
            
    except Exception as e:
        with log_lock:
            logger.error(f"❌ {os.path.basename(filepath)}: {e}", exc_info=True)
        return {'status': 'error', 'file': os.path.basename(filepath), 'reason': str(e)}

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
    
    logger.info(f"Totale: {total_prep} preparati, {total_fail} falliti")
    
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
    
    logger.info("=" * 60)
    logger.info("REGISTRAZIONE CON RISOLUZIONE NATIVA")
    logger.info("=" * 60)
    
    print("\n" + "🔭"*35)
    print(f"STEP 3: REGISTRAZIONE (RISOLUZIONE NATIVA)".center(70))
    print("🔭"*35)
    
    print(f"\n📂 CONFIGURAZIONE:")
    print(f"   Input Hubble: {INPUT_HUBBLE}")
    print(f"   Input Observatory: {INPUT_OBSERVATORY}")
    print(f"   Output: {BASE_DIR / '3_registered_native'}")
    
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
    
    # Riepilogo
    print(f"\n{'='*70}")
    print("📊 RIEPILOGO REGISTRAZIONE")
    print(f"{'='*70}")
    print(f"\n   Totale registrate: {total_success}")
    print(f"   Totale errori: {total_error}")
    
    if total_success > 0:
        print(f"\n✅ REGISTRAZIONE COMPLETATA!")
        print(f"\n   ✨ VANTAGGIO: Ogni immagine ha mantenuto la sua risoluzione nativa!")
    
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
        
        # INPUT STEP 2 (stesso degli output Step 1)
        INPUT_HUBBLE = OUTPUT_LITH_WCS
        INPUT_OBSERVATORY = OUTPUT_OSSERVATORIO_WCS
        
        # OUTPUT STEP 2
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

    # RIEPILOGO FINALE BATCH
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

    # TRANSIZIONE AI PROSSIMI SCRIPT (da v1)
    if ask_continue_to_cropping():
        try:
            # Usa la cartella dinamica degli script
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
    print(f"\n⏱️ Tempo totale esecuzione script: {elapsed:.2f}s")