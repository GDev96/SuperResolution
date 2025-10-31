"""
AstroRegister.py - Registrazione (Allineamento) con Siril CLI
MULTITHREADING VERSION - Elabora 2 immagini alla volta
"""

import os
import sys
import glob
import time
import logging
import subprocess
import shutil
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

# --- CONFIGURAZIONE TEST ---
MAX_IMAGES = 15  # Riduci per test veloce
NUM_THREADS = 1  # Numero di thread paralleli non superare gli 8
# --- CONFIGURAZIONE PERCORSI ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ✅ PATH CORRETTI per la tua struttura reale
INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_cropped_3')     # Se esiste, altrimenti fallback
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_register_4')   # Output registrazione
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Fallback per input (se img_cropped_3 non esiste)
FALLBACK_INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_lights_1')

# --- CONFIGURAZIONE SIRIL ---
SIRIL_CLI = "C:\\Program Files\\SiriL\\bin\\siril-cli.exe"

# Lock per il logging thread-safe
log_lock = threading.Lock()

def setup_logging():
    """Configura il sistema di logging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'registration_fixed_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Log system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Numpy version: {np.__version__}")
    logger.info(f"Astropy version: {astropy.__version__}")
    logger.info(f"Threads: {NUM_THREADS}")
    
    return logger

def extract_wcs_info_safe(hdu, logger):
    """Estrae informazioni WCS in modo sicuro evitando errori di unità."""
    try:
        wcs = WCS(hdu.header)
        
        if not wcs.has_celestial:
            return None, "No celestial coordinates"
        
        # Estrai informazioni base dall'header direttamente
        header = hdu.header
        
        # Metodo sicuro per ottenere centro
        try:
            # Usa pixel al centro dell'immagine
            center_x = hdu.data.shape[1] / 2.0
            center_y = hdu.data.shape[0] / 2.0
            
            # Converti usando wcs_pix2world (più robusto)
            world_coords = wcs.wcs_pix2world([[center_x, center_y]], 1)
            center_ra = float(world_coords[0][0])
            center_dec = float(world_coords[0][1])
            
        except Exception as e:
            logger.debug(f"Fallback to header values: {e}")
            # Fallback ai valori dell'header
            center_ra = float(header.get('CRVAL1', 0))
            center_dec = float(header.get('CRVAL2', 0))
        
        # Calcola scala pixel in modo sicuro
        try:
            # Usa CD matrix se disponibile
            if 'CD1_1' in header and 'CD2_2' in header:
                cd1_1 = float(header['CD1_1'])
                cd2_2 = float(header['CD2_2'])
                pixel_scale_deg = np.sqrt(cd1_1**2 + cd2_2**2)
            elif 'CDELT1' in header and 'CDELT2' in header:
                cdelt1 = float(header['CDELT1'])
                cdelt2 = float(header['CDELT2'])
                pixel_scale_deg = np.sqrt(cdelt1**2 + cdelt2**2)
            else:
                # Stima predefinita
                pixel_scale_deg = 0.04 / 3600.0  # 0.04 arcsec/pixel
                
            pixel_scale_arcsec = pixel_scale_deg * 3600.0
            
        except Exception as e:
            logger.debug(f"Pixel scale fallback: {e}")
            pixel_scale_arcsec = 0.04
            pixel_scale_deg = pixel_scale_arcsec / 3600.0
        
        info = {
            'center_ra': center_ra,
            'center_dec': center_dec,
            'pixel_scale': pixel_scale_arcsec,
            'pixel_scale_deg': pixel_scale_deg,
            'wcs': wcs,
            'shape': hdu.data.shape
        }
        
        return info, None
        
    except Exception as e:
        return None, str(e)

def diagnose_wcs_files_fixed(input_files, logger):
    """Diagnostica WCS migliorata che evita errori di unità."""
    logger.info("=" * 60)
    logger.info("DIAGNOSTICA WCS FIXED")
    logger.info("=" * 60)
    
    valid_files = []
    wcs_info = []
    
    print("\n🔍 Diagnostica WCS delle immagini (versione corretta)...")
    
    for i, filepath in enumerate(input_files):
        try:
            with fits.open(filepath) as hdul:
                # Trova HDU con dati
                data_hdu = None
                for hdu_idx, hdu in enumerate(hdul):
                    if hdu.data is not None:
                        data_hdu = (hdu_idx, hdu)
                        break
                
                if data_hdu is None:
                    logger.warning(f"No data in {os.path.basename(filepath)}")
                    print(f"\n📄 File {i+1}: {os.path.basename(filepath)}")
                    print(f"   ❌ Nessun dato trovato")
                    continue
                
                hdu_idx, hdu = data_hdu
                
                print(f"\n📄 File {i+1}: {os.path.basename(filepath)}")
                print(f"   Shape: {hdu.data.shape}")
                print(f"   HDU index: {hdu_idx}")
                
                # Estrai info WCS in modo sicuro
                wcs_data, error = extract_wcs_info_safe(hdu, logger)
                
                if wcs_data is None:
                    print(f"   ❌ WCS Error: {error}")
                    logger.warning(f"WCS error in {os.path.basename(filepath)}: {error}")
                    continue
                
                print(f"   ✓ WCS: Coordinate celestiali presenti")
                print(f"   📍 Centro: RA={wcs_data['center_ra']:.3f}°, DEC={wcs_data['center_dec']:.3f}°")
                print(f"   📏 Scala: {wcs_data['pixel_scale']:.3f} arcsec/pixel")
                
                # Aggiungi file e path
                wcs_data['file'] = filepath
                wcs_data['hdu_index'] = hdu_idx
                
                wcs_info.append(wcs_data)
                valid_files.append(filepath)
                
                logger.info(f"✓ {os.path.basename(filepath)}: "
                           f"RA={wcs_data['center_ra']:.3f}, DEC={wcs_data['center_dec']:.3f}, "
                           f"scale={wcs_data['pixel_scale']:.3f}")
                
        except Exception as e:
            print(f"\n📄 File {i+1}: {os.path.basename(filepath)}")
            print(f"   ❌ Errore file: {e}")
            logger.error(f"File error {os.path.basename(filepath)}: {e}")
            continue
    
    # Riepilogo
    print(f"\n📊 RIEPILOGO DIAGNOSTICA:")
    print(f"   File totali: {len(input_files)}")
    print(f"   File con WCS valido: {len(valid_files)}")
    print(f"   File scartati: {len(input_files) - len(valid_files)}")
    
    if len(wcs_info) > 0:
        # Analizza distribuzione coordinate
        ra_values = [info['center_ra'] for info in wcs_info]
        dec_values = [info['center_dec'] for info in wcs_info]
        
        print(f"\n🎯 DISTRIBUZIONE COORDINATE:")
        print(f"   RA:  {min(ra_values):.3f}° - {max(ra_values):.3f}° (span: {max(ra_values)-min(ra_values):.3f}°)")
        print(f"   DEC: {min(dec_values):.3f}° - {max(dec_values):.3f}° (span: {max(dec_values)-min(dec_values):.3f}°)")
        
        # Scala pixel
        scales = [info['pixel_scale'] for info in wcs_info]
        print(f"   Scala pixel: {min(scales):.3f} - {max(scales):.3f} arcsec/pixel")
    
    logger.info(f"Valid files found: {len(valid_files)}/{len(input_files)}")
    
    return valid_files, wcs_info

def create_optimal_wcs_fixed(wcs_info, logger):
    """
    Crea WCS ottimale per il mosaico con pixel scale aumentato.
    MODIFICATO: pixel_scale_target moltiplicato per 5 per ridurre canvas.
    """
    logger.info("=" * 60)
    logger.info("CREAZIONE WCS OTTIMALE")
    logger.info("=" * 60)
    
    if not wcs_info:
        logger.error("Nessuna informazione WCS disponibile")
        return None, None
    
    # Estrai coordinate e scale
    ra_values = [info['center_ra'] for info in wcs_info]
    dec_values = [info['center_dec'] for info in wcs_info]
    scales = [info['pixel_scale_deg'] for info in wcs_info]
    
    # Calcola bounds del mosaico
    ra_min = min(ra_values)
    ra_max = max(ra_values)
    dec_min = min(dec_values)
    dec_max = max(dec_values)
    
    ra_center = (ra_min + ra_max) / 2.0
    dec_center = (dec_min + dec_max) / 2.0
    
    ra_span = ra_max - ra_min
    dec_span = dec_max - dec_min
    
    logger.info(f"Bounds mosaico:")
    logger.info(f"  RA:  {ra_min:.4f}° - {ra_max:.4f}° (span: {ra_span:.4f}°)")
    logger.info(f"  DEC: {dec_min:.4f}° - {dec_max:.4f}° (span: {dec_span:.4f}°)")
    logger.info(f"  Centro: RA={ra_center:.4f}°, DEC={dec_center:.4f}°")
    
    # ============================================================
    # MODIFICA CRITICA: Aumenta pixel scale per ridurre canvas
    # ============================================================
    pixel_scale_median = np.median(scales)
    
    # MOLTIPLICATORE: aumenta questo valore per canvas più piccolo
    SCALE_MULTIPLIER = 1  # 5x = canvas 5 volte più piccolo
                          # Prova 3 se vuoi più dettaglio
                          # Prova 10 per test velocissimi
    
    pixel_scale_target = pixel_scale_median * SCALE_MULTIPLIER
    pixel_scale_arcsec = pixel_scale_target * 3600.0
    
    logger.info(f"Pixel scale:")
    logger.info(f"  Originale (mediano): {pixel_scale_median * 3600:.4f} arcsec/px")
    logger.info(f"  Target (x{SCALE_MULTIPLIER}): {pixel_scale_arcsec:.4f} arcsec/px")
    
    print(f"\n📐 CONFIGURAZIONE WCS:")
    print(f"   Pixel scale originale: {pixel_scale_median * 3600:.4f}\"/px")
    print(f"   Pixel scale target: {pixel_scale_arcsec:.4f}\"/px (x{SCALE_MULTIPLIER})")
    print(f"   Centro: RA={ra_center:.4f}°, DEC={dec_center:.4f}°")
    
    # Calcola dimensioni canvas
    # Aggiungi margine del 10% per sicurezza
    margin_factor = 1
    ra_size = ra_span * margin_factor
    dec_size = dec_span * margin_factor
    
    # Dimensioni in pixel
    nx = int(np.ceil(ra_size / pixel_scale_target))
    ny = int(np.ceil(dec_size / pixel_scale_target))
    
    # Limiti di sicurezza
    MAX_DIMENSION = 8000  # Massimo assoluto
    if nx > MAX_DIMENSION or ny > MAX_DIMENSION:
        logger.warning(f"Canvas troppo grande ({nx}x{ny})! Ridimensiono...")
        scale_down = max(nx, ny) / MAX_DIMENSION
        pixel_scale_target *= scale_down
        pixel_scale_arcsec = pixel_scale_target * 3600.0
        nx = int(np.ceil(ra_size / pixel_scale_target))
        ny = int(np.ceil(dec_size / pixel_scale_target))
        logger.info(f"Nuovo pixel scale: {pixel_scale_arcsec:.4f} arcsec/px")
        logger.info(f"Nuove dimensioni: {nx}x{ny}")
    
    # Calcola memoria richiesta
    size_mb = (nx * ny * 4) / (1024**2)  # float32 = 4 bytes
    
    logger.info(f"Dimensioni canvas: {nx}x{ny} pixel")
    logger.info(f"Memoria per immagine: {size_mb:.1f} MB")
    
    print(f"   Canvas: {nx}x{ny} pixel")
    print(f"   Memoria: {size_mb:.1f} MB/immagine")
    
    # Warning se ancora troppo grande
    if size_mb > 500:
        print(f"\n⚠️  Canvas ancora grande! Considera SCALE_MULTIPLIER = 10")
        logger.warning(f"Canvas grande: {size_mb:.1f} MB")
    
    # Crea WCS
    try:
        from astropy.wcs import WCS
        
        wcs = WCS(naxis=2)
        
        # Reference pixel (centro del canvas)
        wcs.wcs.crpix = [nx / 2.0, ny / 2.0]
        
        # Reference coordinate (centro del mosaico)
        wcs.wcs.crval = [ra_center, dec_center]
        
        # Pixel scale (negativo per RA per convenzione)
        wcs.wcs.cdelt = [-pixel_scale_target, pixel_scale_target]
        
        # Projection type
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        # Additional metadata
        wcs.wcs.equinox = 2000.0
        wcs.wcs.radesys = 'ICRS'
        
        logger.info("✓ WCS creato con successo")
        logger.info(f"Shape output: {ny}x{nx}")
        
        return wcs, (ny, nx)
        
    except Exception as e:
        logger.error(f"Errore creazione WCS: {e}", exc_info=True)
        print(f"\n❌ Errore creazione WCS: {e}")
        return None, None








def process_single_image(filepath, target_wcs, shape_out, wcs_info, logger, min_coverage, force_all):
    """
    Processa una singola immagine (funzione per multithreading).
    
    Returns:
        dict: Risultato dell'elaborazione con stato e statistiche
    """
    try:
        from reproject import reproject_interp
        
        filename = os.path.basename(filepath)
        
        with fits.open(filepath) as hdul:
            # Trova HDU con dati (usa info dalla diagnostica)
            hdu_info = next((info for info in wcs_info if info['file'] == filepath), None)
            if hdu_info is None:
                return {'status': 'skip', 'file': filename, 'reason': 'No WCS info'}
            
            hdu_idx = hdu_info['hdu_index']
            data_hdu = hdul[hdu_idx]
            
            # Reproietta con reproject_interp
            with log_lock:
                logger.debug(f"Reproiezione {filename}...")
            
            reprojected_data, footprint = reproject_interp(
                data_hdu,
                target_wcs,
                shape_out=shape_out
            )
            
            # Verifica risultato
            if reprojected_data is None:
                with log_lock:
                    logger.warning(f"Reproiezione fallita per {filename}")
                return {'status': 'error', 'file': filename, 'reason': 'Reprojection failed'}
            
            # Analizza copertura
            valid_mask = ~np.isnan(reprojected_data)
            valid_pixels = np.sum(valid_mask)
            total_pixels = reprojected_data.size
            coverage = (valid_pixels / total_pixels) * 100
            
            # Controllo copertura
            skip_image = False
            if not force_all and coverage < min_coverage:
                with log_lock:
                    logger.warning(f"Skip {filename}: copertura troppo bassa ({coverage:.1f}%)")
                return {'status': 'skip', 'file': filename, 'reason': f'Low coverage ({coverage:.1f}%)'}
            
            # =================================================================
            # === MODIFICA PER BORDI ===
            # Non convertire i NaN in zero. Lasciali come NaN.
            # Lo script del mosaico (step 4) li gestirà correttamente.
            
            # Sostituisci NaN con zero (VECCHIO CODICE)
            # reprojected_data = np.nan_to_num(reprojected_data, nan=0.0)
            
            # (NUOVO CODICE)
            # Nessuna operazione. reprojected_data mantiene i NaN.
            # =================================================================
            
            # Crea header
            new_header = target_wcs.to_header()
            
            # Copia metadati importanti
            important_keys = ['OBJECT', 'DATE-OBS', 'EXPTIME', 'FILTER', 
                            'INSTRUME', 'TELESCOP', 'OBSERVER']
            
            for key in important_keys:
                if key in data_hdu.header:
                    new_header[key] = data_hdu.header[key]
            
            # Aggiungi metadati registrazione
            new_header['REGMTHD'] = 'reproject_interp'
            new_header['REGSRC'] = filename
            new_header['REGDATE'] = datetime.now().isoformat()
            new_header['REGCOVER'] = coverage
            new_header['REGPIXEL'] = valid_pixels
            new_header['REGFORCE'] = force_all
            
            # Salva con prefisso diverso se copertura bassa
            is_low_coverage = coverage < 1.0
            if is_low_coverage:
                prefix = "lowcov_"
            else:
                prefix = "register_"
            
            output_filename = f"{prefix}{os.path.splitext(filename)[0]}_{datetime.now().strftime('%H%M%S')}.fits"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # Salva il file (che ora CONTIENE NaN)
            fits.PrimaryHDU(
                data=reprojected_data.astype(np.float32),
                header=new_header
            ).writeto(output_path, overwrite=True)
            
            # Log thread-safe
            with log_lock:
                if is_low_coverage:
                    logger.info(f"⚠️ {filename}: copertura BASSA ({coverage:.1f}%), {valid_pixels:,} pixel validi - SALVATO")
                else:
                    logger.info(f"✓ {filename}: copertura {coverage:.1f}%, {valid_pixels:,} pixel validi")
            
            return {
                'status': 'success',
                'file': filename,
                'coverage': coverage,
                'valid_pixels': valid_pixels,
                'low_coverage': is_low_coverage
            }
            
    except Exception as e:
        error_msg = f"Errore {os.path.basename(filepath)}: {str(e)}"
        with log_lock:
            logger.error(error_msg, exc_info=True)
        return {'status': 'error', 'file': os.path.basename(filepath), 'reason': str(e)}

def manual_registration_fixed(logger, input_files, min_coverage=0.1, force_all=False):
    """
    Registrazione manuale con diagnostica WCS corretta e multithreading.
    
    Parameters:
    - min_coverage: Copertura minima richiesta (default 0.1%)
    - force_all: Se True, salva tutte le immagini anche con 0% copertura
    """
    logger.info("=" * 50)
    logger.info(f"REGISTRAZIONE MANUALE MULTITHREADING ({NUM_THREADS} threads)")
    logger.info("=" * 50)
    logger.info(f"Copertura minima: {min_coverage}%")
    logger.info(f"Forza tutte: {force_all}")
    
    try:
        from reproject import reproject_interp
        logger.info("Libreria reproject caricata (usando interp)")
    except ImportError as e:
        logger.error("Libreria reproject mancante", exc_info=True)
        return False, 0
    
    # Diagnostica WCS corretta
    valid_files, wcs_info = diagnose_wcs_files_fixed(input_files, logger)
    
    if not valid_files:
        logger.error("Nessun file con WCS valido trovato!")
        print("❌ Nessun file con WCS valido!")
        return False, 0
    
    # Crea WCS ottimale
    target_wcs, shape_out = create_optimal_wcs_fixed(wcs_info, logger)
    
    if target_wcs is None:
        logger.error("Impossibile creare WCS ottimale!")
        return False, 0
    
    # Registrazione con multithreading
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    success_count = 0
    low_coverage_count = 0
    skip_count = 0
    error_count = 0
    
    print(f"\n🔄 Registrazione di {len(valid_files)} immagini con {NUM_THREADS} thread...")
    if force_all:
        print("⚠️  MODALITÀ FORZATA: Salva tutte le immagini indipendentemente dalla copertura")
    else:
        print(f"📏 Copertura minima richiesta: {min_coverage}%")
    
    # Usa ThreadPoolExecutor per processare in parallelo
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Sottometti tutti i task
        future_to_file = {
            executor.submit(
                process_single_image,
                filepath,
                target_wcs,
                shape_out,
                wcs_info,
                logger,
                min_coverage,
                force_all
            ): filepath for filepath in valid_files
        }
        
        # Progress bar
        with tqdm(total=len(valid_files), desc="Registrazione") as pbar:
            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    result = future.result()
                    
                    if result['status'] == 'success':
                        success_count += 1
                        if result.get('low_coverage', False):
                            low_coverage_count += 1
                        pbar.set_description(f"✓ {success_count} registrate")
                    elif result['status'] == 'skip':
                        skip_count += 1
                        pbar.set_description(f"⊘ {skip_count} saltate")
                    else:  # error
                        error_count += 1
                        pbar.set_description(f"❌ {error_count} errori")
                    
                except Exception as exc:
                    error_count += 1
                    with log_lock:
                        logger.error(f"Exception processing {os.path.basename(filepath)}: {exc}")
                
                pbar.update(1)
    
    # Riepilogo
    print(f"\n📊 RISULTATI REGISTRAZIONE:")
    print(f"   File processati: {len(valid_files)}")
    print(f"   Registrati con successo: {success_count}")
    print(f"   Con copertura bassa: {low_coverage_count}")
    print(f"   Saltati: {skip_count}")
    print(f"   Errori: {error_count}")
    
    if low_coverage_count > 0:
        print(f"\n⚠️  {low_coverage_count} immagini salvate con prefisso 'lowcov_' per bassa copertura")
    
    logger.info(f"Registrazione completata: {success_count}/{len(valid_files)} (lowcov: {low_coverage_count}, skip: {skip_count}, errors: {error_count})")
    return success_count > 0, success_count

def run_registration():
    """Funzione principale con verifica percorsi."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info(f"ASTRO REGISTER - MULTITHREADING VERSION ({MAX_IMAGES} immagini, {NUM_THREADS} threads)")
    logger.info("=" * 60)
    
    print("=" * 70)
    print("🔭 ASTRO REGISTER - VERSIONE MULTITHREADING".center(70))
    print("=" * 70)
    
    # ✅ VERIFICA E MOSTRA PERCORSI REALI
    print(f"Input primario: {os.path.abspath(INPUT_DIR)}")
    print(f"Input fallback:  {os.path.abspath(FALLBACK_INPUT_DIR)}")
    print(f"Output:         {os.path.abspath(OUTPUT_DIR)}")
    print(f"Limite: {MAX_IMAGES} immagini")
    print(f"Thread: {NUM_THREADS} paralleli")
    
    # Determina directory input da usare
    current_input_dir = INPUT_DIR
    if not os.path.exists(INPUT_DIR):
        print(f"\n⚠️  Directory {INPUT_DIR} non trovata")
        if os.path.exists(FALLBACK_INPUT_DIR):
            current_input_dir = FALLBACK_INPUT_DIR
            print(f"✓ Uso fallback: {FALLBACK_INPUT_DIR}")
        else:
            print(f"❌ Neppure {FALLBACK_INPUT_DIR} esiste!")
            logger.error(f"Neither input directory exists")
            return
    
    # ✅ OPZIONI CONFIGURABILI PER COPERTURA
    print("\n🔧 OPZIONI REGISTRAZIONE:")
    print("1. Standard (copertura min 1%)")
    print("2. Permissiva (copertura min 0.1%)")
    print("3. Ultra-permissiva (copertura min 0.01%)")
    print("4. Forzata (salva tutto, anche 0%)")
    
    try:
        choice = input("\nScegli opzione (1-4, default=2): ").strip()
        if choice == "1":
            min_coverage = 1.0
            force_all = False
            print("📏 Modalità STANDARD: copertura minima 1%")
        elif choice == "3":
            min_coverage = 0.01
            force_all = False
            print("🔍 Modalità ULTRA-PERMISSIVA: copertura minima 0.01%")
        elif choice == "4":
            min_coverage = 0.0
            force_all = True
            print("⚠️  Modalità FORZATA: salva tutte le immagini")
        else:  # Default
            min_coverage = 0.1
            force_all = False
            print("📏 Modalità PERMISSIVA: copertura minima 0.1%")
    except:
        # Fallback se non può fare input (script automatico)
        min_coverage = 0.1
        force_all = False
        print("📏 Modalità AUTOMATICA: copertura minima 0.1%")
    
    # Setup directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Get input files con pattern HST
    patterns = ['*.fit*', 'hst_*.fits', 'plate_*.fits', 'prep_*.fits']
    input_files = []
    pattern_used = None
    
    for pattern in patterns:
        files = glob.glob(os.path.join(current_input_dir, pattern))
        if files:
            input_files = sorted(files)[:MAX_IMAGES]
            pattern_used = pattern
            print(f"✓ Trovati {len(files)} file con pattern: {pattern}")
            break
    
    if not input_files:
        logger.error(f"Nessun file trovato in {current_input_dir}")
        print(f"\n❌ Nessun file trovato in {current_input_dir}")
        print("💡 Pattern cercati:")
        for pattern in patterns:
            full_pattern = os.path.join(current_input_dir, pattern)
            print(f"   - {full_pattern}")
        return
    
    print(f"\n✓ Trovati {len(input_files)} file da processare")
    
    # Mostra alcuni file di esempio
    print(f"\n📄 Primi file trovati:")
    for i, f in enumerate(input_files[:3]):
        print(f"   {i+1}. {os.path.basename(f)}")
    if len(input_files) > 3:
        print(f"   ... e altri {len(input_files)-3} file")
    
    logger.info(f"Processing {len(input_files)} files with pattern: {pattern_used}")
    
    # Registrazione manuale corretta con multithreading
    print(f"\n🔄 Avvio registrazione con canvas ottimizzato...")
    success, count = manual_registration_fixed(logger, input_files, min_coverage, force_all)
    
    if success:
        print(f"\n✅ REGISTRAZIONE COMPLETATA!")
        print(f"📁 {count} immagini registrate salvate in: {os.path.abspath(OUTPUT_DIR)}")
        print(f"📄 Log dettagliato: {os.path.abspath(LOG_DIR)}")
        print(f"\n💡 SUGGERIMENTO:")
        print(f"   - File 'register_*.fits': Copertura normale (MANTIENE NaN per bordi)")
        print(f"   - File 'lowcov_*.fits': Copertura bassa ma salvati")
        print(f"\n🎯 PROSSIMO PASSO:")
        print(f"   Esegui AstroMosaic.py per creare il mosaico finale")
    else:
        print(f"\n❌ REGISTRAZIONE FALLITA")
        print(f"📄 Controlla i log per dettagli: {os.path.abspath(LOG_DIR)}")
    
    logger.info("Elaborazione completata")



if __name__ == "__main__":
    print("\n🔭 ASTRO REGISTER - VERSIONE MULTITHREADING")
    print(f"Limite test: {MAX_IMAGES} immagini")
    print(f"Thread paralleli: {NUM_THREADS}\n")
    
    start_time = time.time()
    run_registration()
    elapsed = time.time() - start_time
    
    print(f"\n⏱️ Tempo totale: {elapsed:.1f} secondi")