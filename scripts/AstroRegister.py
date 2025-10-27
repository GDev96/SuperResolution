"""
AstroRegister.py - Registrazione (Allineamento) con Siril CLI
Allinea tutte le immagini FITS su una proiezione comune usando le WCS.

METODO 1 (Preferito): Usa il comando 'register' di Siril
METODO 2 (Fallback): Usa reproject Python con astropy
"""

import os
import glob
import time
import logging
import subprocess
import shutil
from datetime import datetime
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from tqdm import tqdm

# --- CONFIGURAZIONE PERCORSI ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_plate_2')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_register_4')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# --- CONFIGURAZIONE SIRIL ---
SIRIL_CLI = "C:\\Program Files\\SiriL\\bin\\siril-cli.exe"

# --- PARAMETRI DI RIFERIMENTO ---
M42_RA_CENTER = 83.835
M42_DEC_CENTER = -5.391
PIXEL_SCALE_ARCSEC = 0.04
PIXEL_SCALE_DEG = PIXEL_SCALE_ARCSEC / 3600.0
FINAL_CANVAS_SHAPE = (4500, 4500)

# --- CONFIGURAZIONE LOGGING ---
def setup_logging():
    """Configura il sistema di logging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'registration_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def extract_image_number(filename):
    """Estrae il numero identificativo dal nome del file."""
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    # Per file plate_image_0a_timestamp.fits
    if name_without_ext.startswith('plate_image_'):
        parts = name_without_ext.split('_')
        if len(parts) >= 3:
            return parts[2]
    
    # Per file standard
    parts = name_without_ext.split('_')
    if len(parts) >= 3:
        return parts[2]
    
    return 'unknown'

def generate_output_filename(input_filename):
    """Genera il nome del file di output: register_image_number_timestamp.fits"""
    image_number = extract_image_number(input_filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    
    _, ext = os.path.splitext(input_filename)
    if not ext:
        ext = '.fits'
    
    return f"register_image_{image_number}_{timestamp}{ext}"

def check_siril_available():
    """Verifica se Siril CLI è disponibile."""
    try:
        result = subprocess.run(
            [SIRIL_CLI, '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False

def run_siril_registration(logger):
    """
    Esegue registrazione usando Siril CLI.
    
    Comandi Siril utilizzati:
    - convert: converte file FITS in sequenza
    - register: allinea le immagini usando stelle o WCS
    - framing: definisce come gestire le dimensioni finali
    """
    
    logger.info("Tentativo registrazione con Siril...")
    
    # Crea directory di lavoro temporanea per Siril
    work_dir = os.path.join(PROJECT_ROOT, 'data', 'temp_siril_register')
    os.makedirs(work_dir, exist_ok=True)
    
    # Script Siril per registrazione
    siril_script = f"""# Script Siril per Registrazione Immagini
cd {INPUT_DIR}

# Converti file FITS in sequenza Siril
# Cerca tutti i file che iniziano con 'plate_image_'
convert plate_image -out={work_dir}/reg_seq

# Carica la sequenza
cd {work_dir}
load reg_seq

# Registrazione delle immagini
# Opzioni:
#   -prefix=nome : prefisso per i file output
#   -framing=max : mantiene tutte le aree comuni (canvas massimo)
#   -framing=min : ritaglia al minimo comune
#   -framing=cog : centra sul centro di gravità
# 
# Metodi di registrazione:
#   register reg_seq : registrazione standard usando stelle
#   register reg_seq -2pass : registrazione a due passaggi (più precisa)
#   register reg_seq -norot : registra senza rotazione

# Usa registrazione globale per deep-sky con framing massimo
register reg_seq -framing=max -prefix=r_

# Salva i file registrati nella cartella finale
savefits {OUTPUT_DIR}/register_image

close
"""
    
    # Salva script temporaneo
    script_file = os.path.join(work_dir, 'register_script.ssf')
    with open(script_file, 'w') as f:
        f.write(siril_script)
    
    try:
        logger.info(f"Esecuzione comando Siril: {SIRIL_CLI} -s {script_file}")
        
        result = subprocess.run(
            [SIRIL_CLI, '-s', script_file],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=3600  # 60 minuti timeout
        )
        
        logger.info(f"Output Siril:\n{result.stdout}")
        
        if result.returncode == 0:
            logger.info("Registrazione Siril completata con successo")
            
            # Verifica che ci siano file output
            output_files = glob.glob(os.path.join(OUTPUT_DIR, 'register_image*.fits'))
            if output_files:
                logger.info(f"Trovati {len(output_files)} file registrati")
                return True, len(output_files)
            else:
                logger.warning("Nessun file output trovato dopo registrazione Siril")
                return False, 0
        else:
            logger.warning(f"Registrazione Siril fallita:\n{result.stderr}")
            return False, 0
            
    except subprocess.TimeoutExpired:
        logger.error("Timeout durante registrazione Siril (>60 min)")
        return False, 0
    except Exception as e:
        logger.error(f"Errore esecuzione Siril: {e}")
        return False, 0
    finally:
        # Pulizia file temporanei (opzionale)
        pass

def manual_registration_reproject(logger):
    """
    METODO FALLBACK: Registrazione manuale usando reproject di astropy.
    Riproietta tutte le immagini su un WCS target comune.
    """
    
    logger.info("Uso metodo manuale per registrazione (reproject)...")
    
    try:
        from reproject import reproject_exact
    except ImportError:
        logger.error("Libreria 'reproject' non installata. Installa con: pip install reproject")
        print("\nERRORE: Libreria 'reproject' mancante")
        print("Installa con: pip install reproject")
        return False, 0
    
    input_files = glob.glob(os.path.join(INPUT_DIR, '*.fit')) + \
                  glob.glob(os.path.join(INPUT_DIR, '*.fits'))
    
    if not input_files:
        logger.error(f"Nessun file trovato in {INPUT_DIR}")
        return False, 0
    
    logger.info(f"Trovati {len(input_files)} file da registrare")
    
    # Crea WCS target
    logger.info("Creazione WCS target...")
    target_wcs = WCS(naxis=2)
    target_wcs.wcs.crpix = [FINAL_CANVAS_SHAPE[1] / 2, FINAL_CANVAS_SHAPE[0] / 2]
    target_wcs.wcs.cdelt = [-PIXEL_SCALE_DEG, PIXEL_SCALE_DEG]
    target_wcs.wcs.crval = [M42_RA_CENTER, M42_DEC_CENTER]
    target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    logger.info(f"WCS Target - CRPIX: {target_wcs.wcs.crpix}")
    logger.info(f"WCS Target - CRVAL: {target_wcs.wcs.crval}")
    logger.info(f"Canvas finale: {FINAL_CANVAS_SHAPE}")
    
    registered_count = 0
    error_count = 0
    
    with tqdm(total=len(input_files), desc="Registrazione Manuale", unit=" file") as pbar:
        for filename in input_files:
            try:
                with fits.open(filename, ignore_missing_end=True) as hdul:
                    input_hdu = hdul[0]
                    
                    # Verifica WCS
                    try:
                        input_wcs = WCS(input_hdu.header)
                        if not input_wcs.has_celestial:
                            raise Exception("WCS non valido o mancante")
                    except Exception as wcs_error:
                        raise Exception(f"WCS non valido: {wcs_error}")
                    
                    # Esegui riproiezione
                    logger.debug(f"Riproiezione di {os.path.basename(filename)}...")
                    aligned_array, footprint = reproject_exact(
                        input_hdu,
                        target_wcs,
                        shape_out=FINAL_CANVAS_SHAPE,
                    )
                    
                    # Verifica pixel validi
                    valid_pixels = np.count_nonzero(~np.isnan(aligned_array))
                    logger.debug(f"Pixel validi: {valid_pixels}/{FINAL_CANVAS_SHAPE[0]*FINAL_CANVAS_SHAPE[1]}")
                    
                    if valid_pixels == 0:
                        raise Exception("Nessun pixel valido dopo riproiezione")
                    
                    # Crea HDU con WCS target
                    aligned_hdu = fits.PrimaryHDU(aligned_array, target_wcs.to_header())
                    
                    # Salva con nuovo nome
                    output_filename = generate_output_filename(filename)
                    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
                    aligned_hdu.writeto(output_filepath, overwrite=True, output_verify='warn')
                    
                    registered_count += 1
                    logger.debug(f"Salvato: {output_filename}")
                    
            except Exception as e:
                logger.error(f"Errore registrazione {os.path.basename(filename)}: {e}")
                tqdm.write(f"ERRORE: {os.path.basename(filename)}: {e}")
                error_count += 1
            
            pbar.update(1)
    
    logger.info(f"Registrazione completata: {registered_count}/{len(input_files)} file")
    logger.info(f"File con errori: {error_count}")
    
    return registered_count > 0, registered_count

# -----------------------------------------------------------
# FLUSSO PRINCIPALE
# -----------------------------------------------------------
def run_registration():
    """
    Esegue la registrazione usando prima Siril, poi fallback manuale se necessario.
    """
    
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("ASTRO REGISTER - ALLINEAMENTO IMMAGINI")
    logger.info("=" * 60)
    
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"Cartella input: {os.path.abspath(INPUT_DIR)}")
    logger.info(f"Cartella output: {os.path.abspath(OUTPUT_DIR)}")
    logger.info(f"Cartella log: {os.path.abspath(LOG_DIR)}")
    logger.info(f"Parametri Target - RA: {M42_RA_CENTER}°, DEC: {M42_DEC_CENTER}°")
    logger.info(f"Scala pixel: {PIXEL_SCALE_ARCSEC} arcsec/pixel")
    logger.info(f"Canvas finale: {FINAL_CANVAS_SHAPE}")
    
    # Verifica disponibilità file
    input_files = glob.glob(os.path.join(INPUT_DIR, '*.fit')) + \
                  glob.glob(os.path.join(INPUT_DIR, '*.fits'))
    
    if not input_files:
        error_msg = f"Nessun file FITS trovato in {INPUT_DIR}"
        logger.error(error_msg)
        print(f"\nERRORE: {error_msg}")
        print("Assicurati di aver eseguito prima AstroPlateSolver.py")
        return
    
    logger.info(f"Trovati {len(input_files)} file da processare")
    
    # Verifica disponibilità Siril
    siril_available = check_siril_available()
    
    if siril_available:
        logger.info(f"✓ Siril CLI trovato: {SIRIL_CLI}")
        print("\n--- METODO 1: Registrazione con Siril ---")
        print("Allineamento automatico delle immagini...\n")
        
        success, count = run_siril_registration(logger)
        
        if success and count > 0:
            print(f"\n✓ Registrazione completata con Siril!")
            print(f"File allineati: {count}")
            print(f"File salvati in: {os.path.abspath(OUTPUT_DIR)}")
            logger.info(f"Registrazione Siril completata: {count} file")
            
            # Suggerimento per prossimo step
            print(f"\nLog salvato in: {LOG_DIR}")
            print("\nPROSSIMO PASSO: Stacking delle immagini registrate")
            print("Puoi usare Siril per lo stacking finale con il comando 'stack'")
            return
        else:
            print("\n⚠ Registrazione Siril fallita. Uso metodo manuale...")
            logger.warning("Registrazione Siril fallita, uso metodo manuale")
    else:
        logger.warning(f"Siril CLI non trovato: {SIRIL_CLI}")
        print(f"\n⚠ Siril CLI non disponibile in: {SIRIL_CLI}")
        print("Uso metodo manuale per registrazione...\n")
    
    # FALLBACK: Metodo manuale
    print("\n--- METODO 2: Registrazione Manuale (reproject) ---")
    print("Riproiezione delle immagini su WCS comune...\n")
    
    success, count = manual_registration_reproject(logger)
    
    if success and count > 0:
        print(f"\n✓ Registrazione manuale completata!")
        print(f"File allineati: {count}/{len(input_files)}")
        print(f"File salvati in: {os.path.abspath(OUTPUT_DIR)}")
        print("\nNOTA: Per risultati ottimali, configura Siril per registrazione automatica.")
    else:
        print("\n❌ Errore durante la registrazione")
        logger.error("Fallimento sia Siril che metodo manuale")
        return
    
    # Suggerimento per prossimo step
    print(f"\nLog salvato in: {LOG_DIR}")
    print("\nPROSSIMO PASSO: Stacking delle immagini registrate")
    print("Puoi usare Siril, PixInsight o DeepSkyStacker per lo stacking finale.")
    
    logger.info("=" * 60)
    logger.info("ELABORAZIONE COMPLETATA")
    logger.info("=" * 60)


if __name__ == "__main__":
    start_time = time.time()
    run_registration()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"\nTempo totale di esecuzione: {elapsed_time:.2f} secondi")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Tempo totale di esecuzione: {elapsed_time:.2f} secondi")