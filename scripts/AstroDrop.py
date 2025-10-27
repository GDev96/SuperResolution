"""
AstroDrop.py - Crop automatico delle immagini FITS
Trova le dimensioni minime comuni e ritaglia tutte le immagini centralmente.
Nota: Siril non ha un comando nativo per auto-crop, quindi usa logica Python + Astropy
"""

import os
import glob
import time
import logging
from datetime import datetime
from astropy.io import fits
from tqdm import tqdm

# --- CONFIGURAZIONE PERCORSI ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_lights_1')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_cropped_3')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# --- CONFIGURAZIONE LOGGING ---
def setup_logging():
    """Configura il sistema di logging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'fits_cropping_{timestamp}.log')
    
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
    parts = name_without_ext.split('_')
    
    if len(parts) >= 3:
        return parts[2]
    else:
        return 'unknown'

def generate_output_filename(input_filename):
    """Genera il nome del file di output: crop_images_number_timestamp.fits"""
    image_number = extract_image_number(input_filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    
    _, ext = os.path.splitext(input_filename)
    if not ext:
        ext = '.fits'
    
    return f"crop_images_{image_number}_{timestamp}{ext}"

# -----------------------------------------------------------
# FUNZIONE PRINCIPALE: CROP AUTOMATICO
# -----------------------------------------------------------
def process_fits_files():
    """
    Trova le dimensioni minime comuni tra tutti i file FITS
    e ritaglia centralmente tutte le immagini a quella dimensione.
    
    NOTA: Siril non ha un comando nativo per questo tipo di crop automatico,
    quindi usiamo astropy direttamente. Alternativamente, potresti usare
    il comando 'crop' di Siril per ogni file individualmente se conosci
    già le dimensioni finali.
    """
    
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("ASTRODROP - CROP AUTOMATICO IMMAGINI FITS")
    logger.info("=" * 60)
    
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"Cartella input: {os.path.abspath(INPUT_DIR)}")
    logger.info(f"Cartella output: {os.path.abspath(OUTPUT_DIR)}")
    logger.info(f"Cartella log: {os.path.abspath(LOG_DIR)}")

    print(f"Ricerca file FITS in: {os.path.abspath(INPUT_DIR)}")

    # Cerca tutti i file FITS
    fits_files = glob.glob(os.path.join(INPUT_DIR, '*.fit')) + \
                 glob.glob(os.path.join(INPUT_DIR, '*.fits'))

    if not fits_files:
        error_msg = f"Nessun file FITS trovato in {INPUT_DIR}"
        logger.error(error_msg)
        print(f"\nERRORE: {error_msg}")
        print("Assicurati che le tue immagini siano nella cartella 'data/img_lights_1/'.")
        return

    logger.info(f"Trovati {len(fits_files)} file FITS")

    # --- FASE 1: Ricerca delle Dimensioni Minime ---
    print("\nFase 1: Analisi delle dimensioni delle immagini...")
    logger.info("-" * 60)
    logger.info("FASE 1: Analisi dimensioni")
    logger.info("-" * 60)

    min_width = float('inf')
    min_height = float('inf')
    dimensions = {}
    skipped_files = []
    
    for filename in tqdm(fits_files, desc="Analisi File (1/2)", unit=" file"):
        try:
            with fits.open(filename, ignore_missing_end=True) as hdul:
                hdu_index = 0
                data = hdul[0].data
                if data is None and len(hdul) > 1:
                    data = hdul[1].data
                    hdu_index = 1
                
                if data is not None:
                    height, width = data.shape[-2:] 
                    min_width = min(min_width, width)
                    min_height = min(min_height, height)
                    dimensions[filename] = (width, height, hdu_index)
                    logger.debug(f"File: {os.path.basename(filename)} - Dimensioni: {width}x{height}, HDU: {hdu_index}")
                
        except Exception as e:
            error_msg = f"Errore lettura file {os.path.basename(filename)}: {e}"
            logger.warning(error_msg)
            tqdm.write(f"ATTENZIONE: Salto file {os.path.basename(filename)} - Errore: {e}")
            skipped_files.append((filename, str(e)))

    if min_width == float('inf'):
        error_msg = "Nessun file FITS valido trovato con dati immagine"
        logger.error(error_msg)
        print(f"\nERRORE: {error_msg}")
        return

    logger.info(f"Dimensioni minime trovate: {min_width} x {min_height}")
    logger.info(f"File validi: {len(dimensions)}")
    logger.info(f"File saltati: {len(skipped_files)}")

    print(f"\nDimensioni minime trovate (Larghezza x Altezza): {min_width} x {min_height}")

    # --- FASE 2: Ritaglio Centrale ---
    print("\nFase 2: Ritaglio centrale delle immagini...")
    logger.info("-" * 60)
    logger.info("FASE 2: Ritaglio e salvataggio")
    logger.info("-" * 60)

    count_cropped = 0
    count_errors = 0
    total_files = len(dimensions)

    with tqdm(total=total_files, desc="Ritaglio FITS (2/2)", unit=" file") as pbar:
        for filename, (width, height, hdu_index) in dimensions.items():
            
            # Calcola offset per centrare il crop
            dx = (width - min_width) // 2
            dy = (height - min_height) // 2
            y_start = dy
            y_end = height - dy
            x_start = dx
            x_end = width - dx
            
            try:
                with fits.open(filename, ignore_missing_end=True) as hdul:
                    hdu = hdul[hdu_index]
                    original_data = hdu.data
                    original_header = hdu.header
                    
                    # Ritaglia i dati
                    cropped_data = original_data[..., y_start:y_end, x_start:x_end]
                    
                    # Crea nuovo HDU con header aggiornato
                    new_hdu = fits.PrimaryHDU(cropped_data, original_header)
                    new_hdu.header['NAXIS1'] = min_width
                    new_hdu.header['NAXIS2'] = min_height
                    
                    # Salva il file
                    output_filename = generate_output_filename(filename)
                    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
                    new_hdu.writeto(output_filepath, overwrite=True, output_verify='warn')
                    
                    count_cropped += 1
                    logger.debug(f"Ritagliato: {os.path.basename(filename)} -> {output_filename}")
                    
            except Exception as e:
                error_msg = f"Errore ritaglio {os.path.basename(filename)}: {e}"
                logger.error(error_msg)
                tqdm.write(f"ERRORE: {error_msg}")
                count_errors += 1

            pbar.update(1)

    # --- RIEPILOGO FINALE ---
    logger.info("=" * 60)
    logger.info("RIEPILOGO ELABORAZIONE")
    logger.info("=" * 60)
    logger.info(f"File totali trovati: {len(fits_files)}")
    logger.info(f"File validi analizzati: {total_files}")
    logger.info(f"File ritagliati con successo: {count_cropped}")
    logger.info(f"File con errori: {count_errors}")
    logger.info(f"Cartella output: {os.path.abspath(OUTPUT_DIR)}")
    
    print(f"\n--- Elaborazione Completata ---")
    print(f"Totale file ritagliati: {count_cropped} / {total_files}")
    print(f"File con errori: {count_errors}")
    print(f"I file ritagliati si trovano in: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Log salvato in: {LOG_DIR}")
    
    # Suggerimento per prossimo step
    print(f"\nProssimo passo: esegui AstroPlateSolver.py per il plate solving")


if __name__ == "__main__":
    start_time = time.time()
    process_fits_files()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"\nTempo totale di esecuzione: {elapsed_time:.2f} secondi")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Tempo totale di esecuzione: {elapsed_time:.2f} secondi")
    logger.info("=" * 60)
    logger.info("ELABORAZIONE TERMINATA")
    logger.info("=" * 60)