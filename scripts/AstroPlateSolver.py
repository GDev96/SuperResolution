"""
STEP 1: ESTRAZIONE E PREPARAZIONE IMMAGINI HST
Estrae i dati e il WCS corretto dai file drizzle multi-estensione HST
e crea file FITS singoli pronti per la registrazione.
"""

import os
import glob
import time
import logging
from datetime import datetime
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm
import numpy as np

# --- CONFIGURAZIONE PERCORSI ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# CONFIGURAZIONE OGGETTO CELESTE
# Cambia questo valore per elaborare oggetti diversi (M42, M33, NGC2024, etc.)
TARGET_OBJECT = "M42"  # <-- MODIFICA QUI IL NOME DELL'OGGETTO

# PATH AUTOMATICI BASATI SU TARGET_OBJECT
INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_lights_1', TARGET_OBJECT)      # Input originale HST
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_plate_2', TARGET_OBJECT)     # Output processato  
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs', TARGET_OBJECT)

def setup_logging():
    """Configura il sistema di logging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'preparation_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def extract_hst_data(filename, logger):
    """
    Estrae dati e WCS da file HST multi-estensione.
    
    I file drizzle HST hanno struttura:
    - HDU 0: Primary (header principale, dati spesso vuoti)
    - HDU 1: SCI (Science data)
    - HDU 2: WHT (Weight map)
    - HDU 3: CTX (Context)
    """
    try:
        with fits.open(filename) as hdul:
            # Mostra struttura file per debug
            logger.debug(f"Struttura {os.path.basename(filename)}:")
            for i, hdu in enumerate(hdul):
                logger.debug(f"  HDU {i}: {hdu.name}, shape: {hdu.data.shape if hdu.data is not None else 'None'}")
            
            # Cerca l'HDU SCI (Science)
            sci_data = None
            sci_header = None
            
            # Metodo 1: cerca per nome 'SCI'
            if 'SCI' in hdul:
                sci_data = hdul['SCI'].data
                sci_header = hdul['SCI'].header
                logger.debug(f"  Trovato HDU 'SCI'")
            
            # Metodo 2: cerca prima estensione con dati 2D
            if sci_data is None:
                for i, hdu in enumerate(hdul):
                    if hdu.data is not None and len(hdu.data.shape) == 2:
                        sci_data = hdu.data
                        sci_header = hdu.header
                        logger.debug(f"  Usando HDU {i} con dati 2D")
                        break
            
            if sci_data is None or sci_header is None:
                logger.error(f"Nessun dato valido in {os.path.basename(filename)}")
                return None, None, None
            
            # Verifica WCS
            wcs = WCS(sci_header)
            if not wcs.has_celestial:
                logger.warning(f"WCS non valido in {os.path.basename(filename)}")
                return None, None, None
            
            # Estrai informazioni
            shape = sci_data.shape
            ra, dec = wcs.wcs.crval
            
            # Calcola pixel scale (protezione contro errori)
            try:
                pixel_scale = abs(wcs.wcs.cdelt[0] * 3600)  # in arcsec
            except:
                # Fallback: usa CD matrix
                cd = wcs.wcs.cd
                pixel_scale = np.sqrt(cd[0,0]**2 + cd[0,1]**2) * 3600
            
            info = {
                'shape': shape,
                'ra': ra,
                'dec': dec,
                'pixel_scale': pixel_scale
            }
            
            logger.info(f"✓ {os.path.basename(filename)}: "
                       f"{shape[1]}x{shape[0]}px, "
                       f"RA={ra:.4f}°, DEC={dec:.4f}°, "
                       f"scale={pixel_scale:.4f}\"/px")
            
            return sci_data, sci_header, info
            
    except Exception as e:
        logger.error(f"✗ Errore su {os.path.basename(filename)}: {e}")
        return None, None, None

def prepare_images():
    """
    Prepara tutte le immagini HST estraendo i dati corretti.
    """
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("STEP 1: PREPARAZIONE IMMAGINI HST")
    logger.info("=" * 60)
    
    print("=" * 70)
    print("🔭 STEP 1: PREPARAZIONE IMMAGINI HST".center(70))
    print("=" * 70)
    print(f"\n🎯 Oggetto Target: {TARGET_OBJECT}")
    print("ℹ️  Estrazione dati da file drizzle multi-estensione...\n")
    
    # Setup directories
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find FITS files
    fits_files = glob.glob(os.path.join(INPUT_DIR, '*.fit')) + \
                 glob.glob(os.path.join(INPUT_DIR, '*.fits'))
    
    if not fits_files:
        error_msg = f"Nessun file FITS trovato in {INPUT_DIR}"
        logger.error(error_msg)
        print(f"\n❌ ERRORE: {error_msg}")
        return
    
    logger.info(f"Trovati {len(fits_files)} file FITS")
    print(f"✓ Trovati {len(fits_files)} file da preparare")
    
    # Process files
    prepared_count = 0
    failed_count = 0
    
    ra_list = []
    dec_list = []
    scale_list = []
    
    print(f"\n🔄 Estrazione e preparazione...")
    
    with tqdm(total=len(fits_files), desc="Preparazione", unit="file") as pbar:
        for input_file in fits_files:
            # Estrai dati
            data, header, info = extract_hst_data(input_file, logger)
            
            if data is not None:
                # Crea file di output
                basename = os.path.basename(input_file)
                name, ext = os.path.splitext(basename)
                output_file = os.path.join(OUTPUT_DIR, f"prep_{name}.fits")
                
                try:
                    # Crea nuovo FITS con un solo HDU
                    primary_hdu = fits.PrimaryHDU(data=data, header=header)
                    
                    # Aggiungi metadati
                    primary_hdu.header['ORIGINAL'] = basename
                    primary_hdu.header['PREPDATE'] = datetime.now().isoformat()
                    primary_hdu.header['PREPBY'] = 'step1_prepare_hst.py'
                    
                    # Salva
                    primary_hdu.writeto(output_file, overwrite=True)
                    
                    prepared_count += 1
                    ra_list.append(info['ra'])
                    dec_list.append(info['dec'])
                    scale_list.append(info['pixel_scale'])
                    
                    logger.debug(f"Salvato: {output_file}")
                    
                except Exception as e:
                    logger.error(f"Errore salvataggio {basename}: {e}")
                    failed_count += 1
            else:
                failed_count += 1
            
            pbar.update(1)
    
    # Calculate field coverage
    if ra_list:
        ra_min, ra_max = min(ra_list), max(ra_list)
        dec_min, dec_max = min(dec_list), max(dec_list)
        avg_scale = np.mean(scale_list)
        
        logger.info("=" * 60)
        logger.info("STATISTICHE CAMPO")
        logger.info("=" * 60)
        logger.info(f"RA range: {ra_min:.4f}° - {ra_max:.4f}° (span: {ra_max-ra_min:.4f}°)")
        logger.info(f"DEC range: {dec_min:.4f}° - {dec_max:.4f}° (span: {dec_max-dec_min:.4f}°)")
        logger.info(f"Scala media: {avg_scale:.4f} arcsec/pixel")
        
        print(f"\n📊 STATISTICHE CAMPO:")
        print(f"   RA: {ra_min:.4f}° → {ra_max:.4f}° (span: {ra_max-ra_min:.4f}°)")
        print(f"   DEC: {dec_min:.4f}° → {dec_max:.4f}° (span: {dec_max-dec_min:.4f}°)")
        print(f"   Scala media: {avg_scale:.4f}\"/px")
    
    # Summary
    logger.info("=" * 60)
    logger.info("RIEPILOGO PREPARAZIONE")
    logger.info("=" * 60)
    logger.info(f"File totali: {len(fits_files)}")
    logger.info(f"File preparati: {prepared_count}")
    logger.info(f"File falliti: {failed_count}")
    logger.info(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    
    print(f"\n📊 RIEPILOGO:")
    print(f"   File preparati: {prepared_count}/{len(fits_files)}")
    print(f"   Falliti: {failed_count}")
    print(f"   Output: {os.path.abspath(OUTPUT_DIR)}")
    
    if prepared_count > 0:
        print(f"\n✅ PREPARAZIONE COMPLETATA!")
        print(f"   File pronti per la registrazione.")
        print(f"\n   Prossimo passo: esegui step2_registration.py")
    else:
        print(f"\n⚠️ ATTENZIONE: Nessun file preparato con successo!")

if __name__ == "__main__":
    start_time = time.time()
    prepare_images()
    elapsed_time = time.time() - start_time
    
    print(f"\n⏱️ Tempo totale: {elapsed_time:.2f} secondi")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Tempo totale: {elapsed_time:.2f} secondi")
    logger.info("PREPARAZIONE COMPLETATA")