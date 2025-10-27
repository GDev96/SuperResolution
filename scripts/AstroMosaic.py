"""
HST MOSAIC CREATOR - Script di Stacking (Integrazione)
Versione aggiornata per compatibilità con nuovo AstroRegister
"""

import os
import sys
import glob
import time 
import logging
import gc
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from datetime import datetime
from tqdm import tqdm

# --- CONFIGURAZIONE PERCORSI ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = 'C:\\Users\\dell\\Desktop\\Super resolution Gaia\\M42\\4_register'
OUTPUT_DIR = 'C:\\Users\\dell\\Desktop\\Super resolution Gaia\\M42\\5_mosaic'
LOG_DIR = 'C:\\Users\\dell\\Desktop\\Super resolution Gaia\\logs'

def setup_logging():
    """Configura il sistema di logging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'mosaic_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Log system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Numpy version: {np.__version__}")
    
    return logger

def stack_mosaic(combine_function='mean'):
    """Esegue lo stacking delle immagini registrate."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("HST MOSAIC CREATOR - STACKING")
    logger.info("=" * 60)

    print("=" * 70)
    print("🔭 HST MOSAIC CREATOR - STACKING".center(70))
    print("=" * 70)

    # Setup directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find input files
    input_pattern = os.path.join(INPUT_DIR, 'register_*.fits')
    files = sorted(glob.glob(input_pattern))
    
    if not files:
        msg = f"Nessuna immagine registrata trovata in {INPUT_DIR}"
        logger.error(msg)
        print(f"\n❌ ERRORE: {msg}")
        return None

    logger.info(f"Trovate {len(files)} immagini registrate")
    print(f"\n✓ Trovate {len(files)} immagini da integrare")

    # Load and verify reference image
    logger.info("Caricamento immagine di riferimento...")
    try:
        with fits.open(files[0]) as hdul:
            ref_data = hdul[0].data
            ref_header = hdul[0].header
            
            if ref_data is None:
                raise ValueError("Nessun dato nell'immagine di riferimento")
                
            final_shape = ref_data.shape
            logger.info(f"Dimensioni riferimento: {final_shape}")
            print(f"✓ Dimensioni canvas: {final_shape[1]}x{final_shape[0]} pixel")
            
            # Extract WCS info
            wcs = WCS(ref_header)
            if wcs.has_celestial:
                ra, dec = wcs.wcs.crval
                logger.info(f"Centro WCS: RA={ra:.3f}°, DEC={dec:.3f}°")
                print(f"✓ Centro: RA={ra:.3f}°, DEC={dec:.3f}°")
    
    except Exception as e:
        logger.error(f"Errore caricamento riferimento: {e}", exc_info=True)
        print(f"\n❌ ERRORE: Impossibile caricare immagine di riferimento")
        return None

    # Initialize accumulators
    logger.info("Inizializzazione accumulatori...")
    final_sum = np.zeros(final_shape, dtype=np.float64)
    count_array = np.zeros(final_shape, dtype=np.int32)
    
    # Stack images
    logger.info(f"Avvio stacking di {len(files)} immagini...")
    print(f"\n🔄 Integrazione di {len(files)} immagini...")
    
    stack_start = time.time()
    errors = []

    with tqdm(total=len(files), desc="Stacking", unit="img") as pbar:
        for i, filename in enumerate(files):
            try:
                with fits.open(filename) as hdul:
                    data = hdul[0].data
                    
                    if data.shape != final_shape:
                        msg = f"Dimensioni non corrispondenti: {data.shape}"
                        logger.warning(f"Skip {os.path.basename(filename)}: {msg}")
                        errors.append(f"Dimensioni errate in {filename}")
                        continue

                    # Check data validity
                    valid_pixels = np.isfinite(data)
                    n_valid = np.sum(valid_pixels)
                    coverage = (n_valid / data.size) * 100
                    
                    logger.info(f"Frame {i+1}: {n_valid:,} pixel validi ({coverage:.1f}%)")
                    
                    if coverage < 1.0:
                        msg = f"Copertura insufficiente: {coverage:.1f}%"
                        logger.warning(f"Skip {os.path.basename(filename)}: {msg}")
                        errors.append(f"Copertura bassa in {filename}")
                        continue
                    
                    # Add to stack
                    final_sum[valid_pixels] += data[valid_pixels]
                    count_array[valid_pixels] += 1
                    
                    # Update progress
                    pbar.set_description(f"Stack: {i+1}/{len(files)}")
                    pbar.update(1)

            except Exception as e:
                msg = f"Errore in {os.path.basename(filename)}: {str(e)}"
                logger.error(msg, exc_info=True)
                errors.append(msg)
            
            # Memory cleanup
            gc.collect()

    # Calculate final mosaic
    logger.info("Calcolo mosaico finale...")
    final_mosaic = np.divide(final_sum, count_array, 
                            out=np.zeros_like(final_sum), 
                            where=count_array > 0)

    # Update header
    ref_header['NCOMBINE'] = (len(files), 'Number of images combined')
    ref_header['COMBFUNC'] = (combine_function, 'Combination method')
    ref_header['DATE'] = datetime.now().isoformat()
    ref_header['CREATOR'] = 'AstroMosaic.py'

    # Save mosaic
    output_file = os.path.join(OUTPUT_DIR, 'mosaic_final.fits')
    logger.info(f"Salvataggio mosaico: {output_file}")
    
    fits.PrimaryHDU(final_mosaic.astype(np.float32), 
                    header=ref_header).writeto(output_file, overwrite=True)

    # Print summary
    print("\n📊 RIEPILOGO INTEGRAZIONE:")
    print(f"   Immagini processate: {len(files)}")
    print(f"   Errori: {len(errors)}")
    if errors:
        print("\n⚠️ ERRORI RISCONTRATI:")
        for err in errors[:5]:
            print(f"   - {err}")
        if len(errors) > 5:
            print(f"   ... e altri {len(errors)-5} errori")

    logger.info("Stacking completato")
    return output_file

if __name__ == "__main__":
    start_time = time.time()
    
    output = stack_mosaic(combine_function='mean')
    
    if output:
        print(f"\n✅ MOSAICO COMPLETATO")
        print(f"📁 Salvato in: {output}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Tempo totale: {elapsed:.1f} secondi")