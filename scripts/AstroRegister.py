"""
AstroRegister.py - Registrazione (Allineamento) con Siril CLI
FIXED VERSION - Risolve problemi con unità Astropy
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

# --- CONFIGURAZIONE PERCORSI ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = 'C:\\Users\\dell\\Desktop\\Super resolution Gaia\\M42\\3_plate'
OUTPUT_DIR = 'C:\\Users\\dell\\Desktop\\Super resolution Gaia\\M42\\4_register'
LOG_DIR = 'C:\\Users\\dell\\Desktop\\Super resolution Gaia\\logs'
WORK_DIR = os.path.join(PROJECT_ROOT, 'M42', 'temp_siril_register')

# --- CONFIGURAZIONE SIRIL ---
SIRIL_CLI = "C:\\Program Files\\SiriL\\bin\\siril-cli.exe"

# --- CONFIGURAZIONE TEST ---
MAX_IMAGES = 20  # Riduci per test veloce

def setup_logging():
    """Configura il sistema di logging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'registration_fixed_{timestamp}.log')
    
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
    logger.info(f"Astropy version: {astropy.__version__}")
    
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
    """Crea WCS ottimale usando dati estratti in modo sicuro."""
    if not wcs_info:
        return None, None
    
    logger.info("Calcolo WCS ottimale...")
    
    # Calcola bounds
    ra_values = [info['center_ra'] for info in wcs_info]
    dec_values = [info['center_dec'] for info in wcs_info]
    scales = [info['pixel_scale'] for info in wcs_info]
    
    # Centro ottimale
    center_ra = np.mean(ra_values)
    center_dec = np.mean(dec_values)
    
    # Scala ottimale
    pixel_scale_arcsec = np.median(scales)
    pixel_scale_deg = pixel_scale_arcsec / 3600.0
    
    # Calcola dimensioni necessarie
    ra_span = max(ra_values) - min(ra_values)
    dec_span = max(dec_values) - min(dec_values)
    
    # Aggiungi margine del 50% per essere sicuri
    ra_span *= 1.5
    dec_span *= 1.5
    
    # Calcola dimensioni canvas
    width_pixels = int(ra_span / pixel_scale_deg) + 1000  # Margine extra
    height_pixels = int(dec_span / pixel_scale_deg) + 1000
    
    # Limiti ragionevoli
    width_pixels = min(max(width_pixels, 2500), 6000)
    height_pixels = min(max(height_pixels, 2500), 6000)
    
    print(f"\n🎯 WCS OTTIMALE:")
    print(f"   Centro: RA={center_ra:.3f}°, DEC={center_dec:.3f}°")
    print(f"   Scala: {pixel_scale_arcsec:.3f} arcsec/pixel")
    print(f"   Canvas: {width_pixels} x {height_pixels} pixel")
    print(f"   Span: RA={ra_span:.3f}°, DEC={dec_span:.3f}°")
    
    # Crea WCS
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [width_pixels/2, height_pixels/2]
    wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]  # RA decresce verso destra
    wcs.wcs.crval = [center_ra, center_dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    
    logger.info(f"Optimal WCS: center=({center_ra:.3f}, {center_dec:.3f}), "
               f"scale={pixel_scale_arcsec:.3f}\"/px, size=({width_pixels}, {height_pixels})")
    
    return wcs, (height_pixels, width_pixels)

def manual_registration_fixed(logger, input_files):
    """Registrazione manuale con diagnostica WCS corretta."""
    logger.info("=" * 50)
    logger.info("REGISTRAZIONE MANUALE CORRETTA")
    logger.info("=" * 50)
    
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
    
    # Registrazione
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    success_count = 0
    errors = []
    
    print(f"\n🔄 Registrazione di {len(valid_files)} immagini...")
    
    with tqdm(total=len(valid_files), desc="Registrazione") as pbar:
        for filepath in valid_files:
            try:
                filename = os.path.basename(filepath)
                pbar.set_description(f"Reg: {filename[:30]}")
                
                with fits.open(filepath) as hdul:
                    # Trova HDU con dati (usa info dalla diagnostica)
                    hdu_info = next((info for info in wcs_info if info['file'] == filepath), None)
                    if hdu_info is None:
                        continue
                    
                    hdu_idx = hdu_info['hdu_index']
                    data_hdu = hdul[hdu_idx]
                    
                    # Reproietta con reproject_interp
                    logger.debug(f"Reproiezione {filename}...")
                    reprojected_data, footprint = reproject_interp(
                        data_hdu,
                        target_wcs,
                        shape_out=shape_out
                    )
                    
                    # Verifica risultato
                    if reprojected_data is None:
                        logger.warning(f"Reproiezione fallita per {filename}")
                        continue
                    
                    # Analizza copertura
                    valid_mask = ~np.isnan(reprojected_data)
                    valid_pixels = np.sum(valid_mask)
                    total_pixels = reprojected_data.size
                    coverage = (valid_pixels / total_pixels) * 100
                    
                    if coverage < 1.0:  # Almeno 1% di copertura
                        logger.warning(f"Skip {filename}: copertura troppo bassa ({coverage:.1f}%)")
                        continue
                    
                    # Sostituisci NaN con zero
                    reprojected_data = np.nan_to_num(reprojected_data, nan=0.0)
                    
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
                    
                    # Salva
                    output_filename = f"register_{os.path.splitext(filename)[0]}_{datetime.now().strftime('%H%M%S')}.fits"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    
                    fits.PrimaryHDU(
                        data=reprojected_data.astype(np.float32),
                        header=new_header
                    ).writeto(output_path, overwrite=True)
                    
                    success_count += 1
                    logger.info(f"✓ {filename}: copertura {coverage:.1f}%, {valid_pixels:,} pixel validi")
                    pbar.set_description(f"✓ {success_count} registrate")
                    
            except Exception as e:
                error_msg = f"Errore {os.path.basename(filepath)}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
                pbar.set_description(f"❌ Errore: {os.path.basename(filepath)}")
            
            pbar.update(1)
    
    # Riepilogo
    print(f"\n📊 RISULTATI REGISTRAZIONE:")
    print(f"   File processati: {len(valid_files)}")
    print(f"   Registrati con successo: {success_count}")
    print(f"   Errori: {len(errors)}")
    
    if errors:
        logger.error("\nERRORI:")
        for err in errors[:5]:  # Mostra solo primi 5
            logger.error(err)
        if len(errors) > 5:
            logger.error(f"... e altri {len(errors)-5} errori")
    
    logger.info(f"Registrazione completata: {success_count}/{len(valid_files)}")
    return success_count > 0, success_count

def run_registration():
    """Funzione principale di registrazione."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info(f"ASTRO REGISTER - FIXED VERSION ({MAX_IMAGES} immagini)")
    logger.info("=" * 60)
    
    print("=" * 70)
    print("🔭 ASTRO REGISTER - VERSIONE CORRETTA".center(70))
    print("=" * 70)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Limite: {MAX_IMAGES} immagini")
    
    # Setup directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get input files
    all_files = glob.glob(os.path.join(INPUT_DIR, '*.fit*'))
    input_files = sorted(all_files)[:MAX_IMAGES]
    
    if not input_files:
        logger.error(f"Nessun file trovato in {INPUT_DIR}")
        print(f"\n❌ Nessun file trovato in {INPUT_DIR}")
        return
    
    print(f"\n✓ Trovati {len(input_files)} file da processare")
    logger.info(f"Processing {len(input_files)} files")
    
    # Registrazione manuale corretta
    print(f"\n🔄 Avvio registrazione con diagnostica WCS corretta...")
    success, count = manual_registration_fixed(logger, input_files)
    
    if success:
        print(f"\n✅ REGISTRAZIONE COMPLETATA!")
        print(f"📁 {count} immagini registrate salvate in: {OUTPUT_DIR}")
        print(f"📄 Log dettagliato: {LOG_DIR}")
    else:
        print(f"\n❌ REGISTRAZIONE FALLITA")
        print(f"📄 Controlla i log per dettagli: {LOG_DIR}")
    
    logger.info("Elaborazione completata")

if __name__ == "__main__":
    print("\n🔭 ASTRO REGISTER - VERSIONE CORRETTA")
    print(f"Limite test: {MAX_IMAGES} immagini\n")
    
    start_time = time.time()
    run_registration()
    elapsed = time.time() - start_time
    
    print(f"\n⏱️ Tempo totale: {elapsed:.1f} secondi")