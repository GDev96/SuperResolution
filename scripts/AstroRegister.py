"""
AstroRegister.py - Registrazione (Allineamento) con Siril CLI
ENHANCED VERSION - Canvas espanso e gestione copertura migliorata
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

# ✅ PATH CORRETTI per la tua struttura
INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_plate_2')      # Da img_plate_2
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_register_4')   # A img_register_4
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
WORK_DIR = os.path.join(PROJECT_ROOT, 'temp_siril_register')

# --- CONFIGURAZIONE SIRIL ---
SIRIL_CLI = "C:\\Program Files\\SiriL\\bin\\siril-cli.exe"

# --- CONFIGURAZIONE TEST ---
MAX_IMAGES = 101  # Tutti i file disponibili

def setup_logging():
    """Configura il sistema di logging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'registration_enhanced_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Log system info e path
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Numpy version: {np.__version__}")
    logger.info(f"Astropy version: {astropy.__version__}")
    logger.info(f"Script dir: {SCRIPT_DIR}")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Input dir: {INPUT_DIR}")
    logger.info(f"Output dir: {OUTPUT_DIR}")
    
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
                # Usa valore dall'header se disponibile, altrimenti default
                pixel_scale_arcsec = float(header.get('PIXSCALE', 0.1))  # Default per Hubble
                pixel_scale_deg = pixel_scale_arcsec / 3600.0
                
            pixel_scale_arcsec = pixel_scale_deg * 3600.0
            
        except Exception as e:
            logger.debug(f"Pixel scale fallback: {e}")
            pixel_scale_arcsec = 0.1  # Default per immagini Hubble
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
    logger.info("DIAGNOSTICA WCS ENHANCED")
    logger.info("=" * 60)
    
    valid_files = []
    wcs_info = []
    
    print("\n🔍 Diagnostica WCS delle immagini (versione migliorata)...")
    
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

def create_optimal_wcs_enhanced(wcs_info, logger):
    """Crea WCS ottimale con canvas espanso per catturare tutte le immagini."""
    if not wcs_info:
        return None, None
    
    logger.info("Calcolo WCS ottimale con canvas espanso...")
    
    # Calcola bounds estesi per ogni immagine
    ra_bounds = []
    dec_bounds = []
    
    for info in wcs_info:
        center_ra = info['center_ra']
        center_dec = info['center_dec']
        pixel_scale_deg = info['pixel_scale_deg']
        shape = info['shape']
        
        # Calcola i bounds reali di ogni immagine
        half_width_deg = (shape[1] / 2.0) * pixel_scale_deg
        half_height_deg = (shape[0] / 2.0) * pixel_scale_deg
        
        # Correzione per distorsione ai poli
        cos_dec = np.cos(np.radians(center_dec))
        
        ra_min = center_ra - half_width_deg / cos_dec
        ra_max = center_ra + half_width_deg / cos_dec
        dec_min = center_dec - half_height_deg
        dec_max = center_dec + half_height_deg
        
        ra_bounds.extend([ra_min, ra_max])
        dec_bounds.extend([dec_min, dec_max])
    
    # Trova bounds globali
    global_ra_min = min(ra_bounds)
    global_ra_max = max(ra_bounds)
    global_dec_min = min(dec_bounds)
    global_dec_max = max(dec_bounds)
    
    # Centro ottimale
    center_ra = (global_ra_min + global_ra_max) / 2.0
    center_dec = (global_dec_min + global_dec_max) / 2.0
    
    # Scala ottimale (usa la mediana per robustezza)
    scales = [info['pixel_scale'] for info in wcs_info]
    pixel_scale_arcsec = np.median(scales)
    pixel_scale_deg = pixel_scale_arcsec / 3600.0
    
    # Calcola dimensioni necessarie con margine generoso
    ra_span = global_ra_max - global_ra_min
    dec_span = global_dec_max - global_dec_min
    
    # Correzione per distorsione ai poli
    cos_center_dec = np.cos(np.radians(center_dec))
    ra_span_corrected = ra_span / cos_center_dec
    
    # Margine del 100% invece del 50% per catturare tutto
    margin_factor = 2.0
    ra_span_corrected *= margin_factor
    dec_span *= margin_factor
    
    # Calcola dimensioni canvas
    width_pixels = int(ra_span_corrected / pixel_scale_deg) + 3000  # Margine extra aumentato
    height_pixels = int(dec_span / pixel_scale_deg) + 3000
    
    # Limiti aumentati per canvas più grandi
    min_size = 4000
    max_size = 16000  # Aumentato per catturare tutto il mosaico M42
    
    width_pixels = min(max(width_pixels, min_size), max_size)
    height_pixels = min(max(height_pixels, min_size), max_size)
    
    # Assicurati che le dimensioni siano pari (aiuta con alcuni algoritmi)
    width_pixels = (width_pixels // 2) * 2
    height_pixels = (height_pixels // 2) * 2
    
    print(f"\n🎯 WCS OTTIMALE ESPANSO:")
    print(f"   Centro: RA={center_ra:.3f}°, DEC={center_dec:.3f}°")
    print(f"   Scala: {pixel_scale_arcsec:.3f} arcsec/pixel")
    print(f"   Canvas: {width_pixels} x {height_pixels} pixel")
    print(f"   Campo: {width_pixels * pixel_scale_arcsec / 60:.1f}' x {height_pixels * pixel_scale_arcsec / 60:.1f}'")
    print(f"   Bounds RA: {global_ra_min:.3f}° - {global_ra_max:.3f}°")
    print(f"   Bounds DEC: {global_dec_min:.3f}° - {global_dec_max:.3f}°")
    print(f"   Memoria stimata: ~{(width_pixels * height_pixels * 4) / 1024**3:.1f} GB per immagine")
    
    # Crea WCS
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [width_pixels/2, height_pixels/2]
    wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]  # RA decresce verso destra
    wcs.wcs.crval = [center_ra, center_dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    
    logger.info(f"Optimal WCS EXPANDED: center=({center_ra:.3f}, {center_dec:.3f}), "
               f"scale={pixel_scale_arcsec:.3f}\"/px, size=({width_pixels}, {height_pixels})")
    logger.info(f"Field of view: {width_pixels * pixel_scale_arcsec / 60:.1f}' x {height_pixels * pixel_scale_arcsec / 60:.1f}'")
    
    return wcs, (height_pixels, width_pixels)

def manual_registration_enhanced(logger, input_files, min_coverage=0.1, force_all=False):
    """Registrazione manuale migliorata con gestione copertura flessibile."""
    logger.info("=" * 50)
    logger.info("REGISTRAZIONE MANUALE ENHANCED")
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
    
    # Crea WCS ottimale espanso
    target_wcs, shape_out = create_optimal_wcs_enhanced(wcs_info, logger)
    
    if target_wcs is None:
        logger.error("Impossibile creare WCS ottimale!")
        return False, 0
    
    # Registrazione
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    success_count = 0
    low_coverage_count = 0
    errors = []
    
    print(f"\n🔄 Registrazione di {len(valid_files)} immagini...")
    if force_all:
        print("⚠️  MODALITÀ FORZATA: Salva tutte le immagini indipendentemente dalla copertura")
    else:
        print(f"📏 Copertura minima richiesta: {min_coverage}%")
    
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
                    
                    # ✅ CONTROLLO COPERTURA FLESSIBILE
                    skip_image = False
                    if not force_all and coverage < min_coverage:
                        logger.warning(f"Skip {filename}: copertura troppo bassa ({coverage:.3f}%)")
                        skip_image = True
                    
                    # Se forziamo tutto o la copertura è sufficiente, procedi
                    if not skip_image:
                        # Sostituisci NaN con zero
                        reprojected_data = np.nan_to_num(reprojected_data, nan=0.0)
                        
                        # Crea header
                        new_header = target_wcs.to_header()
                        
                        # Copia metadati importanti
                        important_keys = ['OBJECT', 'DATE-OBS', 'EXPTIME', 'FILTER', 
                                        'INSTRUME', 'TELESCOP', 'OBSERVER', 'PIXSCALE']
                        
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
                        new_header['REGCANVAS'] = f"{shape_out[1]}x{shape_out[0]}"
                        
                        # Salva con prefisso diverso se copertura bassa
                        if coverage < 1.0:
                            prefix = "lowcov_"
                            low_coverage_count += 1
                        else:
                            prefix = "register_"
                        
                        output_filename = f"{prefix}{os.path.splitext(filename)[0]}_{datetime.now().strftime('%H%M%S')}.fits"
                        output_path = os.path.join(OUTPUT_DIR, output_filename)
                        
                        fits.PrimaryHDU(
                            data=reprojected_data.astype(np.float32),
                            header=new_header
                        ).writeto(output_path, overwrite=True)
                        
                        success_count += 1
                        
                        # Log diverso per copertura bassa
                        if coverage < 1.0:
                            logger.info(f"⚠️ {filename}: copertura BASSA ({coverage:.3f}%), {valid_pixels:,} pixel validi - SALVATO")
                        else:
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
    print(f"   Con copertura bassa: {low_coverage_count}")
    print(f"   Errori: {len(errors)}")
    
    if low_coverage_count > 0:
        print(f"\n⚠️  {low_coverage_count} immagini salvate con prefisso 'lowcov_' per bassa copertura")
    
    if errors:
        logger.error("\nERRORI:")
        for err in errors[:5]:
            logger.error(err)
        if len(errors) > 5:
            logger.error(f"... e altri {len(errors)-5} errori")
    
    logger.info(f"Registrazione completata: {success_count}/{len(valid_files)} (lowcov: {low_coverage_count})")
    return success_count > 0, success_count

def run_registration():
    """Funzione principale di registrazione con opzioni configurabili."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info(f"ASTRO REGISTER - ENHANCED VERSION ({MAX_IMAGES} immagini)")
    logger.info("=" * 60)
    
    print("=" * 70)
    print("🔭 ASTRO REGISTER - VERSIONE MIGLIORATA".center(70))
    print("=" * 70)
    
    # ✅ MOSTRA PATH CORRETTI
    print(f"Input:  {os.path.abspath(INPUT_DIR)}")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Logs:   {os.path.abspath(LOG_DIR)}")
    print(f"Limite: {MAX_IMAGES} immagini")
    
    # Verifica esistenza directory di input
    if not os.path.exists(INPUT_DIR):
        print(f"\n❌ ERRORE: Directory input non esiste: {INPUT_DIR}")
        print("💡 Verifica di aver eseguito prima AstroPlateSolver.py")
        logger.error(f"Input directory does not exist: {INPUT_DIR}")
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
        min_coverage = 0.1
        force_all = False
        print("📏 Modalità AUTOMATICA: copertura minima 0.1%")
    
    # Setup directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Get input files
    all_files = glob.glob(os.path.join(INPUT_DIR, '*.fit*'))
    input_files = sorted(all_files)[:MAX_IMAGES]
    
    if not input_files:
        logger.error(f"Nessun file trovato in {INPUT_DIR}")
        print(f"\n❌ Nessun file trovato in {INPUT_DIR}")
        print("💡 Verifica di aver eseguito prima:")
        print("   1. AstroDrop.py (crop)")
        print("   2. AstroPlateSolver.py (plate solving)")
        return
    
    print(f"\n✓ Trovati {len(input_files)} file da processare")
    
    # Mostra alcuni file di esempio
    print(f"\n📄 Primi file trovati:")
    for i, f in enumerate(input_files[:3]):
        print(f"   {i+1}. {os.path.basename(f)}")
    if len(input_files) > 3:
        print(f"   ... e altri {len(input_files)-3} file")
    
    logger.info(f"Processing {len(input_files)} files")
    
    # Registrazione manuale migliorata
    print(f"\n🔄 Avvio registrazione con canvas espanso...")
    success, count = manual_registration_enhanced(logger, input_files, min_coverage, force_all)
    
    if success:
        print(f"\n✅ REGISTRAZIONE COMPLETATA!")
        print(f"📁 {count} immagini registrate salvate in: {os.path.abspath(OUTPUT_DIR)}")
        print(f"📄 Log dettagliato: {os.path.abspath(LOG_DIR)}")
        print(f"\n💡 SUGGERIMENTO:")
        print(f"   - File 'register_*.fits': Copertura normale")
        print(f"   - File 'lowcov_*.fits': Copertura bassa ma salvati")
        print(f"\n🎯 PROSSIMO PASSO:")
        print(f"   Esegui AstroMosaic.py per creare il mosaico finale")
    else:
        print(f"\n❌ REGISTRAZIONE FALLITA")
        print(f"📄 Controlla i log per dettagli: {os.path.abspath(LOG_DIR)}")
    
    logger.info("Elaborazione completata")

if __name__ == "__main__":
    print("\n🔭 ASTRO REGISTER - VERSIONE MIGLIORATA")
    print(f"Limite: {MAX_IMAGES} immagini\n")
    
    start_time = time.time()
    run_registration()
    elapsed = time.time() - start_time
    
    print(f"\n⏱️ Tempo totale: {elapsed:.1f} secondi")