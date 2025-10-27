"""
AstroPlateSolver.py - Plate Solving ottimizzato per Hubble DRZ
Versione per Siril 1.4.0 beta con immagini dall'archivio Hubble

Le immagini DRZ hanno già WCS nei metadata, questo script:
1. Verifica e valida il WCS esistente
2. Usa Siril 1.4 per plate solving avanzato se necessario
3. Ottimizza i metadati per la super-resolution
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
from tqdm import tqdm

# --- CONFIGURAZIONE PERCORSI ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_cropped_3')   # Da img_cropped_3
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_plate_2')    # A img_plate_2
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# --- CONFIGURAZIONE SIRIL 1.4 ---
SIRIL_CLI = "C:\\Program Files\\SiriL\\bin\\siril-cli.exe"

# --- MODALITÀ OPERATIVA ---
# Per immagini Hubble DRZ:
# - Mode "preserve": Mantieni WCS esistente, solo validazione
# - Mode "refine": Usa Siril per affinare il WCS con Astrometry.net
# - Mode "force": Forza plate solving anche se WCS esiste
PROCESSING_MODE = "preserve"  # Cambia in "refine" o "force" se necessario

# --- PARAMETRI HUBBLE (per riferimento) ---
# Le immagini DRZ hanno già questi dati nel header:
# - TELESCOP = 'HST'
# - INSTRUME = 'ACS' o 'WFC3' o altro
# - FILTER = filtro usato
# - EXPTIME = tempo di esposizione
# - WCS keywords: CRVAL1, CRVAL2, CD1_1, CD2_2, etc.

# Pixel scale tipico Hubble (arcsec/pixel):
HST_ACS_SCALE = 0.05  # ACS/WFC
HST_WFC3_SCALE = 0.04  # WFC3/UVIS
HST_WFPC2_SCALE = 0.1  # WFPC2

# --- CONFIGURAZIONE LOGGING ---
def setup_logging():
    """Configura il sistema di logging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'plate_solving_{timestamp}.log')
    
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
    
    if name_without_ext.startswith('crop_images_'):
        parts = name_without_ext.split('_')
        if len(parts) >= 3:
            return parts[2]
    
    parts = name_without_ext.split('_')
    if len(parts) >= 2:
        return parts[1] if parts[1] else 'unknown'
    
    return os.path.splitext(basename)[0]

def generate_output_filename(input_filename):
    """Genera il nome del file di output."""
    image_number = extract_image_number(input_filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    _, ext = os.path.splitext(input_filename)
    if not ext:
        ext = '.fits'
    return f"plate_image_{image_number}_{timestamp}{ext}"

def check_siril_available():
    """Verifica se Siril CLI è disponibile."""
    if not os.path.exists(SIRIL_CLI):
        return False
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

def validate_wcs(header):
    """
    Verifica se il WCS nel header è valido e completo.
    Ritorna: (is_valid, wcs_info_dict)
    """
    required_keys = ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2']
    
    # Controlla keywords obbligatorie
    has_required = all(key in header for key in required_keys)
    
    if not has_required:
        return False, {}
    
    # Controlla tipo di proiezione
    has_ctype = 'CTYPE1' in header and 'CTYPE2' in header
    
    # Controlla matrice di trasformazione (CD o CDELT)
    has_cd_matrix = ('CD1_1' in header and 'CD2_2' in header)
    has_cdelt = ('CDELT1' in header and 'CDELT2' in header)
    
    if not (has_cd_matrix or has_cdelt):
        return False, {}
    
    # Estrai informazioni WCS
    wcs_info = {
        'ra_center': header.get('CRVAL1'),
        'dec_center': header.get('CRVAL2'),
        'crpix1': header.get('CRPIX1'),
        'crpix2': header.get('CRPIX2'),
        'has_cd_matrix': has_cd_matrix,
        'has_cdelt': has_cdelt,
        'ctype1': header.get('CTYPE1', 'N/A'),
        'ctype2': header.get('CTYPE2', 'N/A')
    }
    
    # Calcola pixel scale se possibile
    try:
        if has_cd_matrix:
            cd1_1 = header.get('CD1_1', 0)
            cd2_2 = header.get('CD2_2', 0)
            pixel_scale = abs(cd1_1 * 3600)  # Converti in arcsec
            wcs_info['pixel_scale'] = pixel_scale
        elif has_cdelt:
            cdelt1 = abs(header.get('CDELT1', 0))
            pixel_scale = cdelt1 * 3600
            wcs_info['pixel_scale'] = pixel_scale
    except:
        pass
    
    return True, wcs_info

def extract_hubble_metadata(header):
    """Estrae metadati specifici Hubble dal header."""
    metadata = {
        'telescope': header.get('TELESCOP', 'Unknown'),
        'instrument': header.get('INSTRUME', 'Unknown'),
        'detector': header.get('DETECTOR', 'Unknown'),
        'filter': header.get('FILTER', 'Unknown'),
        'exptime': header.get('EXPTIME', 0),
        'date_obs': header.get('DATE-OBS', 'Unknown'),
        'proposid': header.get('PROPOSID', 'Unknown'),
        'targname': header.get('TARGNAME', 'Unknown')
    }
    return metadata

def preserve_wcs_mode(logger):
    """
    MODALITÀ PRESERVE: Mantieni il WCS esistente, copia solo con validazione.
    Ideale per immagini Hubble DRZ che hanno già WCS accurato.
    """
    logger.info("MODALITÀ: Preservazione WCS esistente (Hubble DRZ)")
    
    input_files = glob.glob(os.path.join(INPUT_DIR, '*.fit')) + \
                  glob.glob(os.path.join(INPUT_DIR, '*.fits'))
    
    if not input_files:
        logger.error(f"Nessun file trovato in {INPUT_DIR}")
        return False
    
    valid_count = 0
    invalid_count = 0
    copied_count = 0
    
    print("\n📊 Analisi WCS immagini Hubble:")
    print("-" * 70)
    
    with tqdm(total=len(input_files), desc="Validazione WCS", unit=" file") as pbar:
        for filename in input_files:
            try:
                with fits.open(filename, ignore_missing_end=True) as hdul:
                    header = hdul[0].header
                    data = hdul[0].data
                    
                    if data is None:
                        logger.warning(f"⚠ Nessun dato: {os.path.basename(filename)}")
                        invalid_count += 1
                        pbar.update(1)
                        continue
                    
                    # Valida WCS
                    is_valid, wcs_info = validate_wcs(header)
                    
                    # Estrai metadata Hubble
                    hubble_meta = extract_hubble_metadata(header)
                    
                    basename = os.path.basename(filename)
                    
                    if is_valid:
                        logger.info(f"✓ WCS valido: {basename}")
                        logger.info(f"  → RA: {wcs_info['ra_center']:.6f}°, "
                                  f"DEC: {wcs_info['dec_center']:.6f}°")
                        if 'pixel_scale' in wcs_info:
                            logger.info(f"  → Pixel scale: {wcs_info['pixel_scale']:.4f} arcsec/px")
                        logger.info(f"  → Instrument: {hubble_meta['instrument']}/{hubble_meta['detector']}")
                        logger.info(f"  → Filter: {hubble_meta['filter']}, Target: {hubble_meta['targname']}")
                        
                        # Copia file con WCS preservato
                        output_filename = generate_output_filename(filename)
                        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
                        
                        # Crea nuovo HDU preservando tutto
                        new_hdu = fits.PrimaryHDU(data, header)
                        new_hdu.writeto(output_filepath, overwrite=True, output_verify='warn')
                        
                        valid_count += 1
                        copied_count += 1
                        
                    else:
                        logger.warning(f"⚠ WCS incompleto/invalido: {basename}")
                        invalid_count += 1
                    
            except Exception as e:
                logger.error(f"❌ Errore elaborazione {os.path.basename(filename)}: {e}")
                invalid_count += 1
            
            pbar.update(1)
    
    print("\n" + "=" * 70)
    print(f"✓ File con WCS valido: {valid_count}/{len(input_files)}")
    print(f"⚠ File con WCS invalido: {invalid_count}/{len(input_files)}")
    print(f"📁 File copiati: {copied_count}")
    print("=" * 70)
    
    logger.info(f"WCS validation: {valid_count} validi, {invalid_count} invalidi")
    return valid_count > 0

def refine_wcs_with_siril(logger):
    """
    MODALITÀ REFINE: Usa Siril 1.4 per affinare il WCS esistente.
    Utile se vuoi migliorare ulteriormente la precisione astrometrica.
    """
    logger.info("MODALITÀ: Affinamento WCS con Siril 1.4 + Astrometry.net")
    
    work_dir = os.path.join(PROJECT_ROOT, 'M42', 'temp_siril_work')
    os.makedirs(work_dir, exist_ok=True)
    
    # Leggi un file per ottenere metadati
    input_files = glob.glob(os.path.join(INPUT_DIR, '*.fit*'))
    if not input_files:
        return False
    
    # Estrai pixel scale dal primo file
    try:
        with fits.open(input_files[0], ignore_missing_end=True) as hdul:
            header = hdul[0].header
            is_valid, wcs_info = validate_wcs(header)
            
            if is_valid and 'pixel_scale' in wcs_info:
                pixel_scale = wcs_info['pixel_scale']
            else:
                # Usa default Hubble
                instrument = header.get('INSTRUME', '')
                if 'ACS' in instrument:
                    pixel_scale = HST_ACS_SCALE
                elif 'WFC3' in instrument:
                    pixel_scale = HST_WFC3_SCALE
                else:
                    pixel_scale = 0.05
            
            logger.info(f"Pixel scale rilevato: {pixel_scale} arcsec/px")
    except:
        pixel_scale = 0.05
    
    # Script Siril 1.4 ottimizzato
    siril_script = f"""#! SIRIL script
requires 1.4.0

# Impostazioni per immagini Hubble
set32bits
set -mag_mode photometry

# Carica sequenza
cd {INPUT_DIR}
convert crop_images -out={work_dir}/hubble_seq

cd {work_dir}
load hubble_seq

# Plate solving con parametri ottimizzati per Hubble
# -platesolve usa il WCS esistente come punto di partenza
# -catalog=gaia usa GAIA DR3 (migliore per Hubble)
# -localasnet usa Astrometry.net locale se disponibile
platesolve -catalog=gaia -platesolve -pixelsize={pixel_scale} -radius=0.5

# Salva con WCS affinato
savefits {OUTPUT_DIR}/plate_

close
"""
    
    script_file = os.path.join(work_dir, 'refine_wcs.ssf')
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(siril_script)
    
    try:
        logger.info(f"Esecuzione Siril 1.4: {SIRIL_CLI} -s {script_file}")
        
        result = subprocess.run(
            [SIRIL_CLI, '-s', script_file],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=1800
        )
        
        logger.info(f"Output Siril:\n{result.stdout}")
        
        if result.stderr:
            logger.warning(f"Stderr Siril:\n{result.stderr}")
        
        # Verifica output
        output_files = glob.glob(os.path.join(OUTPUT_DIR, 'plate_*.fit*'))
        
        if result.returncode == 0 and len(output_files) > 0:
            logger.info(f"✓ WCS affinato con successo: {len(output_files)} file")
            return True
        else:
            logger.warning("⚠ Affinamento WCS fallito o parziale")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Timeout durante plate solving Siril")
        return False
    except Exception as e:
        logger.error(f"Errore esecuzione Siril: {e}")
        return False
    finally:
        if os.path.exists(script_file):
            try:
                os.remove(script_file)
            except:
                pass

def analyze_hubble_files(logger):
    """Analizza i file Hubble e mostra informazioni dettagliate."""
    input_files = glob.glob(os.path.join(INPUT_DIR, '*.fit*'))
    
    if not input_files:
        return
    
    print("\n" + "=" * 70)
    print("📡 ANALISI IMMAGINI HUBBLE")
    print("=" * 70)
    
    for i, filename in enumerate(input_files[:3], 1):  # Mostra prime 3
        try:
            with fits.open(filename, ignore_missing_end=True) as hdul:
                header = hdul[0].header
                data = hdul[0].data
                
                print(f"\n📸 File {i}: {os.path.basename(filename)}")
                
                # Metadata Hubble
                meta = extract_hubble_metadata(header)
                print(f"   Telescope: {meta['telescope']}")
                print(f"   Instrument: {meta['instrument']}/{meta['detector']}")
                print(f"   Filter: {meta['filter']}")
                print(f"   Target: {meta['targname']}")
                print(f"   Exposure: {meta['exptime']} sec")
                print(f"   Date: {meta['date_obs']}")
                
                # WCS info
                is_valid, wcs_info = validate_wcs(header)
                if is_valid:
                    print(f"   ✓ WCS: VALIDO")
                    print(f"   RA/DEC: {wcs_info['ra_center']:.6f}°, {wcs_info['dec_center']:.6f}°")
                    if 'pixel_scale' in wcs_info:
                        print(f"   Pixel scale: {wcs_info['pixel_scale']:.4f} arcsec/px")
                else:
                    print(f"   ⚠ WCS: INCOMPLETO")
                
                # Dimensioni
                if data is not None:
                    print(f"   Dimensioni: {data.shape}")
                
        except Exception as e:
            print(f"   ❌ Errore lettura: {e}")
    
    if len(input_files) > 3:
        print(f"\n   ... e altri {len(input_files) - 3} file")
    
    print("\n" + "=" * 70)

def run_plate_solver():
    """
    Flusso principale ottimizzato per Hubble DRZ.
    """
    logger = setup_logging()
    logger.info("=" * 70)
    logger.info("ASTRO PLATE SOLVER - Ottimizzato per Hubble DRZ")
    logger.info(f"Siril version: 1.4.0 beta")
    logger.info(f"Processing mode: {PROCESSING_MODE}")
    logger.info("=" * 70)
    
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"Input: {os.path.abspath(INPUT_DIR)}")
    logger.info(f"Output: {os.path.abspath(OUTPUT_DIR)}")
    
    # Verifica file input
    input_files = glob.glob(os.path.join(INPUT_DIR, '*.fit')) + \
                  glob.glob(os.path.join(INPUT_DIR, '*.fits'))
    
    if not input_files:
        logger.error(f"Nessun file FITS in {INPUT_DIR}")
        print("\n❌ ERRORE: Nessun file trovato")
        print("Assicurati di aver eseguito prima il cropping delle immagini.")
        return
    
    logger.info(f"Trovati {len(input_files)} file da processare")
    
    # Analizza file Hubble
    analyze_hubble_files(logger)
    
    # Esegui processing in base alla modalità
    success = False
    
    if PROCESSING_MODE == "preserve":
        print("\n🔒 MODALITÀ PRESERVE: Preservazione WCS esistente")
        print("Le immagini Hubble DRZ hanno già WCS accurato.")
        success = preserve_wcs_mode(logger)
        
    elif PROCESSING_MODE == "refine":
        if check_siril_available():
            print("\n🔧 MODALITÀ REFINE: Affinamento WCS con Siril 1.4")
            print("Uso Astrometry.net per migliorare la precisione...")
            success = refine_wcs_with_siril(logger)
            
            if not success:
                print("\n⚠ Affinamento fallito, uso modalità preserve...")
                success = preserve_wcs_mode(logger)
        else:
            logger.warning("Siril non disponibile, uso preserve mode")
            success = preserve_wcs_mode(logger)
            
    elif PROCESSING_MODE == "force":
        if check_siril_available():
            print("\n⚡ MODALITÀ FORCE: Plate solving completo")
            success = refine_wcs_with_siril(logger)
        else:
            logger.error("Siril richiesto per force mode")
            success = False
    
    # Risultati
    if success:
        output_files = glob.glob(os.path.join(OUTPUT_DIR, 'plate_*.fit*'))
        print(f"\n✅ COMPLETATO: {len(output_files)} file processati")
        print(f"📁 Output: {os.path.abspath(OUTPUT_DIR)}")
        print(f"📝 Log: {LOG_DIR}")
        print("\n🚀 Prossimo passo: AstroRegister.py per registrazione immagini")
    else:
        print("\n❌ Elaborazione fallita")
        print("Controlla il log per dettagli")
    
    logger.info("=" * 70)
    logger.info("ELABORAZIONE COMPLETATA")
    logger.info("=" * 70)

if __name__ == "__main__":
    print("\n" + "🔭" * 35)
    print(" " * 10 + "HUBBLE DRZ PLATE SOLVER")
    print("🔭" * 35 + "\n")
    
    start_time = time.time()
    run_plate_solver()
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Tempo totale: {elapsed:.2f} secondi")