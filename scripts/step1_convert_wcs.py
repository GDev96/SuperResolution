"""
STEP 1: PREPARAZIONE IMMAGINI - CONVERSIONE WCS
Converte coordinate esistenti (OBJCTRA/OBJCTDEC) in WCS standard
Legge la configurazione da config.json (creato da step0)
"""

import os
import glob
import time
import json
import logging
from datetime import datetime
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)

# ============================================================================
# CONFIGURAZIONE DINAMICA
# ============================================================================

# Ottieni il percorso assoluto della directory contenente questo script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Cerca la cartella 'data'
if os.path.isdir(os.path.join(SCRIPT_DIR, 'data')):
    PROJECT_ROOT = SCRIPT_DIR
elif os.path.isdir(os.path.join(os.path.dirname(SCRIPT_DIR), 'data')):
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    raise FileNotFoundError(
        f"Impossibile trovare la directory 'data'. "
        f"Verificata in {SCRIPT_DIR} e {os.path.dirname(SCRIPT_DIR)}."
    )

# Definisci i percorsi principali
BASE_DIR = os.path.join(PROJECT_ROOT, 'data')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
CONFIG_FILE = os.path.join(PROJECT_ROOT, 'config.json')

# ============================================================================
# LETTURA CONFIGURAZIONE
# ============================================================================

def load_config():
    """Carica configurazione da config.json creato da step0"""
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(
            f"File config.json non trovato in {PROJECT_ROOT}\n"
            "Esegui prima: python scripts/step0_set_target_object.py"
        )
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if 'target_object' not in config:
            raise ValueError("Chiave 'target_object' mancante in config.json")
        
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Errore parsing config.json: {e}")


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(target_object):
    """Configura logging per l'oggetto specifico."""
    log_dir = os.path.join(LOG_DIR, target_object)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'step1_convert_wcs_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ============================================================================
# FUNZIONI CONVERSIONE WCS
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
            ra_deg = (h + m/60.0 + s/3600.0) * 15.0
            
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
    xpixsz = header.get('XPIXSZ', None)  # micron
    focal = header.get('FOCALLEN', header.get('FOCAL', None))  # mm
    xbin = header.get('XBINNING', 1)
    
    if xpixsz and focal:
        pixel_size_mm = (xpixsz * xbin) / 1000.0
        pixel_scale_arcsec = 206.265 * pixel_size_mm / focal
        pixel_scale_deg = pixel_scale_arcsec / 3600.0
        return pixel_scale_deg
    
    # Fallback: stima per setup comune
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
        
        # Tipo proiezione
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
            
            for key in important_keys:
                if key in header:
                    new_header[key] = header[key]
            
            new_header.update(wcs_header)
            
            # Aggiungi metadati preparazione
            new_header['WCSADDED'] = True
            new_header['WCSSRC'] = 'OBJCTRA/OBJCTDEC conversion'
            new_header['WCSDATE'] = datetime.now().isoformat()
            
            # Copia coordinate originali per riferimento
            if 'OBJCTRA' in header:
                new_header['ORIGOBJR'] = header['OBJCTRA']
            if 'OBJCTDEC' in header:
                new_header['ORIGOBJD'] = header['OBJCTDEC']
            
            # Salva
            fits.PrimaryHDU(data=data, header=new_header).writeto(
                output_file,
                overwrite=True,
                output_verify='silentfix'
            )
            
            # Log info
            ra_deg, dec_deg = wcs.wcs.crval
            pixel_scale_arcsec = abs(wcs.wcs.cdelt[0]) * 3600
            
            logger.info(f"✓ {filename}: WCS creato - RA={ra_deg:.4f}°, DEC={dec_deg:.4f}°, scale={pixel_scale_arcsec:.3f}\"/px")
            
            return True
            
    except Exception as e:
        logger.error(f"✗ {os.path.basename(input_file)}: {e}")
        return False


def extract_lith_data(filename, logger):
    """Estrae dati LITH/HST con WCS esistente."""
    try:
        with fits.open(filename, mode='readonly') as hdul:
            sci_data = None
            sci_header = None
            
            # Cerca estensione SCI o primo HDU con dati 2D
            if 'SCI' in hdul:
                sci_data = hdul['SCI'].data
                sci_header = hdul['SCI'].header
            else:
                for hdu in hdul:
                    if hdu.data is not None and len(hdu.data.shape) == 2:
                        sci_data = hdu.data
                        sci_header = hdu.header
                        break
            
            if sci_data is None:
                return None, None, None
            
            wcs = WCS(sci_header)
            if not wcs.has_celestial:
                return None, None, None
            
            shape = sci_data.shape
            ra, dec = wcs.wcs.crval
            
            try:
                cd = wcs.wcs.cd
                pixel_scale = np.sqrt(cd[0,0]**2 + cd[0,1]**2) * 3600
            except:
                pixel_scale = 0.04  # Default per HST
            
            info = {
                'shape': shape,
                'ra': ra,
                'dec': dec,
                'pixel_scale': pixel_scale
            }
            
            logger.info(f"✓ {os.path.basename(filename)}: {shape[1]}×{shape[0]}px, RA={ra:.4f}°, DEC={dec:.4f}°, scale={pixel_scale:.3f}\"/px")
            
            return sci_data, sci_header, info
            
    except Exception as e:
        logger.error(f"✗ {os.path.basename(filename)}: {e}")
        return None, None, None


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def process_local_images(target_object, logger):
    """Processa immagini local (osservatorio) per l'oggetto target."""
    input_dir = os.path.join(BASE_DIR, 'local_raw', target_object)
    output_dir = os.path.join(BASE_DIR, 'img_converted_wcs', 'local', target_object)
    
    if not os.path.exists(input_dir):
        logger.warning(f"Directory input non trovata: {input_dir}")
        return 0, 0, None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova file FITS
    fits_files = (glob.glob(os.path.join(input_dir, '*.fit')) + 
                  glob.glob(os.path.join(input_dir, '*.fits')) +
                  glob.glob(os.path.join(input_dir, '*.FIT')) +
                  glob.glob(os.path.join(input_dir, '*.FITS')))
    
    if not fits_files:
        logger.warning(f"Nessun file FITS in {input_dir}")
        return 0, 0, None
    
    logger.info(f"Trovati {len(fits_files)} file local per {target_object}")
    
    prepared_count = 0
    failed_count = 0
    ra_list = []
    dec_list = []
    scale_list = []
    
    with tqdm(total=len(fits_files), desc=f"  Local/{target_object}", unit="file") as pbar:
        for input_file in fits_files:
            basename = os.path.basename(input_file)
            name, ext = os.path.splitext(basename)
            output_file = os.path.join(output_dir, f"{name}.fits")
            
            success = add_wcs_to_file(input_file, output_file, logger)
            
            if success:
                try:
                    with fits.open(output_file) as hdul:
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
            
            pbar.update(1)
    
    stats = None
    if ra_list:
        stats = {
            'ra_range': (min(ra_list), max(ra_list)),
            'dec_range': (min(dec_list), max(dec_list)),
            'avg_scale': np.mean(scale_list)
        }
    
    return prepared_count, failed_count, stats


def process_light_images(target_object, logger):
    """Processa immagini light (HST/Hubble) per l'oggetto target."""
    input_dir = os.path.join(BASE_DIR, 'img_lights', target_object)
    output_dir = os.path.join(BASE_DIR, 'img_converted_wcs', 'hubble', target_object)
    
    if not os.path.exists(input_dir):
        logger.warning(f"Directory input non trovata: {input_dir}")
        return 0, 0, None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova file FITS
    fits_files = (glob.glob(os.path.join(input_dir, '*.fit')) + 
                  glob.glob(os.path.join(input_dir, '*.fits')) +
                  glob.glob(os.path.join(input_dir, '*.FIT')) +
                  glob.glob(os.path.join(input_dir, '*.FITS')))
    
    if not fits_files:
        logger.warning(f"Nessun file FITS in {input_dir}")
        return 0, 0, None
    
    logger.info(f"Trovati {len(fits_files)} file light per {target_object}")
    
    prepared_count = 0
    failed_count = 0
    ra_list = []
    dec_list = []
    scale_list = []
    
    with tqdm(total=len(fits_files), desc=f"  Light/{target_object}", unit="file") as pbar:
        for input_file in fits_files:
            data, header, info = extract_lith_data(input_file, logger)
            
            if data is not None:
                basename = os.path.basename(input_file)
                name, ext = os.path.splitext(basename)
                output_file = os.path.join(output_dir, f"{name}.fits")
                
                try:
                    primary_hdu = fits.PrimaryHDU(data=data, header=header)
                    primary_hdu.header['ORIGINAL'] = basename
                    primary_hdu.header['PREPDATE'] = datetime.now().isoformat()
                    primary_hdu.header['SOURCE'] = 'light'
                    primary_hdu.header['TARGET'] = target_object
                    
                    primary_hdu.writeto(output_file, overwrite=True, output_verify='silentfix')
                    
                    prepared_count += 1
                    ra_list.append(info['ra'])
                    dec_list.append(info['dec'])
                    scale_list.append(info['pixel_scale'])
                except Exception as e:
                    logger.error(f"Errore salvando {basename}: {e}")
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
# MAIN
# ============================================================================

def main():
    """Funzione principale."""
    print("=" * 70)
    print("🔭 STEP 1: CONVERSIONE COORDINATE → WCS".center(70))
    print("=" * 70)
    
    # Carica configurazione
    try:
        config = load_config()
        target_object = config['target_object']
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        print("\nEsegui prima: python scripts/step0_set_target_object.py")
        return
    
    print(f"\n🎯 Oggetto target: {target_object}")
    print(f"   (da config.json)")
    
    # Setup logging
    logger = setup_logging(target_object)
    logger.info("=" * 60)
    logger.info(f"STEP 1: CONVERSIONE WCS - {target_object}")
    logger.info("=" * 60)
    
    # Percorsi
    print(f"\n� Struttura output:")
    print(f"   Local:  data/img_converted_wcs/local/{target_object}/")
    print(f"   Light:  data/img_converted_wcs/hubble/{target_object}/")
    
    # Process LOCAL images
    print(f"\n📡 LOCAL (Osservatorio): Conversione OBJCTRA/OBJCTDEC → WCS")
    prep_local, fail_local, stats_local = process_local_images(target_object, logger)
    
    print(f"\n   ✓ Processati: {prep_local}")
    print(f"   ✗ Falliti: {fail_local}")
    
    if stats_local:
        ra_min, ra_max = stats_local['ra_range']
        dec_min, dec_max = stats_local['dec_range']
        print(f"\n   📊 Campo:")
        print(f"      RA: {ra_min:.4f}° → {ra_max:.4f}° (span: {ra_max-ra_min:.4f}°)")
        print(f"      DEC: {dec_min:.4f}° → {dec_max:.4f}° (span: {dec_max-dec_min:.4f}°)")
        print(f"      Scala media: {stats_local['avg_scale']:.3f}\"/px")
    
    # Process LIGHT images
    print(f"\n🛰️  LIGHT (HST/Hubble): Estrazione WCS esistente")
    prep_light, fail_light, stats_light = process_light_images(target_object, logger)
    
    print(f"\n   ✓ Processati: {prep_light}")
    print(f"   ✗ Falliti: {fail_light}")
    
    if stats_light:
        ra_min, ra_max = stats_light['ra_range']
        dec_min, dec_max = stats_light['dec_range']
        print(f"\n   📊 Campo:")
        print(f"      RA: {ra_min:.4f}° → {ra_max:.4f}° (span: {ra_max-ra_min:.4f}°)")
        print(f"      DEC: {dec_min:.4f}° → {dec_max:.4f}° (span: {dec_max-dec_min:.4f}°)")
        print(f"      Scala media: {stats_light['avg_scale']:.3f}\"/px")
    
    # RIEPILOGO
    total_prep = prep_local + prep_light
    total_fail = fail_local + fail_light
    
    print("\n" + "=" * 70)
    print("📊 RIEPILOGO")
    print("=" * 70)
    print(f"   Oggetto: {target_object}")
    print(f"   Local: {prep_local} OK, {fail_local} falliti")
    print(f"   Hubble: {prep_light} OK, {fail_light} falliti")
    print(f"   TOTALE: {total_prep} preparati, {total_fail} falliti")
    
    logger.info(f"Totale: {total_prep} preparati, {total_fail} falliti")
    
    if total_prep > 0:
        print(f"\n✅ STEP 1 COMPLETATO!")
        print(f"\n   📁 File con WCS in:")
        print(f"      • data/img_converted_wcs/local/{target_object}/")
        print(f"      • data/img_converted_wcs/hubble/{target_object}/")
        print(f"\n   ➡️  Prossimo passo: python scripts/step2_analize.py")
    else:
        print(f"\n⚠️  Nessun file processato per {target_object}")
        print(f"   Verifica che le immagini siano in:")
        print(f"      • data/local_raw/{target_object}/")
        print(f"      • data/img_lights/{target_object}/")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️  Tempo: {elapsed:.2f}s")