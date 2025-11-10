"""
STEP 1: PREPARAZIONE IMMAGINI - CONVERSIONE WCS
Converte coordinate esistenti (OBJCTRA/OBJCTDEC) in WCS standard
Processa TUTTI gli oggetti trovati nelle cartelle input
Output organizzato per fonte/oggetto per integrazione con step2
"""

import os
import glob
import time
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

# Output: organizzato per fonte/oggetto (compatibile con step2)
OUTPUT_WCS_DIR = os.path.join(BASE_DIR, 'img_converted_wcs')

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    """Configura logging generale."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'step1_convert_wcs_{timestamp}.log')
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"LOG FILE: {log_filename}")
    logger.info("=" * 80)
    
    return logger


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


def add_wcs_to_file(input_file, output_file, target_object, logger):
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
                # Copia comunque per aggiungere metadati
                data_to_save = data
                header_to_save = header
            else:
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
                
                data_to_save = data
                header_to_save = new_header
                
                # Log info
                ra_deg, dec_deg = wcs.wcs.crval
                pixel_scale_arcsec = abs(wcs.wcs.cdelt[0]) * 3600
                logger.info(f"✓ {filename}: WCS creato - RA={ra_deg:.4f}°, DEC={dec_deg:.4f}°, scale={pixel_scale_arcsec:.3f}\"/px")
            
            # Aggiungi metadati target
            header_to_save['TARGET'] = target_object
            header_to_save['SOURCE'] = 'local'
            header_to_save['PREPDATE'] = datetime.now().isoformat()
            
            # Salva
            fits.PrimaryHDU(data=data_to_save, header=header_to_save).writeto(
                output_file,
                overwrite=True,
                output_verify='silentfix'
            )
            
            return True
            
    except Exception as e:
        logger.error(f"✗ {os.path.basename(input_file)}: {e}")
        return False


def extract_hubble_data(filename, target_object, logger):
    """Estrae dati Hubble con WCS esistente."""
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
                logger.warning(f"Nessun dato 2D in {os.path.basename(filename)}")
                return None, None, None
            
            wcs = WCS(sci_header)
            if not wcs.has_celestial:
                logger.warning(f"WCS non valido in {os.path.basename(filename)}")
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
# DISCOVERY OGGETTI
# ============================================================================

def discover_objects():
    """Scopre automaticamente tutti gli oggetti nelle cartelle input."""
    objects = set()
    
    # Cerca in local_raw
    local_raw_dir = os.path.join(BASE_DIR, 'local_raw')
    if os.path.exists(local_raw_dir):
        for item in os.listdir(local_raw_dir):
            item_path = os.path.join(local_raw_dir, item)
            if os.path.isdir(item_path):
                objects.add(item)
    
    # Cerca in img_lights
    img_lights_dir = os.path.join(BASE_DIR, 'img_lights')
    if os.path.exists(img_lights_dir):
        for item in os.listdir(img_lights_dir):
            item_path = os.path.join(img_lights_dir, item)
            if os.path.isdir(item_path):
                objects.add(item)
    
    return sorted(list(objects))


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def process_local_images(target_object, logger):
    """Processa immagini local (osservatorio) per l'oggetto target."""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"📡 PROCESSING LOCAL IMAGES: {target_object}")
    logger.info("=" * 80)
    
    input_dir = os.path.join(BASE_DIR, 'local_raw', target_object)
    output_dir = os.path.join(OUTPUT_WCS_DIR, 'local', target_object)
    
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info("")
    
    if not os.path.exists(input_dir):
        logger.warning(f"⚠️  Directory input non trovata: {input_dir}")
        return 0, 0, None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova file FITS
    fits_files = (glob.glob(os.path.join(input_dir, '*.fit')) + 
                  glob.glob(os.path.join(input_dir, '*.fits')) +
                  glob.glob(os.path.join(input_dir, '*.FIT')) +
                  glob.glob(os.path.join(input_dir, '*.FITS')))
    
    if not fits_files:
        logger.warning(f"⚠️  Nessun file FITS in {input_dir}")
        return 0, 0, None
    
    logger.info(f"📊 Trovati {len(fits_files)} file FITS")
    logger.info("")
    
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
            
            success = add_wcs_to_file(input_file, output_file, target_object, logger)
            
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
    
    logger.info("")
    logger.info(f"✅ Successi: {prepared_count}")
    logger.info(f"❌ Errori: {failed_count}")
    
    stats = None
    if ra_list:
        ra_min, ra_max = min(ra_list), max(ra_list)
        dec_min, dec_max = min(dec_list), max(dec_list)
        avg_scale = np.mean(scale_list)
        
        stats = {
            'ra_range': (ra_min, ra_max),
            'dec_range': (dec_min, dec_max),
            'avg_scale': avg_scale
        }
        
        logger.info("")
        logger.info("📐 STATISTICHE CAMPO:")
        logger.info(f"  RA:  {ra_min:.6f}° → {ra_max:.6f}° (span: {(ra_max-ra_min)*60:.2f}')")
        logger.info(f"  DEC: {dec_min:.6f}° → {dec_max:.6f}° (span: {(dec_max-dec_min)*60:.2f}')")
        logger.info(f"  Scala media: {avg_scale:.3f}\"/px")
    
    return prepared_count, failed_count, stats


def process_hubble_images(target_object, logger):
    """Processa immagini Hubble per l'oggetto target."""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"🛰️  PROCESSING HUBBLE IMAGES: {target_object}")
    logger.info("=" * 80)
    
    input_dir = os.path.join(BASE_DIR, 'img_lights', target_object)
    output_dir = os.path.join(OUTPUT_WCS_DIR, 'hubble', target_object)
    
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info("")
    
    if not os.path.exists(input_dir):
        logger.warning(f"⚠️  Directory input non trovata: {input_dir}")
        return 0, 0, None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova file FITS
    fits_files = (glob.glob(os.path.join(input_dir, '*.fit')) + 
                  glob.glob(os.path.join(input_dir, '*.fits')) +
                  glob.glob(os.path.join(input_dir, '*.FIT')) +
                  glob.glob(os.path.join(input_dir, '*.FITS')))
    
    if not fits_files:
        logger.warning(f"⚠️  Nessun file FITS in {input_dir}")
        return 0, 0, None
    
    logger.info(f"📊 Trovati {len(fits_files)} file FITS")
    logger.info("")
    
    prepared_count = 0
    failed_count = 0
    ra_list = []
    dec_list = []
    scale_list = []
    
    with tqdm(total=len(fits_files), desc=f"  Hubble/{target_object}", unit="file") as pbar:
        for input_file in fits_files:
            data, header, info = extract_hubble_data(input_file, target_object, logger)
            
            if data is not None:
                basename = os.path.basename(input_file)
                name, ext = os.path.splitext(basename)
                output_file = os.path.join(output_dir, f"{name}.fits")
                
                try:
                    primary_hdu = fits.PrimaryHDU(data=data, header=header)
                    primary_hdu.header['ORIGINAL'] = basename
                    primary_hdu.header['PREPDATE'] = datetime.now().isoformat()
                    primary_hdu.header['SOURCE'] = 'hubble'
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
    
    logger.info("")
    logger.info(f"✅ Successi: {prepared_count}")
    logger.info(f"❌ Errori: {failed_count}")
    
    stats = None
    if ra_list:
        ra_min, ra_max = min(ra_list), max(ra_list)
        dec_min, dec_max = min(dec_list), max(dec_list)
        avg_scale = np.mean(scale_list)
        
        stats = {
            'ra_range': (ra_min, ra_max),
            'dec_range': (dec_min, dec_max),
            'avg_scale': avg_scale
        }
        
        logger.info("")
        logger.info("📐 STATISTICHE CAMPO:")
        logger.info(f"  RA:  {ra_min:.6f}° → {ra_max:.6f}° (span: {(ra_max-ra_min)*60:.2f}')")
        logger.info(f"  DEC: {dec_min:.6f}° → {dec_max:.6f}° (span: {(dec_max-dec_min)*60:.2f}')")
        logger.info(f"  Scala media: {avg_scale:.3f}\"/px")
    
    return prepared_count, failed_count, stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Funzione principale."""
    print("=" * 70)
    print("🔭 STEP 1: CONVERSIONE COORDINATE → WCS".center(70))
    print("=" * 70)
    
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("STEP 1: CONVERSIONE WCS - TUTTI GLI OGGETTI")
    logger.info("=" * 80)
    logger.info(f"Data/Ora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # Scopri oggetti automaticamente
    print(f"\n🔍 Ricerca automatica oggetti...")
    objects = discover_objects()
    
    if not objects:
        print(f"\n❌ Nessun oggetto trovato!")
        print(f"   Verifica che le cartelle esistano:")
        print(f"      • data/local_raw/OGGETTO/")
        print(f"      • data/img_lights/OGGETTO/")
        logger.error("Nessun oggetto trovato nelle directory input")
        return
    
    print(f"\n📦 Oggetti trovati: {', '.join(objects)}")
    logger.info(f"Oggetti da processare: {', '.join(objects)}")
    logger.info("")
    
    # Percorsi
    print(f"\n📁 Struttura output: data/img_converted_wcs/{{fonte}}/{{oggetto}}/")
    
    # Process tutti gli oggetti
    total_prep_local = 0
    total_fail_local = 0
    total_prep_hubble = 0
    total_fail_hubble = 0
    
    for obj_idx, target_object in enumerate(objects, 1):
        print(f"\n{'='*70}")
        print(f"📦 OGGETTO {obj_idx}/{len(objects)}: {target_object}")
        print(f"{'='*70}")
        
        logger.info("=" * 80)
        logger.info(f"OGGETTO {obj_idx}/{len(objects)}: {target_object}")
        logger.info("=" * 80)
        
        # Process LOCAL
        print(f"\n📡 LOCAL (Osservatorio)")
        prep_local, fail_local, stats_local = process_local_images(target_object, logger)
        
        total_prep_local += prep_local
        total_fail_local += fail_local
        
        print(f"   ✓ Processati: {prep_local}")
        print(f"   ✗ Falliti: {fail_local}")
        
        if stats_local:
            ra_min, ra_max = stats_local['ra_range']
            dec_min, dec_max = stats_local['dec_range']
            print(f"   📊 Campo: RA {(ra_max-ra_min)*60:.2f}' × DEC {(dec_max-dec_min)*60:.2f}', scala {stats_local['avg_scale']:.3f}\"/px")
        
        # Process HUBBLE
        print(f"\n🛰️  HUBBLE")
        prep_hubble, fail_hubble, stats_hubble = process_hubble_images(target_object, logger)
        
        total_prep_hubble += prep_hubble
        total_fail_hubble += fail_hubble
        
        print(f"   ✓ Processati: {prep_hubble}")
        print(f"   ✗ Falliti: {fail_hubble}")
        
        if stats_hubble:
            ra_min, ra_max = stats_hubble['ra_range']
            dec_min, dec_max = stats_hubble['dec_range']
            print(f"   📊 Campo: RA {(ra_max-ra_min)*60:.2f}' × DEC {(dec_max-dec_min)*60:.2f}', scala {stats_hubble['avg_scale']:.3f}\"/px")
    
    # RIEPILOGO TOTALE
    total_prep = total_prep_local + total_prep_hubble
    total_fail = total_fail_local + total_fail_hubble
    
    print("\n" + "=" * 70)
    print("📊 RIEPILOGO TOTALE")
    print("=" * 70)
    print(f"   Oggetti processati: {len(objects)}")
    print(f"   Local:  {total_prep_local} OK, {total_fail_local} falliti")
    print(f"   Hubble: {total_prep_hubble} OK, {total_fail_hubble} falliti")
    print(f"   TOTALE: {total_prep} preparati, {total_fail} falliti")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 RIEPILOGO FINALE")
    logger.info("=" * 80)
    logger.info(f"Oggetti processati: {len(objects)}")
    logger.info(f"Local:  {total_prep_local} OK, {total_fail_local} falliti")
    logger.info(f"Hubble: {total_prep_hubble} OK, {total_fail_hubble} falliti")
    logger.info(f"TOTALE: {total_prep} preparati, {total_fail} falliti")
    logger.info(f"Data/Ora fine: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    if total_prep > 0:
        print(f"\n✅ STEP 1 COMPLETATO!")
        print(f"\n   📁 File con WCS organizzati per fonte/oggetto in:")
        print(f"      data/img_converted_wcs/")
        print(f"\n   ➡️  Prossimo passo: python scripts/step2_register.py")
        print(f"       (usa menu interattivo per scegliere fonte/oggetto)")
    else:
        print(f"\n⚠️  Nessun file processato!")
        print(f"   Verifica che le immagini siano in:")
        print(f"      • data/local_raw/{{oggetto}}/")
        print(f"      • data/img_lights/{{oggetto}}/")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️  Tempo totale: {elapsed:.2f}s")