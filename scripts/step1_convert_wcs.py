"""
STEP 1: PREPARAZIONE IMMAGINI - CONVERSIONE WCS
Converte coordinate esistenti (OBJCTRA/OBJCTDEC) in WCS standard
Script generalizzato per tutti i target.
"""

import os
import glob
import time
import logging
# import argparse # Rimosso
from datetime import datetime
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = r'F:\Super Revolt Gaia\SuperResolution\SuperResolution\data'
# I percorsi di input ora puntano alle cartelle radice
INPUT_OSSERVATORIO = os.path.join(BASE_DIR, 'local_raw')
INPUT_LITH = os.path.join(BASE_DIR, 'img_lights_1')
OUTPUT_OSSERVATORIO = os.path.join(BASE_DIR, 'osservatorio_con_wcs')
OUTPUT_LITH = os.path.join(BASE_DIR, 'lith_con_wcs')
LOG_DIR = r'F:\Super Revolt Gaia\logs'


def setup_logging():
    """Configura logging generico."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'preparation_wcs_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


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
            ra_deg = (h + m/60.0 + s/3600.0) * 15.0  # * 15 per convertire ore in gradi
            
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
    # Estrai parametri
    xpixsz = header.get('XPIXSZ', None)  # micron
    focal = header.get('FOCALLEN', header.get('FOCAL', None))  # mm
    xbin = header.get('XBINNING', 1)
    
    if xpixsz and focal:
        # Formula: pixel_scale (arcsec/px) = 206.265 * pixel_size (mm) / focal_length (mm)
        pixel_size_mm = (xpixsz * xbin) / 1000.0  # converti micron -> mm e applica binning
        pixel_scale_arcsec = 206.265 * pixel_size_mm / focal
        pixel_scale_deg = pixel_scale_arcsec / 3600.0
        return pixel_scale_deg
    
    # Fallback: stima per setup comune
    # Tipico per piccoli telescopi: ~1-2 arcsec/pixel
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
        
        # Tipo proiezione (TAN = tangente, standard per campi piccoli)
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
    
    Args:
        input_file: File input
        output_file: File output con WCS
        logger: Logger
    
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
                # Copia comunque il file
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
            
            # Prima i campi base
            for key in important_keys:
                if key in header:
                    new_header[key] = header[key]
            
            # Poi il WCS
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
        logger.error(f"✗ {filename}: {e}")
        return False


def extract_lith_data(filename, logger):
    """Estrae dati LITH/HST."""
    try:
        with fits.open(filename, mode='readonly') as hdul:
            sci_data = None
            sci_header = None
            
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
                pixel_scale = 0.04
            
            info = {
                'shape': shape,
                'ra': ra,
                'dec': dec,
                'pixel_scale': pixel_scale
            }
            
            logger.info(f"✓ {os.path.basename(filename)}: {shape[1]}x{shape[0]}px, RA={ra:.4f}°, DEC={dec:.4f}°")
            
            return sci_data, sci_header, info
            
    except Exception as e:
        logger.error(f"✗ {os.path.basename(filename)}: {e}")
        return None, None, None


def process_osservatorio_folder(input_dir, output_dir, logger):
    """Processa osservatorio convertendo coordinate in WCS."""
    # Modificato per cercare ricorsivamente in tutte le sottocartelle
    fits_files = glob.glob(os.path.join(input_dir, '**', '*.fit'), recursive=True) + \
                 glob.glob(os.path.join(input_dir, '**', '*.fits'), recursive=True)
    
    if not fits_files:
        logger.warning(f"Nessun file in {input_dir}")
        return 0, 0, None
    
    logger.info(f"Trovati {len(fits_files)} file osservatorio")
    
    prepared_count = 0
    failed_count = 0
    ra_list = []
    dec_list = []
    scale_list = []
    
    with tqdm(total=len(fits_files), desc="  Osservatorio", unit="file") as pbar:
        for input_file in fits_files:
            basename = os.path.basename(input_file)
            name, ext = os.path.splitext(basename)
            output_file = os.path.join(output_dir, f"{name}_wcs.fits")
            
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


def process_lith_folder(input_dir, output_dir, logger):
    """Processa LITH/HST."""
    # Modificato per cercare ricorsivamente in tutte le sottocartelle
    fits_files = glob.glob(os.path.join(input_dir, '**', '*.fit'), recursive=True) + \
                 glob.glob(os.path.join(input_dir, '**', '*.fits'), recursive=True)
    
    if not fits_files:
        return 0, 0, None
    
    logger.info(f"Trovati {len(fits_files)} file LITH")
    
    prepared_count = 0
    failed_count = 0
    ra_list = []
    dec_list = []
    scale_list = []
    
    with tqdm(total=len(fits_files), desc="  LITH", unit="file") as pbar:
        for input_file in fits_files:
            data, header, info = extract_lith_data(input_file, logger)
            
            if data is not None:
                basename = os.path.basename(input_file)
                name, ext = os.path.splitext(basename)
                output_file = os.path.join(output_dir, f"{name}_wcs.fits")
                
                try:
                    primary_hdu = fits.PrimaryHDU(data=data, header=header)
                    primary_hdu.header['ORIGINAL'] = basename
                    primary_hdu.header['PREPDATE'] = datetime.now().isoformat()
                    primary_hdu.header['SOURCE'] = 'lith'
                    
                    primary_hdu.writeto(output_file, overwrite=True, output_verify='silentfix')
                    
                    prepared_count += 1
                    ra_list.append(info['ra'])
                    dec_list.append(info['dec'])
                    scale_list.append(info['pixel_scale'])
                except Exception as e:
                    logger.error(f"Errore {basename}: {e}")
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


def main():
    """Funzione principale."""
    # Rimosso input interattivo
    # TARGET_NAME = input("Inserisci il nome del target (es. M33, M31): ")
    # ...
    # TARGET_NAME = TARGET_NAME.strip()
    
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info(f"PREPARAZIONE: CONVERSIONE WCS DA COORDINATE (Tutti i target)")
    logger.info("=" * 60)
    
    print("=" * 70)
    print(f"🔭 PREPARAZIONE: CONVERSIONE COORDINATE → WCS (Tutti i target)".center(70))
    print("=" * 70)
    
    # I percorsi di input sono ora definiti globalmente
    print(f"\nInput Osservatorio (ricorsivo): {INPUT_OSSERVATORIO}")
    print(f"Input LITH/HST (ricorsivo): {INPUT_LITH}")
    
    # Setup
    os.makedirs(OUTPUT_OSSERVATORIO, exist_ok=True)
    os.makedirs(OUTPUT_LITH, exist_ok=True)
    
    # OSSERVATORIO
    print("\n📡 OSSERVATORIO (Conversione OBJCTRA/OBJCTDEC → WCS)")
    
    prep_oss, fail_oss, stats_oss = process_osservatorio_folder(
        INPUT_OSSERVATORIO,
        OUTPUT_OSSERVATORIO,
        logger
    )
    
    print(f"\n   ✓ Processati: {prep_oss}")
    print(f"   ✗ Falliti: {fail_oss}")
    
    if stats_oss:
        ra_min, ra_max = stats_oss['ra_range']
        dec_min, dec_max = stats_oss['dec_range']
        print(f"\n   📊 Campo:")
        print(f"      RA: {ra_min:.4f}° → {ra_max:.4f}° (span: {ra_max-ra_min:.4f}°)")
        print(f"      DEC: {dec_min:.4f}° → {dec_max:.4f}° (span: {dec_max-dec_min:.4f}°)")
        print(f"      Scala media: {stats_oss['avg_scale']:.3f}\"/px")
    
    # LITH
    print("\n🛰️  LITH/HST (Estrazione WCS esistente)")
    
    prep_lith, fail_lith, stats_lith = process_lith_folder(
        INPUT_LITH,
        OUTPUT_LITH,
        logger
    )
    
    print(f"\n   ✓ Processati: {prep_lith}")
    print(f"   ✗ Falliti: {fail_lith}")
    
    if stats_lith:
        ra_min, ra_max = stats_lith['ra_range']
        dec_min, dec_max = stats_lith['dec_range']
        print(f"\n   📊 Campo:")
        print(f"      RA: {ra_min:.4f}° → {ra_max:.4f}° (span: {ra_max-ra_min:.4f}°)")
        print(f"      DEC: {dec_min:.4f}° → {dec_max:.4f}° (span: {dec_max-dec_min:.4f}°)")
        print(f"      Scala media: {stats_lith['avg_scale']:.3f}\"/px")
    
    # RIEPILOGO
    total_prep = prep_oss + prep_lith
    total_fail = fail_oss + fail_lith
    
    print("\n" + "=" * 70)
    print("📊 RIEPILOGO")
    print("=" * 70)
    print(f"   Osservatorio: {prep_oss} OK, {fail_oss} falliti")
    print(f"   LITH: {prep_lith} OK, {fail_lith} falliti")
    print(f"   TOTALE: {total_prep} preparati, {total_fail} falliti")
    
    logger.info(f"Totale: {total_prep} preparati, {total_fail} falliti")
    
    if total_prep > 0:
        print(f"\n✅ COMPLETATO!")
        print(f"\n   📁 File con WCS in:")
        print(f"      • {OUTPUT_OSSERVATORIO}")
        print(f"      • {OUTPUT_LITH}")
        print(f"\n   ➡️  Prossimo passo: Esegui step 2 e 3")
    else:
        print(f"\n⚠️ Nessun file processato.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️ Tempo: {elapsed:.2f}s")