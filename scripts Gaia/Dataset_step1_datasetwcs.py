import os
import sys
import time
import logging
from datetime import datetime
import numpy as np
import astropy
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings
from pathlib import Path
import subprocess

# Gestione importazioni opzionali
try:
    from reproject import reproject_interp
    REPROJECT_AVAILABLE = True
except ImportError:
    REPROJECT_AVAILABLE = False
    print("⚠️ Libreria 'reproject' non trovata. La registrazione fallirà.")

try:
    import astroalign as aa
    ASTROALIGN_AVAILABLE = True
except ImportError:
    ASTROALIGN_AVAILABLE = False

warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR_ROOT = ROOT_DATA_DIR / "logs"

NUM_THREADS = 6 
log_lock = threading.Lock()

# ================= UTILITY & SETUP =================

def setup_logging():
    os.makedirs(LOG_DIR_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR_ROOT / f'pipeline_smart_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def select_target_directory():
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET".center(70))
    print("📂"*35)
    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    except: return []
    if not subdirs: return []
    
    print("\nTarget disponibili:")
    for i, d in enumerate(subdirs): print(f" {i+1}: {d.name}")
    try:
        idx = int(input("Scelta: ")) - 1
        return [subdirs[idx]] if 0 <= idx < len(subdirs) else []
    except: return []

# ================= STEP 1: WCS CONVERSION =================

def parse_coordinates(ra_str, dec_str):
    try:
        c = SkyCoord(f"{ra_str} {dec_str}", unit=(u.hourangle, u.deg))
        return c.ra.deg, c.dec.deg
    except: raise ValueError

def calculate_pixel_scale(header):
    xpixsz = header.get('XPIXSZ')
    focal = header.get('FOCALLEN', header.get('FOCAL'))
    if xpixsz and focal:
        # Fix formula scala: mm/mm * 206265
        return ((xpixsz * header.get('XBINNING', 1)) / 1000.0 / focal) * 206265.0 / 3600.0
    return 1.5 / 3600.0

def create_wcs_from_header_robust(header, shape):
    """
    Tenta di creare un WCS valido leggendo vari formati di header.
    """
    # TENTATIVO 1: Se l'header ha già chiavi WCS standard (CRVAL1, CD1_1, etc.)
    try:
        w = WCS(header)
        if w.has_celestial:
            return w
    except:
        pass

    # TENTATIVO 2: Coordinate testuali (OBJCTRA/DEC) - Formato Amatoriale
    try:
        if 'OBJCTRA' in header and 'OBJCTDEC' in header:
            ra, dec = parse_coordinates(header['OBJCTRA'], header['OBJCTDEC'])
            scale = calculate_pixel_scale(header)
            w = WCS(naxis=2)
            w.wcs.crpix = [shape[1]/2, shape[0]/2]
            w.wcs.crval = [ra, dec]
            w.wcs.cdelt = [-scale, scale]
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            return w
    except:
        pass

    # TENTATIVO 3: Coordinate Decimali dirette (RA/DEC) - Formato MaxIm DL/TheSkyX
    try:
        if 'RA' in header and 'DEC' in header:
            # A volte sono in gradi, a volte stringhe. Proviamo float diretto.
            w = WCS(naxis=2)
            w.wcs.crpix = [shape[1]/2, shape[0]/2]
            w.wcs.crval = [float(header['RA']), float(header['DEC'])]
            scale = calculate_pixel_scale(header)
            w.wcs.cdelt = [-scale, scale]
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            return w
    except:
        pass

    return None

def add_wcs_to_file(inp, out, log):
    try:
        # Apri il file. Gestisce file con estensioni multiple cercando la prima immagine valida.
        with fits.open(inp) as hdul:
            # Cerca il primo HDU che contiene dati immagine (2D o 3D)
            valid_hdu = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim >= 2:
                    valid_hdu = hdu
                    break
            
            if valid_hdu is None: 
                # log.warning(f"Nessun dato immagine trovato in {inp.name}")
                return False

            data = valid_hdu.data
            header = valid_hdu.header.copy()

            # Se ha già un WCS valido, salva e basta
            try:
                existing_wcs = WCS(header)
                if existing_wcs.has_celestial:
                    fits.PrimaryHDU(data=data, header=header).writeto(out, overwrite=True)
                    return True
            except: pass
            
            # Altrimenti prova a crearlo dai metadati
            w = create_wcs_from_header_robust(header, data.shape)
            
            if not w:
                # Ultimo tentativo: Cerca chiavi nell'header primario se siamo in un'estensione
                if '0' in hdul: # Se esiste un PrimaryHDU separato
                    w = create_wcs_from_header_robust(hdul[0].header, data.shape)
            
            if not w: 
                return False
            
            # Aggiorna header
            header.update(w.to_header())
            fits.PrimaryHDU(data=data, header=header).writeto(out, overwrite=True)
            return True

    except Exception as e:
        # log.error(f"Errore elaborazione {inp.name}: {e}")
        return False

def process_step1_folder(inp_dir, out_dir, logger):
    # Cerca fits, FIT, fit, FITS
    files = list(inp_dir.glob('*.fit')) + list(inp_dir.glob('*.fits')) + \
            list(inp_dir.glob('*.FIT')) + list(inp_dir.glob('*.FITS'))
    files = sorted(list(set(files)))
    
    if not files:
        logger.warning(f"Nessun file trovato in {inp_dir}")
        return 0

    success = 0
    for f in tqdm(files, desc=f"WCS {inp_dir.name}"):
        if add_wcs_to_file(f, out_dir / f"{f.stem}_wcs.fits", logger): success += 1
    return success

# ================= STEP 2: REGISTRAZIONE SMART =================

def extract_wcs_info(f, logger=None):
    try:
        with fits.open(f) as h:
            w = WCS(h[0].header)
            if not w.has_celestial: return None
            return {'file': f, 'wcs': w, 'shape': h[0].data.shape, 
                    'ra': w.wcs.crval[0], 'dec': w.wcs.crval[1],
                    'scale': abs(w.wcs.cdelt[0])*3600}
    except: return None

def register_single_image_smart(info, ref_wcs, ref_data_h, out_dir, logger):
    try:
        fname = info['file'].name
        with fits.open(info['file']) as hdul:
            data = np.nan_to_num(hdul[0].data)
            header = hdul[0].header
            wcs_orig = WCS(header)
            if data.ndim == 3: data = data[0]

        # Target WCS: Usa il centro del Master ma mantiene la scala nativa dell'immagine
        native_scale = info['scale'] / 3600.0
        target_wcs = WCS(naxis=2)
        target_wcs.wcs.crval = ref_wcs.wcs.crval # Centriamo tutto sul master comune
        target_wcs.wcs.ctype = ref_wcs.wcs.ctype
        target_wcs.wcs.cdelt = [-native_scale, native_scale]
        target_wcs.wcs.crpix = [data.shape[1]/2, data.shape[0]/2] 
        
        output_data = None
        method = "WCS_Reproject"

        if output_data is None:
            output_data, _ = reproject_interp((data, wcs_orig), target_wcs, shape_out=data.shape)

        # Salva il file registrato
        out_name = f"reg_{fname}"
        hdr_new = target_wcs.to_header()
        hdr_new['REG_METH'] = method
        fits.PrimaryHDU(data=output_data.astype(np.float32), header=hdr_new).writeto(out_dir/out_name, overwrite=True)
        
        return {'status': 'ok', 'file': out_name}

    except Exception as e:
        return {'status': 'err', 'file': fname, 'err': str(e)}

def main_registration(h_in, o_in, h_out, o_out, logger):
    # 1. Trova i file generati dallo Step 1
    h_files = list(h_in.glob('*.fits'))
    o_files = list(o_in.glob('*.fits'))
    
    # 2. Estrai Info WCS
    h_infos = [extract_wcs_info(f, logger) for f in h_files]
    o_infos = [extract_wcs_info(f, logger) for f in o_files]
    
    # Filtra i None (file corrotti)
    h_infos = [x for x in h_infos if x]
    o_infos = [x for x in o_infos if x]
    
    if not h_infos: 
        logger.error("Nessun file Hubble valido per la registrazione (Step 1 fallito o file vuoti).")
        return False
    
    # Usa il primo Hubble come "Master Reference" per il centro WCS
    common_wcs = h_infos[0]['wcs']
    
    print(f"   Avvio registrazione su {len(h_infos)} file Hubble e {len(o_infos)} file Osservatorio...")
    
    success = 0
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as ex:
        futures = []
        
        # Registra Hubble (HR) -> h_out
        for info in h_infos:
            futures.append(ex.submit(register_single_image_smart, info, common_wcs, None, h_out, logger))
            
        # Registra Osservatorio (LR) -> o_out
        for info in o_infos:
            futures.append(ex.submit(register_single_image_smart, info, common_wcs, None, o_out, logger))
            
        for f in tqdm(as_completed(futures), total=len(futures), desc="Registrazione"):
            res = f.result()
            if res['status'] == 'ok': 
                success += 1
            else:
                logger.warning(f"Errore su {res['file']}: {res['err']}")
            
    return success > 0

# ================= MAIN =================

def main():
    logger = setup_logging()
    targets = select_target_directory()
    if not targets: return

    for BASE_DIR in targets:
        print(f"\n🚀 ELABORAZIONE: {BASE_DIR.name}")
        
        # Paths
        in_o = BASE_DIR / '1_originarie/local_raw'
        in_h = BASE_DIR / '1_originarie/img_lights'
        
        # Output Step 1 (Temporanei)
        out_wcs_o = BASE_DIR / '2_wcs/observatory'
        out_wcs_h = BASE_DIR / '2_wcs/hubble'
        
        # Output Step 2 (Finali per estrazione patch)
        out_reg_o = BASE_DIR / '3_registered_native/observatory'
        out_reg_h = BASE_DIR / '3_registered_native/hubble'
        
        for p in [out_wcs_o, out_wcs_h, out_reg_o, out_reg_h]: p.mkdir(parents=True, exist_ok=True)

        # Step 1: Conversione WCS
        print("   [1/2] Conversione WCS...")
        s1 = process_step1_folder(in_o, out_wcs_o, logger)
        s2 = process_step1_folder(in_h, out_wcs_h, logger)
        
        if s1+s2 == 0:
            print("   ❌ Nessun file convertito (Input vuoto o errori). Salto.")
            continue

        # Step 2: Registrazione
        print("   [2/2] Registrazione...")
        # Input per step 2 sono gli output di step 1
        if main_registration(out_wcs_h, out_wcs_o, out_reg_h, out_reg_o, logger):
            print("   ✅ Registrazione Completata. File pronti in 3_registered_native.")
        else:
            print("   ❌ Errore Registrazione.")

if __name__ == "__main__":
    main()