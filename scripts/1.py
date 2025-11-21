"""
STEP 1 & 2: PIPELINE UNIFICATA (MULTI-THREADING UNLIMITED)
Ottimizzato per usare il 100% della CPU durante la riproiezione.
"""

import os
import sys
import time
import logging
from datetime import datetime
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings
from pathlib import Path

# Configurazione Hardware Automatica
NUM_THREADS = os.cpu_count() or 16  # Usa TUTTI i core
print(f"🚀 Rilevati {NUM_THREADS} Core CPU - Attivazione Multi-Threading Totale")

warnings.filterwarnings('ignore')

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR_ROOT = ROOT_DATA_DIR / "logs"

try:
    from reproject import reproject_interp
    REPROJECT_AVAILABLE = True
except ImportError:
    REPROJECT_AVAILABLE = False
    print("❌ MANCA 'reproject'. Installa: pip install reproject")

# Lock veloce solo per scrivere su disco/log
io_lock = threading.Lock()

# ============================================================================
# LOGGING & SETUP
# ============================================================================
def setup_logging():
    os.makedirs(LOG_DIR_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[logging.FileHandler(LOG_DIR_ROOT / f'pipeline_{timestamp}.log'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

# ============================================================================
# UTILS WCS (Thread Safe)
# ============================================================================
def parse_coordinates(ra_str, dec_str):
    try:
        coord = SkyCoord(f"{ra_str} {dec_str}", unit=(u.hourangle, u.deg))
        return coord.ra.degree, coord.dec.degree
    except:
        return 0.0, 0.0

def create_wcs_from_header(header, shape):
    try:
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [shape[1]/2.0, shape[0]/2.0]
        # Recupera coordinate o usa default
        ra_txt = header.get('OBJCTRA', '00 00 00')
        dec_txt = header.get('OBJCTDEC', '00 00 00')
        ra, dec = parse_coordinates(ra_txt, dec_txt)
        wcs.wcs.crval = [ra, dec]
        
        # Pixel Scale stima
        pix_sz = header.get('XPIXSZ', 3.75) 
        focal = header.get('FOCALLEN', header.get('FOCAL', 1000))
        scale = (206.265 * (pix_sz/1000.0) / focal) / 3600.0
        wcs.wcs.cdelt = [-scale, scale]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return wcs
    except: return None

# ============================================================================
# STEP 1: WCS ADDITION (Threaded)
# ============================================================================
def process_single_wcs(args):
    """Worker per aggiungere WCS a un singolo file."""
    filepath, output_dir, mode = args
    fname = filepath.name
    out_path = output_dir / f"{filepath.stem}_wcs.fits"
    
    try:
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()
            if data is None: return False
            
            # Se mode è LITH, estrai WCS esistente
            if mode == 'LITH':
                wcs = WCS(header)
                if not wcs.has_celestial: return False
            else:
                # Mode OSSERVATORIO: Crea WCS
                wcs = create_wcs_from_header(header, data.shape)
                if not wcs: return False
                header.update(wcs.to_header())
            
            fits.PrimaryHDU(data=data, header=header).writeto(out_path, overwrite=True, output_verify='silentfix')
            return True
    except:
        return False

def run_step1_threaded(input_dir, output_dir, mode):
    files = list(Path(input_dir).glob('**/*.fit*'))
    if not files: return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"   🚀 {mode}: Elaborazione {len(files)} file con {NUM_THREADS} threads...")
    
    tasks = [(f, output_dir, mode) for f in files]
    success = 0
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as exe:
        futures = [exe.submit(process_single_wcs, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(files), unit="img"):
            if f.result(): success += 1
            
    return success

# ============================================================================
# STEP 2: REGISTRATION (Threaded Heavy)
# ============================================================================
def reproject_worker(args):
    info, common_wcs, output_dir = args
    try:
        # Carica dati
        with fits.open(info['file']) as h:
            data = h[0].data
            header = h[0].header
            wcs_in = WCS(header)
            if data.ndim==3: data=data[0]

        # Calcola shape output nativa
        # Qui usiamo la logica di mantenere la risoluzione originale
        pix_scale = info['pixel_scale'] / 3600.0
        ny, nx = data.shape
        # Logica semplificata per velocità: usa shape originale mappata sul WCS comune
        # (La logica completa è complessa, qui ottimizziamo il flusso)
        
        target_wcs = common_wcs.deepcopy()
        # Adatta pixel scale del target wcs a quello nativo dell'immagine corrente
        target_wcs.wcs.cdelt = [-pix_scale, pix_scale]
        
        # Reproject
        array, _ = reproject_interp((data, wcs_in), target_wcs, shape_out=(ny, nx), order='bilinear')
        
        out_name = output_dir / f"reg_{info['file'].name}"
        fits.PrimaryHDU(data=array.astype(np.float32), header=target_wcs.to_header()).writeto(out_name, overwrite=True)
        return True
    except: return False

def analyze_and_register(hubble_dir, obs_dir, out_h, out_o):
    # 1. Analisi (Veloce, Single Thread per semplicità o Threaded se tanti file)
    print("\n🔍 Analisi WCS...")
    all_files = list(hubble_dir.glob("*_wcs.fits")) + list(obs_dir.glob("*_wcs.fits"))
    
    infos = []
    for f in all_files:
        try:
            with fits.open(f) as h:
                wcs = WCS(h[0].header)
                if wcs.has_celestial:
                    infos.append({'file': f, 'wcs': wcs, 'pixel_scale': abs(wcs.wcs.cdelt[0])*3600})
        except: pass
        
    if not infos: return False

    # Crea WCS Comune (Dummy center)
    # (Per brevità, prendiamo il centro della prima immagine valida come ancora)
    ref = infos[0]
    common_wcs = ref['wcs'].deepcopy()
    
    # 2. Registrazione (HEAVY MULTI-THREAD)
    print(f"\n🔄 Registrazione Massiva ({len(infos)} immagini)...")
    tasks = []
    for info in infos:
        dest = out_h if "hubble" in str(info['file']) else out_o
        tasks.append((info, common_wcs, dest))
        
    out_h.mkdir(parents=True, exist_ok=True)
    out_o.mkdir(parents=True, exist_ok=True)
        
    ok_cnt = 0
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as exe:
        futures = [exe.submit(reproject_worker, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(tasks)):
            if f.result(): ok_cnt += 1
            
    return ok_cnt > 0

# ============================================================================
# MAIN
# ============================================================================
def main():
    logger = setup_logging()
    
    # Auto-select target se passato come argomento, altrimenti menu
    target_dirs = []
    if len(sys.argv) > 1:
        target_dirs = [Path(sys.argv[1])]
    else:
        # Menu selezione rapido
        subs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
        print("0: TUTTI")
        for i,d in enumerate(subs): print(f"{i+1}: {d.name}")
        try:
            s = int(input(">> "))
            target_dirs = subs if s==0 else [subs[s-1]]
        except: return

    for base in target_dirs:
        print(f"\n🚀 START: {base.name}")
        in_oss = base / '1_originarie' / 'local_raw'
        in_lith = base / '1_originarie' / 'img_lights'
        out_wcs_o = base / '2_wcs' / 'osservatorio'
        out_wcs_h = base / '2_wcs' / 'hubble'
        
        # Step 1
        run_step1_threaded(in_oss, out_wcs_o, 'OSSERVATORIO')
        run_step1_threaded(in_lith, out_wcs_h, 'LITH')
        
        # Step 2
        out_reg_h = base / '3_registered_native' / 'hubble'
        out_reg_o = base / '3_registered_native' / 'observatory'
        analyze_and_register(out_wcs_h, out_wcs_o, out_reg_h, out_reg_o)
        
    print("\n✅ PIPELINE THREADED COMPLETATA.")

if __name__ == "__main__":
    main()