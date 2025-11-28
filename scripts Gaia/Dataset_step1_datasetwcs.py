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
import shutil

# Gestione importazioni opzionali
try:
    from reproject import reproject_interp
    REPROJECT_AVAILABLE = True
except ImportError:
    REPROJECT_AVAILABLE = False
    print("⚠️ Libreria 'reproject' non trovata. La registrazione fallirà.")

warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR_ROOT = ROOT_DATA_DIR / "logs"

# === CONFIGURAZIONE ASTAP ===
# Modifica questo percorso se ASTAP è installato altrove
if sys.platform == 'win32':
    ASTAP_PATH = r"C:\Program Files\astap\astap.exe"
else:
    ASTAP_PATH = "astap" # Su Linux/Mac si assume sia nel PATH

NUM_THREADS = 2 # Ridotto leggermente per evitare sovraccarico durante il solving
log_lock = threading.Lock()

# ================= UTILITY & SETUP =================

def setup_logging():
    os.makedirs(LOG_DIR_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR_ROOT / f'pipeline_smart_astap_{timestamp}.log'
    
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

# ================= STEP 1: PLATE SOLVING CON ASTAP =================

def solve_with_astap(inp_file, out_file, logger):
    """
    Esegue il plate solving usando ASTAP CLI.
    Copia il file, risolve e aggiorna l'header.
    """
    try:
        # 1. Copia il file originale nella destinazione (per non toccare il raw)
        shutil.copy2(inp_file, out_file)
        
        # 2. Costruzione comando ASTAP
        # -f: file input
        # -update: aggiorna l'header del file esistente con la soluzione WCS
        # -r 30: cerca in un raggio di 30 gradi (utile se le coordinate header sono imprecise)
        # -fov 0: lascia che ASTAP calcoli o legga il FOV dall'header
        cmd = [
            ASTAP_PATH,
            "-f", str(out_file),
            "-update",
            "-r", "30",  # Raggio di ricerca in gradi (aumentare a 180 per blind solving totale)
            "-z", "0"    # Downsampling (0=auto, 1=no). Auto è più veloce per immagini grandi
        ]

        # 3. Esecuzione (silenziosa)
        # Su Windows bisogna gestire la creazione della finestra se non si vuole popup
        startupinfo = None
        if sys.platform == 'win32':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        process = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            startupinfo=startupinfo
        )

        # 4. Verifica successo
        # Controlliamo se il file ha ora una WCS valida
        with fits.open(out_file) as hdul:
            header = hdul[0].header
            # ASTAP inserisce solitamente PLTSOLVD=T o CTYPE1 validi
            w = WCS(header)
            if w.has_celestial:
                # Opzionale: Pulizia file temporanei generati da ASTAP (.wcs, .ini)
                wcs_file = out_file.with_suffix('.wcs')
                ini_file = out_file.with_suffix('.ini')
                if wcs_file.exists(): os.remove(wcs_file)
                if ini_file.exists(): os.remove(ini_file)
                return True
            else:
                # logger.warning(f"ASTAP fallito su {inp_file.name}: Nessuna WCS trovata.")
                return False

    except Exception as e:
        logger.error(f"Errore ASTAP su {inp_file.name}: {e}")
        # Se fallisce, cancelliamo il file di output per non lasciare file corrotti
        if out_file.exists():
            os.remove(out_file)
        return False

def process_step1_folder(inp_dir, out_dir, logger):
    # Cerca fits, FIT, fit, FITS
    files = list(inp_dir.glob('*.fit')) + list(inp_dir.glob('*.fits')) + \
            list(inp_dir.glob('*.FIT')) + list(inp_dir.glob('*.FITS'))
    files = sorted(list(set(files)))
    
    if not files:
        logger.warning(f"Nessun file trovato in {inp_dir}")
        return 0

    print(f"   Avvio ASTAP Plate Solving su {len(files)} immagini in {inp_dir.name}...")
    
    success = 0
    # Usiamo ThreadPool per parallelizzare ASTAP (ASTAP supporta istanze multiple)
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        for f in files:
            out_f = out_dir / f"{f.stem}_solved.fits"
            futures.append(executor.submit(solve_with_astap, f, out_f, logger))
            
        for f in tqdm(as_completed(futures), total=len(files), desc=f"Solving {inp_dir.name}"):
            if f.result():
                success += 1
                
    return success

# ================= STEP 2: REGISTRAZIONE SMART (REPROJECT) =================

def extract_wcs_info(f, logger=None):
    try:
        with fits.open(f) as h:
            # Carica WCS ignorando distorsioni SIP per leggere solo centro e scala approssimata
            w = WCS(h[0].header)
            if not w.has_celestial: return None
            
            # Calcola scala media
            if w.wcs.cdelt[0] != 0:
                scale = abs(w.wcs.cdelt[0]) * 3600
            else:
                # Fallback se CDELT è 0 (es. se usa matrice CD)
                scale = astropy.wcs.utils.proj_plane_pixel_scales(w)[0] * 3600

            return {
                'file': f, 
                'wcs': w, 
                'shape': h[0].data.shape, 
                'ra': w.wcs.crval[0], 
                'dec': w.wcs.crval[1],
                'scale': scale
            }
    except Exception as e: 
        # logger.warning(f"Errore lettura WCS {f.name}: {e}")
        return None

def register_single_image_smart(info, ref_wcs, out_dir, logger):
    try:
        fname = info['file'].name
        with fits.open(info['file']) as hdul:
            data = np.nan_to_num(hdul[0].data)
            header = hdul[0].header
            wcs_orig = WCS(header)
            if data.ndim == 3: data = data[0]

        # --- SETUP SISTEMA DI RIFERIMENTO TARGET ---
        # 1. Centro: Usa il centro del Master (Hubble)
        # 2. Scala: Mantiene la scala nativa dell'immagine corrente
        # 3. Proiezione: FORZA 'TAN' (Gnomonica) per eliminare distorsioni SIP/esotiche
        
        native_scale_deg = info['scale'] / 3600.0
        
        target_wcs = WCS(naxis=2)
        target_wcs.wcs.crval = ref_wcs.wcs.crval # Centro RA/DEC del Master
        target_wcs.wcs.crpix = [data.shape[1]/2, data.shape[0]/2] # Centro pixel immagine
        target_wcs.wcs.cdelt = [-native_scale_deg, native_scale_deg] # Scala nativa
        target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"] # Proiezione Standard Piatta
        
        # Orientamento: Assumiamo Nord in alto (rotazione 0) per standardizzare
        # Se si vuole preservare la rotazione originale del master, bisognerebbe copiare PC/CD matrix
        # Qui forziamo Nord-Su per facilitare l'estrazione patch
        
        output_data, _ = reproject_interp((data, wcs_orig), target_wcs, shape_out=data.shape)

        # Salva il file registrato
        out_name = f"reg_{fname}"
        hdr_new = target_wcs.to_header()
        hdr_new['REG_METH'] = "ASTAP_SOLVE+REPROJECT"
        
        # Mantiene alcune chiavi utili originali
        for k in ['EXPTIME', 'FILTER', 'OBJECT']:
            if k in header: hdr_new[k] = header[k]

        fits.PrimaryHDU(data=output_data.astype(np.float32), header=hdr_new).writeto(out_dir/out_name, overwrite=True)
        
        return {'status': 'ok', 'file': out_name}

    except Exception as e:
        return {'status': 'err', 'file': fname, 'err': str(e)}

def main_registration(h_in, o_in, h_out, o_out, logger):
    # 1. Trova i file Solved dello Step 1
    h_files = list(h_in.glob('*_solved.fits'))
    o_files = list(o_in.glob('*_solved.fits'))
    
    # 2. Estrai Info WCS
    h_infos = [extract_wcs_info(f, logger) for f in h_files]
    o_infos = [extract_wcs_info(f, logger) for f in o_files]
    
    h_infos = [x for x in h_infos if x]
    o_infos = [x for x in o_infos if x]
    
    if not h_infos: 
        logger.error("Nessun file Hubble risolto (ASTAP ha fallito o cartella vuota).")
        return False
    
    # MASTER REFERENCE: Primo file Hubble
    common_wcs = h_infos[0]['wcs']
    
    print(f"   Master Reference RA/DEC: {common_wcs.wcs.crval}")
    print(f"   Avvio registrazione su {len(h_infos)} Hubble e {len(o_infos)} Osservatorio...")
    
    success = 0
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as ex:
        futures = []
        
        # Registra Hubble
        for info in h_infos:
            futures.append(ex.submit(register_single_image_smart, info, common_wcs, h_out, logger))
            
        # Registra Osservatorio
        for info in o_infos:
            futures.append(ex.submit(register_single_image_smart, info, common_wcs, o_out, logger))
            
        for f in tqdm(as_completed(futures), total=len(futures), desc="Registrazione"):
            res = f.result()
            if res['status'] == 'ok': success += 1
            else: logger.warning(f"Err {res['file']}: {res['err']}")
            
    return success > 0

# ================= MAIN =================

def main():
    logger = setup_logging()
    
    # Verifica presenza ASTAP
    if not Path(ASTAP_PATH).exists() and shutil.which("astap") is None:
        print(f"❌ ERRORE CRITICO: ASTAP non trovato in: {ASTAP_PATH}")
        print("   Installa ASTAP o correggi la variabile ASTAP_PATH nello script.")
        return

    targets = select_target_directory()
    if not targets: return

    for BASE_DIR in targets:
        print(f"\n🚀 ELABORAZIONE: {BASE_DIR.name}")
        
        in_o = BASE_DIR / '1_originarie/local_raw'
        in_h = BASE_DIR / '1_originarie/img_lights'
        
        # Cartelle output Step 1 (ora contengono i file "Solved")
        out_solved_o = BASE_DIR / '2_solved_astap/observatory'
        out_solved_h = BASE_DIR / '2_solved_astap/hubble'
        
        # Cartelle output Step 2 (Registered)
        out_reg_o = BASE_DIR / '3_registered_native/observatory'
        out_reg_h = BASE_DIR / '3_registered_native/hubble'
        
        for p in [out_solved_o, out_solved_h, out_reg_o, out_reg_h]: 
            p.mkdir(parents=True, exist_ok=True)

        # Step 1: Plate Solving
        print("   [1/2] Astrometric Solving (ASTAP)...")
        s1 = process_step1_folder(in_o, out_solved_o, logger)
        s2 = process_step1_folder(in_h, out_solved_h, logger)
        
        if s1+s2 == 0:
            print("   ❌ Nessun file risolto. Controlla i database stellari (H17/H18) di ASTAP.")
            continue

        # Step 2: Registrazione
        print("   [2/2] Registrazione e Riproiezione...")
        if main_registration(out_solved_h, out_solved_o, out_reg_h, out_reg_o, logger):
            print("   ✅ Pipeline Completata.")
        else:
            print("   ❌ Errore nella fase di registrazione.")

if __name__ == "__main__":
    main()