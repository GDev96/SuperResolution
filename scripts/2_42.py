"""
STEP 2 (Mosaico): VERSIONE MULTI-THREADED
Parallelizza la proiezione (reproject) in memoria prima di sommare.
"""
import sys
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

# Config
NUM_THREADS = os.cpu_count() or 16
PROJECT_ROOT = Path("/root/SuperResolution")
if not PROJECT_ROOT.exists(): PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

try:
    from reproject import reproject_interp
except:
    print("Manca reproject"); sys.exit()

# Lock per la somma finale nel mosaico (necessario per evitare race conditions)
mosaic_lock = threading.Lock()

def process_image_projection(args):
    """Proietta una singola immagine sul WCS target."""
    fpath, target_header, target_shape = args
    try:
        with fits.open(fpath) as h:
            data = h[0].data
            wcs_in = WCS(h[0].header)
            if data.ndim==3: data=data[0]
            
        # Reproject (parte lenta, ora parallela)
        array, footprint = reproject_interp((data, wcs_in), target_header, shape_out=target_shape, order='bilinear')
        return array, footprint
    except:
        return None, None

def create_mosaic(base_dir):
    print(f"\n🖼️  MOSAICO THREADED: {base_dir.name}")
    
    # Raccogli file
    files = list((base_dir/'3_registered_native'/'hubble').glob('*.fits')) + \
            list((base_dir/'3_registered_native'/'observatory').glob('*.fits'))
            
    if not files: return

    # Crea WCS Master (prendi il primo valido e allargalo o usa logica 1.py)
    # Per semplicità, usiamo il primo file come riferimento geometrico
    # (In produzione useresti una logica di bounding box globale come in 1.py)
    with fits.open(files[0]) as h:
        target_wcs = WCS(h[0].header)
        target_shape = h[0].data.shape[-2:] # Y, X
        if len(target_shape)!=2: target_shape = (2000, 2000) # Fallback

    target_header = target_wcs.to_header()
    
    # Array Accumulatori
    master_flux = np.zeros(target_shape, dtype=np.float32)
    master_weight = np.zeros(target_shape, dtype=np.float32)
    
    print(f"   Proiezione parallela di {len(files)} immagini...")
    
    tasks = [(f, target_header, target_shape) for f in files]
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as exe:
        futures = [exe.submit(process_image_projection, t) for t in tasks]
        
        for f in tqdm(as_completed(futures), total=len(files)):
            arr, foot = f.result()
            if arr is not None:
                # La somma deve essere protetta o seriale
                arr = np.nan_to_num(arr)
                foot = np.nan_to_num(foot)
                
                # Lock breve solo per la somma
                with mosaic_lock:
                    master_flux += arr
                    master_weight += foot

    # Normalizza
    with np.errstate(divide='ignore', invalid='ignore'):
        final = master_flux / master_weight
        final[np.isnan(final)] = 0
        
    # Salva
    out = base_dir / '5_mosaics'
    out.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=final, header=target_header).writeto(out/'final_mosaic.fits', overwrite=True)
    print("✅ Mosaico Salvato.")

if __name__ == "__main__":
    subs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir()]
    print("0: TUTTI")
    for i,d in enumerate(subs): print(f"{i+1}: {d.name}")
    try:
        s = int(input(">> "))
        if s==0: [create_mosaic(d) for d in subs]
        else: create_mosaic(subs[s-1])
    except: pass