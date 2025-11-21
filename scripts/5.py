"""
STEP 4: ALLINEAMENTO & DOWNSCALE (MULTIPROCESSING ULTRA-FAST)
Ottimizzato per bypassare il GIL di Python e usare il 100% della CPU.
"""

import os

# ============================================================================
# 1. MAGIC FIX: BLOCCA I SOTTO-THREAD DI NUMPY
# ============================================================================
# Questo è CRUCIALE. Impedisce che ogni processo lanci a sua volta 32 thread,
# intasando la CPU. Ogni processo userà 1 solo core al 100%.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import shutil
import numpy as np
from pathlib import Path
from astropy.io import fits
from tqdm import tqdm
import warnings
import scipy.ndimage
from skimage.transform import resize
from concurrent.futures import ProcessPoolExecutor, as_completed # NOTA: ProcessPoolExecutor!

# Import specifici
try:
    import astroalign
    from skimage.registration import phase_cross_correlation
except ImportError:
    sys.exit("❌ ERRORE: Installa le librerie: pip install scikit-image astroalign scipy")

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE
# ============================================================================
# Usa tutti i core fisici disponibili
NUM_WORKERS = os.cpu_count()
if not NUM_WORKERS: NUM_WORKERS = 16 

# Path
PROJECT_ROOT = Path("/root/SuperResolution")
if not PROJECT_ROOT.exists(): PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "data"

TARGET_SIZE_LR = 80
ENABLE_ALIGNMENT_FILTER = True 
ENABLE_SIMULATED_NOISE = True   
NOISE_STD_DEV = 0.005           

# ============================================================================
# FUNZIONI (Devono essere "Picklable" per il Multiprocessing)
# ============================================================================

def normalize_data(data):
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = data.min(), data.max()
    if mx - mn == 0: return data
    return (data - mn) / (mx - mn)

def process_single_pair(pair_dir):
    """
    Questa funzione viene eseguita in un PROCESSO separato.
    Ha la sua memoria isolata e non condivide il GIL.
    """
    f_hub = pair_dir / "hubble.fits"
    f_obs = pair_dir / "observatory.fits"

    if not f_hub.exists() or not f_obs.exists():
        return False, "MISSING"

    try:
        # Carica dati
        with fits.open(f_hub) as h: hub = h[0].data.astype(np.float32)
        with fits.open(f_obs) as o: obs = o[0].data.astype(np.float32)
        
        if hub.ndim == 3: hub = hub[0]
        if obs.ndim == 3: obs = obs[0]

        # --- 1. ALLINEAMENTO ---
        source = normalize_data(obs)
        target = normalize_data(hub)
        aligned_image = obs 
        method = "NONE"
        ok = False

        try:
            # Astroalign è pesante sulla CPU -> Qui brilla il multiprocessing
            transf, _ = astroalign.find_transform(source, target, detection_sigma=2, min_area=5)
            aligned_image = astroalign.apply_transform(transf, obs, target.shape, fill_value=np.median(obs)) 
            method = "ASTROALIGN"
            ok = True
        except:
            try:
                # Fallback FFT
                shift, _, _ = phase_cross_correlation(target, source, upsample_factor=10)
                aligned_image = scipy.ndimage.shift(obs, shift, mode='reflect')
                method = "FFT"
                ok = True
            except:
                method = "FAILED"
                ok = False

        # Filtro
        if not ok and ENABLE_ALIGNMENT_FILTER:
            # Rimuovi cartella (Attenzione: operazione su disco da processo figlio)
            shutil.rmtree(pair_dir)
            return False, "FILTERED"

        # --- 2. DOWNSCALE ---
        # Resize è pesante sulla CPU
        downscaled = resize(aligned_image, (TARGET_SIZE_LR, TARGET_SIZE_LR), anti_aliasing=True, preserve_range=True)
        
        if ENABLE_SIMULATED_NOISE:
            mn, mx = downscaled.min(), downscaled.max()
            if mx - mn > 1e-8:
                norm_down = (downscaled - mn) / (mx - mn)
                noise = np.random.normal(0, NOISE_STD_DEV, norm_down.shape).astype(norm_down.dtype)
                final_data = np.clip((norm_down + noise) * (mx - mn) + mn, mn, mx)
            else:
                final_data = downscaled
        else:
            final_data = downscaled

        # --- 3. SALVATAGGIO ---
        header_new = fits.Header()
        header_new['HISTORY'] = f"Aligned {method}"
        fits.PrimaryHDU(data=final_data.astype(np.float32), header=header_new).writeto(f_obs, overwrite=True)
        
        return True, method

    except Exception as e:
        return False, str(e)

# ============================================================================
# BATCH MANAGER
# ============================================================================
def process_target(target_dir):
    patches_dir = target_dir / "6_patches_aligned"
    if not patches_dir.exists(): return
    
    # Raccogli tutti i path prima
    pairs = sorted(list(patches_dir.glob("pair_*")))
    if not pairs: return
    
    print(f"\n🚀 Processing {target_dir.name}")
    print(f"   🔥 MULTIPROCESSING: Attivo su {NUM_WORKERS} Core CPU reali")
    print(f"   📦 Patch totali: {len(pairs)}")
    
    stats = {"OK": 0, "FAIL": 0, "FILTERED": 0}
    
    # ProcessPoolExecutor lancia processi VERI, non thread.
    # Chunksize aiuta a ridurre l'overhead di comunicazione tra processi.
    chunk_size = max(1, len(pairs) // (NUM_WORKERS * 4))
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit massivo
        futures = {executor.submit(process_single_pair, p): p for p in pairs}
        
        with tqdm(total=len(pairs), unit="img") as pbar:
            for future in as_completed(futures):
                try:
                    ok, msg = future.result()
                    if ok: stats["OK"] += 1
                    elif msg == "FILTERED": stats["FILTERED"] += 1
                    else: stats["FAIL"] += 1
                except Exception as e:
                    stats["FAIL"] += 1
                pbar.update(1)

    print(f"✅ Report: {stats['OK']} OK | {stats['FILTERED']} Filtrate | {stats['FAIL']} Errori")

def main():
    subdirs = [d for d in BASE_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    if not subdirs: return
    print("Seleziona Target:")
    print("0: TUTTI")
    for i, d in enumerate(subdirs): print(f"{i+1}: {d.name}")
    try:
        sel = int(input(">> "))
        if sel == 0: [process_target(d) for d in subdirs]
        elif 0 < sel <= len(subdirs): process_target(subdirs[sel-1])
    except: pass

if __name__ == "__main__":
    main()