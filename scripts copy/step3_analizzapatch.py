"""
STEP 3: ESTRAZIONE PATCH
"""
import sys
import time
import json
import logging
import shutil
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

warnings.filterwarnings('ignore')
# ============================================================================
# CONFIGURAZIONE PATH DINAMICI (UNIVERSALI)
# ============================================================================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent

# Risale alla cartella principale del progetto (es. .../SuperResolution)
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent

LOG_DIR = PROJECT_ROOT / "logs"
# ============================================================================
# ============================================================================

TARGET_FOV = 1 # Arcmin
OVERLAP = 25
MATCH_THR = 0.5 / 60.0

def extract_patches(f, d_out, prefix):
    patches = []
    try:
        with fits.open(f) as h:
            d, head = h[0].data, h[0].header
            w = WCS(head)
            if not w.has_celestial: return []
            
            scale = np.sqrt(w.wcs.cd[0,0]**2 + w.wcs.cd[0,1]**2) if hasattr(w.wcs, 'cd') else abs(w.wcs.cdelt[0])
            # Calcolo dimensione in pixel per 1 arcmin
            ps = int((TARGET_FOV/60)/scale)
            # Arrotonda a multiplo di 8
            ps = ((ps+7)//8)*8
            step = int(ps * (1 - OVERLAP/100))
            
            ny, nx = d.shape
            for y in range(0, ny-ps+1, step):
                for x in range(0, nx-ps+1, step):
                    p_data = d[y:y+ps, x:x+ps]
                    # Scarta se troppi NaN
                    if np.isnan(p_data).mean() > 0.5: continue
                    
                    c = w.pixel_to_world(x+ps/2, y+ps/2)
                    name = f"{prefix}_{f.stem}_p{len(patches):04d}.fits"
                    
                    head['NAXIS1'], head['NAXIS2'] = ps, ps
                    fits.PrimaryHDU(p_data, head).writeto(d_out/name, overwrite=True)
                    
                    patches.append({'file': name, 'path': str(d_out/name), 'ra': c.ra.deg, 'dec': c.dec.deg})
        return patches
    except: return []

def main():
    if len(sys.argv) < 2: 
        print("⚠️ Questo script deve essere chiamato da step2 o con un argomento path.")
        # Fallback per test manuale
        return

    BASE = Path(sys.argv[1])
    print(f"\n🚀 Patching {BASE.name}...")
    
    # Input: Immagini croppate
    IN_H = BASE / '4_cropped' / 'hubble'
    IN_O = BASE / '4_cropped' / 'observatory'
    
    # Output
    OUT = BASE / '6_patches_from_cropped'
    (OUT/'hubble_patches').mkdir(parents=True, exist_ok=True)
    (OUT/'observatory_patches').mkdir(parents=True, exist_ok=True)
    
    H_patches, O_patches = [], []
    
    with ThreadPoolExecutor(4) as exc:
        f_h = [exc.submit(extract_patches, f, OUT/'hubble_patches', 'hr') for f in IN_H.glob('*.fits')]
        f_o = [exc.submit(extract_patches, f, OUT/'observatory_patches', 'lr') for f in IN_O.glob('*.fits')]
        
        for f in tqdm(as_completed(f_h), total=len(f_h), desc="Hubble"): H_patches.extend(f.result())
        for f in tqdm(as_completed(f_o), total=len(f_o), desc="Obs"): O_patches.extend(f.result())
        
    print(f"   Estratte: {len(H_patches)} HR, {len(O_patches)} LR")
    
    # Matching
    PAIRS_DIR = OUT / 'paired_patches_folders'
    PAIRS_DIR.mkdir(exist_ok=True)
    pairs = []
    
    print("   Matching...")
    for h in tqdm(H_patches):
        hc = SkyCoord(h['ra'], h['dec'], unit='deg')
        best, m_dist = None, float('inf')
        
        for o in O_patches:
            dist = hc.separation(SkyCoord(o['ra'], o['dec'], unit='deg')).deg
            if dist < m_dist: m_dist, best = dist, o
            
        if m_dist < MATCH_THR and best:
            pid = f"pair_{len(pairs):05d}"
            d = PAIRS_DIR / pid
            d.mkdir()
            shutil.copy(h['path'], d/h['file'])
            shutil.copy(best['path'], d/best['file'])
            pairs.append(pid)
            
    print(f"✅ Create {len(pairs)} coppie.")

if __name__ == "__main__":
    import shutil
    main()