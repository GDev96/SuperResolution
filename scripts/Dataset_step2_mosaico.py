"""
STEP 2: CREAZIONE DOPPIO MOSAICO (HUBBLE & OSSERVATORIO)
Combina due logiche diverse:
1. HUBBLE: Mosaico basato su WCS (allineamento coordinate reali).
2. OSSERVATORIO: Mosaico basato su Ritaglio + Media (Crop & Mean).

INPUT: 
  - '3_registered_native/hubble'
  - '3_registered_native/observatory'

OUTPUT: 
  - '5_mosaics/hubble_mosaic.fits'
  - '5_mosaics/observatory_mosaic.fits'
  - (Intermedio) '4_cropped/observatory'
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm
import warnings
import subprocess

warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE PATH =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR_ROOT = ROOT_DATA_DIR / "logs"
SCRIPTS_DIR = CURRENT_SCRIPT_DIR

# ================= LOGGING =================
def setup_logging():
    os.makedirs(LOG_DIR_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_DIR_ROOT / f'mosaic_combined_{timestamp}.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ================= MENU =================
def select_target_directory():
    print("\n" + "📂"*35)
    print("SELEZIONE TARGET (MOSAICO COMBINATO)".center(70))
    print("📂"*35)
    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    except: return []

    if not subdirs: return []
    print("\nCartelle disponibili:")
    print(f"   0: ✨ Processa TUTTI")
    for i, d in enumerate(subdirs): print(f"   {i+1}: {d.name}")

    try:
        ans = input("\n👉 Scelta (0-N o 'q'): ").strip()
        if ans == 'q': return []
        ans = int(ans)
        return subdirs if ans==0 else [subdirs[ans-1]]
    except: return []

# ============================================================================
# PARTE 1: LOGICA HUBBLE (WCS MOSAIC)
# ============================================================================

def calculate_common_grid(image_infos):
    """Calcola la griglia comune WCS per Hubble."""
    print("\n🔍 [Hubble] Calcolo griglia WCS comune...")
    
    ra_min, ra_max = float('inf'), float('-inf')
    dec_min, dec_max = float('inf'), float('-inf')
    pixel_scales = []
    
    for info in image_infos:
        wcs, shape = info['wcs'], info['shape']
        ny, nx = shape
        corners_pix = np.array([[0, 0], [nx-1, 0], [0, ny-1], [nx-1, ny-1]])
        
        try:
            corners_world = wcs.pixel_to_world(corners_pix[:, 0], corners_pix[:, 1])
            for coord in corners_world:
                ra, dec = coord.ra.deg, coord.dec.deg
                ra_min, ra_max = min(ra_min, ra), max(ra_max, ra)
                dec_min, dec_max = min(dec_min, dec), max(dec_max, dec)
            
            # Pixel scale
            if hasattr(wcs.wcs, 'cd'):
                cd = wcs.wcs.cd
                scale = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
            else:
                scale = abs(wcs.wcs.cdelt[0])
            pixel_scales.append(scale)
        except: continue

    # Centro e Span
    ra_center = (ra_min + ra_max) / 2.0
    dec_center = (dec_min + dec_max) / 2.0
    ra_span = ra_max - ra_min
    dec_span = dec_max - dec_min
    
    target_scale = min(pixel_scales) if pixel_scales else 0.0001
    
    nx_out = int(np.ceil(abs(ra_span / np.cos(np.radians(dec_center))) / target_scale))
    ny_out = int(np.ceil(abs(dec_span) / target_scale))
    
    # Forzatura quadrata
    max_dim = max(nx_out, ny_out)
    # Limitazione dimensioni eccessive (es. > 20k)
    if max_dim > 20000:
        target_scale *= (max_dim / 20000)
        max_dim = 20000
        
    print(f"   Griglia Finale: {max_dim} x {max_dim} px (Scale: {target_scale*3600:.3f}\"/px)")

    output_wcs = WCS(naxis=2)
    output_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    output_wcs.wcs.crval = [ra_center, dec_center]
    output_wcs.wcs.crpix = [max_dim / 2.0, max_dim / 2.0]
    output_wcs.wcs.cdelt = [-target_scale, target_scale]
    
    return output_wcs, (max_dim, max_dim)

def create_hubble_mosaic_wcs(base_dir):
    """Crea il mosaico Hubble usando riproiezione WCS."""
    print("\n" + "🛰️ "*20)
    print(f"HUBBLE MOSAIC (WCS): {base_dir.name}".center(60))
    print("🛰️ "*20)

    input_dir = base_dir / '3_registered_native' / 'hubble'
    output_dir = base_dir / '5_mosaics'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'hubble_mosaic.fits'

    # 1. Carica immagini
    files = list(input_dir.glob('*.fits')) + list(input_dir.glob('*.fit'))
    if not files:
        # Fallback
        input_dir = base_dir / '1_originarie' / 'img_lights'
        files = list(input_dir.glob('*.fits')) + list(input_dir.glob('*.fit'))
    
    if not files:
        print("❌ Nessun file Hubble trovato.")
        return False

    image_infos = []
    print(f"📂 [Hubble] Lettura WCS da {len(files)} file...")
    for f in tqdm(files, unit="file"):
        try:
            with fits.open(f) as hdul:
                # Cerca estensione corretta
                if 'SCI' in hdul: hdu = hdul['SCI']
                else: hdu = hdul[0]
                
                data = hdu.data
                if data is None: continue
                if len(data.shape) == 3: data = data[0]
                
                wcs = WCS(hdu.header)
                if not wcs.has_celestial: continue
                
                image_infos.append({'file': f, 'data': data, 'wcs': wcs, 'shape': data.shape})
        except: continue
    
    if not image_infos: return False

    # 2. Calcola griglia e proietta
    out_wcs, out_shape = calculate_common_grid(image_infos)
    
    mosaic_data = np.zeros(out_shape, dtype=np.float32)
    mosaic_weight = np.zeros(out_shape, dtype=np.float32)
    
    print("🔄 [Hubble] Riproiezione e Stacking...")
    for info in tqdm(image_infos, unit="img"):
        try:
            # Proiezione WCS manuale (per coerenza con il tuo script hubble.py)
            data = info['data']
            ny, nx = info['shape']
            y, x = np.mgrid[:ny, :nx]
            
            coords = info['wcs'].pixel_to_world(x.ravel(), y.ravel())
            xo, yo = out_wcs.world_to_pixel(coords)
            
            xo = np.round(xo).astype(int)
            yo = np.round(yo).astype(int)
            
            valid = (xo>=0) & (xo<out_shape[1]) & (yo>=0) & (yo<out_shape[0]) & ~np.isnan(data.ravel())
            
            mosaic_data[yo[valid], xo[valid]] += data.ravel()[valid]
            mosaic_weight[yo[valid], xo[valid]] += 1
        except: continue

    # 3. Media e Salvataggio
    with np.errstate(divide='ignore', invalid='ignore'):
        final = np.where(mosaic_weight > 0, mosaic_data / mosaic_weight, 0)
        
    header = out_wcs.to_header()
    header['HISTORY'] = "Hubble WCS Mosaic"
    fits.PrimaryHDU(data=final.astype(np.float32), header=header).writeto(output_file, overwrite=True)
    print(f"✅ Mosaico Hubble creato: {output_file}")
    return True

# ============================================================================
# PARTE 2: LOGICA OSSERVATORIO (CROP + MEAN)
# ============================================================================

def find_smallest_dim_obs(files):
    """Trova dimensioni minime per crop."""
    min_h, min_w = float('inf'), float('inf')
    print("\n🔍 [Osservatorio] Ricerca dimensioni minime...")
    for f in tqdm(files, unit="file"):
        try:
            with fits.open(f) as h:
                d = h[0].data
                if d is None: continue
                if len(d.shape)==3: sh = d[0].shape
                else: sh = d.shape
                min_h, min_w = min(min_h, sh[0]), min(min_w, sh[1])
        except: continue
    print(f"   Target Crop: {min_w} x {min_h}")
    return int(min_h), int(min_w)

def crop_obs_image(fpath, outpath, th, tw):
    try:
        with fits.open(fpath) as h:
            d = h[0].data
            head = h[0].header.copy()
            if len(d.shape)==3: d = d[0]
            ch, cw = d.shape
            yo, xo = (ch-th)//2, (cw-tw)//2
            crop = d[yo:yo+th, xo:xo+tw]
            
            head['NAXIS1'], head['NAXIS2'] = tw, th
            fits.PrimaryHDU(crop, head).writeto(outpath, overwrite=True)
            return True
    except: return False

def create_obs_mosaic_crop(base_dir):
    """Crea mosaico Osservatorio tramite crop e media."""
    print("\n" + "🔭 "*20)
    print(f"OBSERVATORY MOSAIC (CROP+MEAN): {base_dir.name}".center(60))
    print("🔭 "*20)

    in_dir = base_dir / '3_registered_native' / 'observatory'
    crop_dir = base_dir / '4_cropped' / 'observatory'
    crop_dir.mkdir(parents=True, exist_ok=True)
    out_dir = base_dir / '5_mosaics'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(in_dir.glob('*.fits')) + list(in_dir.glob('*.fit'))
    if not files:
        print("❌ Nessun file Osservatorio trovato.")
        return False

    # 1. Ritaglio
    th, tw = find_smallest_dim_obs(files)
    print("✂️ [Osservatorio] Esecuzione Ritaglio...")
    cropped_files = []
    for f in tqdm(files, desc="Cropping"):
        out = crop_dir / f.name
        if crop_obs_image(f, out, th, tw):
            cropped_files.append(out)

    # 2. Media
    if not cropped_files: return False
    print("🔄 [Osservatorio] Calcolo Media...")
    
    acc = np.zeros((th, tw), dtype=np.float64)
    count = np.zeros((th, tw), dtype=np.int32)
    
    # Template header
    with fits.open(cropped_files[0]) as h: head = h[0].header.copy()
    
    for f in tqdm(cropped_files, desc="Stacking"):
        try:
            with fits.open(f) as h:
                d = np.nan_to_num(h[0].data, nan=0.0)
                if d.shape != (th, tw): continue
                acc += d
                count += (d!=0).astype(int)
        except: continue
        
    with np.errstate(divide='ignore', invalid='ignore'):
        final = np.where(count > 0, acc / count, 0)
        
    out_file = out_dir / 'observatory_mosaic.fits'
    head['HISTORY'] = "Observatory Mosaic (Crop+Mean)"
    fits.PrimaryHDU(final.astype(np.float32), head).writeto(out_file, overwrite=True)
    print(f"✅ Mosaico Osservatorio creato: {out_file}")
    return True

# ================= MAIN =================
def main():
    logger = setup_logging()
    
    if len(sys.argv) > 1:
        p = Path(sys.argv[1]).resolve()
        targets = [p] if p.exists() else []
    else:
        targets = select_target_directory()

    if not targets: return

    successful = []
    print("\n" + "="*70)
    print("PIPELINE: CREAZIONE DOPPIO MOSAICO".center(70))
    print("="*70)

    for base_dir in targets:
        logger.info(f"Processing {base_dir.name}")
        
        # 1. Mosaico Hubble (WCS)
        ok_h = create_hubble_mosaic_wcs(base_dir)
        
        # 2. Mosaico Osservatorio (Crop+Mean)
        ok_o = create_obs_mosaic_crop(base_dir)
        
        if ok_h and ok_o:
            successful.append(base_dir)
            
    print("\n" + "="*60)
    print(f"✅ Completati {len(successful)} target.")
    
    # Next Step
    next_script = SCRIPTS_DIR / 'Dataset_step3_analizzapatch.py'
    if successful and next_script.exists():
        ask = input("\n👉 Avviare Step 3 (Estrazione Patch)? [S/n]: ").lower()
        if ask in ['s', 'y', '']:
            for t in successful:
                subprocess.run([sys.executable, str(next_script), str(t)])

if __name__ == "__main__":
    main()