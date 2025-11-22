"""
STEP 3 (FINAL 95% FILTER): TAGLIO COPPIE + FILTRO COPERTURA + 8-PANEL VISUAL
------------------------------------------------------------------------
Features:
1. Scansiona i mosaici allineati (Step 2.5).
2. FILTRO RIGOROSO: Scarta qualsiasi patch che non abbia almeno il 
   95% di dati validi (elimina bordi neri e zone vuote).
3. Genera la Context Card a 8 pannelli per verifica visiva.

INPUT: aligned_hubble.fits, aligned_observatory.fits, final_mosaic_observatory.fits
OUTPUT: 6_patches_final/pair_XXXXX/
------------------------------------------------------------------------
"""

import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
from skimage.transform import resize
from reproject import reproject_interp
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE =================
HR_SIZE = 512         
LR_SIZE = 80          
STRIDE = 64           

# SOGLIE DI QUALITÀ
MIN_PIXEL_VALUE = 0.0001 # Valore minimo per considerare un pixel "pieno"
MIN_COVERAGE = 0.95      # 95% della patch deve essere piena (NO BORDI)

SAVE_MAP_EVERY_N = 5  

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

def select_target_directory():
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET (Filtro 95%)".center(70))
    print("📂"*35)
    subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    if not subdirs: return None
    for i, d in enumerate(subdirs):
        print(f"   {i+1}: {d.name}")
    try:
        choice = int(input(f"\n👉 Seleziona (1-{len(subdirs)}): ").strip())
        if 0 < choice <= len(subdirs): return subdirs[choice-1]
    except: pass
    return None

# --- FUNZIONI DI NORMALIZZAZIONE E PLOT (Invariate per brevità, ma incluse) ---
def get_global_normalization(data):
    data = np.nan_to_num(data)
    h, w = data.shape
    cy, cx = h//2, w//2
    if h > 2000 and w > 2000: sample = data[cy-1000:cy+1000, cx-1000:cx+1000]
    else: sample = data
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(sample)
    return vmin, vmax

def normalize_with_limits(data, vmin, vmax):
    data = np.nan_to_num(data)
    if vmax == vmin: return np.zeros_like(data)
    return np.clip((data - vmin) / (vmax - vmin), 0, 1)

def normalize_local(data):
    try:
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(np.nan_to_num(data))
        return np.clip((np.nan_to_num(data) - vmin) / (vmax - vmin), 0, 1)
    except: return np.clip(np.nan_to_num(data), 0, 1)

def save_8panel_card(mosaic_h, mosaic_o_aligned, mosaic_o_visual, mosaic_o_raw, 
                     patch_h, patch_o_in, x, y, wcs_h, wcs_orig, 
                     vmin_h, vmax_h, vmin_o, vmax_o, save_path):
    
    fig = plt.figure(figsize=(28, 12))
    fig.suptitle(f"DATASET SAMPLE: X={x} Y={y} (Coverage > 95%)", fontsize=22, fontweight='bold')
    gs = fig.add_gridspec(2, 4) 
    sc = 8 

    # 1. Hubble
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(normalize_with_limits(mosaic_h[::sc, ::sc], vmin_h, vmax_h), origin='lower', cmap='gray')
    ax1.set_title("1. Hubble Master", color='green')
    ax1.add_patch(patches.Rectangle((x/sc, y/sc), HR_SIZE/sc, HR_SIZE/sc, lw=2, edgecolor='lime', facecolor='none'))
    ax1.axis('off')

    # 2. Obs Aligned
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(normalize_with_limits(mosaic_o_aligned[::sc, ::sc], vmin_o, vmax_o), origin='lower', cmap='magma')
    ax2.set_title("2. Obs Allineato", color='magenta')
    ax2.add_patch(patches.Rectangle((x/sc, y/sc), HR_SIZE/sc, HR_SIZE/sc, lw=2, edgecolor='cyan', facecolor='none'))
    ax2.axis('off')

    # 3. Obs Rotated Visual
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(normalize_with_limits(mosaic_o_visual[::sc, ::sc], vmin_o, vmax_o), origin='lower', cmap='gray')
    ax3.set_title("3. Obs Ruotato (Solo Visual)", color='orange')
    ax3.add_patch(patches.Rectangle((x/sc, y/sc), HR_SIZE/sc, HR_SIZE/sc, lw=2, edgecolor='orange', facecolor='none'))
    ax3.axis('off')

    # 4. Obs Native
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(normalize_with_limits(mosaic_o_raw[::sc, ::sc], vmin_o, vmax_o), origin='lower', cmap='gray')
    ax4.set_title("4. Obs Originale (Nativo)", color='white')
    # Calcolo box distorto
    corners_pix = np.array([[x, y], [x+HR_SIZE, y+HR_SIZE]])
    corners_world = wcs_h.pixel_to_world(corners_pix[:,0], corners_pix[:,1])
    cx_orig, cy_orig = wcs_orig.world_to_pixel(corners_world)
    ox, oy = min(cx_orig), min(cy_orig)
    ow, oh = abs(cx_orig[1]-cx_orig[0]), abs(cy_orig[1]-cy_orig[0])
    ax4.add_patch(patches.Rectangle((ox/sc, oy/sc), ow/sc, oh/sc, lw=2, edgecolor='red', facecolor='none'))
    ax4.axis('off')

    # 5. Patch H
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(normalize_local(patch_h), origin='lower', cmap='viridis')
    ax5.set_title("5. Hubble Target")
    ax5.axis('off')

    # 6. Patch O
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(normalize_local(patch_o_in), origin='lower', cmap='magma')
    ax6.set_title("6. Obs Input")
    ax6.axis('off')

    # 7. Overlay
    ax7 = fig.add_subplot(gs[1, 2])
    inp_s = resize(normalize_local(patch_o_in), (HR_SIZE, HR_SIZE), order=0)
    tar_n = normalize_local(patch_h)
    rgb = np.zeros((HR_SIZE, HR_SIZE, 3))
    rgb[..., 0] = inp_s * 0.8
    rgb[..., 1] = tar_n
    rgb[..., 2] = inp_s * 0.8
    ax7.imshow(rgb, origin='lower')
    ax7.set_title("7. Check Overlay")
    ax7.axis('off')

    # 8. Info
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    txt = f"COPERTURA OK (>95%)\nValid H: {np.count_nonzero(patch_h > MIN_PIXEL_VALUE)/(HR_SIZE**2)*100:.1f}%"
    ax8.text(0.1, 0.5, txt, fontsize=14, family='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=90)
    plt.close(fig)

def create_dataset_filtered(base_dir):
    aligned_dir = base_dir / '5_mosaics' / 'aligned_ready_for_crop'
    mosaics_dir = base_dir / '5_mosaics'
    output_dir = base_dir / '6_patches_final'
    
    f_h_align = aligned_dir / 'aligned_hubble.fits'
    f_o_align = aligned_dir / 'aligned_observatory.fits'
    f_o_orig  = mosaics_dir / 'final_mosaic_observatory.fits'

    if not f_h_align.exists() or not f_o_align.exists() or not f_o_orig.exists():
        print("❌ Mancano file necessari.")
        return

    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n⚙️  Caricamento Dati...")
    with fits.open(f_h_align) as h:
        data_h = np.nan_to_num(h[0].data)
        if data_h.ndim==3: data_h=data_h[0]
        wcs_h = WCS(h[0].header)
        header_h = h[0].header

    with fits.open(f_o_align) as h:
        data_o_aligned = np.nan_to_num(h[0].data)
        if data_o_aligned.ndim==3: data_o_aligned=data_o_aligned[0]

    with fits.open(f_o_orig) as h:
        data_o_raw = np.nan_to_num(h[0].data)
        if data_o_raw.ndim==3: data_o_raw=data_o_raw[0]
        wcs_orig = WCS(h[0].header)

    print("⚖️  Calcolo Contrasto...")
    vmin_h, vmax_h = get_global_normalization(data_h)
    vmin_o, vmax_o = get_global_normalization(data_o_raw)

    print("🔄 Generazione Mappa Ruotata Visuale...")
    visual_rotated, _ = reproject_interp((data_o_raw, wcs_orig), wcs_h, shape_out=data_h.shape)
    visual_rotated = np.nan_to_num(visual_rotated)

    h_dim, w_dim = data_h.shape
    y_list = list(range(0, h_dim - HR_SIZE + 1, STRIDE))
    x_list = list(range(0, w_dim - HR_SIZE + 1, STRIDE))
    total = len(y_list) * len(x_list)
    count = 0
    skipped = 0
    
    print(f"\n✂️  Estrazione con Filtro {MIN_COVERAGE*100}%...")
    with tqdm(total=total, desc="Processing", unit="patch") as pbar:
        for y in y_list:
            for x in x_list:
                patch_h = data_h[y:y+HR_SIZE, x:x+HR_SIZE]
                patch_o_dat = data_o_aligned[y:y+HR_SIZE, x:x+HR_SIZE]
                
                # ==================================================
                # 🔥 NUOVO FILTRO DI COPERTURA
                # ==================================================
                # Conta quanti pixel NON sono zero/neri
                valid_pixels_h = np.count_nonzero(patch_h > MIN_PIXEL_VALUE)
                total_pixels = HR_SIZE * HR_SIZE
                coverage = valid_pixels_h / total_pixels
                
                # Se la copertura è inferiore al 95%, SCARTA
                if coverage < MIN_COVERAGE:
                    skipped += 1
                    pbar.update(1)
                    continue
                # ==================================================

                patch_o_in = resize(patch_o_dat, (LR_SIZE, LR_SIZE), 
                                  anti_aliasing=True, preserve_range=True).astype(np.float32)

                pair_dir = output_dir / f"pair_{count:05d}"
                pair_dir.mkdir(exist_ok=True)
                
                fits.PrimaryHDU(data=patch_h, header=header_h).writeto(pair_dir/"hubble.fits", overwrite=True)
                h_lr = header_h.copy()
                h_lr['NAXIS1'], h_lr['NAXIS2'] = LR_SIZE, LR_SIZE
                fits.PrimaryHDU(data=patch_o_in, header=h_lr).writeto(pair_dir/"observatory.fits", overwrite=True)
                
                if count % SAVE_MAP_EVERY_N == 0:
                    save_8panel_card(
                        mosaic_h=data_h, mosaic_o_aligned=data_o_aligned,
                        mosaic_o_visual=visual_rotated, mosaic_o_raw=data_o_raw,
                        patch_h=patch_h, patch_o_in=patch_o_in,
                        x=x, y=y, wcs_h=wcs_h, wcs_orig=wcs_orig,
                        vmin_h=vmin_h, vmax_h=vmax_h, vmin_o=vmin_o, vmax_o=vmax_o,
                        save_path=pair_dir/"context_card.png"
                    )
                
                count += 1
                pbar.update(1)
                
    print("\n✅ Finito.")
    print(f"   Patch Valide: {count}")
    print(f"   Patch Scartate (Bordi/Vuote): {skipped}")

if __name__ == "__main__":
    target = select_target_directory()
    if target:
        create_dataset_filtered(target)