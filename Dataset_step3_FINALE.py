"""
STEP 3 (FINAL): TAGLIO COPPIE + FILTRO QUALITÀ + VISUALIZZAZIONE GRAYSCALE OTTIMIZZATA
------------------------------------------------------------------------
MIGLIORAMENTO VISIVO:
1. Normalizzazione con STRETCH (Radice Quadrata): Rende visibili i grigi e le nebulosità 
   invece di mostrare solo bianco/nero netto.
2. Visualizzazione: Scala di Grigi ricca di dettagli + Puntini Gialli su stelle allineate.
3. Organizzazione: PNG in cartella separata, FITS in cartella dataset.

INPUT: 
  - aligned_hubble.fits, aligned_observatory.fits (Dati geometricamente corretti)
  - final_mosaic_hubble.fits, final_mosaic_observatory.fits (Dati Raw per visualizzazione)
  
OUTPUT: 
  - Dati Training: 6_patches_final/pair_XXXXX/ (FITS)
  - Controllo Visivo: coppie_patch_png/pair_XXXXX_context.png (PNG)
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
from astropy.visualization import PercentileInterval
from skimage.transform import resize
from reproject import reproject_interp
from tqdm import tqdm
import warnings
import subprocess
from scipy.ndimage import maximum_filter

warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE GLOBALE =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# PARAMETRI DATASET
HR_SIZE = 512         # Dimensione patch Hubble
LR_SIZE = 80          # Dimensione patch Osservatorio
STRIDE = 64           # Passo di scorrimento

# SOGLIE DI QUALITÀ
MIN_PIXEL_VALUE = 0.0001 
MIN_COVERAGE = 0.95      # Scarta se meno del 95% di dati validi
SAVE_MAP_EVERY_N = 1     # Frequenza salvataggio PNG

# PARAMETRI VISUALIZZAZIONE STELLE
PEAK_THRESHOLD = 0.8      # Soglia rilevamento picchi
ALIGNMENT_THRESHOLD = 0.8 # Soglia sovrapposizione per considerare "allineato"
MARKER_COLOR = 'yellow'   
MARKER_SIZE = 20
# ==========================================================

# --- FUNZIONI DI NORMALIZZAZIONE E UTILITY ---

def get_robust_normalization(data: np.ndarray) -> tuple[float, float]:
    """
    Calcola vmin/vmax ignorando i bordi neri e usando percentili.
    """
    data = np.nan_to_num(data)
    valid_mask = data > MIN_PIXEL_VALUE
    
    if np.sum(valid_mask) < 100:
        return 0.0, 1.0
        
    valid_pixels = data[valid_mask]
    
    # Percentile 99.5% per tagliare le stelle saturate
    interval = PercentileInterval(99.5)
    vmin, vmax = interval.get_limits(valid_pixels)
    
    if vmax <= vmin: 
        return np.min(valid_pixels), np.max(valid_pixels)
        
    return vmin, vmax

def normalize_with_stretch(data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """
    Normalizza e applica uno STRETCH (Radice Quadrata) per mostrare i grigi.
    """
    data = np.nan_to_num(data)
    if vmax <= vmin: return np.zeros_like(data)
    
    # 1. Clipping Lineare
    clipped = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    
    # 2. Stretch Non-Lineare (Gamma Correction / Sqrt)
    # Questo espande i toni scuri trasformandoli in grigi visibili
    stretched = np.sqrt(clipped) 
    
    return stretched

def normalize_local_stretch(data: np.ndarray) -> np.ndarray:
    """Normalizza la singola patch con stretch per massima visibilità."""
    try:
        data = np.nan_to_num(data)
        # Usa percentili locali
        interval = PercentileInterval(99.5)
        vmin, vmax = interval.get_limits(data)
        
        if vmax <= vmin: return np.zeros_like(data)
        
        clipped = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        # Applica lo stesso stretch (Radice Quadrata)
        return np.sqrt(clipped)
    except:
        return np.zeros_like(data)

def find_aligned_stars(patch_h: np.ndarray, patch_o_in: np.ndarray) -> tuple:
    """Trova le coordinate delle stelle che coincidono in entrambe le immagini."""
    tar_n = normalize_local_stretch(patch_h)
    inp_s = resize(normalize_local_stretch(patch_o_in), (HR_SIZE, HR_SIZE), order=0)
    
    overlap_intensity = np.minimum(inp_s, tar_n)
    
    footprint = np.ones((3, 3))
    local_max = maximum_filter(tar_n, footprint=footprint)
    is_peak = (tar_n == local_max) & (tar_n > PEAK_THRESHOLD * tar_n.max())
    
    aligned_mask = is_peak & (overlap_intensity > ALIGNMENT_THRESHOLD)
    
    y, x = np.where(aligned_mask)
    return x, y

# --- FUNZIONE DI PLOT (CONTEXT CARD) ---

def save_8panel_card(mosaic_h, mosaic_o_aligned, mosaic_h_raw, mosaic_o_raw, 
                     patch_h, patch_o_in, 
                     x, y, wcs_h, wcs_orig_h, wcs_orig_o,
                     vmin_h, vmax_h, vmin_o, vmax_o, save_path):
    
    fig = plt.figure(figsize=(28, 12))
    coverage_perc = np.count_nonzero(patch_h > MIN_PIXEL_VALUE) / (HR_SIZE * HR_SIZE) * 100
    fig.suptitle(f"DATASET SAMPLE: X={x} Y={y} | Coverage: {coverage_perc:.1f}%", fontsize=18, fontweight='bold')
    gs = fig.add_gridspec(2, 4)
    sc = 8 

    # Helper per visualizzare con stretch
    def plot_mosaic(ax, data, title, color, box_col, v_min, v_max, wcs_curr=None, wcs_target=None):
        # Usa normalize_with_stretch invece di quella lineare
        ax.imshow(normalize_with_stretch(data[::sc, ::sc], v_min, v_max), origin='lower', cmap='gray')
        ax.set_title(title, color=color, fontsize=12)
        ax.axis('off')
        
        if wcs_curr and wcs_target:
            try:
                corners_pix = np.array([[x, y], [x+HR_SIZE, y], [x+HR_SIZE, y+HR_SIZE], [x, y+HR_SIZE]])
                corners_world = wcs_target.pixel_to_world(corners_pix[:,0], corners_pix[:,1])
                px_curr = wcs_curr.world_to_pixel(corners_world)
                poly_coords = np.column_stack((px_curr[0]/sc, px_curr[1]/sc))
                rect = patches.Polygon(poly_coords, linewidth=2, edgecolor=box_col, facecolor='none')
                ax.add_patch(rect)
            except:
                pass

    # --- RIGA 1: Mosaici (con Stretch) ---
    plot_mosaic(fig.add_subplot(gs[0, 0]), mosaic_h, "1. Hubble Master (HR)", 'green', 'lime', vmin_h, vmax_h, wcs_h, wcs_h)
    plot_mosaic(fig.add_subplot(gs[0, 1]), mosaic_o_aligned, "2. Obs Aligned (LR)", 'magenta', 'cyan', vmin_o, vmax_o, wcs_h, wcs_h)
    plot_mosaic(fig.add_subplot(gs[0, 2]), mosaic_h_raw, "3. Hubble Native (Raw)", 'orange', 'orange', vmin_h, vmax_h, wcs_orig_h, wcs_h)
    plot_mosaic(fig.add_subplot(gs[0, 3]), mosaic_o_raw, "4. Obs Native (Raw)", 'white', 'red', vmin_o, vmax_o, wcs_orig_o, wcs_h)

    # --- RIGA 2: Patch e Dettagli (con Stretch) ---
    
    x_stars, y_stars = find_aligned_stars(patch_h, patch_o_in)
    x_stars_lr = x_stars * (LR_SIZE / HR_SIZE)
    y_stars_lr = y_stars * (LR_SIZE / HR_SIZE)

    # 5. Patch Hubble
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(normalize_local_stretch(patch_h), origin='lower', cmap='gray')
    ax5.scatter(x_stars, y_stars, s=MARKER_SIZE, c=MARKER_COLOR, marker='.', alpha=0.7)
    ax5.set_title("5. Hubble Target (HR)", color='black')
    ax5.axis('off')

    # 6. Patch Obs
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(normalize_local_stretch(patch_o_in), origin='lower', cmap='gray')
    ax6.scatter(x_stars_lr, y_stars_lr, s=MARKER_SIZE, c=MARKER_COLOR, marker='.', alpha=0.7)
    ax6.set_title("6. Obs Input (LR)", color='black')
    ax6.axis('off')

    # 7. Check Overlay
    ax7 = fig.add_subplot(gs[1, 2])
    inp_s = resize(normalize_local_stretch(patch_o_in), (HR_SIZE, HR_SIZE), order=0)
    tar_n = normalize_local_stretch(patch_h)
    rgb = np.zeros((HR_SIZE, HR_SIZE, 3))
    rgb[..., 0] = inp_s * 0.9 # Rosso
    rgb[..., 1] = tar_n * 0.9 # Verde
    ax7.imshow(rgb, origin='lower')
    ax7.set_title("7. Overlay (R=Inp, G=Tar)")
    ax7.axis('off')

    # 8. Info
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    txt = (f"INFO PATCH:\n"
           f"- HR Size: {HR_SIZE}x{HR_SIZE}\n"
           f"- LR Size: {LR_SIZE}x{LR_SIZE}\n"
           f"- Stride:  {STRIDE}\n\n"
           f"STATISTICHE:\n"
           f"- Copertura: {coverage_perc:.2f}%\n"
           f"- Stelle Allineate: {len(x_stars)}\n"
           f"- Norm H: [{vmin_h:.4f}, {vmax_h:.4f}]")
    ax8.text(0.1, 0.5, txt, fontsize=13, family='monospace', verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=90)
    plt.close(fig)

# ================= MAIN PROCESSING =================

def create_dataset_filtered(base_dir: Path) -> tuple[bool, Path] | None:
    """Pipeline principale di estrazione."""
    
    # 1. Definizione Cartelle
    aligned_dir = base_dir / '5_mosaics' / 'aligned_ready_for_crop'
    mosaics_dir = base_dir / '5_mosaics'
    
    output_fits_dir = base_dir / '6_patches_final'      
    output_png_dir = base_dir / 'coppie_patch_png'      
    
    # 2. File Richiesti
    f_h_align = aligned_dir / 'aligned_hubble.fits'
    f_o_align = aligned_dir / 'aligned_observatory.fits'
    f_h_raw   = mosaics_dir / 'final_mosaic_hubble.fits'
    f_o_raw   = mosaics_dir / 'final_mosaic_observatory.fits'

    missing = [f.name for f in [f_h_align, f_o_align, f_h_raw, f_o_raw] if not f.exists()]
    if missing:
        print(f"\n❌ ERRORE: Mancano i file: {missing}")
        return None

    # Setup Cartelle Output
    if output_fits_dir.exists(): shutil.rmtree(output_fits_dir)
    output_fits_dir.mkdir(parents=True, exist_ok=True)
    
    if output_png_dir.exists(): shutil.rmtree(output_png_dir)
    output_png_dir.mkdir(parents=True, exist_ok=True)

    # 3. Caricamento Dati
    print(f"\n⚙️  Caricamento Dati da {base_dir.name}...")
    try:
        def load_fits(p):
            with fits.open(p) as h:
                d = np.nan_to_num(h[0].data)
                if d.ndim > 2: d = d[0]
                return d, WCS(h[0].header), h[0].header

        data_h, wcs_h, header_h = load_fits(f_h_align)
        data_o, _, _ = load_fits(f_o_align)
        data_h_raw, wcs_h_raw, _ = load_fits(f_h_raw)
        data_o_raw, wcs_o_raw, _ = load_fits(f_o_raw)
        
    except Exception as e:
        print(f"❌ Errore caricamento FITS: {e}")
        return None

    # 4. Calcolo Normalizzazione Globale (Robusta)
    print("⚖️  Calcolo livelli di normalizzazione...")
    vmin_h, vmax_h = get_robust_normalization(data_h)
    vmin_o, vmax_o = get_robust_normalization(data_o_raw)
    print(f"   Hubble Range: {vmin_h:.4f} - {vmax_h:.4f}")
    print(f"   Obs Range:    {vmin_o:.4f} - {vmax_o:.4f}")

    # 5. Estrazione Patch
    h_dim, w_dim = data_h.shape
    y_list = list(range(0, h_dim - HR_SIZE + 1, STRIDE))
    x_list = list(range(0, w_dim - HR_SIZE + 1, STRIDE))
    total = len(y_list) * len(x_list)
    
    count = 0
    skipped = 0
    
    print(f"\n✂️  Estrazione Patch (Filtro > {MIN_COVERAGE*100:.0f}%)...")
    with tqdm(total=total, unit="patch") as pbar:
        for y in y_list:
            for x in x_list:
                # Estrai Crop
                patch_h = data_h[y:y+HR_SIZE, x:x+HR_SIZE]
                patch_o_big = data_o[y:y+HR_SIZE, x:x+HR_SIZE]
                
                # Filtro Copertura
                valid_px = np.count_nonzero(patch_h > MIN_PIXEL_VALUE)
                if (valid_px / patch_h.size) < MIN_COVERAGE:
                    skipped += 1
                    pbar.update(1)
                    continue

                # Resize Obs per input modello
                patch_o_lr = resize(patch_o_big, (LR_SIZE, LR_SIZE), 
                                  anti_aliasing=True, preserve_range=True).astype(np.float32)

                # Salva FITS
                pair_path = output_fits_dir / f"pair_{count:05d}"
                pair_path.mkdir()
                fits.PrimaryHDU(patch_h.astype(np.float32), header=header_h).writeto(pair_path/"hubble.fits")
                
                h_lr = header_h.copy()
                h_lr['NAXIS1'], h_lr['NAXIS2'] = LR_SIZE, LR_SIZE
                fits.PrimaryHDU(patch_o_lr, header=h_lr).writeto(pair_path/"observatory.fits")
                
                # Salva PNG (Context Card)
                if count % SAVE_MAP_EVERY_N == 0:
                    save_8panel_card(
                        data_h, data_o, data_h_raw, data_o_raw,
                        patch_h, patch_o_lr,
                        x, y, wcs_h, wcs_h_raw, wcs_o_raw,
                        vmin_h, vmax_h, vmin_o, vmax_o,
                        output_png_dir / f"pair_{count:05d}_context.png"
                    )
                
                count += 1
                pbar.update(1)

    print(f"\n✅ Completato. {count} patch valide generate.")
    print(f"📂 Dataset: {output_fits_dir}")
    print(f"🖼️  Visual:  {output_png_dir}")
    return True, base_dir

# --- UTILITY DI INPUT ---

def select_target_directory():
    subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    if not subdirs: return None
    print("\nSELEZIONA TARGET:")
    for i, d in enumerate(subdirs): print(f" {i+1}: {d.name}")
    try:
        idx = int(input("Scelta: ")) - 1
        return subdirs[idx] if 0 <= idx < len(subdirs) else None
    except: return None

def ask_continue(base_dir):
    next_script = CURRENT_SCRIPT_DIR / 'Modello_2_pre_da_usopatch_dataset_step3.py'
    print(f"\nVuoi lanciare lo split dataset? (Script: {next_script.name})")
    if input("S/n: ").lower() in ['s', '', 'y']:
        subprocess.run([sys.executable, str(next_script), str(base_dir)])

# ================= ENTRY POINT =================

if __name__ == "__main__":
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else select_target_directory()
    
    if target:
        res = create_dataset_filtered(target)
        if res:
            ask_continue(res[1])