"""
STEP 3 (FINAL 95% FILTER): TAGLIO COPPIE + FILTRO COPERTURA + 8-PANEL VISUAL
------------------------------------------------------------------------
Features:
1. Scansiona i mosaici allineati (Step 2.5).
2. FILTRO RIGOROSO: Scarta qualsiasi patch che non abbia almeno il 
   95% di dati validi (elimina bordi neri e zone vuote).
3. Genera la Context Card a 8 pannelli per verifica visiva.

MODIFICA: I file PNG vengono salvati in una cartella separata 'coppie_patch_png'.

INPUT: aligned_hubble.fits, aligned_observatory.fits, final_mosaic_observatory.fits
       (Attesi in [data_dir]/5_mosaics/aligned_ready_for_crop/)
OUTPUT: 
  - Dati: 6_patches_final/pair_XXXXX/ (FITS)
  - Visual: coppie_patch_png/pair_XXXXX_context.png (PNG)
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
import subprocess

warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE GLOBALE =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# PARAMETRI DATASET
HR_SIZE = 512         # Dimensione patch Alta Risoluzione (Hubble)
LR_SIZE = 80          # Dimensione patch Bassa Risoluzione (Osservatorio Input)
STRIDE = 64           # Passo di scorrimento (overlap)

# SOGLIE DI QUALITÀ
MIN_PIXEL_VALUE = 0.0001 # Soglia per considerare un pixel come "dato valido"
MIN_COVERAGE = 0.99    # Filtro: copertura minima del 99%
SAVE_MAP_EVERY_N = 1     # Salva la Context Card ogni N patch
# ==========================================================

# --- FUNZIONI DI NORMALIZZAZIONE E PLOT ---

def get_global_normalization(data: np.ndarray) -> tuple[float, float]:
    data = np.nan_to_num(data)
    h, w = data.shape
    cy, cx = h//2, w//2
    sample_size = 2000
    
    if h > sample_size and w > sample_size: 
        sample = data[cy-sample_size//2:cy+sample_size//2, cx-sample_size//2:cx+sample_size//2]
    else: 
        sample = data
        
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(sample)
    return vmin, vmax

def normalize_with_limits(data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    data = np.nan_to_num(data)
    if vmax == vmin: return np.zeros_like(data)
    return np.clip((data - vmin) / (vmax - vmin), 0, 1)

def normalize_local(data: np.ndarray) -> np.ndarray:
    try:
        data_clean = np.nan_to_num(data)
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(data_clean)
        return np.clip((data_clean - vmin) / (vmax - vmin), 0, 1)
    except Exception: 
        return np.clip(np.nan_to_num(data), 0, 1)

def save_8panel_card(mosaic_h: np.ndarray, mosaic_o_aligned: np.ndarray, mosaic_o_visual: np.ndarray, 
                     mosaic_o_raw: np.ndarray, patch_h: np.ndarray, patch_o_in: np.ndarray, 
                     x: int, y: int, wcs_h: WCS, wcs_orig: WCS, 
                     vmin_h: float, vmax_h: float, vmin_o: float, vmax_o: float, save_path: Path):
    """Genera e salva la Context Card di 8 pannelli per verifica visiva."""
    
    fig = plt.figure(figsize=(28, 12))
    coverage_perc = np.count_nonzero(patch_h > MIN_PIXEL_VALUE) / (HR_SIZE * HR_SIZE) * 100
    fig.suptitle(f"DATASET SAMPLE: X={x} Y={y} | Coverage H: {coverage_perc:.1f}% (Required > {MIN_COVERAGE*100:.0f}%)", 
                 fontsize=18, fontweight='bold')
    gs = fig.add_gridspec(2, 4) 
    sc = 8 # Fattore di downsampling per le viste panoramiche

    # --- Riga 1: Viste Mosaico ---
    def plot_mosaic_view(ax, data, title, color, box_color, vmin, vmax, draw_box=True):
        ax.imshow(normalize_with_limits(data[::sc, ::sc], vmin, vmax), origin='lower', cmap='gray')
        ax.set_title(title, color=color, fontsize=12)
        if draw_box:
            ax.add_patch(patches.Rectangle((x/sc, y/sc), HR_SIZE/sc, HR_SIZE/sc, lw=2, edgecolor=box_color, facecolor='none'))
        ax.axis('off')

    # 1. Hubble (Master HR)
    plot_mosaic_view(fig.add_subplot(gs[0, 0]), mosaic_h, "1. Hubble Master (HR)", 'green', 'lime', vmin_h, vmax_h)
    
    # 2. Obs Aligned (Input LR - Allineato)
    plot_mosaic_view(fig.add_subplot(gs[0, 1]), mosaic_o_aligned, "2. Obs Allineato (Input LR)", 'magenta', 'cyan', vmin_o, vmax_o)

    # 3. Obs Rotated Visual (Solo per Visualizzazione)
    plot_mosaic_view(fig.add_subplot(gs[0, 2]), mosaic_o_visual, "3. Obs Ruotato (Visual Check)", 'orange', 'orange', vmin_o, vmax_o)

    # 4. Obs Native (Mosaico Originale NON riproiettato)
    ax4 = fig.add_subplot(gs[0, 3])
    plot_mosaic_view(ax4, mosaic_o_raw, "4. Obs Originale (Nativo)", 'white', 'red', vmin_o, vmax_o, draw_box=False)
    # Aggiungi box distorto in coordinate native
    try:
        corners_pix = np.array([[x, y], [x+HR_SIZE, y+HR_SIZE]])
        corners_world = wcs_h.pixel_to_world(corners_pix[:,0], corners_pix[:,1])
        cx_orig, cy_orig = wcs_orig.world_to_pixel(corners_world[0], corners_world[1]) 
        
        ox, oy = np.min(cx_orig), np.min(cy_orig)
        ow, oh = np.ptp(cx_orig), np.ptp(cy_orig)
        ax4.add_patch(patches.Rectangle((ox/sc, oy/sc), ow/sc, oh/sc, lw=2, edgecolor='red', facecolor='none'))
    except Exception:
        pass 

    # 5. Patch H (Target HR)
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(normalize_local(patch_h), origin='lower', cmap='viridis')
    ax5.set_title("5. Hubble Target (HR)")
    ax5.axis('off')

    # 6. Patch O (Input LR)
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(normalize_local(patch_o_in), origin='lower', cmap='magma')
    ax6.set_title(f"6. Obs Input (LR {LR_SIZE}x{LR_SIZE})")
    ax6.axis('off')

    # 7. Overlay (Check di Allineamento)
    ax7 = fig.add_subplot(gs[1, 2])
    inp_s = resize(normalize_local(patch_o_in), (HR_SIZE, HR_SIZE), order=0) 
    tar_n = normalize_local(patch_h)
    rgb = np.zeros((HR_SIZE, HR_SIZE, 3))
    rgb[..., 0] = inp_s * 0.8
    rgb[..., 1] = tar_n
    rgb[..., 2] = inp_s * 0.8
    ax7.imshow(rgb, origin='lower')
    ax7.set_title("7. Check Overlay (Input vs Target)")
    ax7.axis('off')

    # 8. Info
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    txt = f"Patch Estratta:\nHR_SIZE: {HR_SIZE}\nLR_SIZE: {LR_SIZE}\nSTRIDE: {STRIDE}\n"
    ax8.text(0.1, 0.5, txt, fontsize=12, family='monospace', verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=90)
    plt.close(fig)


# ================= FUNZIONE PRINCIPALE =================

def create_dataset_filtered(base_dir: Path) -> tuple[bool, Path] | None:
    """
    Estrae le coppie di patch (Hubble HR, Observatory LR).
    Ritorna (True, base_dir) se successo, altrimenti None.
    """
    
    # 1. Definizione Path
    aligned_dir = base_dir / '5_mosaics' / 'aligned_ready_for_crop'
    mosaics_dir = base_dir / '5_mosaics'
    
    # Cartella per i dati FITS (per il training)
    output_dir = base_dir / '6_patches_final'
    
    # Cartella separata per le immagini PNG (Visualizzazione)
    png_output_dir = base_dir / 'coppie_patch_png'
    
    f_h_align = aligned_dir / 'aligned_hubble.fits'
    f_o_align = aligned_dir / 'aligned_observatory.fits'
    f_o_orig  = mosaics_dir / 'final_mosaic_observatory.fits'

    # 2. Controllo file necessari
    missing_files = []
    if not f_h_align.exists(): missing_files.append(f_h_align.name)
    if not f_o_align.exists(): missing_files.append(f_o_align.name)
    if not f_o_orig.exists(): missing_files.append(f_o_orig.name)

    if missing_files:
        print("\n" + "="*70)
        print(f"❌ ERRORE: Step 3 interrotto per {base_dir.name}.")
        print("I seguenti file non sono stati trovati. Assicurati di aver eseguito lo Step 2.5:")
        for f in missing_files:
            print(f"   -> Manca: {f} (Atteso in: {aligned_dir} o {mosaics_dir})")
        print("="*70 + "\n")
        return None

    # Pulisci e crea la cartella di output dati (FITS)
    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pulisci e crea la cartella di output PNG
    if png_output_dir.exists(): shutil.rmtree(png_output_dir)
    png_output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Caricamento Dati e WCS
    print(f"\n⚙️  Caricamento Dati da {base_dir.name}...")
    try:
        # Funzione helper per caricare e pulire i dati
        def load_fits_data(filepath):
            with fits.open(filepath) as h:
                data = np.nan_to_num(h[0].data)
                if data.ndim > 2: data = data[0]
                return data, WCS(h[0].header), h[0].header

        data_h, wcs_h, header_h = load_fits_data(f_h_align)
        data_o_aligned, _, _ = load_fits_data(f_o_align)
        data_o_raw, wcs_orig, _ = load_fits_data(f_o_orig)
        
    except Exception as e:
        print(f"❌ ERRORE grave durante la lettura/parsing dei file FITS: {e}")
        return None

    # 4. Pre-calcolo (Contrasto e Rotazione Visuale)
    print("⚖️  Calcolo Contrasto e Mappa Rotata per Visualizzazione...")
    vmin_h, vmax_h = get_global_normalization(data_h)
    vmin_o, vmax_o = get_global_normalization(data_o_raw)

    visual_rotated, _ = reproject_interp((data_o_raw, wcs_orig), wcs_h, shape_out=data_h.shape)
    visual_rotated = np.nan_to_num(visual_rotated)

    h_dim, w_dim = data_h.shape
    y_list = list(range(0, h_dim - HR_SIZE + 1, STRIDE))
    x_list = list(range(0, w_dim - HR_SIZE + 1, STRIDE))
    total = len(y_list) * len(x_list)
    count = 0
    skipped = 0
    
    # 5. Loop di Estrazione e Filtro
    print(f"\n✂️  Estrazione con Filtro {MIN_COVERAGE*100:.0f}%...")
    with tqdm(total=total, desc="Processing", unit="patch") as pbar:
        for y in y_list:
            for x in x_list:
                patch_h = data_h[y:y+HR_SIZE, x:x+HR_SIZE]
                patch_o_dat = data_o_aligned[y:y+HR_SIZE, x:x+HR_SIZE]
                
                # --- FILTRO DI COPERTURA ---
                valid_pixels_h = np.count_nonzero(patch_h > MIN_PIXEL_VALUE)
                coverage = valid_pixels_h / (HR_SIZE * HR_SIZE)
                
                if coverage < MIN_COVERAGE:
                    skipped += 1
                    pbar.update(1)
                    continue
                # --------------------------

                # Resample da HR (allineato) a LR (input del modello)
                patch_o_in = resize(patch_o_dat, (LR_SIZE, LR_SIZE), 
                                  anti_aliasing=True, preserve_range=True).astype(np.float32)

                # 6. Salvataggio Coppie FITS (Cartella strutturata per training)
                pair_dir = output_dir / f"pair_{count:05d}"
                pair_dir.mkdir(exist_ok=True)
                
                fits.PrimaryHDU(data=patch_h.astype(np.float32), header=header_h).writeto(pair_dir/"hubble.fits", overwrite=True)
                
                h_lr = header_h.copy()
                h_lr['NAXIS1'], h_lr['NAXIS2'] = LR_SIZE, LR_SIZE
                fits.PrimaryHDU(data=patch_o_in, header=h_lr).writeto(pair_dir/"observatory.fits", overwrite=True)
                
                # 7. Salvataggio Context Card PNG (Cartella separata dedicata)
                if count % SAVE_MAP_EVERY_N == 0:
                    png_filename = f"pair_{count:05d}_context.png"
                    save_8panel_card(
                        mosaic_h=data_h, mosaic_o_aligned=data_o_aligned,
                        mosaic_o_visual=visual_rotated, mosaic_o_raw=data_o_raw,
                        patch_h=patch_h, patch_o_in=patch_o_in,
                        x=x, y=y, wcs_h=wcs_h, wcs_orig=wcs_orig,
                        vmin_h=vmin_h, vmax_h=vmax_h, vmin_o=vmin_o, vmax_o=vmax_o,
                        save_path=png_output_dir / png_filename
                    )
                
                count += 1
                pbar.update(1)
                
    print("\n✅ Finito.")
    print(f"   Patch Valide: {count}")
    print(f"   Patch Scartate (Bordi/Vuote): {skipped}")
    print(f"   PNG salvati in: {png_output_dir}")
    
    return True, base_dir

# --- FUNZIONI DI INPUT E AVANZAMENTO ---

def select_target_directory_manual():
    """Permette all'utente di selezionare la cartella target dei dati (Modalità Manuale)."""
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET (Filtro 95%)".center(70))
    print("📂"*35)
    subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    if not subdirs: 
        print(f"❌ ERRORE: Nessuna cartella target trovata in {ROOT_DATA_DIR}")
        return None
        
    for i, d in enumerate(subdirs):
        print(f"   {i+1}: {d.name}")
        
    try:
        choice = int(input(f"\n👉 Seleziona (1-{len(subdirs)}): ").strip())
        if 0 < choice <= len(subdirs): return subdirs[choice-1]
    except ValueError:
        print("❌ Scelta non valida.")
        pass
    
    return None

def ask_continue_to_split(base_dir: Path) -> bool:
    """Chiede se proseguire con lo Step 4 (Split)."""
    next_script_name = 'Modello_2_pre_da_usopatch_dataset_step3.py'
    
    print("\n" + "="*70)
    print("🎯 TAGLIO PATCH COMPLETATO!")
    print("="*70)
    print(f"\n📋 Vuoi eseguire lo Step 4 (Split Train/Val/Test) ora?")
    print(f"   Avvierà: **{next_script_name}**")
    
    while True:
        print("\n" + "─"*70)
        choice = input(f"👉 Continua con lo Split? [S/n, default=S]: ").strip().lower()
        if choice in ('', 's', 'si', 'y', 'yes'):
            return True
        elif choice in ('n', 'no'):
            return False
        else:
            print("❌ Scelta non valida.")

# ================= PUNTO DI INGRESSO =================

if __name__ == "__main__":
    
    # Gestione input automatico o manuale
    if len(sys.argv) > 1:
        target = Path(sys.argv[1]).resolve()
    else:
        target = select_target_directory_manual()
    
    if target and target.is_dir():
        # Utilizziamo 'result' per gestire l'errore NoneType
        result = create_dataset_filtered(target)
        
        if result is not None:
            success, base_dir = result
            if success and ask_continue_to_split(base_dir):
                try:
                    split_script_name = 'Modello_2_pre_da_usopatch_dataset_step3.py'
                    next_script = CURRENT_SCRIPT_DIR / split_script_name
                    
                    if next_script.exists():
                        print(f"\n🚀 Avvio Split per {base_dir.name}...")
                        subprocess.run([sys.executable, str(next_script), str(base_dir)], check=True)
                    else:
                        print(f"\n⚠️  Script {next_script.name} non trovato in {CURRENT_SCRIPT_DIR}")
                except Exception as e:
                    print(f"❌ Errore avvio script di split: {e}")