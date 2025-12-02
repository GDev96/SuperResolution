import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from skimage.transform import resize
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings

warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE =================
# Punta alla cartella creata dallo step precedente
DATASET_DIR = Path(r"c:\Users\fratt\Documents\GitHub\SuperResolution\data\M1\6_patches_aligned")
NUM_SAMPLES = 5  # Quante coppie vuoi controllare
# ==================================================

def normalize(data):
    """Normalizza tra 0 e 1 per visualizzazione (taglia outlier luminosi)"""
    d = np.nan_to_num(data)
    vmin, vmax = np.percentile(d, [1, 99.5])
    return np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1)

def get_corner_coords(header, data_shape):
    """Calcola le coordinate RA/DEC dei quattro angoli della patch."""
    try:
        wcs = WCS(header)
        ny, nx = data_shape
        
        # Le coordinate dei pixel degli angoli (0-based)
        corner_pixels = np.array([
            [0, 0],       # Top-Left (TL)
            [nx, 0],      # Top-Right (TR)
            [0, ny],      # Bottom-Left (BL)
            [nx, ny]      # Bottom-Right (BR)
        ])
        
        # Converte pixel in coordinate celesti
        corner_world = wcs.pixel_to_world(corner_pixels[:, 0], corner_pixels[:, 1])
        
        # Formatta le stringhe degli angoli (usiamo solo 4 decimali per non sovraffollare il grafico)
        coords = {}
        for i, label in enumerate(['TL', 'TR', 'BL', 'BR']):
            coords[label] = f"RA: {corner_world[i].ra.deg:.4f} DEC: {corner_world[i].dec.deg:.4f}"
            
        return coords, corner_world
    except Exception as e:
        return {'Error': str(e)}, None

def calculate_mismatch(h_coords, o_coords):
    """Calcola lo scostamento angolare tra i centri delle due patch."""
    try:
        # Calcoliamo la differenza tra i due angoli TL (Top-Left) in arcsec
        h_tl = SkyCoord(h_coords[0].ra, h_coords[0].dec)
        o_tl = SkyCoord(o_coords[0].ra, o_coords[0].dec)
        sep = h_tl.separation(o_tl).arcsec
        return f"{sep:.2f} arcsec"
    except:
        return "N/A"

def inspect_dataset():
    if not DATASET_DIR.exists():
        print(f"❌ Cartella non trovata: {DATASET_DIR}")
        return

    # Cerca tutte le cartelle 'pair_XXXXXX' ricorsivamente (in train/val/test)
    pair_folders = list(DATASET_DIR.glob("**/pair_*"))
    
    if not pair_folders:
        print("❌ Nessuna coppia trovata. Hai confermato il salvataggio nello step precedente?")
        return

    print(f"📦 Trovate {len(pair_folders)} coppie nel dataset.")
    print(f"🔍 Ne mostro {NUM_SAMPLES} a caso...")

    selected = random.sample(pair_folders, min(len(pair_folders), NUM_SAMPLES))

    for pair_dir in selected:
        h_path = pair_dir / "hubble.fits"
        o_path = pair_dir / "observatory.fits"

        if not h_path.exists() or not o_path.exists(): continue

        with fits.open(h_path) as h, fits.open(o_path) as o:
            hr_data = h[0].data
            hr_header = h[0].header
            lr_data = o[0].data
            lr_header = o[0].header
        
        # 1. Calcola Coordinate Angolari
        h_coords_str, h_coords_obj = get_corner_coords(hr_header, hr_data.shape)
        o_coords_str, o_coords_obj = get_corner_coords(lr_header, lr_data.shape)
        
        # 2. Calcola Mismatch tra i centri (o TL)
        mismatch = calculate_mismatch(h_coords_obj, o_coords_obj)

        # Normalizzazione
        hr_show = normalize(hr_data)
        lr_show = normalize(lr_data)

        # Resize LR to HR size per l'overlay (solo visivo)
        lr_resized = resize(lr_show, hr_show.shape, order=1) # Bilineare
        
        # --- Creazione Plot ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Allarghiamo un po' la figura

        # 1. Hubble HR
        axes[0].imshow(hr_show, origin='lower', cmap='inferno')
        axes[0].set_title(f"Hubble HR (Target)\n{hr_data.shape}")
        axes[0].axis('off')
        
        # Testo Coordinate (Hubble)
        axes[0].text(0.02, 0.98, f"TL: {h_coords_str['TL']}\nBL: {h_coords_str['BL']}", 
                     transform=axes[0].transAxes, color='white', fontsize=7, verticalalignment='top')
        axes[0].text(0.98, 0.98, f"TR: {h_coords_str['TR']}\nBR: {h_coords_str['BR']}", 
                     transform=axes[0].transAxes, color='white', fontsize=7, horizontalalignment='right', verticalalignment='top')


        # 2. Observatory LR
        axes[1].imshow(lr_show, origin='lower', cmap='viridis')
        axes[1].set_title(f"Observatory LR (Input)\n{lr_data.shape} (Origin: ~28px crop)")
        axes[1].axis('off')
        
        # Testo Coordinate (Osservatorio)
        axes[1].text(0.02, 0.98, f"TL: {o_coords_str['TL']}\nBL: {o_coords_str['BL']}", 
                     transform=axes[1].transAxes, color='white', fontsize=7, verticalalignment='top')
        axes[1].text(0.98, 0.98, f"TR: {o_coords_str['TR']}\nBR: {o_coords_str['BR']}", 
                     transform=axes[1].transAxes, color='white', fontsize=7, horizontalalignment='right', verticalalignment='top')


        # 3. Overlay (Controllo Allineamento)
        rgb = np.zeros((hr_show.shape[0], hr_show.shape[1], 3))
        rgb[..., 0] = hr_show     # Rosso = Hubble
        rgb[..., 1] = lr_resized  # Verde = Observatory
        
        axes[2].imshow(rgb, origin='lower')
        axes[2].set_title(f"Overlay Check\n{pair_dir.name} | Mismatch TL: {mismatch}")
        axes[2].axis('off')

        plt.suptitle(f"Validazione WCS Post-Estrazione | Coppia {pair_dir.name}", fontsize=14)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    inspect_dataset()