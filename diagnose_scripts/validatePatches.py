import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from skimage.transform import resize
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
            lr_data = o[0].data

        # Normalizzazione
        hr_show = normalize(hr_data)
        lr_show = normalize(lr_data)

        # Resize LR to HR size per l'overlay (solo visivo)
        lr_resized = resize(lr_show, hr_show.shape, order=1) # order=1 = bilineare

        # Creazione Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Hubble HR
        axes[0].imshow(hr_show, origin='lower', cmap='inferno')
        axes[0].set_title(f"Hubble HR (Target)\n{hr_data.shape}")
        axes[0].axis('off')

        # 2. Observatory LR
        axes[1].imshow(lr_show, origin='lower', cmap='viridis')
        axes[1].set_title(f"Observatory LR (Input)\n{lr_data.shape}\n(Origin: ~28px crop)")
        axes[1].axis('off')

        # 3. Overlay (Controllo Allineamento)
        rgb = np.zeros((hr_show.shape[0], hr_show.shape[1], 3))
        rgb[..., 0] = hr_show    # Rosso = Hubble
        rgb[..., 1] = lr_resized # Verde = Observatory
        
        axes[2].imshow(rgb, origin='lower')
        axes[2].set_title(f"Overlay Check\n{pair_dir.name}")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    inspect_dataset()