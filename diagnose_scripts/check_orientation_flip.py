import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from skimage.transform import resize
import warnings

warnings.filterwarnings('ignore')

# CONFIGURAZIONE PATH
ROOT_DATA_DIR = Path(r"c:\Users\fratt\Documents\GitHub\SuperResolution\data")

def normalize(data):
    d = np.nan_to_num(data)
    return (d - np.min(d)) / (np.max(d) - np.min(d) + 1e-6)

def check_orientation():
    # Cerchiamo il target M1
    target_dir = ROOT_DATA_DIR / "M1"
    h_dir = target_dir / '3_registered_native' / 'hubble'
    o_dir = target_dir / '3_registered_native' / 'observatory'
    
    # Prendiamo i file
    h_file = list(h_dir.glob("*.fits"))[0]
    # Cerchiamo un file centrale (es. "ha 1") o il primo
    o_files = list(o_dir.glob("*ha 1*wcs.fits"))
    o_file = o_files[0] if o_files else list(o_dir.glob("*.fits"))[0]

    print(f"Visualizing: {h_file.name} vs {o_file.name}")

    with fits.open(h_file) as h, fits.open(o_file) as o:
        hr = h[0].data
        lr = o[0].data
        if hr.ndim==3: hr=hr[0]
        if lr.ndim==3: lr=lr[0]

    # Normalizza
    hr = normalize(hr)
    lr = normalize(lr)

    # Resize LR to HR dimensions for overlay
    lr_big = resize(lr, hr.shape, anti_aliasing=True)

    # Setup Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    transformations = [
        ("Originale", lr_big),
        ("Flip Left-Right (Orizzontale)", np.fliplr(lr_big)),
        ("Flip Up-Down (Verticale)", np.flipud(lr_big)),
        ("Ruotato 180°", np.rot90(lr_big, 2))
    ]

    for ax, (title, img_trans) in zip(axes.flatten(), transformations):
        # Create Red-Green Overlay
        rgb = np.zeros((hr.shape[0], hr.shape[1], 3))
        rgb[..., 0] = hr        # RED = Hubble
        rgb[..., 1] = img_trans # GREEN = Obs (Trasformato)
        
        ax.imshow(rgb, origin='lower')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle("Test Orientamento: Cerca il pannello dove Rosso e Verde COINCIDONO (Giallo)", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_orientation()