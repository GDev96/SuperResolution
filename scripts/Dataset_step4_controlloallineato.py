import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
from astropy.visualization import ZScaleInterval
from scipy.ndimage import zoom, center_of_mass
from scipy.stats import pearsonr
from tqdm import tqdm 

# ================= CONFIGURAZIONE PATH =================
CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR.parent / "data"

# ================= SOGLIE DI QUALITÀ =================
# Se la correlazione è sotto 0.5 o lo shift è sopra 20px, viene segnato come DUBBIO
THRESHOLD_CORR = 0.40
THRESHOLD_SHIFT = 30.0 

def normalize_image(data):
    """Normalizza l'immagine per la visualizzazione (0-1)"""
    try:
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(data)
        norm_data = (data - vmin) / (vmax - vmin)
        return np.clip(norm_data, 0, 1)
    except:
        data = np.nan_to_num(data)
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)

def calculate_alignment_metrics(img_lr, img_hr):
    """
    Calcola metriche matematiche di allineamento.
    """
    # Upscale LR
    scale_factor = img_hr.shape[0] / img_lr.shape[0]
    lr_upscaled = zoom(img_lr, scale_factor, order=1)
    
    # Taglio di sicurezza
    h, w = img_hr.shape
    lr_upscaled = lr_upscaled[:h, :w]
    
    # 1. Correlazione di Pearson
    lr_flat = np.nan_to_num(lr_upscaled.flatten())
    hr_flat = np.nan_to_num(img_hr.flatten())
    
    if np.std(lr_flat) == 0 or np.std(hr_flat) == 0:
        return 0.0, 999.0 # Evita divisioni per zero su immagini nere
        
    correlation, _ = pearsonr(lr_flat, hr_flat)
    
    # 2. Distanza Baricentro
    lr_bg = lr_upscaled - np.median(lr_upscaled)
    hr_bg = img_hr - np.median(img_hr)
    lr_bg[lr_bg < 0] = 0
    hr_bg[hr_bg < 0] = 0
    
    if np.sum(lr_bg) == 0 or np.sum(hr_bg) == 0:
        return correlation, 999.0
        
    cy_lr, cx_lr = center_of_mass(lr_bg)
    cy_hr, cx_hr = center_of_mass(hr_bg)
    
    centroid_shift = np.sqrt((cx_lr - cx_hr)**2 + (cy_lr - cy_hr)**2)
    
    return correlation, centroid_shift

def find_valid_targets():
    valid_targets = []
    if not DATA_DIR.exists():
        print(f"❌ Errore: {DATA_DIR} non esiste")
        return []
    for target_dir in DATA_DIR.iterdir():
        if target_dir.is_dir() and target_dir.name not in ['logs', 'splits', '__pycache__']:
            pairs_path = target_dir / '6_patches_from_cropped' / 'paired_patches_folders'
            if pairs_path.exists() and any(pairs_path.iterdir()):
                valid_targets.append((target_dir.name, pairs_path))
    return valid_targets

def select_target_menu():
    targets = find_valid_targets()
    if not targets:
        print("\n❌ Nessun target trovato.")
        return None, None
    
    print("\n" + "="*60)
    print("🔍 ANALISI MASSIVA ALLINEAMENTO".center(60))
    print("="*60)
    
    for i, (name, path) in enumerate(targets):
        num_pairs = len(list(path.glob("pair_*")))
        print(f"   {i+1}. {name} ({num_pairs} coppie)")
    print("-" * 30)
    
    while True:
        try:
            choice = input(f"👉 Scegli (1-{len(targets)}) o 'q': ").strip().lower()
            if choice == 'q': return None, None
            idx = int(choice) - 1
            if 0 <= idx < len(targets):
                return targets[idx][1], targets[idx][0]
        except ValueError: pass

def visualize_bad_pair(pair_dir, img_lr, img_hr, corr, shift):
    """Visualizza solo se richiesto (per le coppie brutte)"""
    norm_lr = normalize_image(img_lr)
    norm_hr = normalize_image(img_hr)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"⚠️ BAD PAIR: {pair_dir.name} - Corr: {corr:.2f} - Shift: {shift:.1f}px", 
                 fontsize=14, fontweight='bold', color='red')

    axes[0].imshow(norm_lr, origin='lower', cmap='magma')
    axes[0].set_title("LR (Observatory)")
    axes[1].imshow(norm_hr, origin='lower', cmap='viridis')
    axes[1].set_title("HR (Hubble)")
    
    # Overlay
    scale_factor = img_hr.shape[0] / img_lr.shape[0]
    resampled_lr = zoom(norm_lr, scale_factor, order=0)[:img_hr.shape[0], :img_hr.shape[1]]
    overlay = np.zeros((img_hr.shape[0], img_hr.shape[1], 3))
    overlay[..., 0] = resampled_lr * 0.8
    overlay[..., 1] = norm_hr * 1.0
    overlay[..., 2] = resampled_lr * 0.8
    axes[2].imshow(overlay, origin='lower')
    axes[2].set_title("Overlay")
    
    plt.show(block=False)
    plt.pause(2) # Mostra per 2 secondi poi chiudi
    plt.close()

def analyze_all_pairs(pairs_root, target_name):
    pair_folders = sorted([d for d in pairs_root.iterdir() if d.is_dir()])
    if not pair_folders: return

    results_csv = pairs_root.parent / f"alignment_report_{target_name}.csv"
    
    print(f"\n📊 ANALISI TOTALE SU {len(pair_folders)} COPPIE...")
    print(f"📄 Report salvato in: {results_csv}")
    print(f"⚙️  Soglie: Corr > {THRESHOLD_CORR} | Shift < {THRESHOLD_SHIFT}px")
    
    count_good = 0
    count_bad = 0
    bad_pairs_data = []

    with open(results_csv, 'w', newline='') as csvfile:
        fieldnames = ['Pair_ID', 'Correlation', 'Shift_px', 'Status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # TQDM per la barra di caricamento
        for pair_dir in tqdm(pair_folders, desc="Analisi in corso"):
            fits_files = list(pair_dir.glob("*.fits"))
            img_lr, img_hr = None, None
            
            for f in fits_files:
                try:
                    with fits.open(f) as hdul:
                        data = hdul[0].data
                        if len(data.shape) == 3: data = data[0]
                        if data is None: continue
                        data = np.nan_to_num(data)
                        
                        if data.shape[0] < 200: img_lr = data
                        else: img_hr = data
                except: continue

            if img_lr is None or img_hr is None:
                writer.writerow({'Pair_ID': pair_dir.name, 'Correlation': 0, 'Shift_px': 0, 'Status': 'ERROR'})
                continue

            # Calcolo
            corr, shift = calculate_alignment_metrics(img_lr, img_hr)
            
            # Valutazione
            is_good = (corr >= THRESHOLD_CORR) and (shift <= THRESHOLD_SHIFT)
            status = "OK" if is_good else "BAD"
            
            if is_good:
                count_good += 1
            else:
                count_bad += 1
                bad_pairs_data.append((pair_dir, img_lr, img_hr, corr, shift))

            writer.writerow({
                'Pair_ID': pair_dir.name, 
                'Correlation': f"{corr:.4f}", 
                'Shift_px': f"{shift:.2f}", 
                'Status': status
            })

    print("\n" + "="*60)
    print("🏁 RISULTATI FINALI")
    print("="*60)
    print(f"✅ Coppie Valide:  {count_good}")
    print(f"⚠️ Coppie Dubbie:  {count_bad}")
    print(f"📊 Percentuale OK: {(count_good/len(pair_folders))*100:.1f}%")
    
    if count_bad > 0:
        ask = input(f"\n👉 Vuoi vedere le prime 5 coppie DUBBIE? [s/n]: ").lower()
        if ask == 's':
            for i, (p_dir, lr, hr, c, s) in enumerate(bad_pairs_data):
                if i >= 5: break
                visualize_bad_pair(p_dir, lr, hr, c, s)

if __name__ == "__main__":
    pairs_path, t_name = select_target_menu()
    if pairs_path:
        analyze_all_pairs(pairs_path, t_name)