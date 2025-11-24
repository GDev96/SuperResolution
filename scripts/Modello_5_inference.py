"""
MODELLO - STEP 5: INFERENZA E TEST (CORRETTO PER HAT/TRANSFORMER)
Genera immagini Super-Risolte dal dataset di TEST.
Gestisce il padding automatico per evitare errori di dimensione con le Window Attention.

POSIZIONE FILE: scripts/Modello_5_inference.py
"""

import argparse
import sys
import os
import torch
import numpy as np
import json
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from astropy.io import fits
from tqdm import tqdm
import math

# ============================================================================
# 1. CONFIGURAZIONE PATH DINAMICA
# ============================================================================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
ROOT_DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

print(f"📂 Project Root: {PROJECT_ROOT}")

try:
    from src.architecture import HybridSuperResolutionModel
    from src.metrics import Metrics
    from src.dataset import AstronomicalDataset
except ImportError as e:
    sys.exit(f"❌ Errore Import 'src': {e}. Assicurati di essere nella root corretta.")

# ============================================================================
# 2. FUNZIONI DI UTILITÀ E PADDING
# ============================================================================

def pad_tensor(x, window_size=64):
    """
    Applica padding riflettente affinché H e W siano divisibili per window_size.
    Necessario per evitare crash con architetture Transformer (HAT/Swin).
    """
    _, _, h, w = x.size()
    h_pad = (window_size - h % window_size) % window_size
    w_pad = (window_size - w % window_size) % window_size
    
    if h_pad == 0 and w_pad == 0:
        return x, (0, 0)

    # Pad: (left, right, top, bottom)
    x_padded = F.pad(x, (0, w_pad, 0, h_pad), mode='reflect')
    return x_padded, (h_pad, w_pad)

def crop_tensor(x, pads, scale):
    """
    Rimuove il padding aggiunto precedentemente, considerando lo scale factor.
    """
    h_pad, w_pad = pads
    if h_pad == 0 and w_pad == 0:
        return x
    
    # Calcola le dimensioni finali attese
    _, _, h_curr, w_curr = x.size()
    h_final = h_curr - (h_pad * scale)
    w_final = w_curr - (w_pad * scale)
    
    return x[:, :, :h_final, :w_final]

def select_target_directory():
    """Menu per selezionare il target se non passato come argomento."""
    print("\n" + "📂"*35)
    print("SELEZIONE TARGET PER INFERENZA".center(70))
    print("📂"*35)
    
    candidates = []
    if OUTPUTS_ROOT.exists():
        for d in OUTPUTS_ROOT.iterdir():
            if d.is_dir() and (d / "checkpoints" / "best_model.pth").exists():
                candidates.append(d.name)
            elif d.is_dir() and (d / "checkpoints" / "best.pth").exists():
                candidates.append(d.name)
    
    if not candidates:
        print("❌ Nessun modello addestrato trovato in outputs/.")
        return None

    print("\nModelli disponibili:")
    for i, name in enumerate(candidates):
        print(f"   {i+1}: {name}")

    while True:
        try:
            choice = input(f"\n👉 Seleziona (1-{len(candidates)}) o 'q': ").strip()
            if choice.lower() == 'q': return None
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except ValueError: pass
    return None

def create_test_json(target_name):
    """Crea un JSON temporaneo per il dataset di TEST."""
    split_dir = ROOT_DATA_DIR / target_name / "6_patches_final" / "splits"
    if not split_dir.exists():
        split_dir = ROOT_DATA_DIR / target_name / "6_patches_aligned" / "splits"
        
    test_dir = split_dir / "test"
    if not test_dir.exists():
        print(f"⚠️  Cartella test non trovata in: {split_dir}")
        return None

    test_list = []
    for p in sorted(test_dir.glob("pair_*")):
        lr = p / "observatory.fits"
        hr = p / "hubble.fits"
        if lr.exists() and hr.exists():
            test_list.append({
                "patch_id": p.name, 
                "ground_path": str(lr), 
                "hubble_path": str(hr)
            })
    
    if not test_list:
        print("❌ Nessuna coppia trovata nel folder test.")
        return None

    json_path = split_dir / "test_temp_inference.json"
    with open(json_path, 'w') as f:
        json.dump(test_list, f, indent=4)
    
    return json_path

# ============================================================================
# 3. MAIN INFERENCE
# ============================================================================

def run_inference(target_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Gestione Errori CUDA precedenti ---
    # Se la GPU è in stato di errore per il crash precedente, svuotiamo la cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Path Checkpoint
    ckpt_dir = OUTPUTS_ROOT / target_name / "checkpoints"
    ckpt_path = ckpt_dir / "best_model.pth"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "best.pth" 
    
    if not ckpt_path.exists():
        print(f"❌ Checkpoint non trovato in: {ckpt_dir}")
        return

    # Path Output
    out_test_dir = OUTPUTS_ROOT / target_name / "test_results"
    (out_test_dir / "fits").mkdir(parents=True, exist_ok=True)
    (out_test_dir / "comparison_png").mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 START INFERENCE: {target_name}")
    print(f"   📥 Modello: {ckpt_path.name}")
    print(f"   💾 Output:  {out_test_dir}")

    # 1. Dataset
    json_file = create_test_json(target_name)
    if not json_file: return

    # Nota: augment=False fondamentale per il test
    test_ds = AstronomicalDataset(json_file, base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    # 2. Carica Modello
    print("   🔧 Caricamento Architettura...")
    model = HybridSuperResolutionModel(smoothing='balanced', device=device).to(device)
    
    print("   📂 Caricamento Pesi...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        psnr = checkpoint.get('best_psnr', 0.0)
        print(f"      -> Checkpoint Epoca: {epoch}, Best PSNR: {psnr:.2f} dB")
    else:
        model.load_state_dict(checkpoint)
        print("      -> Pesi raw caricati.")
        
    model.eval()

    # 3. Inferenza
    metrics = Metrics()
    print(f"\n   ⚡ Elaborazione {len(test_loader)} immagini...")
    
    # Window size per HAT/Transformers (Importante: deve essere coerente con l'architettura)
    # HAT usa tipicamente window_size=16, usiamo 64 come multiplo sicuro.
    WINDOW_SIZE = 64 
    SCALE = 4 # Assumiamo scala 4x, ideale recuperarlo dalla config se possibile

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            # --- FIX DIMENSIONI (PADDING) ---
            lr_padded, pads = pad_tensor(lr, window_size=WINDOW_SIZE)
            
            # Esegui SR
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    sr_padded = model(lr_padded)
            else:
                sr_padded = model(lr_padded)
            
            # --- FIX DIMENSIONI (CROP) ---
            # Rimuoviamo il padding dall'output SR
            sr = crop_tensor(sr_padded, pads, scale=SCALE)

            # Controllo sicurezza dimensioni per metriche
            if sr.shape != hr.shape:
                # Se c'è discrepanza di 1-2 pixel per arrotondamenti
                sr = F.interpolate(sr, size=hr.shape[2:], mode='bicubic', align_corners=False)

            # Aggiorna metriche (su CPU float32 per precisione)
            metrics.update(sr.float(), hr.float())
            
            # --- SALVATAGGIO ---
            # A. FITS (Dati scientifici)
            sr_cpu = sr.squeeze().float().cpu().numpy() # [H, W]
            fits_out = out_test_dir / "fits" / f"sr_{i:04d}.fits"
            fits.PrimaryHDU(data=sr_cpu).writeto(fits_out, overwrite=True)
            
            # B. PNG (Confronto Visivo)
            lr_up = F.interpolate(lr, size=hr.shape[2:], mode='nearest')
            
            viz_lr = lr_up.clamp(0, 1)
            viz_sr = sr.clamp(0, 1)
            viz_hr = hr.clamp(0, 1)
            
            comparison = torch.cat((viz_lr, viz_sr, viz_hr), dim=3) 
            png_out = out_test_dir / "comparison_png" / f"compare_{i:04d}.png"
            save_image(comparison, png_out, normalize=False)

    # 4. Report Finale
    final = metrics.compute()
    print("\n" + "="*50)
    print("📊 REPORT TEST SET")
    print("="*50)
    print(f"   PSNR Medio: {final['psnr']:.2f} dB")
    print(f"   SSIM Medio: {final['ssim']:.4f}")
    print(f"\n✅ Completato. Risultati in: {out_test_dir}")
    
    if json_file.exists(): json_file.unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, help="Nome del target (es. M33)")
    args = parser.parse_args()
    
    target = args.target
    if not target:
        target = select_target_directory()
    
    if target:
        try:
            run_inference(target)
        except RuntimeError as e:
            if "device-side assert triggered" in str(e):
                print("\n❌ ERRORE CRITICO CUDA: La GPU è in uno stato instabile.")
                print("👉 Soluzione: Riavvia l'ambiente Python (Kernel Restart) e riprova.")
            else:
                raise e
    else:
        print("Nessun target selezionato.")