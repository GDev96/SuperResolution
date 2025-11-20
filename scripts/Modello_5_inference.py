import argparse
import sys
import os
import torch
import numpy as np
from astropy.io import fits
from pathlib import Path
import torch.nn.functional as F

# ============================================================
# 0. FIX MEMORIA & NUMPY
# ============================================================
if not hasattr(np, 'float'):
    np.float = float

# ============================================================
# 1. CONFIGURAZIONE PATH (CRUCIALE)
# ============================================================
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MODELS_DIR = PROJECT_ROOT / "models"

sys.path.append(str(PROJECT_ROOT))

if MODELS_DIR.exists():
    if str(MODELS_DIR) not in sys.path:
        sys.path.insert(0, str(MODELS_DIR))
    for subfolder in MODELS_DIR.iterdir():
        if subfolder.is_dir() and str(subfolder) not in sys.path:
            sys.path.insert(0, str(subfolder))

# Ora possiamo importare
try:
    from src.architecture import HybridSuperResolutionModel
except ImportError as e:
    print(f"❌ Errore Import Architecture: {e}")
    sys.exit(1)

def pad_image_for_hat(tensor, window_size=16):
    """Aggiunge padding se l'immagine non è divisibile per window_size (per HAT)"""
    _, _, h, w = tensor.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return tensor, pad_h, pad_w

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🧪 Inferenza su: {device}")
    
    ckpt_path = Path(args.checkpoint)
    in_path = Path(args.input)
    out_path = Path(args.output)
    
    if not ckpt_path.exists(): print(f"❌ Checkpoint mancante: {ckpt_path}"); return
    if not in_path.exists(): print(f"❌ Input mancante: {in_path}"); return

    print(f"📥 Caricamento Modello...")
    try:
        # Carica checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)
        
        # Configurazione smoothing
        conf = ckpt.get('config', {})
        smoothing = args.force_smoothing if args.force_smoothing else conf.get('smoothing', 'balanced')
        print(f"   Mode: {smoothing}")

        # Inizializza modello
        model = HybridSuperResolutionModel(smoothing=smoothing, device=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
    except Exception as e:
        print(f"❌ Errore caricamento modello: {e}")
        return

    print(f"📂 Input: {in_path.name}")
    with fits.open(in_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header

    # Gestione array 3D (se presente)
    if len(data.shape) == 3:
        data = data[0]

    # Normalizzazione P1-P99 (Robusta)
    p1, p99 = np.percentile(data, 1), np.percentile(data, 99)
    norm = np.clip(data, p1, p99)
    norm = (norm - p1) / (p99 - p1 + 1e-8)
    
    # Preparazione Tensore
    inp = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)
    
    # Padding per HAT (Evita crash su dimensioni strane)
    inp_padded, pad_h, pad_w = pad_image_for_hat(inp, window_size=16)
    
    print(f"✨ Elaborazione... (Input: {data.shape} -> Padded: {inp_padded.shape})")
    
    with torch.no_grad():
        # INFERENZA
        out = model(inp_padded)
        
        # Rimozione Padding (Crop finale)
        # Il modello scala 6.4x (approx). Calcoliamo il crop esatto.
        # Poiché HAT fa x4 (2 stage x2) e poi interpolate finale,
        # il padding viene amplificato.
        
        # Calcolo dimensione target esatta basata sull'input originale
        h, w = data.shape
        target_h = int(h * 6.4)
        target_w = int(w * 6.4)
        
        # Interpoliamo l'output finale alla dimensione esatta desiderata
        # Questo corregge eventuali piccoli errori di padding/scaling
        out = F.interpolate(out, size=(target_h, target_w), mode='bicubic', align_corners=False)

    # Denormalizzazione
    out_np = out.squeeze().cpu().numpy()
    out_np = out_np * (p99 - p1) + p1
    
    # Salvataggio
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header['HISTORY'] = f"SR Hybrid 6.4x (Smoothing: {smoothing})"
    fits.writeto(out_path, out_np, header, overwrite=True)
    print(f"✅ Salvato: {out_path}")
    print(f"   Dimensione Finale: {out_np.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help="Path al file .pth (best_model.pth)")
    parser.add_argument('--input', required=True, help="Path immagine FITS Low-Res")
    parser.add_argument('--output', required=True, help="Path dove salvare l'output FITS")
    parser.add_argument('--force_smoothing', choices=['none', 'light', 'balanced', 'strong'])
    run(parser.parse_args())