import argparse
import sys
import os
import torch
import numpy as np
from astropy.io import fits
from pathlib import Path
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# ============================================================
# 0. FIX MEMORIA & NUMPY
# ============================================================
if not hasattr(np, 'float'):
    np.float = float

# ============================================================
# 1. CONFIGURAZIONE PATH (ROBUSTA PER SRC/LIGHT)
# ============================================================
CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent.parent  # Risale da scripts/ a root

print(f"📂 Repo Root: {PROJECT_ROOT}")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importa specificamente da src.light come nel training
try:
    from src.light.architecture import HybridSuperResolutionModel
    print("✅ Moduli caricati da src.light")
except ImportError as e:
    print(f"❌ Errore Import Architecture: {e}")
    print(f"   Assicurati che src/light esista in {PROJECT_ROOT}")
    sys.exit(1)

# ============================================================
# 2. FUNZIONI UTILI
# ============================================================
def save_png_preview(lr, sr, path):
    """Salva un confronto visivo rapido in PNG"""
    # Ridimensiona LR alle dimensioni di SR per affiancarle
    lr_resized = F.interpolate(lr, size=sr.shape[2:], mode='nearest')
    
    # Crea griglia: LR | SR
    comparison = torch.cat((lr_resized, sr), dim=3)
    
    # Salva
    vutils.save_image(comparison, path, normalize=False)
    print(f"   🖼️  PNG Preview salvata: {path}")

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🧪 Inizio Inferenza su: {device}")
    
    ckpt_path = Path(args.checkpoint)
    in_path = Path(args.input)
    out_path = Path(args.output)
    
    if not ckpt_path.exists(): print(f"❌ Checkpoint mancante: {ckpt_path}"); return
    if not in_path.exists(): print(f"❌ Input mancante: {in_path}"); return

    # --- 1. CARICAMENTO DATI FITS ---
    print(f"📂 Input: {in_path.name}")
    with fits.open(in_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header

    # Gestione array 3D (se presente prende il primo canale)
    if len(data.shape) == 3:
        data = data[0]

    # --- 2. NORMALIZZAZIONE (Coerente con Training) ---
    # Il training usa Min-Max puro. Usiamo lo stesso qui.
    data_clean = np.nan_to_num(data, nan=0.0)
    min_val, max_val = np.min(data_clean), np.max(data_clean)
    
    if max_val - min_val < 1e-8:
        norm = np.zeros_like(data_clean)
    else:
        norm = (data_clean - min_val) / (max_val - min_val)
    
    # Preparazione Tensore (1, 1, H, W)
    inp = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)
    
    # Calcolo dimensioni target
    h, w = inp.shape[2], inp.shape[3]
    target_h = int(h * args.scale)
    target_w = int(w * args.scale)
    
    print(f"   📏 Dimensioni: {h}x{w} -> {target_h}x{target_w} (Scale x{args.scale})")

    # --- 3. CARICAMENTO MODELLO ---
    print(f"📥 Caricamento Modello...")
    try:
        # Importante: Passiamo output_size al costruttore!
        model = HybridSuperResolutionModel(
            smoothing=args.smoothing, 
            device=device,
            output_size=(target_h, target_w)
        )
        
        # Carica pesi
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
            
        model.eval()
    except Exception as e:
        print(f"❌ Errore caricamento modello: {e}")
        return

    # --- 4. INFERENZA ---
    print(f"✨ Elaborazione...")
    with torch.no_grad():
        # Supporto Mixed Precision se disponibile
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            out = model(inp)
            
            # Sicurezza: Assicuriamoci che l'output sia esattamente della dimensione richiesta
            # (Il modello lo fa già internamente, ma questo è un double-check)
            if out.shape[-2:] != (target_h, target_w):
                out = F.interpolate(out, size=(target_h, target_w), mode='bicubic')

    # --- 5. SALVATAGGIO FITS (SCIENTIFICO) ---
    out_np = out.squeeze().cpu().numpy()
    
    # Denormalizzazione (Riporta ai valori fisici originali)
    out_np = out_np * (max_val - min_val) + min_val
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header['HISTORY'] = f"SR Hybrid x{args.scale} (Smoothing: {args.smoothing})"
    fits.writeto(out_path, out_np, header, overwrite=True)
    print(f"✅ FITS Salvato: {out_path}")

    # --- 6. SALVATAGGIO PNG (VISUALIZZAZIONE) ---
    # Salviamo il confronto nella stessa cartella del fits, ma con estensione .png
    png_path = out_path.with_suffix('.png')
    save_png_preview(inp, out, png_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help="Path al file .pth")
    parser.add_argument('--input', required=True, help="Path immagine FITS Low-Res")
    parser.add_argument('--output', required=True, help="Path dove salvare l'output FITS")
    
    # Parametri opzionali modificati
    parser.add_argument('--scale', type=float, default=2.0, help="Fattore di scala (es. 2.0, 1.6)")
    parser.add_argument('--smoothing', type=str, default='balanced', choices=['none', 'light', 'balanced', 'strong'])
    
    args = parser.parse_args()
    run(args)