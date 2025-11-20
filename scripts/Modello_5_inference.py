import argparse
import sys
import os
import torch
import numpy as np
from astropy.io import fits
from pathlib import Path
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm
import math

# ============================================================
# 0. CONFIGURAZIONI
# ============================================================
if not hasattr(np, 'float'): np.float = float

# ============================================================
# 1. CONFIGURAZIONE PATH
# ============================================================
CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print(f"📂 Repo Root: {PROJECT_ROOT}")

# Import Ibrido: Tenta prima src (Heavy), poi src.light
try:
    from src.architecture import HybridSuperResolutionModel
    print("✅ Caricato Modello STANDARD (src.architecture)")
except ImportError:
    try:
        from src.light.architecture import HybridSuperResolutionModel
        print("⚠️ Usato Modello LIGHT (src.light.architecture)")
    except ImportError as e:
        print(f"❌ Errore Import: {e}"); sys.exit(1)

# ============================================================
# 2. MOTORE DI INFERENZA (TILING)
# ============================================================
def process_tiled(model, img_tensor, tile_size=128, overlap=16, scale=4.0, device='cuda'):
    """
    Esegue l'inferenza dividendo l'immagine in tile per risparmiare VRAM.
    Fondamentale per il modello HEAVY.
    """
    b, c, h, w = img_tensor.shape
    target_h, target_w = int(h * scale), int(w * scale)
    output = torch.zeros((b, c, target_h, target_w), device=device)
    weights = torch.zeros((b, c, target_h, target_w), device=device)

    # Calcolo step
    stride = tile_size - overlap
    h_steps = (h + stride - 1) // stride
    w_steps = (w + stride - 1) // stride

    pbar = tqdm(total=h_steps*w_steps, desc="🧩 Elaborazione Tiled", leave=False)

    for i in range(h_steps):
        for j in range(w_steps):
            # Coordinate Input
            h_start = i * stride
            h_end = min(h_start + tile_size, h)
            w_start = j * stride
            w_end = min(w_start + tile_size, w)
            
            # Aggiustamento per bordi (se l'ultimo tile è più piccolo, torniamo indietro)
            if h_end - h_start < tile_size and h > tile_size:
                h_start = h - tile_size
            if w_end - w_start < tile_size and w > tile_size:
                w_start = w - tile_size

            # Estrai Tile Input
            in_tile = img_tensor[:, :, h_start:h_end, w_start:w_end]
            
            # Inferenza
            with torch.no_grad():
                # Calcola dimensione target del tile corrente
                tile_out_h = int(in_tile.size(2) * scale)
                tile_out_w = int(in_tile.size(3) * scale)
                
                # Per il modello Heavy, potremmo dover forzare output_size nel forward se supportato,
                # ma qui usiamo la dimensione naturale scalata
                out_tile = model(in_tile)
                
                # Resize di sicurezza se il modello non ha prodotto esattamente scale x
                if out_tile.shape[2] != tile_out_h or out_tile.shape[3] != tile_out_w:
                     out_tile = F.interpolate(out_tile, size=(tile_out_h, tile_out_w), mode='bicubic')

            # Coordinate Output
            h_start_out = int(h_start * scale)
            h_end_out = int(h_end * scale)
            w_start_out = int(w_start * scale)
            w_end_out = int(w_end * scale)

            # Accumula
            output[:, :, h_start_out:h_end_out, w_start_out:w_end_out] += out_tile
            weights[:, :, h_start_out:h_end_out, w_start_out:w_end_out] += 1.0
            pbar.update(1)

    # Media nelle zone di sovrapposizione
    output /= weights
    return output

# ============================================================
# 3. MAIN RUN
# ============================================================
def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🧪 INFERENZA HEAVY (Scala x{args.scale}) su {device}")
    
    in_path = Path(args.input)
    out_path = Path(args.output)
    
    if not in_path.exists(): print("❌ Input non trovato"); return

    # 1. CARICAMENTO
    print(f"📂 Input: {in_path.name}")
    with fits.open(in_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header
    if len(data.shape) == 3: data = data[0]

    # 2. NORMALIZZAZIONE
    data_clean = np.nan_to_num(data, nan=0.0)
    min_val, max_val = np.min(data_clean), np.max(data_clean)
    norm = (data_clean - min_val) / (max_val - min_val) if (max_val - min_val) > 1e-8 else np.zeros_like(data_clean)
    
    inp = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)
    
    # 3. MODELLO
    print(f"📥 Caricamento pesi: {Path(args.checkpoint).name}")
    try:
        # Istanzia il modello. Nota: output_size qui è un hint, il tiling gestisce il grosso.
        target_full_h = int(inp.shape[2] * args.scale)
        target_full_w = int(inp.shape[3] * args.scale)
        
        model = HybridSuperResolutionModel(
            smoothing=args.smoothing, 
            device=device,
            output_size=(target_full_h, target_full_w) # Passiamo la dim finale desiderata
        )
        
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"❌ Errore Modello: {e}"); return

    # 4. ESECUZIONE (Tiled o Direct)
    # Se l'immagine è piccola (<200px) la facciamo diretta, altrimenti Tiled
    if inp.shape[2] < 200 and inp.shape[3] < 200:
        print("🚀 Modalità Diretta (Immagine piccola)")
        with torch.no_grad():
            out = model(inp)
    else:
        print(f"🧩 Modalità Tiled (Input > 200px) - Tile Size: {args.tile_size}")
        out = process_tiled(model, inp, tile_size=args.tile_size, scale=args.scale, device=device)

    # 5. SALVATAGGIO
    out_np = out.squeeze().cpu().numpy()
    out_np = out_np * (max_val - min_val) + min_val # Denormalizza
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header['HISTORY'] = f"SR Heavy x{args.scale}"
    fits.writeto(out_path, out_np, header, overwrite=True)
    print(f"✅ FITS Salvato: {out_path}")

    # PNG Preview
    png_path = out_path.with_suffix('.png')
    lr_resized = F.interpolate(inp, size=out.shape[2:], mode='nearest')
    vutils.save_image(torch.cat((lr_resized, out), dim=3), png_path, normalize=False)
    print(f"🖼️  PNG Preview: {png_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scale', type=float, default=4.0) # DEFAULT 4.0 per Heavy
    parser.add_argument('--smoothing', type=str, default='balanced')
    parser.add_argument('--tile_size', type=int, default=128, help="Dimensione tile per risparmiare VRAM")
    args = parser.parse_args()
    run(args)