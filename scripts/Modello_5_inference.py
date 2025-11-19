import argparse
import sys
import torch
import numpy as np
from astropy.io import fits
from pathlib import Path

# ============================================================
# FIX IMPORT: src è nella cartella PADRE
# ============================================================
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.append(str(PROJECT_ROOT))
# ============================================================
from src.architecture import HybridSuperResolutionModel

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🧪 Inferenza su: {device}")
    
    ckpt_path = Path(args.checkpoint)
    in_path = Path(args.input)
    out_path = Path(args.output)
    
    if not ckpt_path.exists(): print(f"❌ Checkpoint mancante: {ckpt_path}"); return
    if not in_path.exists(): print(f"❌ Input mancante: {in_path}"); return

    print(f"📥 Loading Checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    conf = ckpt.get('config', {})
    smoothing = args.force_smoothing if args.force_smoothing else conf.get('smoothing', 'balanced')
    print(f"   Mode: {smoothing}")

    model = HybridSuperResolutionModel(target_scale=15, smoothing=smoothing, device=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"📂 Input: {in_path.name}")
    with fits.open(in_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header

    # Norm P1-P99
    p1, p99 = np.percentile(data, 1), np.percentile(data, 99)
    norm = np.clip(data, p1, p99)
    norm = (norm - p1) / (p99 - p1 + 1e-8)
    
    inp = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)
    
    print("✨ Processing...")
    with torch.no_grad():
        out = model(inp)

    # Denorm
    out_np = out.squeeze().cpu().numpy()
    out_np = out_np * (p99 - p1) + p1
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header['HISTORY'] = f"SR Hybrid (Smoothing: {smoothing})"
    fits.writeto(out_path, out_np, header, overwrite=True)
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--force_smoothing', choices=['none', 'light', 'balanced', 'strong'])
    run(parser.parse_args())