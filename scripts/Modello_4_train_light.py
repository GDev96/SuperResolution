"""
MODELLO - STEP 4: TRAINING (LIGHT) - ULTIMATE LIVE + CONTINUOUS GRAPH
Versione Light con grafico PSNR che si aggiorna AD OGNI FOTO.

FEATURES:
1. LIVE_VAL_CURRENT.png: Si sovrascrive ad ogni singola immagine.
2. Grafico 'Live_Validation/PSNR_Continuous_Stream': 
   Si aggiorna per OGNI FOTO (non per epoca), creando un grafico continuo in tempo reale.
"""

import argparse
import sys
import os
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np
from math import log10

# ============================================================================
# 0. CONFIGURAZIONI MEMORIA
# ============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if not hasattr(np, 'float'): np.float = float

# ============================================================================
# 1. CONFIGURAZIONE PATH ASSOLUTO (REPO ROOT)
# ============================================================================
CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent.parent
print(f"📂 Repo Root Assoluta: {PROJECT_ROOT}")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# IMPORT SPECIFICO DA SRC.LIGHT
try:
    from src.light.architecture import HybridSuperResolutionModel
    from src.light.dataset import AstronomicalDataset
    from src.light.losses import CombinedLoss
    from src.light.metrics import Metrics
    print("✅ Moduli caricati correttamente da src.light")
except ImportError as e:
    print(f"\n❌ ERRORE IMPORT: {e}"); sys.exit(1)

ROOT_DATA_DIR = PROJECT_ROOT / "data"

# ============================================================================
# CONFIGURAZIONE HARDWARE
# ============================================================================
HARDWARE_CONFIG = {
    'batch_size': 1,          
    'accumulation_steps': 1,  
    'target_lr_size': 128,     
    'target_hr_size': 512,
    'num_workers': 4,        
    'prefetch_factor': 2,    
}

# ============================================================================
# UTILS
# ============================================================================
def calc_psnr_tensor(pred, target):
    """Calcola PSNR su tensori GPU senza staccarli (veloce)"""
    mse = F.mse_loss(pred, target)
    if mse == 0: return 100.0
    return 10 * log10(1.0 / mse.item())

def save_preview_png(lr, pred, hr, path):
    """Salva un trittico PNG: LR | PRED | HR"""
    lr_resized = F.interpolate(lr, size=hr.shape[2:], mode='nearest')
    comparison = torch.cat((lr_resized, pred, hr), dim=3)
    vutils.save_image(comparison, path, normalize=False)

def select_target_directory():
    if not ROOT_DATA_DIR.exists(): return None
    subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    if not subdirs: return None
    print("\nDataset disponibili:"); valid_targets = []
    for d in subdirs:
        if (d / "6_patches_from_cropped" / "splits" / "train").exists():
            valid_targets.append(d)
            print(f"   {len(valid_targets)}: {d.name}")
    if not valid_targets: return None
    choice = input(f"👉 Seleziona (1-{len(valid_targets)}) o 'q': ").strip().lower()
    if choice == 'q': return None
    try: return valid_targets[int(choice) - 1]
    except: return None

def create_json_from_split_folder(split_folder):
    pairs = []
    for pair_dir in sorted(split_folder.glob("pair_*")):
        fits_files = list(pair_dir.glob("*.fits"))
        if len(fits_files) != 2: continue
        try:
            from astropy.io import fits as astro_fits
            dims = []
            for f in fits_files:
                with astro_fits.open(f) as hdul:
                    if hdul[0].data is None: continue
                    shape = hdul[0].data.shape
                    if len(shape) == 3: area = shape[-2]*shape[-1]
                    else: area = shape[0]*shape[1]
                    dims.append((f, area))
            dims.sort(key=lambda x: x[1])
            pairs.append({"patch_id": pair_dir.name, "ground_path": str(dims[0][0]), "hubble_path": str(dims[1][0])})
        except: continue
    return pairs

# ============================================================================
# TRAIN LOOP
# ============================================================================
def train(args, target_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = args.use_amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    hr_size = HARDWARE_CONFIG['target_hr_size']
    
    # --- SETUP PATH OUTPUT ---
    log_dir = PROJECT_ROOT / "outputs" / target_dir.name / "tensorboard_light"
    save_dir = PROJECT_ROOT / "outputs" / target_dir.name / "checkpoints_light"
    png_out_dir = PROJECT_ROOT / "outputs" / target_dir.name / "predictions_light"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    png_out_dir.mkdir(parents=True, exist_ok=True)
    
    # File LIVE che si sovrascrivono
    live_val_path = png_out_dir / "LIVE_VAL_CURRENT.png"
    live_train_path = png_out_dir / "LIVE_TRAIN_STEP.png"
    
    writer = SummaryWriter(log_dir=str(log_dir))
    
    print("\n" + "="*70)
    print("📈  MONITORAGGIO LIVE CONTINUO ATTIVO".center(70))
    print("="*70)
    print(f"   Log TensorBoard: {log_dir}")
    print(f"   📄 LIVE VAL:      {live_val_path}")
    print(f"   👉 APRI BROWSER:  http://localhost:6006/")
    print("="*70 + "\n")
    
    # --- DATASET ---
    splits_dir = target_dir / "6_patches_from_cropped" / "splits"
    temp_train = splits_dir / "temp_train_light.json"
    temp_val = splits_dir / "temp_val_light.json"
    
    train_pairs = create_json_from_split_folder(splits_dir / "train")
    val_pairs = create_json_from_split_folder(splits_dir / "val")
    
    with open(temp_train, 'w') as f: json.dump(train_pairs, f)
    with open(temp_val, 'w') as f: json.dump(val_pairs, f)
    
    train_ds = AstronomicalDataset(temp_train, PROJECT_ROOT, augment=True, force_hr_size=hr_size)
    val_ds = AstronomicalDataset(temp_val, PROJECT_ROOT, augment=False, force_hr_size=hr_size)
    
    if temp_train.exists(): temp_train.unlink()
    if temp_val.exists(): temp_val.unlink()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=HARDWARE_CONFIG['num_workers'], prefetch_factor=HARDWARE_CONFIG['prefetch_factor'], persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # --- MODEL ---
    print(f"\n🏗️  Model Setup (Output: {hr_size}x{hr_size})...")
    model = HybridSuperResolutionModel(smoothing=args.smoothing, device=device, output_size=hr_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = CombinedLoss().to(device)
    
    print(f"🔥 Inizio Training Light ({args.epochs} epoche)")
    
    global_step = 0
    # CONTATORE GLOBALE PER VALIDAZIONE (NON SI AZZERA MAI)
    val_global_counter = 0 

    for epoch in range(args.epochs):
        # 1. TRAIN
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1} Train")
        
        for batch in pbar:
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(lr)
                loss, _ = criterion(pred, hr)
            
            if use_amp:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
                
            epoch_loss += loss.item()
            global_step += 1
            
            # Salva PNG LIVE Training
            if global_step % 5 == 0: # Ogni 5 step per non rallentare troppo
                with torch.no_grad(): save_preview_png(lr, pred, hr, live_train_path)
            
            writer.add_scalar('1_Metrics/Training_Loss', loss.item(), global_step)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # 2. VALIDATION (CONTINUA E DETTAGLIATA)
        model.eval()
        metrics = Metrics()
        
        epoch_dir = png_out_dir / f"epoch_{epoch+1}"
        epoch_dir.mkdir(exist_ok=True)

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Ep {epoch+1} Valid", leave=False)
            for i, batch in enumerate(val_pbar):
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
                pred = model(lr)
                
                metrics.update(pred, hr)
                
                # --- GRAFICO PSNR CONTINUO (OGNI FOTO) ---
                single_psnr = calc_psnr_tensor(pred, hr)
                
                # QUESTO è il grafico che cercavi: Asse X = Numero Foto Totale Processata
                writer.add_scalar('Live_Validation/PSNR_Continuous_Stream', single_psnr, val_global_counter)
                val_global_counter += 1 # Incremento continuo
                
                # Grafici per singola foto (Asse X = Epoca)
                writer.add_scalar(f'Single_Images_Epoch_History/Img_{i:03d}', single_psnr, epoch)

                # SALVATAGGIO PNG
                save_preview_png(lr, pred, hr, live_val_path)
                save_preview_png(lr, pred, hr, epoch_dir / f"val_{i:04d}.png")

        # 3. FINE EPOCA
        res = metrics.compute()
        avg_train_loss = epoch_loss/len(train_loader)
        
        writer.add_scalar('1_Metrics/Avg_Val_PSNR_Epoch', res['psnr'], epoch)
        print(f"   Loss: {avg_train_loss:.4f} | Avg PSNR: {res['psnr']:.2f} dB")
        
        torch.save(model.state_dict(), save_dir / "last_model_light.pth")

    writer.close()
    print(f"\n✅ Training Finito.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--smoothing', type=str, default='balanced')
    parser.add_argument('--resize', type=int, default=512)
    parser.add_argument('--use_amp', type=bool, default=True)
    args = parser.parse_args()
    
    HARDWARE_CONFIG['target_hr_size'] = args.resize
    HARDWARE_CONFIG['batch_size'] = args.batch_size
    
    if args.target: target_dir = ROOT_DATA_DIR / args.target
    else: target_dir = select_target_directory()
    
    if target_dir: train(args, target_dir)