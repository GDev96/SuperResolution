"""
MODELLO - STEP 4: TRAINING (HEAVY) - SMART DATA & LIVE MONITOR
Fix: Legge correttamente tutte le cartelle pair (Smart Sorting).
Features: Live Preview PNG, TensorBoard dettagliato, Anti-Freeze.
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
# 0. CONFIGURAZIONI BASE
# ============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if not hasattr(np, 'float'): np.float = float

CURRENT_SCRIPT = Path(__file__).resolve()
HERE = CURRENT_SCRIPT.parent                        
PROJECT_ROOT = HERE.parent                          
MODELS_DIR = PROJECT_ROOT / "models"                

print(f"📂 Project Root: {PROJECT_ROOT}")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if MODELS_DIR.exists():
    if str(MODELS_DIR) not in sys.path:
        sys.path.insert(0, str(MODELS_DIR))
    for subfolder in MODELS_DIR.iterdir():
        if subfolder.is_dir():
            if str(subfolder) not in sys.path:
                sys.path.insert(0, str(subfolder))

try:
    from src.architecture import HybridSuperResolutionModel
    from src.dataset import AstronomicalDataset
    from src.losses import CombinedLoss
    from src.metrics import Metrics
except ImportError as e:
    print(f"\n❌ ERRORE IMPORT CRITICO: {e}")
    sys.exit(1)

ROOT_DATA_DIR = PROJECT_ROOT / "data"

# ============================================================================
# CONFIGURAZIONE HARDWARE (HEAVY)
# ============================================================================
HARDWARE_CONFIG = {
    'gpu_model': 'RTX 2060',
    'vram_gb': 6,
    'batch_size': 1,          
    'accumulation_steps': 4,  
    'use_amp': True,
  'target_lr_size': 80,    # CAMBIA QUESTO (era 128)
    'target_hr_size': 512,
    'scale_ratio': 6.4,      # CAMBIA QUESTO (era 4.0)
}

# ============================================================================
# UTILS
# ============================================================================
def calc_psnr_tensor(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0: return 100.0
    return 10 * log10(1.0 / mse.item())

def save_preview_png(lr, pred, hr, path):
    try:
        lr_resized = F.interpolate(lr, size=hr.shape[2:], mode='nearest')
        comparison = torch.cat((lr_resized, pred, hr), dim=3)
        vutils.save_image(comparison, path, normalize=False)
    except: pass

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
    """LOGICA SMART: Trova LR e HR ordinando per dimensione, ignora file extra."""
    pairs = []
    for pair_dir in sorted(split_folder.glob("pair_*")):
        fits_files = list(pair_dir.glob("*.fits"))
        
        # Se ci sono meno di 2 file, salta
        if len(fits_files) < 2: continue
        
        try:
            from astropy.io import fits as astro_fits
            candidates = []
            for f in fits_files:
                with astro_fits.open(f) as hdul:
                    if hdul[0].data is None: continue
                    shape = hdul[0].data.shape
                    if len(shape) == 3: area = shape[-2]*shape[-1]
                    else: area = shape[0]*shape[1]
                    candidates.append((f, area))
            
            if len(candidates) < 2: continue
            
            # Ordina per area: il più piccolo è LR (indice 0), il più grande è HR (indice -1)
            candidates.sort(key=lambda x: x[1])
            
            lr_file = candidates[0][0]
            hr_file = candidates[-1][0]
            
            pairs.append({
                "patch_id": pair_dir.name, 
                "ground_path": str(lr_file), 
                "hubble_path": str(hr_file)
            })
        except: continue
    return pairs

# ============================================================================
# TRAIN LOOP
# ============================================================================
def train(args, target_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = False
    scaler = None
    
    if torch.cuda.is_available():
        try:
            from torch.amp import GradScaler, autocast
            scaler = GradScaler('cuda')
            use_amp_context = lambda: autocast('cuda')
        except ImportError:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
            use_amp_context = lambda: autocast()
        use_amp = True
        print(f"\n🚀 Training su: {torch.cuda.get_device_name(0)}")
    else:
        use_amp_context = None

    # SETUP OUTPUT
    save_dir = PROJECT_ROOT / "outputs" / target_dir.name / "checkpoints"
    log_dir = PROJECT_ROOT / "outputs" / target_dir.name / "tensorboard"
    png_out_dir = PROJECT_ROOT / "outputs" / target_dir.name / "predictions_heavy"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    png_out_dir.mkdir(parents=True, exist_ok=True)
    
    live_preview_path = png_out_dir / "LIVE_VAL_CURRENT.png"
    live_train_path = png_out_dir / "LIVE_TRAIN_STEP.png"
    
    writer = SummaryWriter(log_dir=str(log_dir))
    
    print("\n" + "="*70)
    print("📈  MONITORAGGIO HEAVY ATTIVO".center(70))
    print("="*70)
    print(f"   Log: {log_dir}")
    print(f"   Live Val: {live_preview_path}")
    print(f"   👉 APRI BROWSER: http://localhost:6006/")
    print("="*70 + "\n")
    
    # DATASET (SMART LOADING)
    splits_dir = target_dir / "6_patches_from_cropped" / "splits"
    
    print("🔍 Scansione Smart del dataset...")
    train_pairs = create_json_from_split_folder(splits_dir / "train")
    val_pairs = create_json_from_split_folder(splits_dir / "val")
    
    print(f"   ✅ Trovate {len(train_pairs)} coppie di training")
    print(f"   ✅ Trovate {len(val_pairs)} coppie di validazione")
    
    temp_train = splits_dir / "temp_train_run.json"
    temp_val = splits_dir / "temp_val_run.json"
    with open(temp_train, 'w') as f: json.dump(train_pairs, f)
    with open(temp_val, 'w') as f: json.dump(val_pairs, f)
    
    train_ds = AstronomicalDataset(temp_train, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(temp_val, base_path=PROJECT_ROOT, augment=False)
    
    if temp_train.exists(): temp_train.unlink()
    if temp_val.exists(): temp_val.unlink()

    train_loader = DataLoader(train_ds, batch_size=HARDWARE_CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    # Val loader single thread per sicurezza salvataggio PNG
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # MODEL
    try:
        model = HybridSuperResolutionModel(smoothing=args.smoothing, device=device)
    except Exception as e:
        print(f"❌ ERRORE MODELLO: {e}"); return

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = CombinedLoss().to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_psnr = 0.0
    start_epoch = 0
    global_step = 0
    val_stream_step = 0
    
    if args.resume and Path(args.resume).exists():
        print(f"📥 Resume: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt.get('best_psnr', 0.0)
        global_step = ckpt.get('global_step', 0)

    print(f"\n🔥 Inizio Training: {start_epoch+1} -> {args.epochs} epoche")
    accumulation_steps = HARDWARE_CONFIG['accumulation_steps']
    optimizer.zero_grad(set_to_none=True) 

    for epoch in range(start_epoch, args.epochs):
        # 1. TRAIN
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1} Train")
        
        for i, batch in enumerate(pbar):
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)
            
            if use_amp:
                with use_amp_context():
                    pred = model(lr)
                    loss, _ = criterion(pred, hr)
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
            else:
                pred = model(lr)
                loss, _ = criterion(pred, hr)
                loss = loss / accumulation_steps
                loss.backward()
            
            current_loss = loss.item() * accumulation_steps
            train_loss += current_loss

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                
                if global_step % 10 == 0:
                     with torch.no_grad(): save_preview_png(lr, pred, hr, live_train_path)
                
                writer.add_scalar('1_Global_Metrics/Training_Loss_Step', current_loss, global_step)
            
            pbar.set_postfix({'loss': f"{current_loss:.4f}"})

        # 2. VALIDATION
        model.eval()
        metrics = Metrics()
        epoch_dir = png_out_dir / f"epoch_{epoch+1}"
        epoch_dir.mkdir(exist_ok=True)
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Val Live", leave=False)
            for i, batch in enumerate(val_pbar):
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
                
                if use_amp:
                    with use_amp_context(): pred = model(lr)
                else:
                    pred = model(lr)
                
                metrics.update(pred, hr)
                
                # Live Metrics
                val_stream_step += 1
                single_psnr = calc_psnr_tensor(pred, hr)
                writer.add_scalar('Live_Stream/PSNR_Continuous', single_psnr, val_stream_step)
                writer.add_scalar(f'Single_Image_History/Img_{i:03d}', single_psnr, epoch)

                # Save Images
                save_preview_png(lr, pred, hr, live_preview_path)
                save_preview_png(lr, pred, hr, epoch_dir / f"val_{i:04d}.png")

        # 3. END EPOCH
        res = metrics.compute()
        avg_train_loss = train_loss / len(train_loader)
        
        writer.add_scalar('1_Global_Metrics/Training_Loss_Epoch', avg_train_loss, epoch)
        writer.add_scalar('1_Global_Metrics/Average_Validation_PSNR', res['psnr'], epoch)
        
        print(f"   Loss: {avg_train_loss:.4f} | PSNR: {res['psnr']:.2f} dB")
        
        if res['psnr'] > best_psnr:
            best_psnr = res['psnr']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_psnr': best_psnr, 'global_step': global_step}, save_dir / "best_model.pth")
            
        if (epoch + 1) % 10 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, save_dir / f"ckpt_ep{epoch+1}.pth")

        scheduler.step(res['psnr'])
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    writer.close() 
    print("\n✅ Training Completato!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--smoothing', type=str, default='balanced')
    parser.add_argument('--resume', type=str)
    args = parser.parse_args()
    
    HARDWARE_CONFIG['batch_size'] = args.batch_size
    
    if args.target: target_dir = ROOT_DATA_DIR / args.target
    else: target_dir = select_target_directory()
    
    if target_dir: train(args, target_dir)