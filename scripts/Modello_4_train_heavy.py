"""
TRAINING A6000 BUNKER EDITION (48GB VRAM) + TENSORBOARD IMAGES
Configurazione Blindata anti-OOM.
Batch Size: 2 | Accumulation: 16
NOVITÀ: Le immagini di preview vengono caricate su TensorBoard.
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
from torch.cuda.amp import GradScaler 

# ============================================================================
# 1. CONFIGURAZIONE HARDWARE BLINDATA
# ============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Abilita TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# CPU Workers
CPU_CORES = os.cpu_count() or 16
WORKERS = min(16, CPU_CORES) 

# CONFIGURAZIONE DI SICUREZZA (Batch 2 / Accum 16)
HARDWARE_CONFIG = {
    'batch_size': 2,           
    'accum_steps': 16,         
    'target_hr_size': 512,
}

# ============================================================================
# PATHS & IMPORTS
# ============================================================================
PROJECT_ROOT = Path("/root/SuperResolution")
if not PROJECT_ROOT.exists(): 
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
ROOT_DATA_DIR = PROJECT_ROOT / "data"

try:
    from src.architecture import HybridSuperResolutionModel
    from src.dataset import AstronomicalDataset
    from src.losses import CombinedLoss
    from src.metrics import Metrics
except ImportError:
    sys.exit(f"❌ Errore import src. Assicurati che {PROJECT_ROOT}/src esista.")

# ============================================================================
# UTILS
# ============================================================================
def create_json_files_if_missing(split_dir):
    def scan_split(split_name):
        d = split_dir / split_name
        data = []
        if not d.exists(): return data
        for entry in os.scandir(d):
            if entry.is_dir() and entry.name.startswith("pair_"):
                lr = os.path.join(entry.path, "observatory.fits")
                hr = os.path.join(entry.path, "hubble.fits")
                if os.path.exists(lr) and os.path.exists(hr):
                    data.append({"patch_id": entry.name, "ground_path": lr, "hubble_path": hr})
        return sorted(data, key=lambda x: x['patch_id'])

    print("   📂 Scansione dataset (Train/Val)...")
    train_list = scan_split("train")
    val_list = scan_split("val")

    ft = split_dir / "train_temp.json"
    fv = split_dir / "val_temp.json"
    
    with open(ft, 'w') as f: json.dump(train_list, f)
    with open(fv, 'w') as f: json.dump(val_list, f)
    
    return ft, fv

# ============================================================================
# TRAIN LOOP
# ============================================================================
def train(args, target_dir):
    device = torch.device('cuda')
    torch.cuda.empty_cache() 
    
    print(f"\n🚀 START TRAIN A6000 BUNKER + TB IMAGES: {target_dir.name}")
    eff_batch = HARDWARE_CONFIG['batch_size'] * HARDWARE_CONFIG['accum_steps']
    print(f"   🔥 Batch Efficace: {eff_batch}")
    
    out_dir = PROJECT_ROOT / "outputs" / target_dir.name
    save_dir = out_dir / "checkpoints"
    log_dir = out_dir / "tensorboard"
    img_dir = out_dir / "images"
    for d in [save_dir, log_dir, img_dir]: d.mkdir(parents=True, exist_ok=True)
    
    splits_dir = target_dir / "6_patches_aligned" / "splits"
    if not splits_dir.exists():
        sys.exit(f"❌ Splits non trovati in {splits_dir}.")

    json_train, json_val = create_json_files_if_missing(splits_dir)
    
    # Dataset
    train_ds = AstronomicalDataset(json_train, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(json_val, base_path=PROJECT_ROOT, augment=False)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=HARDWARE_CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    if json_train.exists(): json_train.unlink()
    if json_val.exists(): json_val.unlink()

    print("   🔧 Inizializzazione Modello...")
    model = HybridSuperResolutionModel(
        smoothing=args.smoothing, 
        device=device, 
        output_size=HARDWARE_CONFIG['target_hr_size']
    ).to(device)
    
    if args.resume:
        print(f"   📥 Resuming: {args.resume}")
        model.load_state_dict(torch.load(args.resume))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = CombinedLoss().to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    scaler = torch.amp.GradScaler('cuda')
    writer = SummaryWriter(str(log_dir))
    
    best_loss = float('inf')
    global_step = 0
    accum_steps = HARDWARE_CONFIG['accum_steps']

    print("   🟢 Training Avviato!")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs}", unit="batch")
        
        for i, batch in enumerate(pbar):
            lr = batch['lr'].to(device, non_blocking=True) 
            hr = batch['hr'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                pred = model(lr) 
                loss, _ = criterion(pred, hr)
                loss = loss / accum_steps 
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            current_loss = loss.item() * accum_steps
            epoch_loss += current_loss
            global_step += 1
            
            writer.add_scalar('Train/Loss_Iter', current_loss, global_step)
            pbar.set_postfix({'loss': f"{current_loss:.4f}"})
            
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Train/Loss_Epoch', avg_loss, epoch)
        
        # VALIDATION & IMAGE LOGGING
        model.eval()
        val_loss = 0
        metrics = Metrics()
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                lr = batch['lr'].to(device, non_blocking=True)
                hr = batch['hr'].to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    pred = model(lr)
                    v_loss, _ = criterion(pred, hr)
                
                val_loss += v_loss.item()
                metrics.update(pred, hr)
                
                # === SALVATAGGIO IMMAGINI (DISK + TENSORBOARD) ===
                if i == 0: 
                    # Upscale semplice dell'input per confronto visivo (Nearest Neighbor)
                    lr_up = F.interpolate(lr, size=(512,512), mode='nearest')
                    
                    # Unisci: Input(Sgranato) | Predizione(AI) | Target(Hubble)
                    comp = torch.cat((lr_up, pred, hr), dim=3)
                    
                    # 1. Salva su DISCO
                    vutils.save_image(comp, img_dir / f"val_ep_{epoch+1}.png", normalize=False)
                    
                    # 2. Manda a TENSORBOARD (Tab IMAGES)
                    # .squeeze(0) rimuove la dimensione batch [1, C, H, W] -> [C, H, W]
                    # .clamp(0, 1) assicura che i colori non siano "bruciati" nel visualizzatore
                    writer.add_image('Validation/Preview (Input | AI | Target)', comp.squeeze(0).clamp(0, 1), epoch)
                # =================================================
            
        res = metrics.compute()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"   📉 Val Loss: {avg_val_loss:.4f} | PSNR: {res['psnr']:.2f} | SSIM: {res['ssim']:.3f}")
        
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        writer.add_scalar('Val/PSNR', res['psnr'], epoch)
        writer.add_scalar('Val/SSIM', res['ssim'], epoch)
        
        scheduler.step(avg_val_loss)
        
        torch.save(model.state_dict(), save_dir / "last.pth")
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), save_dir / "best.pth")
            print("   🏆 New Best Model!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--smoothing', type=str, default='balanced') 
    parser.add_argument('--resume', type=str)
    args = parser.parse_args()
    
    if args.target:
        train(args, ROOT_DATA_DIR / args.target)
    else:
        valid = [d for d in ROOT_DATA_DIR.iterdir() if (d / "6_patches_aligned" / "splits").exists()]
        if len(valid) == 1:
            train(args, valid[0])
        elif len(valid) > 1:
            try:
                s = int(input(f"Target ({[d.name for d in valid]}): ")) - 1
                train(args, valid[s])
            except: pass