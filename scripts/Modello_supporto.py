#!/usr/bin/env python3
import os
import argparse
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURAZIONE PATH =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.architecture import HybridSuperResolutionModel
    from src.dataset import AstronomicalDataset
    from src.losses import CombinedLoss
    from src.metrics import Metrics
except ImportError as e:
    sys.exit(f"❌ Errore Import: {e}")

# ================= HYPERPARAMETERS SANITY CHECK (LINUX) =================
BATCH_SIZE = 4        # Se usiamo 2 GPU, 4img/gpu è ok. (Verrà ridotto se dataset=1)
ACCUM_STEPS = 1       # Update immediato per sanity check
LR = 5e-4             # Learning rate aggressivo per "svegliare" le stelle
TOTAL_EPOCHS = 1000

LOG_INTERVAL = 5     
IMAGE_INTERVAL = 20  

# Tuning NVIDIA
torch.backends.cudnn.benchmark = True 
torch.backends.cuda.matmul.allow_tf32 = True 

def train_worker(args):
    # 1. Setup Device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    
    print(f"🚀 TRAINING SU LINUX | GPU Rilevate: {num_gpus}")
    for i in range(num_gpus):
        print(f"   ✅ GPU {i}: {torch.cuda.get_device_name(i)}")
    
    target_name = f"{args.target}_LINUX_STAR_HUNTER"
    
    # 2. Setup Cartelle
    out_dir = PROJECT_ROOT / "outputs" / target_name
    save_dir = out_dir / "checkpoints"
    img_dir = out_dir / "images"
    log_dir = out_dir / "tensorboard"
    for d in [save_dir, img_dir, log_dir]: d.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))

    # 3. Caricamento Dati
    splits_dir = PROJECT_ROOT / "data" / args.target / "8_dataset_split" / "splits_json"
    
    with open(splits_dir / "train.json") as f: train_data = json.load(f)
    with open(splits_dir / "val.json") as f: val_data = json.load(f)
    
    ft_path = splits_dir / f"temp_train_{os.getpid()}.json"
    fv_path = splits_dir / f"temp_val_{os.getpid()}.json"
    with open(ft_path, 'w') as f: json.dump(train_data, f)
    with open(fv_path, 'w') as f: json.dump(val_data, f)

    # Dataset - AUGMENT=FALSE PER SANITY CHECK!
    train_ds = AstronomicalDataset(ft_path, base_path=PROJECT_ROOT, augment=False)
    val_ds = AstronomicalDataset(fv_path, base_path=PROJECT_ROOT, augment=False)
    
    # Smart Batch Size Logic
    curr_batch = BATCH_SIZE
    drop_last = True
    if len(train_ds) < BATCH_SIZE:
        curr_batch = len(train_ds)
        drop_last = False
        print(f"⚠️ Dataset piccolo ({len(train_ds)} img): Batch ridotto a {curr_batch}")

    train_loader = DataLoader(
        train_ds, 
        batch_size=curr_batch, 
        shuffle=False,            # No shuffle per sanity check
        num_workers=4,            
        pin_memory=True,          
        drop_last=drop_last
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    print(f"   🔥 Config: Batch={curr_batch} | Loss=STAR_WEIGHTED (500x) | Augment=FALSE")
    
    # 4. Modello Multi-GPU
    # Smoothing='none' per massima nitidezza sui pixel stellari
    model = HybridSuperResolutionModel(smoothing='none', device=device).to(device)
    
    if num_gpus > 1:
        print(f"   ⚖️  Attivazione DataParallel su {num_gpus} GPU")
        model = nn.DataParallel(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    
    # Loss Cacciatore di Stelle
    criterion = CombinedLoss().to(device)
    
    scaler = torch.cuda.amp.GradScaler() # FP16 per velocità
    best_psnr = 0.0

    # === TRAINING LOOP ===
    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        acc_loss = 0.0
        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Ep {epoch}", ncols=100, colour='cyan', leave=False)
        
        for i, batch in enumerate(pbar):
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                pred = model(lr)
                loss, _ = criterion(pred, hr)
                loss = loss / ACCUM_STEPS
            
            scaler.scale(loss).backward()
            acc_loss += loss.item() * ACCUM_STEPS

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            pbar.set_postfix(loss=f"{loss.item()*ACCUM_STEPS:.4f}")

        if len(train_loader) % ACCUM_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()
        
        # Validation
        if epoch % LOG_INTERVAL == 0:
            avg_loss = acc_loss / len(train_loader) if len(train_loader) > 0 else 0
            writer.add_scalar('Train/Loss', avg_loss, epoch)
            
            model.eval()
            metrics = Metrics()
            with torch.inference_mode(): 
                for v_batch in val_loader:
                    v_lr = v_batch['lr'].to(device)
                    v_hr = v_batch['hr'].to(device)
                    
                    with torch.cuda.amp.autocast():
                        v_pred = model(v_lr)
                    
                    metrics.update(v_pred.float(), v_hr.float())
            
            res = metrics.compute()
            writer.add_scalar('Val/PSNR', res['psnr'], epoch)
            
            print(f"   Ep {epoch:04d} | Loss: {avg_loss:.4f} | PSNR: {res['psnr']:.2f} dB")

            if res['psnr'] > best_psnr:
                best_psnr = res['psnr']
                # Gestione salvataggio DataParallel
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), save_dir / "best_model.pth")
                else:
                    torch.save(model.state_dict(), save_dir / "best_model.pth")
            
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_dir / "last.pth")
            else:
                torch.save(model.state_dict(), save_dir / "last.pth")

            if epoch % IMAGE_INTERVAL == 0:
                v_lr_up = torch.nn.functional.interpolate(v_lr, size=(512,512), mode='nearest')
                comp = torch.cat((v_lr_up, v_pred, v_hr), dim=3).clamp(0,1)
                vutils.save_image(comp, img_dir / f"epoch_{epoch}.png")

    try:
        if ft_path.exists(): ft_path.unlink()
        if fv_path.exists(): fv_path.unlink()
    except: pass
    print(f"\n✅ Finito.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()
    train_worker(args)