"""
TRAINING INDEPENDENT WORKER (TENSORBOARD LOG EVERY 2 EPOCHS)
Batch Size: 2 | Workers: 1 (per GPU)
Logica: Ogni worker prende una frazione del dataset e allena una copia del modello.
Update: Salva 'best_model.pth' e 'last.pth' regolarmente.
"""

import os
# Ottimizzazioni di sistema per evitare colli di bottiglia CPU
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import cv2
cv2.setNumThreads(0)

import argparse
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter 

# ============================================================
# CONFIGURAZIONE PATH & IMPORT
# ============================================================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
ROOT_DATA_DIR = PROJECT_ROOT / "data"

try:
    from src.architecture import HybridSuperResolutionModel
    from src.dataset import AstronomicalDataset
    from src.losses import CombinedLoss
    from src.metrics import Metrics
except ImportError as e:
    sys.exit(f"❌ Errore import src: {e}")

# ============================================================
# FUNZIONI UTILI
# ============================================================
def create_partitioned_json(split_dir, rank, total_gpus):
    """
    Crea un JSON contenente SOLO una frazione (1/total_gpus) dei dati totali.
    Usa lo slicing [rank::total_gpus] per distribuire i file senza sovrapposizioni.
    """
    def scan_and_split(split_name):
        d = split_dir / split_name
        data = []
        if not d.exists(): return []
        
        # Raccogli tutti i folder
        all_files = []
        for entry in os.scandir(d):
            if entry.is_dir() and entry.name.startswith("pair_"):
                all_files.append(entry)
        
        # Ordina per garantire determinismo tra i worker
        all_files.sort(key=lambda x: x.name)
        
        # Prendi solo i file per questo worker
        my_files = all_files[rank::total_gpus]
        
        for entry in my_files:
            lr = (d / entry.name / "observatory.fits").resolve()
            hr = (d / entry.name / "hubble.fits").resolve()
            if lr.exists() and hr.exists():
                data.append({
                    "patch_id": entry.name, 
                    "ground_path": str(lr), 
                    "hubble_path": str(hr)
                })
        return data

    train_list = scan_and_split("train")
    val_list = scan_and_split("val")

    # Nomi file unici per evitare conflitti tra processi
    ft = split_dir / f"train_worker_{rank}.json"
    fv = split_dir / f"val_worker_{rank}.json"
    
    with open(ft, 'w') as f: json.dump(train_list, f, indent=4)
    with open(fv, 'w') as f: json.dump(val_list, f, indent=4)
    
    return ft, fv, len(train_list)

# ============================================================
# MAIN WORKER
# ============================================================
def train_worker(args):
    # Assegna dispositivo (sarà cuda:0 visto che il launcher imposta CUDA_VISIBLE_DEVICES)
    device = torch.device('cuda:0') 
    rank = args.rank
    
    target_name = f"{args.target}_GPU_{rank}"
    out_dir = PROJECT_ROOT / "outputs" / target_name
    save_dir = out_dir / "checkpoints"
    img_dir = out_dir / "images"
    log_dir = out_dir / "tensorboard" 
    for d in [save_dir, img_dir, log_dir]: d.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))

    # Dataset
    splits_dir = ROOT_DATA_DIR / args.target / "6_patches_final" / "splits"
    if not splits_dir.exists():
         # Fallback per compatibilità vecchi nomi
         splits_dir = ROOT_DATA_DIR / args.target / "6_patches_aligned" / "splits"
         
    json_train, json_val, num_samples = create_partitioned_json(splits_dir, rank, 7)
    
    print(f"⚙️  [GPU {rank}] Dataset caricato: {num_samples} immagini")

    # Configurazione Training
    BATCH_SIZE = 2  
    ACCUM_STEPS = 8 
    LOG_INTERVAL = 2 
    TOTAL_EPOCHS = 150 
    
    train_ds = AstronomicalDataset(json_train, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(json_val, base_path=PROJECT_ROOT, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    model = HybridSuperResolutionModel(smoothing='balanced', device=device, output_size=512).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = CombinedLoss().to(device)
    
    # Compatibilità AMP (Old vs New Pytorch)
    try:
        from torch.amp import GradScaler, autocast
        scaler = GradScaler('cuda')
        amp_context = lambda: autocast('cuda')
    except ImportError:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        amp_context = lambda: autocast()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10) # Monitora PSNR

    # Loop
    best_psnr = 0.0
    
    # Usa posizione=rank per evitare sovrapposizione barre tqdm se lanciati insieme
    pbar = tqdm(range(TOTAL_EPOCHS), desc=f"GPU {rank}", position=rank, leave=True,
                bar_format="{desc}: Ep {n_fmt} | {postfix}")

    for epoch in pbar:
        model.train()
        
        acc_loss_total = 0.0
        acc_loss_l1 = 0.0
        acc_loss_astro = 0.0
        acc_loss_perc = 0.0
        
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)
            
            with amp_context():
                pred = model(lr)
                loss, loss_dict = criterion(pred, hr)
                loss_scaled = loss / ACCUM_STEPS
            
            scaler.scale(loss_scaled).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            current_val = loss.item()
            acc_loss_total += current_val
            acc_loss_l1 += loss_dict['l1'].item()
            acc_loss_astro += loss_dict['astro'].item()
            acc_loss_perc += loss_dict['perceptual'].item()

            # Aggiorna testo pbar ogni tot batch
            if i % 5 == 0:
                pbar.set_postfix_str(f"Loss: {current_val:.4f}")

        # --- VALIDATION & LOGGING ---
        if (epoch + 1) % LOG_INTERVAL == 0:
            
            steps = len(train_loader)
            avg_total = acc_loss_total / steps
            
            writer.add_scalar('Train/Loss', avg_total, epoch)
            writer.add_scalar('Train/L1', acc_loss_l1 / steps, epoch)
            writer.add_scalar('Train/Astro', acc_loss_astro / steps, epoch)

            # Validazione
            model.eval()
            val_loss = 0
            metrics = Metrics()
            preview_img = None 

            with torch.no_grad():
                for j, v_batch in enumerate(val_loader):
                    v_lr = v_batch['lr'].to(device)
                    v_hr = v_batch['hr'].to(device)
                    
                    with amp_context():
                        v_pred = model(v_lr)
                        v_loss, _ = criterion(v_pred, v_hr)
                    
                    val_loss += v_loss.item()
                    metrics.update(v_pred.float(), v_hr.float())
                    
                    # Salva la prima immagine come preview
                    if j == 0:
                        v_lr_up = torch.nn.functional.interpolate(v_lr, size=(512,512), mode='nearest')
                        preview_img = torch.cat((v_lr_up, v_pred, v_hr), dim=3)

            res = metrics.compute()
            avg_val_loss = val_loss / len(val_loader)
            
            # Scheduler step su PSNR
            scheduler.step(res['psnr'])

            # Log Validazione
            writer.add_scalar('Val/Loss', avg_val_loss, epoch)
            writer.add_scalar('Val/PSNR', res['psnr'], epoch)
            writer.add_scalar('Val/SSIM', res['ssim'], epoch)
            
            if preview_img is not None:
                preview_img = preview_img.clamp(0, 1)
                writer.add_image('Val/Preview', preview_img[0], epoch)
                if (epoch + 1) % 10 == 0:
                    vutils.save_image(preview_img, img_dir / f"ep_{epoch+1}.png", normalize=False)

            writer.flush()
            pbar.set_postfix_str(f"L:{avg_total:.3f} | PSNR:{res['psnr']:.2f}")
            
            # --- SALVATAGGIO CHECKPOINT ---
            # Salva il migliore
            if res['psnr'] > best_psnr:
                best_psnr = res['psnr']
                torch.save(model.state_dict(), save_dir / "best_model.pth")
        
        # Salva l'ultimo checkpoint AD OGNI EPOCA (sovrascrivendo)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict()
        }, save_dir / "last.pth")

    # Pulizia finale
    if json_train.exists(): json_train.unlink()
    if json_val.exists(): json_val.unlink()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    train_worker(args)