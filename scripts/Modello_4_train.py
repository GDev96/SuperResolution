"""
MODELLO - STEP 4: TRAINING
Training ottimizzato per RTX 2060 (6GB VRAM) + 64GB RAM
Usa GRADIENT ACCUMULATION: Batch 1 (fisico) -> Batch 4 (virtuale)

POSIZIONE FILE: scripts/Modello_4_train.py
"""

import argparse
import sys
import os
import json
import warnings
import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np

# ============================================================================
# 0. CONFIGURAZIONI MEMORIA & FIX NUMPY
# ============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if not hasattr(np, 'float'):
    np.float = float

# ============================================================================
# 1. CONFIGURAZIONE PATH ASSOLUTI E DINAMICI
# ============================================================================
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
# CONFIGURAZIONE HARDWARE (RTX 2060 - GRADIENT ACCUMULATION)
# ============================================================================
HARDWARE_CONFIG = {
    'gpu_model': 'RTX 2060',
    'vram_gb': 6,
    
    # TRUCCO GRADIENT ACCUMULATION:
    'batch_size': 1,          # Quante immagini carica in VRAM (Basso = Sicuro)
    'accumulation_steps': 4,  # Ogni quante immagini aggiorna i pesi (1 * 4 = Batch Virtuale 4)
    
    'use_amp': True,
    'target_lr_size': 80,
    'target_hr_size': 512,
    'scale_ratio': 6.4
}

# ============================================================================
# FUNZIONI DI UTILITÀ
# ============================================================================
def select_target_directory():
    print("\n" + "📂"*35)
    print("SELEZIONE DATASET PER TRAINING".center(70))
    print("📂"*35)
    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() 
                   if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    except Exception as e:
        print(f"\n❌ ERRORE: {e}"); return None

    if not subdirs: print(f"\n❌ Nessun dataset trovato."); return None

    valid_targets = []
    print("\nDataset disponibili:")
    for i, dir_path in enumerate(subdirs):
        splits_dir = dir_path / "6_patches_from_cropped" / "splits"
        if (splits_dir / "train").exists():
            valid_targets.append(dir_path)
            print(f"   {len(valid_targets)}: {dir_path.name}")

    if not valid_targets: return None

    while True:
        print("\n" + "─"*70)
        choice = input(f"👉 Seleziona (1-{len(valid_targets)}) o 'q': ").strip().lower()
        if choice == 'q': return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(valid_targets):
                return valid_targets[idx]
        except ValueError: pass

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
                    shape = hdul[0].data.shape
                    h = shape[-2] if len(shape) >= 2 else shape[0]
                    w = shape[-1] if len(shape) >= 2 else shape[0]
                    dims.append((f, h*w))
            dims.sort(key=lambda x: x[1])
            pairs.append({"patch_id": pair_dir.name, "ground_path": str(dims[0][0]), "hubble_path": str(dims[1][0])})
        except: continue
    return pairs

# ============================================================================
# TRAINING
# ============================================================================
def train(args, target_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = False
    scaler = None
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
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
        print("⚠️  Training su CPU (LENTO!)")
        use_amp_context = None

    save_dir = PROJECT_ROOT / "outputs" / target_dir.name / "checkpoints"
    log_dir = PROJECT_ROOT / "outputs" / target_dir.name / "tensorboard"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    
    # DATASET
    splits_dir = target_dir / "6_patches_from_cropped" / "splits"
    train_pairs = create_json_from_split_folder(splits_dir / "train")
    val_pairs = create_json_from_split_folder(splits_dir / "val")
    
    temp_train = splits_dir / "temp_train_run.json"
    temp_val = splits_dir / "temp_val_run.json"
    with open(temp_train, 'w') as f: json.dump(train_pairs, f)
    with open(temp_val, 'w') as f: json.dump(val_pairs, f)
    
    train_ds = AstronomicalDataset(temp_train, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(temp_val, base_path=PROJECT_ROOT, augment=False)
    
    if temp_train.exists(): temp_train.unlink()
    if temp_val.exists(): temp_val.unlink()

    # DATALOADER - Batch Size Fisico = 1
    train_loader = DataLoader(train_ds, batch_size=HARDWARE_CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"   ✅ Physical Batch: {HARDWARE_CONFIG['batch_size']}")
    print(f"   ✅ Virtual Batch:  {HARDWARE_CONFIG['batch_size'] * HARDWARE_CONFIG['accumulation_steps']} (con Gradient Accumulation)")

    # MODEL
    print(f"\n🏗️  Costruzione Modello...")
    try:
        model = HybridSuperResolutionModel(smoothing=args.smoothing, device=device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   📊 Parametri: {params:,}")
    except Exception as e:
        print(f"❌ ERRORE MODELLO: {e}"); return

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = CombinedLoss().to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_psnr = 0.0
    start_epoch = 0
    global_step = 0
    
    if args.resume and Path(args.resume).exists():
        print(f"📥 Resume: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt.get('best_psnr', 0.0)
        global_step = ckpt.get('global_step', 0)

    # LOOP CON GRADIENT ACCUMULATION
    print(f"\n🔥 Inizio Training: {start_epoch+1} -> {args.epochs} epoche")
    print("="*70)
    
    accumulation_steps = HARDWARE_CONFIG['accumulation_steps']
    optimizer.zero_grad(set_to_none=True) # Reset iniziale

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        
        for i, batch in enumerate(pbar):
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)
            
            # 1. Forward Pass
            if use_amp:
                with use_amp_context():
                    pred = model(lr)
                    loss, _ = criterion(pred, hr)
                    # Normalizza la loss perché la stiamo sommando 4 volte
                    loss = loss / accumulation_steps
                
                # 2. Backward (Accumulo gradienti)
                scaler.scale(loss).backward()
            else:
                pred = model(lr)
                loss, _ = criterion(pred, hr)
                loss = loss / accumulation_steps
                loss.backward()
            
            # Solo per visualizzazione, rimoltiplichiamo
            current_loss = loss.item() * accumulation_steps
            train_loss += current_loss

            # 3. Step dell'Optimizer (Solo ogni 'accumulation_steps' passi)
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                # Reset gradienti DOPO lo step
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            
            pbar.set_postfix({'loss': f"{current_loss:.4f}"})

        # VALIDATION
        model.eval()
        metrics = Metrics()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
                if use_amp:
                    with use_amp_context():
                        pred = model(lr)
                        l, _ = criterion(pred, hr)
                else:
                    pred = model(lr)
                    l, _ = criterion(pred, hr)
                val_loss += l.item()
                metrics.update(pred, hr)

        res = metrics.compute()
        writer.add_scalar('Val/PSNR', res['psnr'], epoch)
        
        print(f"   Loss: {train_loss/len(train_loader):.4f} | PSNR: {res['psnr']:.2f} dB")
        
        if res['psnr'] > best_psnr:
            best_psnr = res['psnr']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_psnr': best_psnr, 'global_step': global_step}, save_dir / "best_model.pth")
            print("   🏆 Best Model!")
            
        if (epoch + 1) % 10 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, save_dir / f"ckpt_ep{epoch+1}.pth")

        scheduler.step(res['psnr'])
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()
    print("\n✅ Training Completato!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1) # OVERRIDE A 1
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--smoothing', type=str, default='balanced')
    parser.add_argument('--resume', type=str)
    args = parser.parse_args()
    
    # Forza batch size a 1 se non specificato diversamente da riga di comando
    HARDWARE_CONFIG['batch_size'] = args.batch_size
    
    if args.target: target_dir = ROOT_DATA_DIR / args.target
    else: target_dir = select_target_directory()
    
    if target_dir: train(args, target_dir)