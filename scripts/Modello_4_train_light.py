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

# ============================================================================
# 0. CONFIGURAZIONI BASE
# ============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if not hasattr(np, 'float'): np.float = float

# ============================================================================
# 1. CONFIGURAZIONE PATH ASSOLUTO DINAMICO (REPO ROOT)
# ============================================================================
# Ottiene il path assoluto dello script corrente: .../SuperResolution/scripts/Modello_4...py
CURRENT_SCRIPT = Path(__file__).resolve()

# Risale di due livelli per trovare la root assoluta della repo:
# scripts/ -> SuperResolution/ (ROOT)
PROJECT_ROOT = CURRENT_SCRIPT.parent.parent

print(f"📂 Repo Root Assoluta: {PROJECT_ROOT}")

# Aggiunge la root al sys.path per le importazioni
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.light.architecture import HybridSuperResolutionModel
    from src.light.dataset import AstronomicalDataset
    from src.light.losses import CombinedLoss
    from src.light.metrics import Metrics
    print("✅ Moduli caricati correttamente da src.light")
except ImportError as e:
    print(f"\n❌ ERRORE IMPORT: {e}")
    print(f"   Verifica che {PROJECT_ROOT}/src/light esista.")
    sys.exit(1)

ROOT_DATA_DIR = PROJECT_ROOT / "data"

# ============================================================================
# CONFIGURAZIONE HARDWARE
# ============================================================================
HARDWARE_CONFIG = {
    'batch_size': 1,          
    'accumulation_steps': 1,  
    'target_lr_size': 80,     
    'target_hr_size': 128,
    'num_workers': 4,        
    'prefetch_factor': 2,    
}

# ============================================================================
# UTILS
# ============================================================================
def save_preview_png(lr, pred, hr, path):
    """Salva un trittico: LR (Resize) | PRED | HR"""
    # Ridimensiona LR alla dimensione di HR per confronto pulito
    lr_resized = F.interpolate(lr, size=hr.shape[2:], mode='nearest')
    # Concatena orizzontalmente
    comparison = torch.cat((lr_resized, pred, hr), dim=3)
    # Salva
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
                    shape = hdul[0].data.shape
                    dims.append((f, shape[-2]*shape[-1] if len(shape)>=2 else shape[0]**2))
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
    
    # --- CONFIGURAZIONE PATH OUTPUT ASSOLUTO (REPO BASED) ---
    # Path: .../SuperResolution/outputs/tensorboard
    png_out_dir = PROJECT_ROOT / "outputs" / "tensorboard"
    png_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Cartella log classica
    log_dir = PROJECT_ROOT / "outputs" / target_dir.name / "tensorboard_light"
    save_dir = PROJECT_ROOT / "outputs" / target_dir.name / "checkpoints_light"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(log_dir))
    
    print("\n" + "="*70)
    print("👀  LIVE PREVIEW ATTIVA (PER OGNI IMMAGINE)".center(70))
    print("="*70)
    print(f"   📂 Cartella Output Assoluta: {png_out_dir}")
    print(f"   📄 File Live (Sovrascritto): {png_out_dir / 'LIVE_PREVIEW.png'}")
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
    model = HybridSuperResolutionModel(smoothing=args.smoothing, device=device, output_size=hr_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = CombinedLoss().to(device)
    
    print(f"🔥 Inizio Training ({args.epochs} epoche)")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        
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
            
            # --- SALVATAGGIO LIVE OGNI SINGOLA IMMAGINE (STEP % 1) ---
            # Sovrascrive il file LIVE_PREVIEW.png ad ogni passo
            save_preview_png(lr, pred, hr, png_out_dir / "LIVE_PREVIEW.png")
            
            # Salva anche lo storico ogni 100 step per riferimento
            if global_step % 100 == 0:
                save_preview_png(lr, pred, hr, png_out_dir / f"step_{global_step:05d}.png")

            writer.add_scalar('Loss/Train_Step', loss.item(), global_step)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # --- VALIDAZIONE ---
        model.eval()
        metrics = Metrics()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc="Val", leave=False)):
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
                pred = model(lr)
                metrics.update(pred, hr)
                
                # Salva TUTTE le immagini di validazione in una sottocartella per epoca
                val_epoch_dir = png_out_dir / f"epoch_{epoch}_val"
                val_epoch_dir.mkdir(exist_ok=True)
                save_preview_png(lr, pred, hr, val_epoch_dir / f"val_{i:03d}.png")

        res = metrics.compute()
        print(f"   Loss: {epoch_loss/len(train_loader):.4f} | PSNR: {res['psnr']:.2f}")
        
        writer.add_scalar('Loss/Train_Epoch', epoch_loss/len(train_loader), epoch)
        writer.add_scalar('Metrics/PSNR', res['psnr'], epoch)
        
        torch.save(model.state_dict(), save_dir / "last_model_light.pth")

    writer.close()
    print(f"\n✅ Training Finito. Controlla {png_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--smoothing', type=str, default='balanced')
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--use_amp', type=bool, default=True)
    args = parser.parse_args()
    
    HARDWARE_CONFIG['target_hr_size'] = args.resize
    HARDWARE_CONFIG['batch_size'] = args.batch_size
    
    if args.target: target_dir = ROOT_DATA_DIR / args.target
    else: target_dir = select_target_directory()
    
    if target_dir: train(args, target_dir)