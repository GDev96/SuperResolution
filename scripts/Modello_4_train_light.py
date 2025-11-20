import argparse
import sys
import os
import json
import torch
import torch.optim as optim
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
# 1. CONFIGURAZIONE PATH E IMPORT DA SRC/LIGHT
# ============================================================================
CURRENT_SCRIPT = Path(__file__).resolve()
# Fallback path se lo script viene spostato
PROJECT_ROOT = Path(r"F:\SuperRevoltGaia\SuperResolution") 
if (CURRENT_SCRIPT.parent.parent / "src").exists():
    PROJECT_ROOT = CURRENT_SCRIPT.parent.parent

print(f"📂 Project Root: {PROJECT_ROOT}")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importiamo specificamente dal pacchetto 'light'
try:
    from src.light.architecture import HybridSuperResolutionModel
    from src.light.dataset import AstronomicalDataset
    from src.light.losses import CombinedLoss
    from src.light.metrics import Metrics
    print("✅ Moduli caricati correttamente da src.light")
except ImportError as e:
    print(f"\n❌ ERRORE IMPORT da src.light: {e}")
    print(f"   Assicurati che i file siano in: {PROJECT_ROOT}/src/light/")
    sys.exit(1)

ROOT_DATA_DIR = PROJECT_ROOT / "data"

# ============================================================================
# HARDWARE CONFIG (LIGHT)
# ============================================================================
HARDWARE_CONFIG = {
    'batch_size': 1,          
    'accumulation_steps': 1,  
    'target_lr_size': 80,     
    'target_hr_size': 128,    
}

# ============================================================================
# UTILS
# ============================================================================
def select_target_directory():
    if not ROOT_DATA_DIR.exists(): return None
    subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    if not subdirs: return None
    
    print("\nDataset disponibili:")
    valid_targets = []
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
    
    # --- SETUP TENSORBOARD LOGS ---
    save_dir = PROJECT_ROOT / "outputs" / target_dir.name / "checkpoints_light"
    log_dir = PROJECT_ROOT / "outputs" / target_dir.name / "tensorboard_light"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # --- STAMPA BANNER TENSORBOARD ---
    print("\n" + "="*70)
    print("📈  MONITORAGGIO TENSORBOARD ATTIVO (LIGHT TEST)".center(70))
    print("="*70)
    print(f"   Cartella Log: {log_dir}")
    print(f"   👉 APRI IL BROWSER SU: http://localhost:6006/")
    print("-" * 70)
    try:
        # Cerca di creare un path relativo per il comando (più corto)
        rel_log_dir = log_dir.relative_to(PROJECT_ROOT)
        cmd_path = rel_log_dir
    except ValueError:
        cmd_path = log_dir
        
    print(f"   Per avviare il server, apri un nuovo terminale ed esegui:")
    print(f"   tensorboard --logdir={cmd_path}")
    print("="*70 + "\n")
    # ---------------------------------
    
    # Setup Dataset
    splits_dir = target_dir / "6_patches_from_cropped" / "splits"
    train_pairs = create_json_from_split_folder(splits_dir / "train")
    val_pairs = create_json_from_split_folder(splits_dir / "val")
    
    temp_train = splits_dir / "temp_train_light.json"
    temp_val = splits_dir / "temp_val_light.json"
    with open(temp_train, 'w') as f: json.dump(train_pairs, f)
    with open(temp_val, 'w') as f: json.dump(val_pairs, f)
    
    # Dataset Load da src.light (con force_hr_size)
    train_ds = AstronomicalDataset(temp_train, PROJECT_ROOT, augment=True, force_hr_size=hr_size)
    val_ds = AstronomicalDataset(temp_val, PROJECT_ROOT, augment=False, force_hr_size=hr_size)
    
    if temp_train.exists(): temp_train.unlink()
    if temp_val.exists(): temp_val.unlink()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    print(f"\n🏗️  Model Setup (Output: {hr_size}x{hr_size})...")
    model = HybridSuperResolutionModel(smoothing=args.smoothing, device=device, output_size=hr_size).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = CombinedLoss().to(device)
    
    print(f"🔥 Inizio Training Light ({args.epochs} epoche)")
    
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        
        for batch in pbar:
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(lr)
                loss, loss_dict = criterion(pred, hr)
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            epoch_loss += loss.item()
            global_step += 1
            
            # Log training loss ogni step
            writer.add_scalar('Loss/Train_Step', loss.item(), global_step)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Validation
        model.eval()
        metrics = Metrics()
        
        # Visualizzazione immagini (solo primo batch della val)
        visualized = False
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc="Val", leave=False)):
                lr, hr = batch['lr'].to(device), batch['hr'].to(device)
                pred = model(lr)
                metrics.update(pred, hr)
                
                if not visualized:
                    writer.add_images('Light/LR_Input', lr, epoch)
                    writer.add_images('Light/HR_Target', hr, epoch)
                    writer.add_images('Light/Prediction', pred, epoch)
                    visualized = True
        
        res = metrics.compute()
        avg_train_loss = epoch_loss/len(train_loader)
        
        print(f"   Loss: {avg_train_loss:.4f} | PSNR: {res['psnr']:.2f} dB")
        
        # Log Epoca
        writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)
        writer.add_scalar('Metrics/PSNR', res['psnr'], epoch)
        writer.add_scalar('Metrics/SSIM', res['ssim'], epoch)
        
        # Salva sempre l'ultimo nel light test
        torch.save(model.state_dict(), save_dir / "last_model_light.pth")

    writer.close()
    print("\n✅ Test Light Completato.")

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