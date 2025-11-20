"""
MODELLO - STEP 4: TRAINING
Training ottimizzato per RTX 2060 (6GB VRAM) + 64GB RAM
Con menu selezione target, path dinamici e supporto src esterno.

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
# 0. MONKEY PATCH PER NUMPY 2.0 (CRITICO PER BASICSR)
# ============================================================================
# BasicSR usa 'np.float' che è stato rimosso in NumPy 2.0. 
# Questa riga lo ripristina temporaneamente.
if not hasattr(np, 'float'):
    np.float = float

# ============================================================================
# 1. CONFIGURAZIONE PATH ASSOLUTI E DINAMICI
# ============================================================================
# Ottiene il percorso assoluto di questo script (es: .../scripts/Modello_4_train.py)
CURRENT_SCRIPT = Path(__file__).resolve()
HERE = CURRENT_SCRIPT.parent                        # Cartella scripts/
PROJECT_ROOT = HERE.parent                          # Cartella SuperResolution/ (ROOT)
MODELS_DIR = PROJECT_ROOT / "models"                # Cartella models/

print(f"📂 Project Root: {PROJECT_ROOT}")

# Aggiungi la ROOT al sistema (per poter fare 'from src import ...')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Aggiungi la cartella 'models' e le sue sottocartelle (BasicSR, HAT, ecc.)
if MODELS_DIR.exists():
    # Aggiunge la cartella madre 'models'
    if str(MODELS_DIR) not in sys.path:
        sys.path.insert(0, str(MODELS_DIR))
    
    # Aggiunge ricorsivamente le sottocartelle immediate (es: models/BasicSR)
    for subfolder in MODELS_DIR.iterdir():
        if subfolder.is_dir():
            if str(subfolder) not in sys.path:
                sys.path.insert(0, str(subfolder))
else:
    print(f"⚠️  ATTENZIONE: La cartella {MODELS_DIR} non esiste! Il modello potrebbe non caricarsi.")

# ============================================================================
# IMPORT MODULI DA SRC
# ============================================================================
try:
    from src.architecture import HybridSuperResolutionModel
    from src.dataset import AstronomicalDataset
    from src.losses import CombinedLoss
    from src.metrics import Metrics
except ImportError as e:
    print(f"\n❌ ERRORE IMPORT CRITICO: {e}")
    print("   Assicurati di aver creato la cartella 'src' nella root del progetto")
    sys.exit(1)

ROOT_DATA_DIR = PROJECT_ROOT / "data"

# ============================================================================
# CONFIGURAZIONE HARDWARE (RTX 2060)
# ============================================================================
HARDWARE_CONFIG = {
    'gpu_model': 'RTX 2060',
    'vram_gb': 6,
    'ram_gb': 64,
    'recommended_batch_size': 4,
    'max_batch_size': 6,
    'use_amp': True,
    'target_lr_size': 80,     # <--- MODIFICATO DA 34 A 80
    'target_hr_size': 512,
    'scale_ratio': 6.4        # <--- MODIFICATO (512/80)

}

# ============================================================================
# FUNZIONI DI UTILITÀ (Menu e JSON)
# ============================================================================
def select_target_directory():
    """Mostra menu per selezionare dataset target per training."""
    print("\n" + "📂"*35)
    print("SELEZIONE DATASET PER TRAINING".center(70))
    print("📂"*35)
    
    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() 
                   if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    except Exception as e:
        print(f"\n❌ ERRORE scansione directory: {e}")
        return None

    if not subdirs:
        print(f"\n❌ Nessuna cartella trovata in {ROOT_DATA_DIR}")
        return None

    valid_targets = []
    print("\nDataset disponibili (con splits pronti):")
    
    for i, dir_path in enumerate(subdirs):
        splits_dir = dir_path / "6_patches_from_cropped" / "splits"
        train_dir = splits_dir / "train"
        
        if train_dir.exists():
            valid_targets.append(dir_path)
            info_txt = ""
            meta_file = splits_dir / "dataset_info.json"
            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        d = json.load(f)
                        info_txt = f"(Train: {d.get('train_pairs', '?')}, Val: {d.get('val_pairs', '?')})"
                except: pass
            
            print(f"   {len(valid_targets)}: {dir_path.name} {info_txt}")

    if not valid_targets:
        print("\n❌ Nessun dataset pronto trovato.")
        print("   Esegui prima 'Modello_2_prepare_data.py'")
        return None

    while True:
        print("\n" + "─"*70)
        choice = input(f"👉 Seleziona (1-{len(valid_targets)}) o 'q': ").strip().lower()
        if choice == 'q': return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(valid_targets):
                print(f"\n✅ Selezionato: {valid_targets[idx].name}")
                return valid_targets[idx]
            print("❌ Numero non valido.")
        except ValueError:
            print("❌ Input non valido.")

def create_json_from_split_folder(split_folder):
    """Genera lista dizionari per il Dataset."""
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
            lr_file = dims[0][0]
            hr_file = dims[1][0]
            
            pairs.append({
                "patch_id": pair_dir.name,
                "ground_path": str(lr_file),
                "hubble_path": str(hr_file)
            })
        except Exception as e:
            print(f"⚠️  Skip {pair_dir.name}: {e}")
            continue
    return pairs

# ============================================================================
# FUNZIONE DI TRAINING PRINCIPALE
# ============================================================================
def train(args, target_dir):
    
    # 1. SETUP DEVICE & AMP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    use_amp = False
    scaler = None
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            from torch.amp import GradScaler, autocast # PyTorch 2.4+
            scaler = GradScaler('cuda')
            use_amp_context = lambda: autocast('cuda')
        except ImportError:
            from torch.cuda.amp import GradScaler, autocast # PyTorch < 2.4
            scaler = GradScaler()
            use_amp_context = lambda: autocast()
            
        use_amp = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n🚀 Training su: {gpu_name}")
        print(f"   Mixed Precision: ✅ ATTIVA")
    else:
        print("⚠️  Training su CPU (LENTO!)")
        use_amp_context = None

    # 2. SETUP CARTELLE OUTPUT
    save_dir = PROJECT_ROOT / "outputs" / target_dir.name / "checkpoints"
    log_dir = PROJECT_ROOT / "outputs" / target_dir.name / "tensorboard"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📊 TensorBoard: tensorboard --logdir={log_dir}")

    # 3. CARICAMENTO DATI
    splits_dir = target_dir / "6_patches_from_cropped" / "splits"
    train_folder = splits_dir / "train"
    val_folder = splits_dir / "val"
    
    if not train_folder.exists():
        print(f"❌ Errore: Cartella train non trovata in {splits_dir}")
        return

    print(f"\n📦 Indicizzazione dataset...")
    train_pairs = create_json_from_split_folder(train_folder)
    val_pairs = create_json_from_split_folder(val_folder)
    
    if not train_pairs:
        print("❌ Nessuna coppia trovata per il training.")
        return

    temp_train = splits_dir / "temp_train_run.json"
    temp_val = splits_dir / "temp_val_run.json"
    
    with open(temp_train, 'w') as f: json.dump(train_pairs, f, indent=2)
    with open(temp_val, 'w') as f: json.dump(val_pairs, f, indent=2)
    
    try:
        train_ds = AstronomicalDataset(temp_train, base_path=PROJECT_ROOT, augment=True)
        val_ds = AstronomicalDataset(temp_val, base_path=PROJECT_ROOT, augment=False)
    except Exception as e:
        print(f"❌ Errore inizializzazione Dataset: {e}")
        return
    finally:
        if temp_train.exists(): temp_train.unlink()
        if temp_val.exists(): temp_val.unlink()

    # 4. DATALOADER
    num_workers = 4 if torch.cuda.is_available() else 0
    prefetch = 2 if num_workers > 0 else None
    persistent = (num_workers > 0)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent, prefetch_factor=prefetch
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2 if num_workers > 0 else 0, pin_memory=True
    )
    
    print(f"   ✅ Train: {len(train_ds)} img | Val: {len(val_ds)} img")
    print(f"   ✅ Batch: {args.batch_size} | Workers: {num_workers}")

    # 5. INIZIALIZZAZIONE MODELLO
    print(f"\n🏗️  Costruzione Modello Hybrid (BasicSR + HAT)...")
    try:
        model = HybridSuperResolutionModel(
            target_scale=HARDWARE_CONFIG['scale_ratio'],
            smoothing=args.smoothing,
            device=device
        )
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   📊 Parametri addestrabili: {params:,}")
    except Exception as e:
        print(f"❌ ERRORE MODELLO: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. OPTIMIZER & LOSS
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = CombinedLoss().to(device)
    
    # --- FIX: Rimosso 'verbose=True' che causa errore in PyTorch nuovo ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # 7. RESUME CHECKPOINT
    start_epoch = 0
    best_psnr = 0.0
    global_step = 0
    
    if args.resume:
        if Path(args.resume).exists():
            print(f"📥 Caricamento checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch'] + 1
            best_psnr = ckpt.get('best_psnr', 0.0)
            global_step = ckpt.get('global_step', 0)
            print(f"   Ripresa da Epoca {start_epoch}, Best PSNR: {best_psnr:.2f}")
        else:
            print(f"⚠️ Checkpoint non trovato: {args.resume}")

    # 8. TRAINING LOOP
    print(f"\n🔥 Inizio Training: {start_epoch+1} -> {args.epochs} epoche")
    print("="*70)
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                with use_amp_context():
                    pred = model(lr)
                    loss, loss_dict = criterion(pred, hr)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(lr)
                loss, loss_dict = criterion(pred, hr)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            train_loss += loss.item()
            global_step += 1
            
            if global_step % 10 == 0:
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                writer.add_scalar('Train/L1', loss_dict['l1'].item(), global_step)
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
        
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # 9. VALIDATION LOOP
        model.eval()
        metrics = Metrics()
        val_loss = 0
        preview_img = None
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validazione", leave=False):
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
                
                if use_amp:
                    with use_amp_context():
                        pred = model(lr)
                        loss_v, _ = criterion(pred, hr)
                else:
                    pred = model(lr)
                    loss_v, _ = criterion(pred, hr)
                
                val_loss += loss_v.item()
                metrics.update(pred, hr)
                
                if preview_img is None:
                    # Salva prima immagine per preview
                    preview_img = torch.cat([pred[0].cpu(), hr[0].cpu()], dim=2)

        res = metrics.compute()
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        writer.add_scalar('Val/PSNR', res['psnr'], epoch)
        writer.add_scalar('Val/SSIM', res['ssim'], epoch)
        if preview_img is not None:
            writer.add_image('Preview/Pred_vs_HR', preview_img, epoch)
        
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   PSNR: {res['psnr']:.2f} dB | SSIM: {res['ssim']:.4f}")
        
        if res['psnr'] > best_psnr:
            best_psnr = res['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_psnr': best_psnr,
                'global_step': global_step
            }, save_dir / "best_model.pth")
            print("   🏆 Best Model aggiornato!")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict()
            }, save_dir / f"ckpt_ep{epoch+1}.pth")

        scheduler.step(res['psnr'])

    writer.close()
    print("\n✅ Training Completato!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, help='Nome cartella target (es: M33)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--smoothing', type=str, default='balanced')
    parser.add_argument('--resume', type=str, help='Path checkpoint .pth')
    
    args = parser.parse_args()
    
    if args.target:
        target_dir = ROOT_DATA_DIR / args.target
        if not target_dir.exists():
            print(f"❌ Target non trovato: {target_dir}")
            sys.exit(1)
    else:
        target_dir = select_target_directory()
    
    if target_dir:
        try:
            train(args, target_dir)
        except KeyboardInterrupt:
            print("\n🛑 Training interrotto dall'utente.")