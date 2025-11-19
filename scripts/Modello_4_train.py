"""
Training Script per Super-Resolution Model con TensorBoard
Features:
- Visualizzazione live su localhost:6006
- Loss, PSNR, SSIM in tempo reale
- Immagini di confronto LR/SR/HR
- Grafici learning rate
"""

import argparse
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import json
import warnings
import torchvision.utils as vutils


from src.architecture import HybridSuperResolutionModel
from src.dataset import AstronomicalDataset
from src.losses import CombinedLoss
from src.metrics import Metrics

# ============================================================
# FIX IMPORT: src è nella cartella PADRE
# ============================================================
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.append(str(PROJECT_ROOT))
# ============================================================
def train(args):
    """Funzione principale di training con TensorBoard"""
    
    # ============================================================
    # SETUP DEVICE E MIXED PRECISION
    # ============================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    
    if use_amp:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        print(f"🚀 Avvio Training su: {device} (Mixed Precision: ON)")
    else:
        scaler = None
        print(f"🚀 Avvio Training su: {device} (Mixed Precision: OFF)")
        warnings.warn("GPU non disponibile. Il training sarà più lento su CPU.")
    
   # ============================================================
    # SETUP DIRECTORIES
    # ============================================================
    if args.output_dir == "./outputs":
        # Salva in SUPERRESOLUTION/outputs invece che scripts/outputs
        save_dir = PROJECT_ROOT / "outputs" / "checkpoints"
        log_dir = PROJECT_ROOT / "outputs" / "tensorboard"
    else:
        # Se l'utente specifica un path, usa quello
        save_dir = Path(args.output_dir) / "checkpoints"
        log_dir = Path(args.output_dir) / "tensorboard"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Output: {save_dir}")
    print(f"📊 TensorBoard logs: {log_dir}")
    
    # ============================================================
    # TENSORBOARD WRITER
    # ============================================================
    writer = SummaryWriter(log_dir=log_dir)
    print(f"\n🌐 Per vedere il training live:")
    print(f"   Apri un nuovo terminale e lancia:")
    print(f"   tensorboard --logdir={log_dir}")
    print(f"   Poi apri: http://localhost:6006")
    print()
    
    # ============================================================
    # VERIFICA DATASET
    # ============================================================
    splits_dir = PROJECT_ROOT / "data" / "splits"
    
    if not (splits_dir / "train.json").exists():
        print(f"❌ Errore: 'train.json' non trovato in {splits_dir}")
        print(f"   Esegui prima lo script 2_prepare_data.py")
        return
    
    if not (splits_dir / "val.json").exists():
        print(f"❌ Errore: 'val.json' non trovato in {splits_dir}")
        return
    
    # ============================================================
    # CARICAMENTO DATASET
    # ============================================================
    print("📦 Caricamento Dataset...")
    try:
        train_ds = AstronomicalDataset(splits_dir / "train.json", base_path=PROJECT_ROOT, augment=True)
        val_ds = AstronomicalDataset(splits_dir / "val.json", base_path=PROJECT_ROOT, augment=False)
    
    except Exception as e:
        print(f"❌ Errore nel caricamento dataset: {e}")
        return
    
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory
    )
    
    print(f"   ✅ Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    print(f"   ✅ Batch size: {args.batch_size}, Workers: {num_workers}")
    
    # ============================================================
    # INIZIALIZZAZIONE MODELLO
    # ============================================================
    print(f"🏗️  Inizializzazione Modello (Smoothing: {args.smoothing})...")
    try:
        model = HybridSuperResolutionModel(
            target_scale=15, 
            smoothing=args.smoothing, 
            device=device
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   📊 Parametri totali: {total_params:,}")
        print(f"   📊 Parametri trainabili: {trainable_params:,}")
        
        # Log architettura su TensorBoard
        writer.add_text('Model/Architecture', f"Total params: {total_params:,}\nTrainable: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ Errore nell'inizializzazione del modello: {e}")
        return
    
    # ============================================================
    # OPTIMIZER E LOSS
    # ============================================================
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    criterion = CombinedLoss().to(device)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5, 
        patience=5
    )
    
    # ============================================================
    # RESUME CHECKPOINT
    # ============================================================
    best_psnr = 0.0
    start_epoch = 0
    global_step = 0
    
    if args.resume:
        if Path(args.resume).exists():
            print(f"📥 Resume da: {args.resume}")
            try:
                ckpt = torch.load(args.resume, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer'])
                start_epoch = ckpt['epoch'] + 1
                best_psnr = ckpt.get('best_psnr', 0.0)
                global_step = ckpt.get('global_step', 0)
                print(f"   ✅ Ripresa dall'epoca {start_epoch}, Best PSNR: {best_psnr:.2f}")
            except Exception as e:
                print(f"⚠️ Errore nel caricamento checkpoint: {e}")
                print("   Continuo con modello fresh...")
        else:
            print(f"⚠️ Checkpoint non trovato: {args.resume}")
    
    # ============================================================
    # TRAINING LOOP
    # ============================================================
    print(f"\n🔥 Inizio Training Loop (Epoche {start_epoch+1} → {args.epochs})...")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs):
        # ========================================
        # TRAINING PHASE
        # ========================================
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoca {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                lr = batch['lr'].to(device, non_blocking=True)
                hr = batch['hr'].to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Forward pass
                if use_amp:
                    from torch.cuda.amp import autocast
                    with autocast():
                        pred = model(lr)
                        loss, loss_components = criterion(pred, hr)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(lr)
                    loss, loss_components = criterion(pred, hr)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                global_step += 1
                
                # ============================================
                # LOG SU TENSORBOARD (ogni 10 step)
                # ============================================
                if global_step % 10 == 0:
                    writer.add_scalar('Train/Loss', loss.item(), global_step)
                    writer.add_scalar('Train/Loss_L1', loss_components[0].item(), global_step)
                    writer.add_scalar('Train/Loss_Astro', loss_components[1].item(), global_step)
                    writer.add_scalar('Train/Loss_Perceptual', loss_components[2].item(), global_step)
                    writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
            except RuntimeError as e:
                print(f"\n❌ Errore al batch {batch_idx}: {e}")
                if "out of memory" in str(e):
                    print("   💡 Suggerimento: riduci --batch_size")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue
        
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # ========================================
        # VALIDATION PHASE
        # ========================================
        model.eval()
        metrics = Metrics()
        val_loss = 0
        
        # Salva primo sample per visualizzazione
        sample_lr, sample_sr, sample_hr = None, None, None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
                try:
                    lr = batch['lr'].to(device)
                    hr = batch['hr'].to(device)
                    
                    pred = model(lr)
                    loss, _ = criterion(pred, hr)
                    val_loss += loss.item()
                    
                    metrics.update(pred, hr)
                    
                    # Salva primo sample
                    if batch_idx == 0:
                        sample_lr = lr[0].cpu()
                        sample_sr = pred[0].cpu()
                        sample_hr = hr[0].cpu()
                    
                except Exception as e:
                    print(f"⚠️ Errore in validazione: {e}")
                    continue
        
        # Compute metrics
        try:
            results = metrics.compute()
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            
            # ============================================
            # LOG METRICHE SU TENSORBOARD
            # ============================================
            writer.add_scalar('Val/Loss', avg_val_loss, epoch)
            writer.add_scalar('Val/PSNR', results['psnr'], epoch)
            writer.add_scalar('Val/SSIM', results['ssim'], epoch)
            writer.add_scalar('Epoch/Train_Loss', avg_train_loss, epoch)
            
            # ============================================
            # LOG IMMAGINI SU TENSORBOARD
            # ============================================
            if sample_lr is not None:
                # Crea griglia di confronto [LR, SR, HR]
                comparison = torch.cat([sample_lr, sample_sr, sample_hr], dim=2)
                writer.add_image('Validation/LR_SR_HR', comparison, epoch)
            
            print(f"\n📊 Epoca {epoch+1}/{args.epochs}:")
            print(f"   Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
            print(f"   PSNR: {results['psnr']:.2f} dB | SSIM: {results['ssim']:.4f}")
            
        except Exception as e:
            print(f"⚠️ Errore nel calcolo metriche: {e}")
            results = {'psnr': 0, 'ssim': 0}
        
        # ========================================
        # SALVATAGGIO CHECKPOINT
        # ========================================
        current_psnr = results.get('psnr', 0)
        
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_psnr': best_psnr,
                'global_step': global_step,
                'config': vars(args)
            }
            torch.save(checkpoint, save_dir / "best_model.pth")
            print(f"   🏆 Best Model Saved! (PSNR: {best_psnr:.2f} dB)")
        
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'psnr': current_psnr,
                'global_step': global_step
            }
            torch.save(checkpoint, save_dir / f"ckpt_ep{epoch+1}.pth")
            print(f"   💾 Checkpoint salvato: ckpt_ep{epoch+1}.pth")
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(current_psnr)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"   📉 Learning rate ridotto: {old_lr:.2e} → {new_lr:.2e}")
        
        print("=" * 70)
    
    print(f"\n✅ Training completato!")
    print(f"🏆 Best PSNR: {best_psnr:.2f} dB")
    print(f"📂 Modelli salvati in: {save_dir}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Super-Resolution Model con TensorBoard")
    
    parser.add_argument('--output_dir', type=str, default="./outputs",
                        help='Directory per salvare checkpoints e logs')
    parser.add_argument('--smoothing', type=str, default='balanced',
                        choices=['none', 'light', 'balanced', 'strong'],
                        help='Livello di smoothing del modello')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Numero di epoche')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (riduci se OOM)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate iniziale')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path al checkpoint da cui riprendere')
    
    args = parser.parse_args()
    
    try:
        train(args)
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrotto dall'utente")
    except Exception as e:
        print(f"\n\n❌ Errore fatale: {e}")
        import traceback
        traceback.print_exc()