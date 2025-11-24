import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CombinedLoss(nn.Module):
    def __init__(self, l1_w=2.0, perceptual_w=0.05, astro_w=2.0):
        """
        Loss Combinata: L1 + Loss Percettiva (VGG) + Loss Astro (Weighted L1)
        
        Pesi Aggiornati per migliorare la fedeltà strutturale (SSIM):
        - l1_w: 2.0 (Aumentato rispetto a 1.0)
        - perceptual_w: 0.05 (Ridotto rispetto a 0.1)
        - astro_w: 2.0 (Aumentato rispetto a 1.0)
        """
        super().__init__()
        self.weights = (l1_w, perceptual_w, astro_w)
        self.l1 = nn.L1Loss()
        
        # VGG Network for Perceptual Loss (VGG19 features up to layer 18)
        self.vgg = models.vgg19(weights='DEFAULT').features[:18].eval()
        for p in self.vgg.parameters(): p.requires_grad = False
        
        # Registra i buffer per la normalizzazione ImageNet (necessari per VGG)
        # Il .view(1,3,1,1) è cruciale per la broadcasting dimensionale
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        
        # 1. Loss L1 (Mean Absolute Error)
        l1 = self.l1(pred, target)
        
        # 2. Loss Astro (Weighted L1)
        # Enfatizza le regioni con segnale forte (dove target > 0)
        # Formula: L1 * (1.0 + 10.0 * target)
        astro = (F.l1_loss(pred, target, reduction='none') * (1.0 + 10.0 * target)).mean()
        
        # 3. Loss Percettiva (VGG)
        # Prepara i tensori monocromatici [B, 1, H, W] per la rete VGG RGB [B, 3, H, W]
        # Ripetizione del canale + Normalizzazione ImageNet
        pr = (pred.repeat(1,3,1,1) - self.mean) / self.std
        tr = (target.repeat(1,3,1,1) - self.mean) / self.std
        
        # Calcola la L1 tra le feature VGG
        perc = F.l1_loss(self.vgg(pr), self.vgg(tr))
        
        # Loss Totale: Somma pesata
        total_loss = self.weights[0] * l1 + self.weights[1] * perc + self.weights[2] * astro
        
        # Ritorna la loss totale e le componenti per il logging
        return total_loss, \
               {'l1': l1, 'astro': astro, 'perceptual': perc}