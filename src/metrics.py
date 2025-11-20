import torch
import torch.nn.functional as F
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_torch(img1, img2, window_size=11, window=None, size_average=True, val_range=None):
    # Gestione del range di valori (se le immagini sono [0,1] o [0,255])
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
            
        if torch.min(img1) < -0.5:
            min_val = -1
            max_val = 2  # Range [-1, 1] -> estensione 2
        else:
            min_val = 0

        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, _, _) = img1.size()
    
    # Crea la finestra se non passata
    if window is None:
        real_size = min(window_size, img1.size(2), img1.size(3))
        window = create_window(real_size, channel)
        
    # --- FIX IMPORTANTE: Assicura che window sia su stesso device e dtype delle immagini ---
    if window.device != img1.device or window.dtype != img1.dtype:
        window = window.to(img1.device).type_as(img1)
    # ------------------------------------------------------------------------------------

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) + (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class Metrics:
    def __init__(self):
        self.psnr = 0.0
        self.ssim = 0.0
        self.count = 0

    def update(self, pred, target):
        """
        Aggiorna le metriche con un batch di predizioni e target.
        """
        # Assicurati che siano float32 per il calcolo delle metriche per massima precisione
        # Questo evita anche problemi con HalfTensor se la metrica non lo supporta bene
        p = pred.detach().float()
        t = target.detach().float()
        
        batch_size = p.size(0)
        
        # PSNR
        mse = F.mse_loss(p, t, reduction='none').view(batch_size, -1).mean(1)
        # Evita log(0)
        mse = torch.clamp(mse, min=1e-8)
        self.psnr += (10 * torch.log10(1.0 / mse)).sum().item()
        
        # SSIM
        self.ssim += ssim_torch(p, t).item() * batch_size
        
        self.count += batch_size

    def compute(self):
        if self.count == 0:
            return {'psnr': 0.0, 'ssim': 0.0}
        return {
            'psnr': self.psnr / self.count,
            'ssim': self.ssim / self.count
        }

    def reset(self):
        self.psnr = 0.0
        self.ssim = 0.0
        self.count = 0