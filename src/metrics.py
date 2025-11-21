import torch
import torch.nn.functional as F
from math import exp

# ============================================================================
# FUNZIONI DI SUPPORTO PER SSIM (MANCAVANO PRIMA)
# ============================================================================
def gaussian(window_size, sigma):
    """Genera una gaussiana 1D."""
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """Crea una finestra 2D gaussiana per il calcolo SSIM."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

# ============================================================================
# METRICHE STANDARD
# ============================================================================
def ssim_torch(img1, img2, window_size=11, window=None, size_average=True, val_range=1.0):
    L = val_range
    padd = window_size // 2  # Padding corretto basato sulla dimensione finestra
    (_, channel, _, _) = img1.size()
    
    # Crea la finestra se non passata
    if window is None:
        real_size = min(window_size, img1.size(2), img1.size(3))
        window = create_window(real_size, channel)
        
    if window.device != img1.device or window.dtype != img1.dtype:
        window = window.to(img1.device).type_as(img1)
    
    # Calcoli SSIM (Convoluzioni)
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
        p = pred.detach().float()
        t = target.detach().float()
        
        # Clamp a [0, 1] per stabilità
        p_norm = torch.clamp(p, 0.0, 1.0)
        t_norm = torch.clamp(t, 0.0, 1.0)
        
        batch_size = p_norm.size(0)
        
        # PSNR
        mse = F.mse_loss(p_norm, t_norm, reduction='none').view(batch_size, -1).mean(1)
        mse = torch.clamp(mse, min=1e-8)
        self.psnr += (10 * torch.log10(1.0 / mse)).sum().item()
        
        # SSIM
        self.ssim += ssim_torch(p_norm, t_norm, val_range=1.0).item() * batch_size
        
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