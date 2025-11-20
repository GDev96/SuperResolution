import torch
import torch.nn as nn
import torch.nn.functional as F
# Import relativo: cerca env_setup nella stessa cartella (src/light)
from .env_setup import import_external_archs

RRDBNet, HAT_Arch = import_external_archs()

class AntiCheckerboardLayer(nn.Module):
    def __init__(self, mode='balanced'):
        super().__init__()
        if mode == 'strong':
            k, p, s = 7, 3, 1600.0
            bk = [[1,6,15,20,15,6,1],[6,36,90,120,90,36,6],[15,90,225,300,225,90,15],
                  [20,120,300,400,300,120,20],[15,90,225,300,225,90,15],[6,36,90,120,90,36,6],[1,6,15,20,15,6,1]]
        elif mode == 'balanced':
            k, p, s = 5, 2, 256.0
            bk = [[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]
        else:
            k, p, s = 3, 1, 16.0
            bk = [[1,2,1],[2,4,2],[1,2,1]]

        kernel = torch.tensor(bk, dtype=torch.float32) / s
        self.conv = nn.Conv2d(1, 1, k, padding=p, bias=False)
        with torch.no_grad(): self.conv.weight[0, 0] = kernel
        self.conv.weight.requires_grad = False

    def forward(self, x): return self.conv(x)

class HybridSuperResolutionModel(nn.Module):
    def __init__(self, target_scale=None, smoothing='balanced', device='cuda', output_size=(512, 512)):
        super().__init__()
        
        # Output size dinamica (Default 512x512)
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

        if RRDBNet is None: raise ImportError("BasicSR mancante o non caricato correttamente.")
        
        # ---------------------------------------------------------
        # STAGE 1: RRDBNet (Upscale x2) - Versione Light
        # Input:  128x128 -> Output: 256x256
        # Ho ridotto num_block da 23 a 6 per risparmiare memoria
        # ---------------------------------------------------------
        self.stage1 = RRDBNet(num_in_ch=1, num_out_ch=1, num_feat=64, num_block=6, num_grow_ch=32, scale=2)
        
        self.has_stage2 = False
        self.stage2 = nn.Identity()
        
        # ---------------------------------------------------------
        # STAGE 2: HAT (Upscale x2) - VERSIONE TINY (6GB VRAM SAFE)
        # Input:  256x256 -> Output: 512x512
        # ---------------------------------------------------------
        if HAT_Arch:
            try:
                self.stage2 = HAT_Arch(
                    img_size=64, 
                    patch_size=1, 
                    in_chans=1, 
                    
                    # --- OTTIMIZZAZIONE MEMORIA PER 128px INPUT ---
                    embed_dim=48,           # RIDOTTO (era 180)
                    depths=[3, 3, 3, 3],    # RIDOTTO (era [6,6,6,6,6,6])
                    num_heads=[4, 4, 4, 4], # RIDOTTO (era [6,6,6,6,6,6])
                    # ----------------------------------------------
                    
                    window_size=16, 
                    compress_ratio=3, 
                    squeeze_factor=30, 
                    conv_scale=0.01, 
                    overlap_ratio=0.5, 
                    mlp_ratio=2., 
                    qkv_bias=True, 
                    upscale=2, 
                    img_range=1., 
                    upsampler='pixelshuffle', 
                    resi_connection='1conv'
                )
                self.has_stage2 = True
                print("   ✅ HAT Architecture: TINY Mode (VRAM Safe)")
            except Exception as e: 
                print(f"   ⚠️ HAT Init fallito: {e}")
                pass

        if smoothing != 'none':
            self.s1 = AntiCheckerboardLayer(smoothing)
            self.s2 = AntiCheckerboardLayer(smoothing)
            self.sf = AntiCheckerboardLayer('light')
        else:
            self.s1 = self.s2 = self.sf = nn.Identity()
            
        self.to(device)

    def forward(self, x):
        # 1. BasicSR: 128 -> 256
        x = self.s1(self.stage1(x))
        
        # 2. HAT: 256 -> 512
        if self.has_stage2: 
            x = self.s2(self.stage2(x))
        
        # 3. Check Dimensionale
        if x.shape[-2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode='bicubic', align_corners=False, antialias=True)
        
        return self.sf(x)