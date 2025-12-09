import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from astropy.io import fits
from astropy.wcs import WCS
import warnings

# Ignora warning FITS non critici
warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE PATH =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import moduli progetto
try:
    from src.architecture import HybridSuperResolutionModel
except ImportError:
    sys.exit("❌ Errore Import src. Assicurati di essere nella cartella scripts/")

# ================= CLASSE RICOSTRUZIONE =================

class MosaicReconstructor:
    def __init__(self, target, device):
        self.target = target
        self.device = device
        
        # Percorsi
        self.data_dir = PROJECT_ROOT / "data" / target
        self.patches_dir = self.data_dir / "6_patches_final"       # Per leggere WCS originale
        self.input_dir = self.data_dir / "7_dataset_ready_LOG"      # Per input LR (TIFF)
        self.master_dir = self.data_dir / "3_registered_native" / "hubble"
        self.out_dir = PROJECT_ROOT / "outputs" / target / "mosaic_reconstruction"
        self.weights_path = PROJECT_ROOT / "outputs" / target / "final_weights" / "best.pth"
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Carica il Master per definire la tela
        self._load_master_info()
        
        # Inizializza Modello
        self._init_model()

    def _load_master_info(self):
        """Trova il file master originale per determinare dimensioni e WCS della tela."""
        print("🔍 Ricerca Master Frame...")
        candidates = sorted(list(self.master_dir.glob("*.fits")))
        if not candidates:
            sys.exit("❌ Nessun file Master Hubble trovato in 3_registered_native/hubble")
            
        self.master_path = candidates[0]
        with fits.open(self.master_path) as hdul:
            data = hdul[0].data
            if data.ndim == 3: data = data[0]
            self.h, self.w = data.shape
            self.master_wcs = WCS(hdul[0].header)
            
        print(f"   ✅ Master trovato: {self.master_path.name}")
        print(f"   📐 Dimensioni Tela: {self.w} x {self.h} px")

    def _init_model(self):
        """Carica il modello con i pesi addestrati."""
        print("🧠 Caricamento Modello...")
        if not self.weights_path.exists():
            sys.exit(f"❌ Pesi non trovati: {self.weights_path}. Esegui Modello_4 prima.")
            
        self.model = HybridSuperResolutionModel(smoothing='balanced', device=self.device, output_size=512).to(self.device)
        
        # Carica pesi (gestisce sia DataParallel che standard)
        state = torch.load(self.weights_path, map_location=self.device)
        if 'module.' in list(state.keys())[0]:
            state = {k.replace('module.', ''): v for k, v in state.items()}
            
        self.model.load_state_dict(state)
        self.model.eval()

    def _load_tiff_tensor(self, path):
        """Carica TIFF 16-bit come tensore normalizzato."""
        img = Image.open(path)
        arr = np.array(img, dtype=np.float32) / 65535.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        return tensor.to(self.device)

    def reconstruct(self):
        print(f"🚀 Avvio Ricostruzione Mosaico per {self.target}...")
        
        # 1. Trova tutte le patch
        pair_folders = sorted(list(self.input_dir.glob("pair_*")))
        if not pair_folders:
            sys.exit("❌ Nessuna patch trovata in 7_dataset_ready_LOG.")
            
        # 2. Prepara Tela e Mappa Pesi (per blending)
        # Usiamo float32 per accumulare precisione
        canvas = np.zeros((self.h, self.w), dtype=np.float32)
        weight_map = np.zeros((self.h, self.w), dtype=np.float32)
        
        print(f"   🧩 Elaborazione di {len(pair_folders)} patch...")
        
        for pair in tqdm(pair_folders, ncols=100):
            patch_id = pair.name
            
            # --- PATH ---
            # Input LR (per inferenza)
            lr_path = pair / "observatory.tiff"
            # Originale FITS (per coordinate WCS)
            fits_path = self.patches_dir / patch_id / "hubble.fits"
            
            if not lr_path.exists() or not fits_path.exists():
                continue
                
            # --- INFERENZA ---
            with torch.no_grad():
                lr_tensor = self._load_tiff_tensor(lr_path)
                with torch.cuda.amp.autocast():
                    sr_tensor = self.model(lr_tensor)
                
                # Converti in numpy [512, 512]
                sr_data = sr_tensor.squeeze().cpu().numpy().clip(0, 1)

            # --- POSIZIONAMENTO (WCS MAPPING) ---
            # Leggiamo il WCS della patch originale per sapere dove va
            with fits.open(fits_path) as h:
                patch_wcs = WCS(h[0].header)
                
            # Calcoliamo la coordinata pixel (0,0) della patch nel sistema del Master
            # pixel_to_world(0,0) ci dà le coordinate celesti dell'angolo
            # world_to_pixel(...) ci dà le coordinate pixel sul master
            origin_sky = patch_wcs.pixel_to_world(0, 0)
            origin_pix = self.master_wcs.world_to_pixel(origin_sky)
            
            # Arrotondiamo all'intero più vicino
            x_start = int(np.round(origin_pix[0]))
            y_start = int(np.round(origin_pix[1]))
            
            h_p, w_p = sr_data.shape
            
            # --- GESTIONE BORDI TELA ---
            # Calcola coordinate di ritaglio se la patch esce dalla tela
            y_end = y_start + h_p
            x_end = x_start + w_p
            
            # Clipping indici tela
            y_s_c = max(0, y_start)
            y_e_c = min(self.h, y_end)
            x_s_c = max(0, x_start)
            x_e_c = min(self.w, x_end)
            
            # Clipping indici patch
            py_s = y_s_c - y_start
            py_e = py_s + (y_e_c - y_s_c)
            px_s = x_s_c - x_start
            px_e = px_s + (x_e_c - x_s_c)
            
            if py_e <= py_s or px_e <= px_s:
                continue

            # --- BLENDING ---
            # Aggiungiamo i dati e incrementiamo il peso per la media
            canvas[y_s_c:y_e_c, x_s_c:x_e_c] += sr_data[py_s:py_e, px_s:px_e]
            weight_map[y_s_c:y_e_c, x_s_c:x_e_c] += 1.0

        # 3. Normalizzazione Finale (Media Ponderata)
        print("   ⚖️  Normalizzazione sovrapposizioni...")
        # Evita divisione per zero dove non ci sono patch
        mask_valid = weight_map > 0
        canvas[mask_valid] /= weight_map[mask_valid]
        
        # 4. Salvataggio
        out_name = f"{self.target}_SuperResolution_Mosaic.tiff"
        out_path = self.out_dir / out_name
        
        print(f"   💾 Salvataggio {out_name}...")
        
        # Converti in uint16
        canvas_u16 = np.clip(canvas * 65535, 0, 65535).astype(np.uint16)
        Image.fromarray(canvas_u16).save(out_path)
        
        # Salvataggio copia PNG per preview rapida
        Image.fromarray((canvas * 255).astype(np.uint8)).save(out_path.with_suffix('.png'))
        
        print(f"\n✅ MOSAICO COMPLETATO!")
        print(f"   TIFF (16-bit): {out_path}")
        print(f"   PNG (Preview): {out_path.with_suffix('.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, help='Nome del target (es. M42)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    worker = MosaicReconstructor(args.target, device)
    worker.reconstruct()