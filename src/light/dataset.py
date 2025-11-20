import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from astropy.io import fits
import numpy as np
import random
import cv2

# ============================================================
# CONFIGURAZIONE DIMENSIONI DEFAULT (ALLINEATE ALLE PATCH)
# ============================================================
DEFAULT_LR_SIZE = (128, 128)  # AGGIORNATO: Prima era (80, 80)
DEFAULT_HR_SIZE = (512, 512)  # HR Standard

class AstronomicalDataset(Dataset):
    def __init__(self, split_file, base_path, augment=True, force_hr_size=None):
        self.base_path = Path(base_path)
        self.augment = augment
        
        # Imposta la dimensione target LR di default
        self.lr_size = DEFAULT_LR_SIZE
        
        # Gestione dimensione HR (se forzata dallo script di training o default)
        if force_hr_size:
             self.hr_size = (force_hr_size, force_hr_size) if isinstance(force_hr_size, int) else force_hr_size
        else:
             self.hr_size = DEFAULT_HR_SIZE

        with open(split_file, 'r') as f:
            self.pairs = json.load(f)
    
    def _load(self, path):
        if Path(path).is_absolute():
            file_path = Path(path)
        else:
            file_path = self.base_path / path
        try:
            with fits.open(file_path, mode='readonly', memmap=False) as hdul:
                return hdul[0].data.astype(np.float32).copy()
        except Exception:
            # Ritorna nero in caso di errore
            return np.zeros(self.lr_size, dtype=np.float32)
    
    def _norm(self, data):
        data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        min_val, max_val = np.min(data_clean), np.max(data_clean)
        if max_val - min_val < 1e-8: return np.zeros_like(data_clean, dtype=np.float32)
        return ((data_clean - min_val) / (max_val - min_val)).astype(np.float32)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        lr = self._load(pair['ground_path'])
        hr = self._load(pair['hubble_path'])
        
        lr = self._norm(lr)
        hr = self._norm(hr)
        
        # Verifica e Resize se necessario (adatta patch anomale al target 128x128)
        if lr.shape != self.lr_size:
            lr = cv2.resize(lr, self.lr_size, interpolation=cv2.INTER_LINEAR)
        
        # Verifica e Resize HR (adatta al target 512x512)
        if hr.shape != self.hr_size:
            hr = cv2.resize(hr, self.hr_size, interpolation=cv2.INTER_CUBIC)
        
        if self.augment:
            if random.random() < 0.5: lr, hr = np.flipud(lr).copy(), np.flipud(hr).copy()
            if random.random() < 0.5: lr, hr = np.fliplr(lr).copy(), np.fliplr(hr).copy()
            k = random.randint(0, 3)
            if k > 0: lr, hr = np.rot90(lr, k).copy(), np.rot90(hr, k).copy()
        
        return {'lr': torch.from_numpy(np.ascontiguousarray(lr)).unsqueeze(0), 
                'hr': torch.from_numpy(np.ascontiguousarray(hr)).unsqueeze(0)}
    
    def __len__(self):
        return len(self.pairs)