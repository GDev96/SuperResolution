"""
Dataset per Super-Resolution Astronomica
Versione HAT OPTIMIZED: 80x80 -> 512x512
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from astropy.io import fits
import numpy as np
import random
import cv2

# DIMENSIONI FORZATE PER HAT
# 80 è perfetto perché divisibile per 16 (window size)
FIXED_LR_SIZE = (80, 80)
FIXED_HR_SIZE = (512, 512)

class AstronomicalDataset(Dataset):
    def __init__(self, split_file, base_path, augment=True):
        self.base_path = Path(base_path)
        self.augment = augment
        
        with open(split_file, 'r') as f:
            self.pairs = json.load(f)
        
        print(f"📦 Dataset: {len(self.pairs)} coppie")
        print(f"   🔧 Force Resize: LR {FIXED_LR_SIZE} -> HR {FIXED_HR_SIZE}")
    
    def _load(self, path):
        if Path(path).is_absolute():
            file_path = Path(path)
        else:
            file_path = self.base_path / path
        
        try:
            with fits.open(file_path, mode='readonly', memmap=False) as hdul:
                data = hdul[0].data
                return data.astype(np.float32).copy()
        except Exception as e:
            print(f"⚠️ Errore caricamento {file_path.name}: {e}")
            return np.zeros(FIXED_LR_SIZE, dtype=np.float32)
    
    def _norm(self, data):
        data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        min_val = np.min(data_clean)
        max_val = np.max(data_clean)
        if max_val - min_val < 1e-8:
            return np.zeros_like(data_clean, dtype=np.float32)
        return ((data_clean - min_val) / (max_val - min_val)).astype(np.float32)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        lr = self._load(pair['ground_path'])
        hr = self._load(pair['hubble_path'])
        
        # 1. NORMALIZZAZIONE
        lr = self._norm(lr)
        hr = self._norm(hr)
        
        # 2. FORCE RESIZE (Sicurezza)
        # Se i dati sono già 80x80 (da Step 3), questo non farà danni.
        if lr.shape != FIXED_LR_SIZE:
            lr = cv2.resize(lr, FIXED_LR_SIZE, interpolation=cv2.INTER_LINEAR)
        if hr.shape != FIXED_HR_SIZE:
            hr = cv2.resize(hr, FIXED_HR_SIZE, interpolation=cv2.INTER_CUBIC)
        
        # 3. DATA AUGMENTATION
        if self.augment:
            if random.random() < 0.5:
                lr = np.flipud(lr).copy()
                hr = np.flipud(hr).copy()
            if random.random() < 0.5:
                lr = np.fliplr(lr).copy()
                hr = np.fliplr(hr).copy()
            k = random.randint(0, 3)
            if k > 0:
                lr = np.rot90(lr, k).copy()
                hr = np.rot90(hr, k).copy()
        
        lr = np.ascontiguousarray(lr)
        hr = np.ascontiguousarray(hr)
        
        return {
            'lr': torch.from_numpy(lr).unsqueeze(0),
            'hr': torch.from_numpy(hr).unsqueeze(0)
        }
    
    def __len__(self):
        return len(self.pairs)