"""
Crea dataset SR da immagini Hubble registrate + osservatorio locale
"""
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from scipy.ndimage import zoom
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# ✅ CONFIGURAZIONE OGGETTO CELESTE
# Cambia questo valore per elaborare oggetti diversi (M42, M33, NGC2024, etc.)
TARGET_OBJECT = "M42"  # <-- MODIFICA QUI IL NOME DELL'OGGETTO

class SRDatasetCreator:
    def __init__(self, 
                 target_object=TARGET_OBJECT,
                 hubble_dir=f'data/img_register_4/{TARGET_OBJECT}',
                 local_dir=f'data/local_processed/{TARGET_OBJECT}',  # Dopo plate solving
                 output_dir=f'data/dataset_sr_patches/{TARGET_OBJECT}',
                 patch_size_hr=256,
                 scale_factor=4):
        
        self.target_object = target_object
        self.hubble_dir = Path(hubble_dir)
        self.local_dir = Path(local_dir)
        self.output_dir = Path(output_dir)
        self.patch_size_hr = patch_size_hr
        self.patch_size_lr = patch_size_hr // scale_factor
        self.scale_factor = scale_factor
    
    def pair_images(self):
        """
        Determina pairing tra immagini locali e Hubble
        basato su overlap WCS
        """
        hubble_files = sorted(list(self.hubble_dir.glob('*.fit*')))
        local_files = sorted(list(self.local_dir.glob('*.fit*')))
        
        print(f"Hubble images: {len(hubble_files)}")
        print(f"Local images: {len(local_files)}")
        
        pairs = []
        
        # Se una sola immagine locale
        if len(local_files) == 1:
            # Usala con tutte le Hubble che hanno overlap
            local_file = local_files[0]
            with fits.open(local_file) as hdul:
                local_wcs = WCS(hdul[0].header)
                local_shape = hdul[0].data.shape
            
            for hf in hubble_files:
                with fits.open(hf) as hdul:
                    hubble_wcs = WCS(hdul[0].header)
                    hubble_shape = hdul[0].data.shape
                
                # Check overlap
                if self.check_overlap(local_wcs, local_shape, 
                                     hubble_wcs, hubble_shape):
                    pairs.append((local_file, hf))
        
        else:
            # Match multipli basato su coordinate
            for lf in local_files:
                best_match = self.find_best_hubble_match(lf, hubble_files)
                if best_match:
                    pairs.append((lf, best_match))
        
        print(f"Created {len(pairs)} valid pairs")
        return pairs
    
    def check_overlap(self, wcs1, shape1, wcs2, shape2, threshold=0.5):
        """Verifica se due immagini hanno overlap sufficiente"""
        # Implementazione semplificata
        # In realtà dovresti calcolare overlap geometrico preciso
        try:
            center1 = wcs1.pixel_to_world(shape1[1]/2, shape1[0]/2)
            center2 = wcs2.pixel_to_world(shape2[1]/2, shape2[0]/2)
            
            separation = center1.separation(center2).degree
            
            # FOV approssimativo
            fov1 = max(shape1) * wcs1.proj_plane_pixel_scales()[0].to('degree').value
            fov2 = max(shape2) * wcs2.proj_plane_pixel_scales()[0].to('degree').value
            
            # Overlap se separazione < somma dei raggi
            return separation < (fov1 + fov2) / 2
            
        except:
            return False
    
    # ... continua con extract_patches, normalize, etc
    # Simile al codice precedente ma usa img_register_4