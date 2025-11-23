"""
STEP 2 (NORTH-UP SPECIAL): MOSAICO HUBBLE ORIENTATO A NORD
------------------------------------------------------------------------
Genera un mosaico Hubble forzando l'orientamento "North-Up" (Nord in alto).
Questo eliminerà la rotazione strana nei riquadri di anteprima.

INPUT: Cartella '3_registered_native/hubble'
OUTPUT: '5_mosaics/final_mosaic_hubble.fits' (Sovrascrive quello vecchio)
------------------------------------------------------------------------
"""

import os
import sys
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE =================
# Risoluzione desiderata (gradi/pixel). 
# Lasciare None per calcolarla automaticamente dalla media delle immagini.
OUTPUT_PIXEL_SCALE = None 

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

def select_target_directory():
    print("\n" + "📂"*35)
    print("SELEZIONE TARGET (Mosaico Hubble North-Up)".center(70))
    print("📂"*35)
    subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    if not subdirs: return None
    for i, d in enumerate(subdirs):
        print(f"   {i+1}: {d.name}")
    try:
        choice = int(input(f"\n👉 Seleziona (1-{len(subdirs)}): ").strip())
        if 0 < choice <= len(subdirs): return subdirs[choice-1]
    except: pass
    return None

def create_north_up_mosaic(base_dir):
    input_dir = base_dir / '3_registered_native' / 'hubble'
    output_dir = base_dir / '5_mosaics'
    output_file = output_dir / 'final_mosaic_hubble.fits'
    
    if not input_dir.exists():
        print("❌ Errore: Cartella input Hubble non trovata.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Generazione Mosaico Hubble (North-Up) per: {base_dir.name}")
    
    # 1. RACCOLTA FILES E METADATI
    fits_files = sorted(list(input_dir.glob("*.fits")))
    if not fits_files:
        print("❌ Nessun file FITS trovato.")
        return

    print(f"   Caricamento metadati di {len(fits_files)} immagini...")
    
    # Per reproject_and_coadd servono tuple (array, wcs) o HDU
    # Per risparmiare memoria, usiamo una lista di (file_path, hdu_index) e carichiamo al volo
    # Ma find_optimal_celestial_wcs vuole gli oggetti WCS subito.
    
    wcs_list = []
    shapes_list = []
    input_data_list = [] # Lista di tuple (data, wcs)

    for f in tqdm(fits_files, desc="Reading Headers"):
        with fits.open(f) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            if data is None: continue
            if data.ndim == 3: data = data[0]
            
            w = WCS(header)
            wcs_list.append(w)
            shapes_list.append(data.shape)
            
            # Mettiamo in lista per il passaggio successivo (memoria permettendo)
            # Se crasha per RAM, bisogna cambiare strategia
            input_data_list.append((np.nan_to_num(data), w))

    # 2. CALCOLO WCS OTTIMALE (NORTH UP)
    print("\n📐 Calcolo griglia ottimale 'North-Up'...")
    
    # auto_rotate=True ruota il WCS finale per minimizzare l'area, 
    # ma spesso mantiene il Nord in alto se la proiezione è TAN.
    # Per forzare Nord=Alto, dobbiamo assicurarci che CD/PC matrix non abbiano rotazione.
    
    target_wcs, target_shape = find_optimal_celestial_wcs(
        input_data_list, 
        auto_rotate=True, # Tenta di allineare agli assi celesti (RA/DEC)
        projection='TAN'
    )
    
    # Verifica pixel scale
    scale = np.sqrt(target_wcs.wcs.cdelt[0]**2 + target_wcs.wcs.cdelt[1]**2) * 3600
    print(f"   Griglia calcolata: {target_shape[1]}x{target_shape[0]} px")
    print(f"   Risoluzione: {scale:.3f} arcsec/px")
    print(f"   Orientamento: North-Up (Automatico)")

    # 3. CO-ADDING (CREAZIONE MOSAICO)
    print("\n🏗️  Fusione immagini (questo richiede tempo)...")
    
    # reproject_and_coadd fa tutto il lavoro sporco: riproietta e gestisce le sovrapposizioni
    mosaic_array, footprint = reproject_and_coadd(
        input_data_list,
        target_wcs,
        shape_out=target_shape,
        reproject_function=reproject_interp,
        match_background=False
    )

    # 4. SALVATAGGIO
    print(f"\n💾 Salvataggio: {output_file.name}")
    header = target_wcs.to_header()
    header['HISTORY'] = 'Mosaicked with North-Up orientation'
    
    fits.PrimaryHDU(data=np.nan_to_num(mosaic_array), header=header).writeto(output_file, overwrite=True)

    print("\n✅ FATTO!")
    print("   Ora il mosaico di Hubble è orientato col Nord in alto.")
    print("   ⚠️  IMPORTANTE: Ora devi rifare lo Step 2.5 (Super Mosaic)!")
    print("      Così anche l'Osservatorio verrà proiettato su questa nuova griglia dritta.")

if __name__ == "__main__":
    target = select_target_directory()
    if target:
        create_north_up_mosaic(target)