"""
STEP 2.5 (ADVANCED): CREAZIONE MOSAICO OSSERVATORIO SU GRIGLIA HUBBLE
------------------------------------------------------------------------
Invece di deformare il mosaico finale, questo script prende ogni singola
immagine dell'osservatorio e la proietta direttamente sulla griglia 
ad alta risoluzione di Hubble.

VANTAGGI:
- Minore perdita di qualità (una sola interpolazione invece di due).
- Allineamento perfetto nativo.

INPUT: 
  - Master: 5_mosaics/final_mosaic_hubble.fits (WCS Target)
  - Files:  3_registered_native/observatory/*.fits (Files Sorgente)

OUTPUT:
  - 5_mosaics/aligned_ready_for_crop/aligned_observatory.fits
  (Sovrascrive quello vecchio con questa versione migliore)
------------------------------------------------------------------------
"""

import os
import sys
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE =================
# Se hai problemi di RAM, riduci questo numero (es. a 5000)
# Se hai 64GB di RAM come dicevi, non toccare nulla.
CHUNK_SIZE = None 

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

def select_target_directory():
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET (Super Mosaico)".center(70))
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

def create_super_mosaic(base_dir):
    # Path
    mosaics_dir = base_dir / '5_mosaics'
    input_obs_dir = base_dir / '3_registered_native' / 'observatory'
    output_dir = mosaics_dir / 'aligned_ready_for_crop'
    
    master_hubble = mosaics_dir / 'final_mosaic_hubble.fits'
    output_file = output_dir / 'aligned_observatory.fits'

    # Controlli
    if not master_hubble.exists():
        print(f"❌ ERRORE: Manca il mosaico Hubble Master: {master_hubble}")
        return
    if not input_obs_dir.exists():
        print(f"❌ ERRORE: Cartella immagini observatory non trovata: {input_obs_dir}")
        return
    
    # Crea output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. CARICA MASTER WCS (HUBBLE)
    print(f"\n📡 Caricamento Master WCS da: {master_hubble.name}")
    with fits.open(master_hubble) as h:
        h_data = h[0].data
        if h_data.ndim == 3: h_data = h_data[0]
        h_header = h[0].header
        h_wcs = WCS(h_header)
        h_shape = h_data.shape
        
    print(f"   Target Grid: {h_shape[1]} x {h_shape[0]} pixel (Alta Risoluzione)")

    # 2. PREPARA ACCUMULATORI
    # Creiamo due matrici giganti vuote
    print("   Allocazione memoria per il Super Mosaico...")
    try:
        mosaic_sum = np.zeros(h_shape, dtype=np.float32)
        mosaic_weights = np.zeros(h_shape, dtype=np.float32)
    except MemoryError:
        print("❌ ERRORE: Memoria insufficiente per questa griglia!")
        return

    # 3. LISTA FILES
    obs_files = sorted(list(input_obs_dir.glob("*.fits")))
    if not obs_files:
        print("❌ Nessun file FITS trovato nella cartella observatory.")
        return
    
    print(f"\n🚀 Inizio fusione di {len(obs_files)} immagini Observatory sulla griglia Hubble...")

    # 4. LOOP DI PROIEZIONE
    success_count = 0
    with tqdm(total=len(obs_files), desc="Projecting", unit="img") as pbar:
        for f in obs_files:
            try:
                with fits.open(f) as hdul:
                    # Cerca dati validi
                    data = None
                    header = None
                    for hdu in hdul:
                        if hdu.data is not None and hdu.data.ndim >= 2:
                            data = hdu.data
                            header = hdu.header
                            break
                    
                    if data is None: continue
                    if data.ndim == 3: data = data[0]
                    
                    # Leggi WCS input
                    wcs_in = WCS(header)
                    data = np.nan_to_num(data)

                    # RIPROIEZIONE DIRETTA SU HUBBLE
                    # Qui sta la magia: ogni pixel observatory viene spalmato sui pixel Hubble
                    reprojected, footprint = reproject_interp(
                        (data, wcs_in),
                        h_wcs,
                        shape_out=h_shape,
                        order='bilinear' # Bilineare è più sicuro per evitare artefatti di ringing
                    )
                    
                    reprojected = np.nan_to_num(reprojected)
                    footprint = np.nan_to_num(footprint)

                    # Somma accumulativa
                    mosaic_sum += reprojected
                    mosaic_weights += footprint
                    
                    success_count += 1

            except Exception as e:
                # print(f"Err: {e}") # Decommenta per debug
                pass
            
            pbar.update(1)

    # 5. NORMALIZZAZIONE (MEDIA)
    print("\n🧮 Calcolo media finale...")
    with np.errstate(divide='ignore', invalid='ignore'):
        final_mosaic = np.where(mosaic_weights > 0, mosaic_sum / mosaic_weights, 0.0)

    # 6. SALVATAGGIO
    print(f"💾 Salvataggio Super Mosaico in: {output_file.name}...")
    
    # Usiamo l'header di Hubble come base (perché la geometria è quella!)
    new_header = h_header.copy()
    new_header['HISTORY'] = 'Super-Mosaic: Created from individual Obs files projected on Hubble Grid'
    new_header['ORIGIN'] = 'Observatory Data'
    
    fits.PrimaryHDU(data=final_mosaic, header=new_header).writeto(output_file, overwrite=True)
    
    # Copiamo anche il file hubble nella cartella aligned, così sono vicini
    hubble_copy = output_dir / 'aligned_hubble.fits'
    if not hubble_copy.exists():
        print("   Copia di sicurezza del master Hubble...")
        # Ricarichiamo per salvare pulito
        fits.PrimaryHDU(data=h_data, header=h_header).writeto(hubble_copy, overwrite=True)

    print("\n✅ COMPLETATO!")
    print("   Ora 'aligned_observatory.fits' è un mosaico ad alta risoluzione generato dai file grezzi.")
    print("   Puoi rieseguire lo Step 3 (Cutting) per ottenere patch di qualità superiore.")

if __name__ == "__main__":
    target = select_target_directory()
    if target:
        create_super_mosaic(target)