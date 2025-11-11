"""
PROGRAMMA DI TEST TEMPORANEO: CREA MOSAICO
Crea un mosaico (media) da tutte le immagini registrate.

INPUT: Cartelle '3_registered/hubble' e '3_registered/observatory'
OUTPUT: Un singolo file FITS 'final_mosaic.fits'
"""

import os
import glob
import time
from pathlib import Path
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

BASE_DIR = r'F:\Super Revolt Gaia\parte 2(patch)\data'

# Input: cartelle con immagini registrate da step3
REGISTERED_DIRS = [
    Path(BASE_DIR) / '4_cropped' / 'hubble',
    Path(BASE_DIR) / '4_cropped' / 'observatory'
]

# Output: una nuova cartella per il mosaico
OUTPUT_DIR = Path(BASE_DIR) / '5_mosaics'
OUTPUT_FILE = OUTPUT_DIR / 'final_mosaic.fits'

# ============================================================================
# MAIN
# ============================================================================

def create_mosaic():
    """Crea il mosaico."""
    print("\n" + "🖼️ "*35)
    print("CREAZIONE MOSAICO DA IMMAGINI REGISTRATE".center(70))
    print("🖼️ "*35)
    
    # Crea cartella output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Cartelle di input:")
    for d in REGISTERED_DIRS:
        print(f"   - {d}")
    print(f"\n📂 File di output:")
    print(f"   - {OUTPUT_FILE}")
    
    # 1. Trova tutti i file FITS registrati
    all_files = []
    for d in REGISTERED_DIRS:
        all_files.extend(glob.glob(str(d / '*.fits')))
        all_files.extend(glob.glob(str(d / '*.fit')))
        
    if not all_files:
        print(f"\n❌ ERRORE: Nessun file FITS trovato nelle cartelle di input.")
        print("   Assicurati di aver eseguito 'step3_register.py' prima.")
        return
        
    print(f"\n✅ Trovati {len(all_files)} file FITS da combinare.")
    
    # 2. Inizializza gli array per la media
    # Prendi le dimensioni e l'header WCS dal primo file
    try:
        with fits.open(all_files[0]) as hdul:
            template_header = hdul[0].header
            shape = hdul[0].data.shape
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile leggere il primo file {all_files[0]}: {e}")
        return
        
    print(f"   Dimensioni mosaico: {shape[1]} x {shape[0]} pixel")
    
    # Array per sommare i valori (usa float64 per precisione)
    total_flux = np.zeros(shape, dtype=np.float64)
    # Array per contare quanti pixel validi ci sono in ogni punto
    n_pixels = np.zeros(shape, dtype=np.int32)
    
    # 3. Itera su tutti i file e combinali
    print("\n🔄 Combinazione immagini in corso...")
    
    for filepath in tqdm(all_files, desc="Combinazione", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                img_data = hdul[0].data
                
                # Assicurati che le dimensioni corrispondano
                if img_data.shape != shape:
                    print(f"\n⚠️  ATTENZIONE: {filepath} ha dimensioni {img_data.shape} diverse da {shape}. Saltato.")
                    continue
                    
                # Trova pixel validi (non NaN)
                valid_mask = ~np.isnan(img_data)
                
                # Sostituisci i NaN con 0 per la somma
                img_data_no_nan = np.nan_to_num(img_data, nan=0.0, copy=False)
                
                # Aggiungi i valori all'array totale
                total_flux += img_data_no_nan
                
                # Incrementa il contatore per i pixel validi
                n_pixels[valid_mask] += 1
                
        except Exception as e:
            print(f"\n⚠️  ATTENZIONE: Errore nel leggere {filepath}: {e}. Saltato.")
            
    # 4. Calcola la media finale
    print("\n🧮 Calcolo della media finale...")
    
    # Inizializza il mosaico finale con NaN
    mosaic_data = np.full(shape, np.nan, dtype=np.float32)
    
    # Trova dove abbiamo almeno un pixel (per evitare divisione per zero)
    valid_stack = n_pixels > 0
    
    # Calcola la media solo dove n_pixels > 0
    mosaic_data[valid_stack] = (total_flux[valid_stack] / n_pixels[valid_stack]).astype(np.float32)
    
    # 5. Salva il file FITS finale
    print(f"\n💾 Salvataggio mosaico in {OUTPUT_FILE}...")
    
    # Aggiorna l'header
    template_header['HISTORY'] = 'Mosaico creato da create_mosaic.py'
    template_header['NCOMBINE'] = (len(all_files), 'Numero di file combinati')
    
    try:
        fits.PrimaryHDU(data=mosaic_data, header=template_header).writeto(OUTPUT_FILE, overwrite=True)
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile salvare il file FITS finale: {e}")
        return

    print(f"\n{'='*70}")
    print("✅ MOSAICO COMPLETATO!")
    print(f"{'='*70}")
    print(f"   File salvato in: {OUTPUT_FILE}")

if __name__ == "__main__":
    start_time = time.time()
    create_mosaic()
    elapsed = time.time() - start_time
    print(f"\n⏱️ Tempo totale: {elapsed:.1f} secondi")