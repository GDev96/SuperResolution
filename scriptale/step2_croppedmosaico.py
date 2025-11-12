"""
STEP 3+4: RITAGLIO IMMAGINI REGISTRATE E CREAZIONE MOSAICO
1. Ritaglia tutte le immagini registrate alle dimensioni dell'immagine più piccola
2. Crea un mosaico (media) da tutte le immagini ritagliate

INPUT: Cartelle '3_registered_native/hubble' e '3_registered_native/observatory'
OUTPUT: 
  - Cartelle '4_cropped/hubble' e '4_cropped/observatory' con immagini ritagliate
  - File '5_mosaics/final_mosaic.fits' con il mosaico finale
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
INPUT_DIRS = {
    'hubble': Path(BASE_DIR) / '3_registered_native' / 'hubble',
    'observatory': Path(BASE_DIR) / '3_registered_native' / 'observatory'
}

# Output: cartelle per le immagini ritagliate
OUTPUT_DIR_BASE = Path(BASE_DIR) / '4_cropped'
OUTPUT_DIRS = {
    'hubble': OUTPUT_DIR_BASE / 'hubble',
    'observatory': OUTPUT_DIR_BASE / 'observatory'
}

# Output: cartella e file per il mosaico finale
MOSAIC_OUTPUT_DIR = Path(BASE_DIR) / '5_mosaics'
MOSAIC_OUTPUT_FILE = MOSAIC_OUTPUT_DIR / 'final_mosaic.fits'

# ============================================================================
# FUNZIONI - RITAGLIO
# ============================================================================

def find_smallest_dimensions(all_files):
    """
    Trova le dimensioni dell'immagine più piccola tra tutti i file.
    
    Args:
        all_files (list): Lista di percorsi ai file FITS
        
    Returns:
        tuple: (min_height, min_width) dimensioni minime trovate
    """
    print("\n🔍 Ricerca dimensioni minime...")
    
    min_height = float('inf')
    min_width = float('inf')
    smallest_file = None
    
    for filepath in tqdm(all_files, desc="Scansione", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                height, width = hdul[0].data.shape
                
                if height < min_height or width < min_width:
                    if height < min_height:
                        min_height = height
                    if width < min_width:
                        min_width = width
                    smallest_file = filepath
                    
        except Exception as e:
            print(f"\n⚠️  ATTENZIONE: Impossibile leggere {filepath}: {e}")
            continue
    
    print(f"\n✅ Dimensioni minime trovate: {min_width} x {min_height} pixel")
    print(f"   File più piccolo: {Path(smallest_file).name}")
    
    return min_height, min_width


def crop_image(input_path, output_path, target_height, target_width):
    """
    Ritaglia un'immagine FITS alle dimensioni target (centrato).
    
    Args:
        input_path (Path): Percorso file di input
        output_path (Path): Percorso file di output
        target_height (int): Altezza target in pixel
        target_width (int): Larghezza target in pixel
    """
    try:
        with fits.open(input_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            
            current_height, current_width = data.shape
            
            # Calcola gli offset per centrare il ritaglio
            y_offset = (current_height - target_height) // 2
            x_offset = (current_width - target_width) // 2
            
            # Ritaglia l'immagine
            cropped_data = data[
                y_offset:y_offset + target_height,
                x_offset:x_offset + target_width
            ]
            
            # Aggiorna l'header WCS se presente
            if 'CRPIX1' in header:
                header['CRPIX1'] -= x_offset
            if 'CRPIX2' in header:
                header['CRPIX2'] -= y_offset
                
            # Aggiorna le dimensioni nell'header
            header['NAXIS1'] = target_width
            header['NAXIS2'] = target_height
            
            # Aggiungi informazioni sul ritaglio
            header['HISTORY'] = 'Cropped by step3_crop_and_mosaic.py'
            header['CROPX'] = (x_offset, 'X offset for cropping')
            header['CROPY'] = (y_offset, 'Y offset for cropping')
            header['ORIGW'] = (current_width, 'Original width')
            header['ORIGH'] = (current_height, 'Original height')
            
            # Salva il file ritagliato
            fits.PrimaryHDU(data=cropped_data, header=header).writeto(
                output_path, overwrite=True
            )
            
            return True
            
    except Exception as e:
        print(f"\n❌ ERRORE nel ritaglio di {input_path.name}: {e}")
        return False


def crop_all_images():
    """Esegue il ritaglio di tutte le immagini."""
    print("\n" + "✂️ "*35)
    print("RITAGLIO IMMAGINI REGISTRATE".center(70))
    print("✂️ "*35)
    
    # Crea cartelle di output
    for output_dir in OUTPUT_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Cartelle di input:")
    for name, path in INPUT_DIRS.items():
        print(f"   - {name}: {path}")
    
    print(f"\n📂 Cartelle di output:")
    for name, path in OUTPUT_DIRS.items():
        print(f"   - {name}: {path}")
    
    # 1. Trova tutti i file FITS registrati
    all_files = []
    file_mapping = {}  # Mappa file -> categoria (hubble/observatory)
    
    for category, input_dir in INPUT_DIRS.items():
        files = glob.glob(str(input_dir / '*.fits'))
        files.extend(glob.glob(str(input_dir / '*.fit')))
        
        for f in files:
            all_files.append(f)
            file_mapping[f] = category
        
        print(f"\n   {category}: {len(files)} file trovati")
    
    if not all_files:
        print(f"\n❌ ERRORE: Nessun file FITS trovato nelle cartelle di input.")
        print("   Assicurati di aver eseguito 'step3_register.py' prima.")
        return False
    
    print(f"\n✅ Totale: {len(all_files)} file da ritagliare")
    
    # 2. Trova le dimensioni minime
    min_height, min_width = find_smallest_dimensions(all_files)
    
    # Conferma dimensioni (dovrebbero essere circa 1800px come menzionato)
    print(f"\n📐 Le immagini verranno ritagliate a: {min_width} x {min_height} pixel")
    
    # 3. Ritaglia tutte le immagini
    print("\n✂️  Ritaglio in corso...\n")
    
    success_count = 0
    failed_count = 0
    
    for filepath in tqdm(all_files, desc="Ritaglio", unit="file"):
        # Determina la categoria e il percorso di output
        category = file_mapping[filepath]
        filename = Path(filepath).name
        output_path = OUTPUT_DIRS[category] / filename
        
        # Esegui il ritaglio
        if crop_image(Path(filepath), output_path, min_height, min_width):
            success_count += 1
        else:
            failed_count += 1
    
    # 4. Riepilogo finale
    print(f"\n{'='*70}")
    print("✅ RITAGLIO COMPLETATO!")
    print(f"{'='*70}")
    print(f"   Immagini ritagliate con successo: {success_count}")
    if failed_count > 0:
        print(f"   ⚠️  Immagini con errori: {failed_count}")
    print(f"\n   Dimensioni finali: {min_width} x {min_height} pixel")
    print(f"\n   File salvati in:")
    for name, path in OUTPUT_DIRS.items():
        print(f"   - {name}: {path}")
    
    return success_count > 0


# ============================================================================
# FUNZIONI - MOSAICO
# ============================================================================

def create_mosaic():
    """Crea il mosaico."""
    print("\n" + "🖼️ "*35)
    print("CREAZIONE MOSAICO DA IMMAGINI REGISTRATE".center(70))
    print("🖼️ "*35)
    
    # Crea cartella output
    MOSAIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Cartelle con immagini ritagliate
    REGISTERED_DIRS = [
        OUTPUT_DIRS['hubble'],
        OUTPUT_DIRS['observatory']
    ]
    
    print(f"\n📂 Cartelle di input:")
    for d in REGISTERED_DIRS:
        print(f"   - {d}")
    print(f"\n📂 File di output:")
    print(f"   - {MOSAIC_OUTPUT_FILE}")
    
    # 1. Trova tutti i file FITS ritagliati
    all_files = []
    for d in REGISTERED_DIRS:
        all_files.extend(glob.glob(str(d / '*.fits')))
        all_files.extend(glob.glob(str(d / '*.fit')))
        
    if not all_files:
        print(f"\n❌ ERRORE: Nessun file FITS trovato nelle cartelle di input.")
        print("   Assicurati di aver completato il ritaglio prima.")
        return False
        
    print(f"\n✅ Trovati {len(all_files)} file FITS da combinare.")
    
    # 2. Inizializza gli array per la media
    # Prendi le dimensioni e l'header WCS dal primo file
    try:
        with fits.open(all_files[0]) as hdul:
            template_header = hdul[0].header
            shape = hdul[0].data.shape
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile leggere il primo file {all_files[0]}: {e}")
        return False
        
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
    print(f"\n💾 Salvataggio mosaico in {MOSAIC_OUTPUT_FILE}...")
    
    # Aggiorna l'header
    template_header['HISTORY'] = 'Mosaico creato da step3_crop_and_mosaic.py'
    template_header['NCOMBINE'] = (len(all_files), 'Numero di file combinati')
    
    try:
        fits.PrimaryHDU(data=mosaic_data, header=template_header).writeto(MOSAIC_OUTPUT_FILE, overwrite=True)
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile salvare il file FITS finale: {e}")
        return False

    print(f"\n{'='*70}")
    print("✅ MOSAICO COMPLETATO!")
    print(f"{'='*70}")
    print(f"   File salvato in: {MOSAIC_OUTPUT_FILE}")
    
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Funzione principale che esegue ritaglio e creazione mosaico."""
    print("\n" + "="*70)
    print("PIPELINE: RITAGLIO IMMAGINI + CREAZIONE MOSAICO".center(70))
    print("="*70)
    
    start_time = time.time()
    
    # STEP 1: Ritaglio delle immagini
    if not crop_all_images():
        print("\n❌ Pipeline interrotta: errore durante il ritaglio.")
        return
    
    print("\n\n")
    
    # STEP 2: Creazione del mosaico
    if not create_mosaic():
        print("\n❌ Pipeline interrotta: errore durante la creazione del mosaico.")
        return
    
    # Riepilogo finale
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETATA CON SUCCESSO!")
    print("="*70)
    print(f"   ⏱️  Tempo totale: {elapsed:.1f} secondi ({elapsed/60:.1f} minuti)")


if __name__ == "__main__":
    main()