"""
STEP 5: RITAGLIO IMMAGINI REGISTRATE
Ritaglia tutte le immagini registrate alle dimensioni dell'immagine più piccola.
Questo passo è necessario prima di creare il mosaico per garantire che tutte 
le immagini abbiano le stesse dimensioni.

INPUT: Cartelle '3_registered_native/hubble' e '3_registered_native/observatory'
OUTPUT: Cartelle '4_cropped/hubble' e '4_cropped/observatory' con immagini ritagliate
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

# ============================================================================
# FUNZIONI
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
            header['HISTORY'] = 'Cropped by step5_crop.py'
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
        return
    
    print(f"\n✅ Totale: {len(all_files)} file da ritagliare")
    
    # 2. Trova le dimensioni minime
    min_height, min_width = find_smallest_dimensions(all_files)
    
    # Conferma dimensioni (dovrebbero essere circa 1800px come menzionato)
    print(f"\n📏 Le immagini verranno ritagliate a: {min_width} x {min_height} pixel")
    
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


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    start_time = time.time()
    crop_all_images()
    elapsed = time.time() - start_time
    print(f"\n⏱️ Tempo totale: {elapsed:.1f} secondi")