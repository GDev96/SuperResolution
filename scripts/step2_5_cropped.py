"""
STEP 5: RITAGLIO IMMAGINI REGISTRATE
Ritaglia tutte le immagini registrate alle dimensioni dell'immagine più piccola.
Questo passo è necessario prima di creare il mosaico per garantire che tutte 
le immagini abbiano le stesse dimensioni.

INPUT: Cartella 'img_register' (output di step2)
OUTPUT: Cartella 'img_cropped' con immagini ritagliate
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
# CONFIGURAZIONE DINAMICA
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.isdir(os.path.join(SCRIPT_DIR, 'data')):
    PROJECT_ROOT = SCRIPT_DIR
elif os.path.isdir(os.path.join(os.path.dirname(SCRIPT_DIR), 'data')):
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    raise FileNotFoundError("Impossibile trovare la directory 'data'.")

BASE_DIR = os.path.join(PROJECT_ROOT, 'data')

# Input: cartella con immagini registrate da step2
INPUT_DIR = os.path.join(BASE_DIR, 'img_register')

# Output: cartella per le immagini ritagliate
OUTPUT_DIR = os.path.join(BASE_DIR, 'img_cropped')

# ============================================================================
# FUNZIONI MENU INTERATTIVO
# ============================================================================

def list_available_sources():
    """Lista le fonti disponibili nell'output di step2."""
    sources = []
    if os.path.exists(INPUT_DIR):
        for item in os.listdir(INPUT_DIR):
            item_path = os.path.join(INPUT_DIR, item)
            if os.path.isdir(item_path):
                sources.append(item)
    return sorted(sources)


def list_available_objects(source):
    """Lista gli oggetti disponibili per una fonte."""
    source_dir = os.path.join(INPUT_DIR, source)
    objects = []
    
    if os.path.exists(source_dir):
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            if os.path.isdir(item_path):
                # Conta file FITS nella cartella
                fits_count = len(glob.glob(os.path.join(item_path, '*.fits')) + 
                               glob.glob(os.path.join(item_path, '*.fit')))
                if fits_count > 0:
                    objects.append((item, fits_count))
    
    return sorted(objects)


def interactive_menu():
    """Menu interattivo per selezionare fonte e oggetto."""
    print("\n" + "=" * 70)
    print("✂️  SELEZIONE IMMAGINI DA RITAGLIARE".center(70))
    print("=" * 70)
    
    # === STEP 1: Selezione Fonte ===
    sources = list_available_sources()
    
    if not sources:
        print("\n❌ Nessuna fonte trovata in data/img_register/")
        print("   Esegui prima: python scripts/step2_register.py")
        return None, None
    
    print("\n📂 FONTI DISPONIBILI:")
    print("-" * 70)
    for i, source in enumerate(sources, 1):
        # Conta oggetti per questa fonte
        objects = list_available_objects(source)
        obj_count = len(objects)
        total_images = sum(count for _, count in objects)
        print(f"   {i}. {source:<15} ({obj_count} oggetti, {total_images} immagini)")
    print("-" * 70)
    
    # Input fonte
    while True:
        try:
            choice = input(f"\n➤ Scegli fonte [1-{len(sources)}]: ").strip()
            source_idx = int(choice) - 1
            
            if 0 <= source_idx < len(sources):
                selected_source = sources[source_idx]
                break
            else:
                print(f"⚠️  Inserisci un numero tra 1 e {len(sources)}")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Operazione annullata.")
            return None, None
    
    print(f"\n✓ Fonte selezionata: {selected_source}")
    
    # === STEP 2: Selezione Oggetto ===
    objects = list_available_objects(selected_source)
    
    if not objects:
        print(f"\n❌ Nessun oggetto con immagini FITS in {selected_source}/")
        return None, None
    
    print(f"\n🎯 OGGETTI DISPONIBILI ({selected_source}):")
    print("-" * 70)
    for i, (obj_name, img_count) in enumerate(objects, 1):
        print(f"   {i}. {obj_name:<20} ({img_count} immagini)")
    print(f"   {len(objects)+1}. TUTTI (ritaglia tutti gli oggetti)")
    print("-" * 70)
    
    # Input oggetto
    while True:
        try:
            choice = input(f"\n➤ Scegli oggetto [1-{len(objects)+1}]: ").strip()
            obj_idx = int(choice) - 1
            
            if obj_idx == len(objects):
                # Tutti gli oggetti
                selected_object = None
                print(f"\n✓ Ritaglierò TUTTI gli oggetti di {selected_source}")
                break
            elif 0 <= obj_idx < len(objects):
                selected_object = objects[obj_idx][0]
                img_count = objects[obj_idx][1]
                print(f"\n✓ Oggetto selezionato: {selected_object} ({img_count} immagini)")
                break
            else:
                print(f"⚠️  Inserisci un numero tra 1 e {len(objects)+1}")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Operazione annullata.")
            return None, None
    
    # === RIEPILOGO ===
    print("\n" + "=" * 70)
    print("📋 RIEPILOGO SELEZIONE")
    print("=" * 70)
    print(f"   Fonte: {selected_source}")
    
    if selected_object:
        print(f"   Oggetto: {selected_object}")
        input_path = os.path.join(INPUT_DIR, selected_source, selected_object)
        output_path = os.path.join(OUTPUT_DIR, selected_source, selected_object)
    else:
        print(f"   Oggetti: TUTTI ({len(objects)} oggetti)")
        input_path = os.path.join(INPUT_DIR, selected_source)
        output_path = os.path.join(OUTPUT_DIR, selected_source)
    
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print("=" * 70)
    
    # Conferma
    confirm = input("\n➤ Confermi e procedi? [S/n]: ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ Operazione annullata.")
        return None, None
    
    return selected_source, selected_object


# ============================================================================
# FUNZIONI RITAGLIO
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
    
    for filepath in tqdm(all_files, desc="  Scansione", unit="file"):
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
            print(f"\n⚠️  ATTENZIONE: Impossibile leggere {os.path.basename(filepath)}: {e}")
            continue
    
    print(f"\n✅ Dimensioni minime trovate: {min_width} x {min_height} pixel")
    print(f"   File più piccolo: {os.path.basename(smallest_file)}")
    
    return min_height, min_width


def crop_image(input_path, output_path, target_height, target_width):
    """
    Ritaglia un'immagine FITS alle dimensioni target (centrato).
    
    Args:
        input_path (str): Percorso file di input
        output_path (str): Percorso file di output
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
        print(f"\n❌ ERRORE nel ritaglio di {os.path.basename(input_path)}: {e}")
        return False


def crop_single_object(source, obj_name):
    """Ritaglia le immagini di un singolo oggetto."""
    input_dir = os.path.join(INPUT_DIR, source, obj_name)
    output_dir = os.path.join(OUTPUT_DIR, source, obj_name)
    
    print(f"\n{'─'*70}")
    print(f"✂️  Ritaglio: {source}/{obj_name}")
    print(f"{'─'*70}")
    print(f"   Input:  {input_dir}")
    print(f"   Output: {output_dir}")
    
    # Crea cartella output
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova file FITS
    all_files = glob.glob(os.path.join(input_dir, '*.fits'))
    all_files.extend(glob.glob(os.path.join(input_dir, '*.fit')))
    
    if not all_files:
        print(f"   ⚠️  Nessun file FITS trovato")
        return 0, 0
    
    print(f"   Trovati: {len(all_files)} file")
    
    # Trova dimensioni minime
    min_height, min_width = find_smallest_dimensions(all_files)
    
    print(f"\n   📏 Ritaglio a: {min_width} x {min_height} pixel")
    
    # Ritaglia
    print("\n   ✂️  Ritaglio in corso...")
    success_count = 0
    failed_count = 0
    
    for filepath in tqdm(all_files, desc="     Ritaglio", unit="file"):
        filename = os.path.basename(filepath)
        output_path = os.path.join(output_dir, filename)
        
        if crop_image(filepath, output_path, min_height, min_width):
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n   ✅ Completato: {success_count} successi")
    if failed_count > 0:
        print(f"   ⚠️  Errori: {failed_count}")
    
    return success_count, failed_count


def crop_all_objects(source):
    """Ritaglia le immagini di tutti gli oggetti di una fonte."""
    objects = list_available_objects(source)
    
    print(f"\n🔄 Ritaglio di {len(objects)} oggetti...")
    
    total_success = 0
    total_failed = 0
    
    for obj_idx, (obj_name, img_count) in enumerate(objects, 1):
        print(f"\n[{obj_idx}/{len(objects)}]")
        success, failed = crop_single_object(source, obj_name)
        total_success += success
        total_failed += failed
    
    return total_success, total_failed


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Funzione principale."""
    print("\n" + "✂️ "*35)
    print("STEP 5: RITAGLIO IMMAGINI REGISTRATE".center(70))
    print("✂️ "*35)
    
    # Menu interattivo
    selected_source, selected_object = interactive_menu()
    
    if not selected_source:
        print("\n❌ Nessuna selezione effettuata.")
        return
    
    start_time = time.time()
    
    # Esegui ritaglio
    if selected_object:
        # Singolo oggetto
        success, failed = crop_single_object(selected_source, selected_object)
        
        # Riepilogo
        print("\n" + "=" * 70)
        print("📊 RIEPILOGO")
        print("=" * 70)
        print(f"   Fonte: {selected_source}")
        print(f"   Oggetto: {selected_object}")
        print(f"   Immagini ritagliate: {success}")
        if failed > 0:
            print(f"   ⚠️  Errori: {failed}")
        print(f"   Output: {os.path.join(OUTPUT_DIR, selected_source, selected_object)}")
        
    else:
        # Tutti gli oggetti
        success, failed = crop_all_objects(selected_source)
        
        # Riepilogo
        print("\n" + "=" * 70)
        print("📊 RIEPILOGO TOTALE")
        print("=" * 70)
        print(f"   Fonte: {selected_source}")
        print(f"   Immagini ritagliate: {success}")
        if failed > 0:
            print(f"   ⚠️  Errori: {failed}")
        print(f"   Output: {os.path.join(OUTPUT_DIR, selected_source)}/")
    
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Tempo totale: {elapsed:.1f} secondi")
    print(f"\n✅ STEP 5 COMPLETATO!")
    print(f"\n   📁 Immagini ritagliate: {OUTPUT_DIR}/{selected_source}/")
    print(f"\n   ➡️  Prossimo passo: Usa le immagini ritagliate per il mosaico")


if __name__ == "__main__":
    main()