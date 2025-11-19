"""
STEP 3+4: RITAGLIO IMMAGINI REGISTRATE E CREAZIONE MOSAICO
1. Ritaglia tutte le immagini registrate alle dimensioni dell'immagine più piccola
2. Crea un mosaico (media) da tutte le immagini ritagliate

INPUT: Cartelle '3_registered_native/hubble' e '3_registered_native/observatory'
OUTPUT: 
  - Cartelle '4_cropped/hubble' e '4_cropped/observatory' con immagini ritagliate
  - File '5_mosaics/final_mosaic.fits' con il mosaico finale

MODIFICATO: Integra gestione path dinamici, menu selezione e loop batch da v2.
"""

import os
import sys
import glob
import time
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import warnings
import subprocess

warnings.filterwarnings('ignore')
# ============================================================================
# CONFIGURAZIONE PATH DINAMICI (UNIVERSALI)
# ============================================================================
# 1. Ottiene la directory corrente dello script
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent

# 2. Risale alla root del progetto
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent

# 3. Dove cercare i dati
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# 4. Dove salvare i log
LOG_DIR_ROOT = ROOT_DATA_DIR / "logs"

# 5. Cartella script
SCRIPTS_DIR = CURRENT_SCRIPT_DIR
# ============================================================================

# ============================================================================
# SETUP LOGGING
# ============================================================================

def setup_logging():
    """Configura logging."""
    os.makedirs(LOG_DIR_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = LOG_DIR_ROOT / f'crop_mosaic_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

# ============================================================================
# FUNZIONI MENU E SELEZIONE (DA V2)
# ============================================================================

def select_target_directory():
    """Mostra un menu per selezionare una o TUTTE le cartelle target."""
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET".center(70))
    print("📂"*35)
    print(f"\nScansione sottocartelle in: {ROOT_DATA_DIR}")

    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile leggere la cartella {ROOT_DATA_DIR}: {e}")
        return []

    if not subdirs:
        print(f"\n❌ ERRORE: Nessuna sottocartella trovata in {ROOT_DATA_DIR}")
        return []

    print("\nCartelle target disponibili:")
    print(f"   0: ✨ Processa TUTTI i {len(subdirs)} target")
    print("   " + "─"*30)
    for i, dir_path in enumerate(subdirs):
        print(f"   {i+1}: {dir_path.name}")

    while True:
        print("\n" + "─"*70)
        try:
            choice_str = input(f"👉 Seleziona un numero (0-{len(subdirs)}) o 'q' per uscire: ").strip()

            if choice_str.lower() == 'q':
                return [] 

            choice = int(choice_str)

            if choice == 0:
                print(f"\n✅ Selezionati TUTTI i {len(subdirs)} target.")
                return subdirs 
            
            choice_idx = choice - 1
            if 0 <= choice_idx < len(subdirs):
                selected_dir = subdirs[choice_idx]
                print(f"\n✅ Cartella selezionata: {selected_dir.name}")
                return [selected_dir]
            else:
                print(f"❌ Scelta non valida.")
        except ValueError:
            print("❌ Input non valido.")

def ask_continue_to_next_step():
    """Chiede se proseguire con il prossimo script."""
    print("\n" + "="*70)
    print("🎯 STEP 3+4 (Crop e Mosaico) COMPLETATI!")
    print("="*70)
    print("\n📋 OPZIONI:")
    print("   1️⃣  Continua con Step 5 (Analisi Patch - step3_analizzapatch.py)")
    print("   2️⃣  Termina qui")
    
    next_script_name = 'step3_analizzapatch.py'
    
    while True:
        print("\n" + "─"*70)
        choice = input(f"👉 Vuoi continuare con '{next_script_name}'? [S/n, default=S]: ").strip().lower()
        if choice in ('', 's', 'si', 'y', 'yes'):
            return True
        elif choice in ('n', 'no'):
            return False
        else:
            print("❌ Scelta non valida.")

# ============================================================================
# FUNZIONI - RITAGLIO (LOGICA ORIGINALE MANTENUTA)
# ============================================================================

def find_smallest_dimensions(all_files):
    """
    Trova le dimensioni dell'immagine più piccola tra tutti i file.
    """
    print("\n🔍 Ricerca dimensioni minime...")
    
    min_height = float('inf')
    min_width = float('inf')
    smallest_file = None
    
    for filepath in tqdm(all_files, desc="Scansione", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                if len(hdul[0].data.shape) == 3:
                    height, width = hdul[0].data[0].shape
                else:
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
    if smallest_file:
        print(f"   File più piccolo: {Path(smallest_file).name}")
    
    return min_height, min_width


def crop_image(input_path, output_path, target_height, target_width):
    """
    Ritaglia un'immagine FITS alle dimensioni target (centrato).
    """
    try:
        with fits.open(input_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy() # Copia header
            
            # Gestione dati 3D se necessario
            if len(data.shape) == 3:
                data = data[0]
            
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
            header['HISTORY'] = 'Cropped by step2_croppedmosaico.py'
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


def crop_all_images_for_target(base_dir):
    """Esegue il ritaglio di tutte le immagini per un target specifico."""
    print("\n" + "✂️ "*35)
    print(f"RITAGLIO: {base_dir.name}".center(70))
    print("✂️ "*35)
    
    # Definizione path relativi al target corrente
    input_dirs = {
        'hubble': base_dir / '3_registered_native' / 'hubble',
        'observatory': base_dir / '3_registered_native' / 'observatory'
    }
    
    output_dir_base = base_dir / '4_cropped'
    output_dirs = {
        'hubble': output_dir_base / 'hubble',
        'observatory': output_dir_base / 'observatory'
    }
    
    # Crea cartelle di output
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Trova tutti i file FITS registrati
    all_files = []
    file_mapping = {}  # Mappa file -> categoria (hubble/observatory)
    
    for category, input_dir in input_dirs.items():
        files = list(input_dir.glob('*.fits')) + list(input_dir.glob('*.fit'))
        
        for f in files:
            all_files.append(f)
            file_mapping[f] = category
        
        print(f"   {category}: {len(files)} file trovati")
    
    if not all_files:
        print(f"\n❌ ERRORE: Nessun file FITS trovato in {base_dir.name}/3_registered_native.")
        return False
    
    # 2. Trova le dimensioni minime
    min_height, min_width = find_smallest_dimensions(all_files)
    
    # Conferma dimensioni
    print(f"\n📐 Dimensioni target: {min_width} x {min_height} pixel")
    
    # 3. Ritaglia tutte le immagini
    print("\n✂️  Ritaglio in corso...\n")
    
    success_count = 0
    failed_count = 0
    
    for filepath in tqdm(all_files, desc="Ritaglio", unit="file"):
        # Determina la categoria e il percorso di output
        category = file_mapping[filepath]
        filename = filepath.name
        output_path = output_dirs[category] / filename
        
        # Esegui il ritaglio
        if crop_image(filepath, output_path, min_height, min_width):
            success_count += 1
        else:
            failed_count += 1
    
    # 4. Riepilogo finale
    print(f"\n   ✅ Ritagliati: {success_count}")
    if failed_count > 0:
        print(f"   ⚠️  Errori: {failed_count}")
    
    return success_count > 0

# ============================================================================
# FUNZIONI - MOSAICO (LOGICA ORIGINALE MANTENUTA)
# ============================================================================

def create_mosaic_for_target(base_dir):
    """Crea il mosaico per un target specifico."""
    print("\n" + "🖼️ "*35)
    print(f"MOSAICO: {base_dir.name}".center(70))
    print("🖼️ "*35)
    
    # Path
    output_dir_base = base_dir / '4_cropped'
    cropped_dirs = [
        output_dir_base / 'hubble',
        output_dir_base / 'observatory'
    ]
    
    mosaic_output_dir = base_dir / '5_mosaics'
    mosaic_output_file = mosaic_output_dir / 'final_mosaic.fits'
    
    # Crea cartella output
    mosaic_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Trova tutti i file FITS ritagliati
    all_files = []
    for d in cropped_dirs:
        all_files.extend(list(d.glob('*.fits')) + list(d.glob('*.fit')))
        
    if not all_files:
        print(f"\n❌ ERRORE: Nessun file FITS ritagliato trovato.")
        return False
        
    print(f"\n✅ Trovati {len(all_files)} file FITS da combinare.")
    
    # 2. Inizializza gli array per la media
    try:
        with fits.open(all_files[0]) as hdul:
            template_header = hdul[0].header.copy()
            if len(hdul[0].data.shape) == 3:
                shape = hdul[0].data[0].shape
            else:
                shape = hdul[0].data.shape
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile leggere il primo file {all_files[0].name}: {e}")
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
                if len(img_data.shape) == 3:
                    img_data = img_data[0]
                
                # Assicurati che le dimensioni corrispondano
                if img_data.shape != shape:
                    print(f"\n⚠️  ATTENZIONE: {filepath.name} ha dimensioni {img_data.shape} diverse da {shape}. Saltato.")
                    continue
                    
                # Trova pixel validi (non NaN e non zero)
                # Nota: nell'originale era solo ~np.isnan, ma spesso 0 è usato come background
                valid_mask = ~np.isnan(img_data)
                
                # Sostituisci i NaN con 0 per la somma
                img_data_no_nan = np.nan_to_num(img_data, nan=0.0, copy=False)
                
                # Aggiungi i valori all'array totale
                total_flux += img_data_no_nan
                
                # Incrementa il contatore per i pixel validi
                n_pixels[valid_mask] += 1
                
        except Exception as e:
            print(f"\n⚠️  ATTENZIONE: Errore nel leggere {filepath.name}: {e}. Saltato.")
            
    # 4. Calcola la media finale
    print("\n🧮 Calcolo della media finale...")
    
    # Inizializza il mosaico finale con NaN
    mosaic_data = np.full(shape, np.nan, dtype=np.float32)
    
    # Trova dove abbiamo almeno un pixel
    valid_stack = n_pixels > 0
    
    # Calcola la media solo dove n_pixels > 0
    mosaic_data[valid_stack] = (total_flux[valid_stack] / n_pixels[valid_stack]).astype(np.float32)
    
    # 5. Salva il file FITS finale
    print(f"\n💾 Salvataggio mosaico in {mosaic_output_file.name}...")
    
    # Aggiorna l'header
    template_header['HISTORY'] = 'Mosaico creato da step2_croppedmosaico.py'
    template_header['NCOMBINE'] = (len(all_files), 'Numero di file combinati')
    
    try:
        fits.PrimaryHDU(data=mosaic_data, header=template_header).writeto(mosaic_output_file, overwrite=True)
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile salvare il file FITS finale: {e}")
        return False

    print(f"\n✅ MOSAICO COMPLETATO: {mosaic_output_file}")
    return True

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Funzione principale."""
    logger = setup_logging()
    
    # Selezione Target
    target_dirs = select_target_directory()
    if not target_dirs:
        return

    logger.info(f"Inizio batch su {len(target_dirs)} target")
    
    print("\n" + "="*70)
    print("PIPELINE: RITAGLIO IMMAGINI + CREAZIONE MOSAICO (BATCH)".center(70))
    print("="*70)
    
    start_time_total = time.time()
    successful_targets = []
    failed_targets = []
    
    for base_dir in target_dirs:
        print("\n" + "🚀"*35)
        print(f"ELABORAZIONE TARGET: {base_dir.name}".center(70))
        print("🚀"*35)
        
        # STEP 1: Ritaglio
        if not crop_all_images_for_target(base_dir):
            print(f"\n❌ Target {base_dir.name} fallito al ritaglio.")
            failed_targets.append(base_dir)
            continue
            
        # STEP 2: Mosaico
        if not create_mosaic_for_target(base_dir):
            print(f"\n❌ Target {base_dir.name} fallito al mosaico.")
            failed_targets.append(base_dir)
            continue
            
        successful_targets.append(base_dir)
        logger.info(f"Target completato: {base_dir.name}")

    # Riepilogo finale
    elapsed_total = time.time() - start_time_total
    print("\n" + "="*70)
    print("📊 RIEPILOGO BATCH")
    print("="*70)
    print(f"   ✅ Completati: {len(successful_targets)}")
    for t in successful_targets: print(f"      - {t.name}")
    print(f"\n   ❌ Falliti: {len(failed_targets)}")
    for t in failed_targets: print(f"      - {t.name}")
    print(f"\n   ⏱️ Tempo totale: {elapsed_total:.2f}s")

    if not successful_targets:
        return

    # Transizione al prossimo step
    if ask_continue_to_next_step():
        try:
            next_script = SCRIPTS_DIR / 'step3_analizzapatch.py'
            if next_script.exists():
                print(f"\n🚀 Avvio Step 5 per {len(successful_targets)} target...")
                for base_dir in successful_targets:
                    print(f"\n--- Avvio per {base_dir.name} ---")
                    subprocess.run([sys.executable, str(next_script), str(base_dir)])
            else:
                print(f"\n⚠️  Script {next_script.name} non trovato in {SCRIPTS_DIR}")
        except Exception as e:
            print(f"❌ Errore avvio script successivo: {e}")

if __name__ == "__main__":
    main()