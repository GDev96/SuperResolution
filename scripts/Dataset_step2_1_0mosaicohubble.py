"""
STEP 3+4: RITAGLIO IMMAGINI REGISTRATE E CREAZIONE MOSAICO (SOLO HUBBLE)

1. Ritaglia tutte le immagini registrate nella cartella hubble alle dimensioni dell'immagine più piccola (centrato).
2. Crea un mosaico (media) da tutte le immagini ritagliate di hubble.

INPUT: Esclusivamente la cartella '3_registered_native/hubble' (la cartella 'observatory' è ignorata).
OUTPUT: 
  - Cartella '4_cropped/hubble' con immagini ritagliate.
  - File '5_mosaics/final_mosaic_hubble.fits' (Nome del mosaico modificato per coerenza).
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

# Suppress warnings, especially from FITS/WCS header handling
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE PATH DINAMICI
# ============================================================================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR_ROOT = ROOT_DATA_DIR / "logs"
SCRIPTS_DIR = CURRENT_SCRIPT_DIR

# ============================================================================
# SETUP LOGGING
# ============================================================================

def setup_logging():
    """Configura il sistema di logging."""
    os.makedirs(LOG_DIR_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = LOG_DIR_ROOT / f'crop_mosaic_hubble_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# ============================================================================
# FUNZIONI MENU E SELEZIONE
# ============================================================================

def select_target_directory(logger):
    """Mostra un menu per selezionare una o TUTTE le cartelle target."""
    logger.info("Avvio selezione cartella target.")
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET".center(70))
    print("📂"*35)
    
    try:
        # Filtra solo le sottocartelle valide in ROOT_DATA_DIR
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    except Exception as e:
        logger.error(f"Impossibile leggere la cartella {ROOT_DATA_DIR}: {e}")
        print(f"\n❌ ERRORE: Impossibile leggere la cartella {ROOT_DATA_DIR}: {e}")
        return []

    if not subdirs:
        logger.warning(f"Nessuna sottocartella target trovata in {ROOT_DATA_DIR}")
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
                logger.info(f"Selezionati TUTTI i {len(subdirs)} target.")
                print(f"\n✅ Selezionati TUTTI i {len(subdirs)} target.")
                return subdirs 
            
            choice_idx = choice - 1
            if 0 <= choice_idx < len(subdirs):
                selected_dir = subdirs[choice_idx]
                logger.info(f"Cartella selezionata: {selected_dir.name}")
                print(f"\n✅ Cartella selezionata: {selected_dir.name}")
                return [selected_dir]
            else:
                print("❌ Scelta non valida.")
        except ValueError:
            print("❌ Input non valido. Inserisci un numero.")

def ask_continue_to_next_step():
    """Chiede se proseguire con il prossimo script (Step 5)."""
    print("\n" + "="*70)
    print("🎯 STEP 3+4 (Crop e Mosaico) COMPLETATI!")
    print("="*70)
    
    next_script_name = 'Dataset_step3_analizzapatch.py'
    print(f"\n📋 Prossimo Step: Analisi Patch ('{next_script_name}')")
    
    while True:
        print("\n" + "─"*70)
        choice = input("👉 Vuoi continuare con lo Step 5? [S/n, default=S]: ").strip().lower()
        if choice in ('', 's', 'si', 'y', 'yes'):
            return True
        elif choice in ('n', 'no'):
            return False
        else:
            print("❌ Scelta non valida.")

# ============================================================================
# FUNZIONI DI RITAGLIO (CROP)
# ============================================================================

def find_smallest_dimensions(all_files, logger):
    """Trova le dimensioni dell'immagine 2D più piccola tra tutti i file FITS."""
    logger.info("Ricerca dimensioni minime tra le immagini registrate.")
    print("\n🔍 Ricerca dimensioni minime...")
    
    min_height, min_width = float('inf'), float('inf')
    smallest_file = None
    
    for filepath in tqdm(all_files, desc="Scansione", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                data_shape = hdul[0].data.shape
                
                if len(data_shape) == 3:
                    height, width = data_shape[1], data_shape[2]
                elif len(data_shape) == 2:
                    height, width = data_shape
                else:
                    continue
                
                # Aggiorna le dimensioni minime
                if height < min_height: min_height = height
                if width < min_width: min_width = width
                
                if height == min_height and width == min_width:
                    smallest_file = filepath
                    
        except Exception as e:
            logger.warning(f"Impossibile leggere {filepath}: {e}")
            continue
    
    if min_height == float('inf') or min_width == float('inf'):
        logger.error("Nessuna dimensione valida trovata.")
        return 0, 0
        
    logger.info(f"Dimensioni minime: {min_width} x {min_height}")
    print(f"\n✅ Dimensioni minime trovate: {min_width} x {min_height} pixel")
    if smallest_file:
        print(f"   File più piccolo: {Path(smallest_file).name}")
    return min_height, min_width


def crop_image(input_path, output_path, target_height, target_width, logger):
    """Ritaglia un'immagine FITS alle dimensioni target (centrato) e aggiorna l'header."""
    try:
        with fits.open(input_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()
            
            if len(data.shape) == 3:
                data = data[0]
            
            current_height, current_width = data.shape
            
            y_offset = (current_height - target_height) // 2
            x_offset = (current_width - target_width) // 2
            
            cropped_data = data[
                y_offset : y_offset + target_height,
                x_offset : x_offset + target_width
            ]
            
            # Aggiorna il WCS (World Coordinate System) nell'header
            if 'CRPIX1' in header: header['CRPIX1'] -= x_offset
            if 'CRPIX2' in header: header['CRPIX2'] -= y_offset
            
            header['NAXIS1'] = target_width
            header['NAXIS2'] = target_height
            header['HISTORY'] = f"Cropped to {target_width}x{target_height} centered"

            fits.PrimaryHDU(data=cropped_data, header=header).writeto(
                output_path, overwrite=True
            )
            return True
            
    except Exception as e:
        logger.error(f"ERRORE nel ritaglio di {input_path.name}: {e}")
        return False


def crop_all_images_for_target(base_dir, logger):
    """Coordina la ricerca delle dimensioni minime e il ritaglio batch (SOLO HUBBLE)."""
    logger.info(f"Avvio ritaglio per target: {base_dir.name} (SOLO HUBBLE)")
    print("\n" + "✂️ "*35)
    print(f"RITAGLIO: {base_dir.name} (SOLO HUBBLE)".center(70))
    print("✂️ "*35)
    
    # Cartella di input UNICA (Hubble)
    input_dir_hubble = base_dir / '3_registered_native' / 'hubble'
    
    # Cartella di output
    output_dir_base = base_dir / '4_cropped'
    output_dir_hubble = output_dir_base / 'hubble'
    
    output_dir_hubble.mkdir(parents=True, exist_ok=True)
    
    all_files = list(input_dir_hubble.glob('*.fits')) + list(input_dir_hubble.glob('*.fit'))
    
    logger.info(f"Trovati {len(all_files)} file in Hubble.")
    print(f"   Hubble: {len(all_files)} file trovati")
    
    if not all_files:
        logger.error("Nessun file trovato per il ritaglio in Hubble.")
        print(f"\n❌ ERRORE: Nessun file FITS in {input_dir_hubble}.")
        return False
    
    # 1. Trova le dimensioni target
    min_height, min_width = find_smallest_dimensions(all_files, logger)
    if min_height == 0 or min_width == 0:
        return False
        
    print(f"\n📐 Target: {min_width} x {min_height} pixel")
    
    # 2. Esegue il ritaglio
    success_count = 0
    for filepath in tqdm(all_files, desc="Ritaglio Hubble", unit="file"):
        output_filepath = output_dir_hubble / filepath.name
        
        if crop_image(filepath, output_filepath, min_height, min_width, logger):
            success_count += 1
        
    logger.info(f"Ritaglio completato. {success_count} file ritagliati.")
    return success_count > 0

# ============================================================================
# FUNZIONI DI MOSAICO (STACKING E MEDIA)
# ============================================================================

def create_mosaic_for_target(base_dir, logger):
    """Crea un mosaico (media) da tutte le immagini ritagliate di Hubble."""
    logger.info(f"Avvio creazione mosaico per target: {base_dir.name} (SOLO HUBBLE)")
    print("\n" + "🖼️ "*35)
    print(f"MOSAICO: {base_dir.name} (SOLO HUBBLE)".center(70))
    print("🖼️ "*35)
    
    output_dir_base = base_dir / '4_cropped'
    
    # Cerca solo i file Hubble ritagliati
    all_files = list((output_dir_base / 'hubble').glob('*.fits'))
        
    if not all_files:
        logger.error("Nessun file ritagliato di Hubble trovato per il mosaico.")
        print(f"\n❌ ERRORE: Nessun file FITS ritagliato di Hubble trovato per il mosaico.")
        return False
    
    try:
        # Usa il primo file ritagliato come template per header e shape
        with fits.open(all_files[0]) as hdul:
            template_header = hdul[0].header.copy()
            data_shape = hdul[0].data.shape
            shape = data_shape if len(data_shape) == 2 else data_shape[1:]
    except Exception as e:
        logger.error(f"Errore lettura file template: {e}")
        return False
        
    # Inizializza le matrici di stacking
    total_flux = np.zeros(shape, dtype=np.float64)
    n_pixels = np.zeros(shape, dtype=np.int32)
    
    logger.info(f"Combinazione {len(all_files)} immagini di dimensione {shape}...")
    print("\n🔄 Combinazione immagini...")
    
    for filepath in tqdm(all_files, desc="Stacking Hubble", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                d = hdul[0].data
                if len(d.shape) == 3: d = d[0]
                
                if d.shape != shape: continue
                
                valid = ~np.isnan(d) & (d != 0) 
                d_clean = np.nan_to_num(d, nan=0.0)
                
                total_flux += d_clean
                n_pixels[valid] += 1
        except Exception as e: 
            logger.warning(f"Skippato file {filepath.name} durante lo stacking: {e}")
            continue
            
    # Calcolo della media
    logger.info("Calcolo della media aritmetica.")
    print("\n🧮 Calcolo media...")
    
    mosaic_data = np.full(shape, np.nan, dtype=np.float32)
    valid_stack = n_pixels > 0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mosaic_data[valid_stack] = (total_flux[valid_stack] / n_pixels[valid_stack]).astype(np.float32)
    
    # Salvataggio
    mosaic_out = base_dir / '5_mosaics'
    mosaic_out.mkdir(parents=True, exist_ok=True)
    
    # Nome del file modificato per riflettere l'origine dei dati (SOLO HUBBLE)
    final_path = mosaic_out / 'final_mosaic_hubble.fits'
    
    template_header['HISTORY'] = 'Mosaic created by cropping and stacking (average) - ONLY HUBBLE'
    fits.PrimaryHDU(data=mosaic_data, header=template_header).writeto(final_path, overwrite=True)
    
    logger.info(f"Mosaico salvato con successo: {final_path.name}")
    print(f"\n✅ MOSAICO SALVATO: {final_path.name}")
    return True

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger = setup_logging()
    
    # GESTIONE INPUT AUTOMATIZZATA (da script precedente) o MANUALE
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1]).resolve()
        if input_path.exists():
            print(f"\n🤖 Modalità Automatica: Target ricevuto {input_path.name}")
            target_dirs = [input_path]
        else:
            logger.error(f"Path fornito non valido: {input_path}")
            return
    else:
        target_dirs = select_target_directory(logger)
        if not target_dirs: return

    logger.info(f"Inizio batch su {len(target_dirs)} target")
    print("\n" + "="*70)
    print("PIPELINE: RITAGLIO + MOSAICO (SOLO HUBBLE)".center(70))
    print("="*70)
    
    start_time_total = time.time()
    successful_targets = []
    
    for base_dir in target_dirs:
        print("\n" + "─"*70)
        logger.info(f"Processing target: {base_dir.name}")
        
        # STEP 3: Ritaglio (Crop) - SOLO HUBBLE
        if not crop_all_images_for_target(base_dir, logger): 
            logger.error(f"STEP 3 (Ritaglio Hubble) fallito per {base_dir.name}")
            continue
            
        # STEP 4: Mosaico (Stacking) - SOLO HUBBLE
        if not create_mosaic_for_target(base_dir, logger): 
            logger.error(f"STEP 4 (Mosaico Hubble) fallito per {base_dir.name}")
            continue
            
        successful_targets.append(base_dir)

    elapsed_total = time.time() - start_time_total
    
    print("\n" + "="*70)
    print("✅ ELABORAZIONE BATCH COMPLETATA")
    print(f"⏱️ Tempo totale: {elapsed_total:.2f}s")

    if not successful_targets:
        print("\n❌ Nessun target completato con successo.")
        return

    # Chiedi di proseguire al prossimo script (Step 5: Analisi Patch)
    if ask_continue_to_next_step():
        try:
            next_script_name = 'Dataset_step3_analizzapatch.py'
            next_script = SCRIPTS_DIR / next_script_name
            if next_script.exists():
                for base_dir in successful_targets:
                    print(f"\n🚀 Avvio Step 5 ({next_script_name}) per {base_dir.name}...")
                    subprocess.run([sys.executable, str(next_script), str(base_dir.resolve())], check=True)
            else:
                print(f"⚠️  Script mancante: {next_script_name} non trovato in {SCRIPTS_DIR}")
        except Exception as e:
            logger.error(f"Errore avvio script successivo: {e}")
            print(f"❌ Errore avvio script successivo: {e}")

if __name__ == "__main__":
    main()