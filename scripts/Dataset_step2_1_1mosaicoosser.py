"""
STEP 3+4: RITAGLIO IMMAGINI REGISTRATE E CREAZIONE MOSAICO (SOLO OSSERVATORIO)

1. Ritaglia tutte le immagini registrate nella cartella observatory alle dimensioni dell'immagine più piccola.
2. Crea un mosaico (media) da tutte le immagini ritagliate di observatory.

INPUT: Esclusivamente la cartella '3_registered_native/observatory' (la cartella 'hubble' è ignorata).
OUTPUT: 
  - Cartella '4_cropped/observatory' con immagini ritagliate.
  - File '5_mosaics/final_mosaic_observatory.fits' con il mosaico finale.
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
    """Configura logging."""
    os.makedirs(LOG_DIR_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = LOG_DIR_ROOT / f'crop_mosaic_observatory_{timestamp}.log'
    
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
# FUNZIONI MENU E SELEZIONE
# ============================================================================

def select_target_directory():
    """Mostra un menu per selezionare una o TUTTE le cartelle target."""
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET".center(70))
    print("📂"*35)
    print(f"\nScansione sottocartelle in: {ROOT_DATA_DIR}")

    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
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
    
    next_script_name = 'Dataset_step3_analizzapatch.py'
    print(f"\n📋 Prossimo Step: Analisi Patch ('{next_script_name}')")
    
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
# FUNZIONI - RITAGLIO
# ============================================================================

def find_smallest_dimensions(all_files):
    """Trova le dimensioni dell'immagine più piccola tra tutti i file."""
    print("\n🔍 Ricerca dimensioni minime...")
    
    min_height = float('inf')
    min_width = float('inf')
    smallest_file = None
    
    for filepath in tqdm(all_files, desc="Scansione", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                data_shape = hdul[0].data.shape
                if len(data_shape) == 3:
                    height, width = data_shape[1], data_shape[2]
                else:
                    height, width = data_shape
                
                if height < min_height or width < min_width:
                    if height < min_height: min_height = height
                    if width < min_width: min_width = width
                    smallest_file = filepath
                    
        except Exception as e:
            print(f"\n⚠️  ATTENZIONE: Impossibile leggere {filepath}: {e}")
            continue
    
    print(f"\n✅ Dimensioni minime trovate: {min_width} x {min_height} pixel")
    if smallest_file:
        print(f"   File più piccolo: {Path(smallest_file).name}")
    return min_height, min_width


def crop_image(input_path, output_path, target_height, target_width):
    """Ritaglia un'immagine FITS alle dimensioni target (centrato)."""
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
                y_offset:y_offset + target_height,
                x_offset:x_offset + target_width
            ]
            
            if 'CRPIX1' in header: header['CRPIX1'] -= x_offset
            if 'CRPIX2' in header: header['CRPIX2'] -= y_offset
            header['NAXIS1'] = target_width
            header['NAXIS2'] = target_height
            
            fits.PrimaryHDU(data=cropped_data, header=header).writeto(
                output_path, overwrite=True
            )
            return True
    except Exception as e:
        print(f"\n❌ ERRORE nel ritaglio di {input_path.name}: {e}")
        return False


def crop_all_images_for_target(base_dir):
    """Esegue il ritaglio di tutte le immagini del SOLO OSSERVATORIO per un target specifico."""
    print("\n" + "✂️ "*35)
    print(f"RITAGLIO: {base_dir.name} (SOLO OSSERVATORIO)".center(70))
    print("✂️ "*35)
    
    # Path assoluto UNICO per l'input (Osservatorio)
    input_dir_observatory = base_dir / '3_registered_native' / 'observatory'
    
    # Path assoluto UNICO per l'output
    output_dir_base = base_dir / '4_cropped'
    output_dir_observatory = output_dir_base / 'observatory'
    
    # Crea la cartella di output (ignoriamo 'hubble')
    output_dir_observatory.mkdir(parents=True, exist_ok=True)
    
    # Cerca solo i file nell'osservatorio
    all_files = list(input_dir_observatory.glob('*.fits')) + list(input_dir_observatory.glob('*.fit'))
    
    print(f"   Osservatorio: {len(all_files)} file trovati")
    
    if not all_files:
        print(f"\n❌ ERRORE: Nessun file FITS in {input_dir_observatory}.")
        return False
    
    min_height, min_width = find_smallest_dimensions(all_files)
    print(f"\n📐 Target: {min_width} x {min_height} pixel")
    
    success_count = 0
    for filepath in tqdm(all_files, desc="Ritaglio Osservatorio", unit="file"):
        # La categoria è implicitamente 'observatory'
        output_filepath = output_dir_observatory / filepath.name
        
        # Logica per saltare i file già ritagliati (omessa per brevità, si basa su crop_image)
        
        if crop_image(filepath, output_filepath, min_height, min_width):
            success_count += 1
            
    return success_count > 0

# ============================================================================
# FUNZIONI - MOSAICO
# ============================================================================

def create_mosaic_for_target(base_dir):
    """Crea il mosaico usando SOLO le immagini ritagliate dell'Osservatorio."""
    print("\n" + "🖼️ "*35)
    print(f"MOSAICO: {base_dir.name} (SOLO OSSERVATORIO)".center(70))
    print("🖼️ "*35)
    
    output_dir_base = base_dir / '4_cropped'
    
    # Includiamo SOLO i file Osservatorio ritagliati
    all_files = list((output_dir_base / 'observatory').glob('*.fits'))
        
    if not all_files:
        print(f"\n❌ ERRORE: Nessun file FITS ritagliato dell'Osservatorio trovato per il mosaico.")
        return False
    
    try:
        # Usiamo il primo file ritagliato come template per header e shape
        with fits.open(all_files[0]) as hdul:
            template_header = hdul[0].header.copy()
            data_shape = hdul[0].data.shape
            shape = data_shape if len(data_shape) == 2 else data_shape[1:]
    except Exception as e:
        print(f"❌ Errore lettura file template: {e}")
        return False
        
    total_flux = np.zeros(shape, dtype=np.float64)
    n_pixels = np.zeros(shape, dtype=np.int32)
    
    print("\n🔄 Combinazione immagini...")
    for filepath in tqdm(all_files, desc="Stacking Osservatorio", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                d = hdul[0].data
                if len(d.shape) == 3: d = d[0]
                if d.shape != shape: continue
                
                valid = ~np.isnan(d) & (d != 0) 
                d_clean = np.nan_to_num(d, nan=0.0)
                total_flux += d_clean
                n_pixels[valid] += 1
        except: 
            continue
            
    print("\n🧮 Calcolo media...")
    mosaic_data = np.full(shape, np.nan, dtype=np.float32)
    valid_stack = n_pixels > 0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mosaic_data[valid_stack] = (total_flux[valid_stack] / n_pixels[valid_stack]).astype(np.float32)
    
    mosaic_out = base_dir / '5_mosaics'
    mosaic_out.mkdir(parents=True, exist_ok=True)
    
    # Nome del file appropriato
    final_path = mosaic_out / 'final_mosaic_observatory.fits'
    
    template_header['HISTORY'] = 'Mosaic created by pipeline - ONLY OBSERVATORY'
    fits.PrimaryHDU(data=mosaic_data, header=template_header).writeto(final_path, overwrite=True)
    
    print(f"\n✅ MOSAICO SALVATO: {final_path.name}")
    return True

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger = setup_logging()
    
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1]).resolve()
        if input_path.exists():
            print(f"\n🤖 Modalità Automatica: Target ricevuto {input_path.name}")
            target_dirs = [input_path]
        else:
            print(f"❌ Errore: Path fornito non valido: {input_path}")
            return
    else:
        target_dirs = select_target_directory()
        if not target_dirs: return

    logger.info(f"Inizio batch su {len(target_dirs)} target")
    
    print("\n" + "="*70)
    print("PIPELINE: RITAGLIO + MOSAICO (SOLO OSSERVATORIO)".center(70))
    print("="*70)
    
    successful_targets = []
    
    for base_dir in target_dirs:
        logger.info(f"Processing: {base_dir}")
        
        # Esegue Ritaglio (SOLO OSSERVATORIO)
        if not crop_all_images_for_target(base_dir): 
            logger.error(f"Ritaglio fallito per {base_dir}")
            continue
            
        # Esegue Mosaico (SOLO OSSERVATORIO)
        if not create_mosaic_for_target(base_dir): 
            logger.error(f"Mosaico fallito per {base_dir}")
            continue
            
        successful_targets.append(base_dir)

    if not successful_targets:
        print("\n❌ Nessun target completato.")
        return

    # Chiedi di proseguire al prossimo script (Step 5)
    if ask_continue_to_next_step():
        try:
            next_script = SCRIPTS_DIR / 'Dataset_step3_analizzapatch.py'
            if next_script.exists():
                for base_dir in successful_targets:
                    print(f"\n🚀 Avvio Step 5 (Analisi Patch) per {base_dir.name}...")
                    subprocess.run([sys.executable, str(next_script), str(base_dir.resolve())])
            else:
                print(f"⚠️  Script mancante: {next_script}")
        except Exception as e:
            print(f"❌ Errore avvio script successivo: {e}")

if __name__ == "__main__":
    main()