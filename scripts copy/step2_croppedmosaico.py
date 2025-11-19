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

def create_mosaic(input_folder, output_folder, target_name):
    """
    Crea un mosaico finale combinando tutte le immagini ritagliate.
    FIX: Combinazione corretta preservando dynamic range astronomico.
    FIX: Path corretto per leggere i file ritagliati.
    FIX: Aggiunto timestamp al nome file.
    """
    print(f"\n🖼️  {'🖼️  '*20}")
    print(f"                             MOSAICO: {target_name}")
    print(f"🖼️  {'🖼️  '*20}\n")
    
    # ✅ FIX: Path corretto per i file ritagliati (4_cropped)
    hubble_folder = input_folder / 'hubble'
    obs_folder = input_folder / 'observatory'
    
    # Raccogli tutti i file
    all_files = []
    if hubble_folder.exists():
        hubble_fits = list(hubble_folder.glob('*.fits'))
        all_files.extend(sorted(hubble_fits))
        print(f"   📂 Hubble: {len(hubble_fits)} file")
    if obs_folder.exists():
        obs_fits = list(obs_folder.glob('*.fits'))
        all_files.extend(sorted(obs_fits))
        print(f"   📂 Observatory: {len(obs_fits)} file")
    
    if not all_files:
        print(f"❌ Nessun file FITS in {input_folder}")
        print(f"   Verificare che esistano file in:")
        print(f"   - {hubble_folder}")
        print(f"   - {obs_folder}")
        return None
    
    print(f"\n✅ Trovati {len(all_files)} file FITS da combinare.")
    
    # Leggi dimensioni dal primo file
    with fits.open(all_files[0]) as hdul:
        reference_shape = hdul[0].data.shape
        reference_header = hdul[0].header.copy()
    
    print(f"   Dimensioni mosaico: {reference_shape[1]} x {reference_shape[0]} pixel")
    
    # ═══════════════════════════════════════════════════════════════
    # FIX: Combinazione manuale preservando dynamic range
    # ═══════════════════════════════════════════════════════════════
    
    print("\n🔄 Combinazione immagini in corso...")
    
    # Array per accumulare somma e conteggio
    sum_array = np.zeros(reference_shape, dtype=np.float64)
    count_array = np.zeros(reference_shape, dtype=np.uint16)
    
    for i, filepath in enumerate(tqdm(all_files, desc="Combinazione", unit="file")):
        try:
            with fits.open(filepath) as hdul:
                data = hdul[0].data.astype(np.float64)
                
                # Maschera valori validi (no NaN, no Inf, no zero)
                valid_mask = np.isfinite(data) & (data != 0)
                
                # Accumula somma solo dove valido
                sum_array[valid_mask] += data[valid_mask]
                count_array[valid_mask] += 1
                
        except Exception as e:
            print(f"\n⚠️  Errore lettura {filepath.name}: {e}")
            continue
    
    # Calcola media preservando dynamic range
    print("\n🧮 Calcolo della media finale...")
    
    # Evita divisione per zero
    count_array[count_array == 0] = 1
    
    # Media finale
    mosaic_data = sum_array / count_array.astype(np.float64)
    
    # Statistiche finali
    valid_final = np.isfinite(mosaic_data) & (mosaic_data != 0)
    if valid_final.any():
        print(f"   📊 Range finale: [{mosaic_data[valid_final].min():.3e}, {mosaic_data[valid_final].max():.3e}]")
        print(f"   📊 Median: {np.median(mosaic_data[valid_final]):.3e}")
        print(f"   📊 Pixel validi: {valid_final.sum():,} / {mosaic_data.size:,} ({valid_final.sum()/mosaic_data.size*100:.1f}%)")
    
    # ═══════════════════════════════════════════════════════════════
    # Salvataggio con header WCS originale + TIMESTAMP
    # ═══════════════════════════════════════════════════════════════
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # ✅ FIX: Aggiungi timestamp al nome file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'mosaic_{target_name}_{timestamp}.fits'
    output_path = output_folder / output_filename
    
    print(f"\n💾 Salvataggio mosaico in {output_filename}...")
    
    # Crea HDU con tipo float32 (preserva dynamic range, più efficiente di float64)
    hdu = fits.PrimaryHDU(data=mosaic_data.astype(np.float32), header=reference_header)
    
    # Aggiungi metadati combinazione
    hdu.header['NCOMBINE'] = (len(all_files), 'Number of images combined')
    hdu.header['COMBMETH'] = ('MEAN', 'Combination method')
    hdu.header['CMBDATE'] = (timestamp, 'Mosaic creation timestamp')
    hdu.header['TARGET'] = (target_name, 'Target name')
    hdu.header['HISTORY'] = f'Combined {len(all_files)} images on {datetime.now().isoformat()}'
    
    hdu.writeto(output_path, overwrite=True)
    
    print(f"\n✅ MOSAICO COMPLETATO: {output_path}")
    
    return output_path


# ============================================================================
# MAIN - FIX: Passa il path corretto a create_mosaic()
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
        
        # ✅ FIX: Passa il path corretto (4_cropped)
        cropped_dir = base_dir / '4_cropped'
        mosaic_output_dir = base_dir / '5_mosaics'
        
        # STEP 2: Mosaico
        mosaic_path = create_mosaic(cropped_dir, mosaic_output_dir, base_dir.name)
        
        if not mosaic_path:
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