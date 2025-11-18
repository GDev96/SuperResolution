"""
STEP 3+4: RITAGLIO IMMAGINI REGISTRATE E CREAZIONE MOSAICO
1. Ritaglia tutte le immagini registrate alle dimensioni dell'immagine più piccola
2. Crea un mosaico (media) da tutte le immagini ritagliate
"""

import os
import glob
import time
from pathlib import Path
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import warnings
import sys
import subprocess

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE PATH ASSOLUTI
# ============================================================================
# Definizione della radice del progetto
PROJECT_ROOT = Path(r"F:\Super Revolt Gaia\SuperResolution")

# Percorsi assoluti derivati
ROOT_DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "finale"
# ============================================================================

# ============================================================================
# NUOVA FUNZIONE: SELEZIONE CARTELLA TARGET
# ============================================================================

def select_target_directory():
    """
    Mostra un menu per selezionare una o TUTTE le cartelle target.
    """
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET".center(70))
    print("📂"*35)
    print(f"\nScansione sottocartelle in: {ROOT_DATA_DIR}")

    try:
        # Usa il percorso assoluto definito in alto
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir()]
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile leggere la cartella {ROOT_DATA_DIR}")
        print(f"   Dettagli: {e}")
        return []

    if not subdirs:
        print(f"\n❌ ERRORE: Nessuna sottocartella trovata in {ROOT_DATA_DIR}")
        print("   Assicurati di aver creato cartelle come 'M33', 'M42', ecc.")
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
                print("👋 Uscita.")
                return []

            choice = int(choice_str)

            if choice == 0:
                print(f"\n✅ Selezionati TUTTI i {len(subdirs)} target.")
                return subdirs
            
            choice_idx = choice - 1
            if 0 <= choice_idx < len(subdirs):
                selected_dir = subdirs[choice_idx]
                print(f"\n✅ Cartella selezionata: {selected_dir.name}")
                print(f"   Percorso completo: {selected_dir}")
                return [selected_dir]
            else:
                print(f"❌ Scelta non valida. Inserisci un numero tra 0 e {len(subdirs)}.")
        except ValueError:
            print("❌ Input non valido. Inserisci un numero.")
        except Exception as e:
            print(f"❌ Errore: {e}")
            return []

# ============================================================================
# FUNZIONI - RITAGLIO (Invariate)
# ============================================================================

def find_smallest_dimensions(all_files):
    print("\n🔍 Ricerca dimensioni minime...")
    min_height = float('inf')
    min_width = float('inf')
    smallest_file = None
    
    for filepath in tqdm(all_files, desc="Scansione", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                data_hdu = None
                for hdu in hdul:
                    if hdu.data is not None and len(hdu.data.shape) >= 2:
                        data_hdu = hdu
                        break
                if data_hdu is None: continue
                if len(data_hdu.data.shape) == 3:
                    shape = data_hdu.data.shape[1:3]
                else:
                    shape = data_hdu.data.shape
                height, width = shape
                if height < min_height or width < min_width:
                    if height < min_height: min_height = height
                    if width < min_width: min_width = width
                    smallest_file = filepath
        except Exception as e:
            print(f"\n⚠️  ATTENZIONE: Impossibile leggere {filepath}: {e}")
            continue
    
    if smallest_file is None:
        print("\n❌ ERRORE: Impossibile determinare le dimensioni minime. Nessun file valido.")
        return None, None
    print(f"\n✅ Dimensioni minime trovate: {min_width} x {min_height} pixel (da {Path(smallest_file).name})")
    return min_height, min_width

def crop_image(input_path, output_path, target_height, target_width):
    try:
        with fits.open(input_path) as hdul:
            data_hdu = None
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    data_hdu = hdu
                    break
            if data_hdu is None: return False
            data = data_hdu.data
            header = data_hdu.header
            if len(data.shape) == 3:
                data = data[0]
            current_height, current_width = data.shape
            y_offset = (current_height - target_height) // 2
            x_offset = (current_width - target_width) // 2
            cropped_data = data[y_offset:y_offset + target_height, x_offset:x_offset + target_width]
            if 'CRPIX1' in header: header['CRPIX1'] -= x_offset
            if 'CRPIX2' in header: header['CRPIX2'] -= y_offset
            header['NAXIS1'] = target_width
            header['NAXIS2'] = target_height
            if 'NAXIS3' in header: del header['NAXIS3']
            header['NAXIS'] = 2
            header['HISTORY'] = 'Cropped by step2_croppedmosaico.py'
            fits.PrimaryHDU(data=cropped_data, header=header).writeto(output_path, overwrite=True)
            return True
    except Exception as e:
        print(f"\n❌ ERRORE nel ritaglio di {input_path.name}: {e}")
        return False

def crop_all_images(INPUT_DIRS, OUTPUT_DIRS):
    print("\n" + "✂️ "*35)
    print("STEP 3: RITAGLIO IMMAGINI REGISTRATE".center(70))
    print("✂️ "*35)
    
    for output_dir in OUTPUT_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Input: {INPUT_DIRS['hubble'].parent}")
    print(f"📂 Output: {OUTPUT_DIRS['hubble'].parent}")
    
    all_files = []
    file_mapping = {}
    
    for category, input_dir in INPUT_DIRS.items():
        if not input_dir.exists():
            print(f"\n⚠️  ATTENZIONE: Directory input non trovata: {input_dir}")
            continue
        files = glob.glob(str(input_dir / '*.fits')) + glob.glob(str(input_dir / '*.fit'))
        for f in files:
            all_files.append(f)
            file_mapping[f] = category
        print(f"\n   {category}: {len(files)} file trovati")
    
    if not all_files:
        print(f"\n❌ ERRORE: Nessun file FITS trovato nelle cartelle di input.")
        return False
    
    min_height, min_width = find_smallest_dimensions(all_files)
    if min_height is None or min_width is None:
        print(f"\n❌ ERRORE: Impossibile procedere.")
        return False

    print(f"\n✂️  Ritaglio in corso a {min_width} x {min_height} pixel...\n")
    success_count, failed_count = 0, 0
    
    for filepath in tqdm(all_files, desc="Ritaglio", unit="file"):
        category = file_mapping[filepath]
        filename = Path(filepath).name
        output_path = OUTPUT_DIRS[category] / filename
        if crop_image(Path(filepath), output_path, min_height, min_width):
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n{'='*70}\n✅ RITAGLIO COMPLETATO!\n{'='*70}")
    print(f"   Immagini ritagliate: {success_count} (Errori: {failed_count})")
    print(f"   Dimensioni finali: {min_width} x {min_height} pixel")
    return success_count > 0

# ============================================================================
# FUNZIONI - MOSAICO (Invariate)
# ============================================================================

def create_mosaic(INPUT_DIRS_CROPPED, MOSAIC_OUTPUT_FILE, MOSAIC_OUTPUT_DIR):
    print("\n" + "🖼️ "*35)
    print("STEP 4: CREAZIONE MOSAICO DA IMMAGINI RITAGLIATE".center(70))
    print("🖼️ "*35)
    
    MOSAIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIRS = [INPUT_DIRS_CROPPED['hubble'], INPUT_DIRS_CROPPED['observatory']]
    print(f"\n📂 Input: {INPUT_DIRS_CROPPED['hubble'].parent}")
    print(f"📂 Output: {MOSAIC_OUTPUT_FILE}")
    
    all_files = []
    for d in INPUT_DIRS:
        if not d.exists(): continue
        all_files.extend(glob.glob(str(d / '*.fits')))
        all_files.extend(glob.glob(str(d / '*.fit')))
        
    if not all_files:
        print(f"\n❌ ERRORE: Nessun file FITS ritagliato trovato.")
        return False
    print(f"\n✅ Trovati {len(all_files)} file FITS da combinare.")
    
    try:
        with fits.open(all_files[0]) as hdul:
            template_header = hdul[0].header
            shape = hdul[0].data.shape
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile leggere il primo file {all_files[0]}: {e}")
        return False
        
    print(f"   Dimensioni mosaico: {shape[1]} x {shape[0]} pixel")
    
    total_flux = np.zeros(shape, dtype=np.float64)
    n_pixels = np.zeros(shape, dtype=np.int32)
    
    print("\n🔄 Combinazione immagini in corso...")
    for filepath in tqdm(all_files, desc="Combinazione", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                img_data = hdul[0].data
                if img_data.shape != shape:
                    print(f"\n⚠️  ATTENZIONE: {filepath} ha dimensioni errate. Saltato.")
                    continue
                valid_mask = ~np.isnan(img_data)
                img_data_no_nan = np.nan_to_num(img_data, nan=0.0, copy=False)
                total_flux += img_data_no_nan
                n_pixels[valid_mask] += 1
        except Exception as e:
            print(f"\n⚠️  ATTENZIONE: Errore nel leggere {filepath}: {e}. Saltato.")
            
    print("\n🧮 Calcolo della media finale...")
    mosaic_data = np.full(shape, np.nan, dtype=np.float32)
    valid_stack = n_pixels > 0
    mosaic_data[valid_stack] = (total_flux[valid_stack] / n_pixels[valid_stack]).astype(np.float32)
    
    print(f"\n💾 Salvataggio mosaico in {MOSAIC_OUTPUT_FILE}...")
    template_header['HISTORY'] = 'Mosaico creato da step2_croppedmosaico.py'
    template_header['NCOMBINE'] = (len(all_files), 'Numero di file combinati')
    
    try:
        fits.PrimaryHDU(data=mosaic_data, header=template_header).writeto(MOSAIC_OUTPUT_FILE, overwrite=True)
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile salvare il file FITS finale: {e}")
        return False

    print(f"\n{'='*70}\n✅ MOSAICO COMPLETATO!\n{'='*70}")
    print(f"   File salvato in: {MOSAIC_OUTPUT_FILE}")
    return True

# ============================================================================
# MENU DI PROSEGUIMENTO (MODIFICATO)
# ============================================================================

def ask_continue_to_mosaic():
    """Chiede all'utente se vuole proseguire con la creazione del mosaico."""
    print("\n" + "="*70); print("🎯 RITAGLIO COMPLETATO!"); print("="*70)
    print("\n📋 OPZIONI:\n   1️⃣  Continua con creazione Mosaico\n   2️⃣  Salta Mosaico (passa a Step 5+6)")
    while True:
        print("\n" + "─"*70)
        choice = input("👉 Vuoi creare il Mosaico ora? [S/n, default=S]: ").strip().lower()
        if choice in ('', 's', 'si', 'y', 'yes'):
            print("\n✅ Proseguimento con creazione Mosaico...")
            return True
        elif choice in ('n', 'no'):
            print("\n✅ Mosaico saltato.")
            return False
        else:
            print("❌ Scelta non valida. Inserisci S per Sì o N per No.")

def ask_continue_to_step3(target_list):
    """
    Chiede all'utente se vuole proseguire con Step 3 (Analisi Patches).
    """
    if not target_list:
        return False
        
    print("\n" + "="*70); print("🎯 FASI PRECEDENTI COMPLETATE"); print("="*70)
    print("\n📋 OPZIONI PROSSIMO STEP:\n   1️⃣  Continua con Step 5+6 (Analisi e Patches)\n   2️⃣  Termina qui")
    
    if len(target_list) > 1:
        prompt_msg = f"per i {len(target_list)} target processati?"
    else:
        prompt_msg = f"per {target_list[0].name}?"

    while True:
        print("\n" + "─"*70)
        choice = input(f"👉 Vuoi continuare con 'step3_analizzapatch.py' {prompt_msg} [S/n, default=S]: ").strip().lower()
        
        if choice in ('', 's', 'si', 'y', 'yes'):
            print("\n✅ Avvio Step 5+6 (step3_analizzapatch.py)...")
            return True
        elif choice in ('n', 'no'):
            print("\n✅ Pipeline completata")
            return False
        else:
            print("❌ Scelta non valida. Inserisci S per Sì o N per No.")

# ============================================================================
# MAIN (MODIFICATA PER LOOP)
# ============================================================================

def main():
    """Funzione principale che esegue ritaglio e creazione mosaico."""
    
    # Gestione lista target
    target_dirs = []
    if len(sys.argv) > 1:
        # Se avviato da step1, prende UN SOLO BASE_DIR dall'argomento
        target_dirs = [Path(sys.argv[1])]
        print(f"🚀 Avviato da script precedente. Target: {target_dirs[0].name}")
    else:
        # Se avviato da solo, mostra il menu
        target_dirs = select_target_directory()
    
    if not target_dirs:
        print("Nessun target selezionato. Uscita.")
        return

    print("\n" + "="*70)
    print("PIPELINE: RITAGLIO IMMAGINI + CREAZIONE MOSAICO".center(70))
    if len(target_dirs) > 1:
        print(f"Modalità Batch: {len(target_dirs)} target".center(70))
    print("="*70)
    
    start_time_total = time.time()
    successful_targets_to_pass = []

    for BASE_DIR in target_dirs:
        print("\n" + "🚀"*35)
        print(f"INIZIO ELABORAZIONE TARGET: {BASE_DIR.name}".center(70))
        print("🚀"*35)

        # Definizione percorsi per QUESTO target
        INPUT_DIRS = {
            'hubble': BASE_DIR / '3_registered_native' / 'hubble',
            'observatory': BASE_DIR / '3_registered_native' / 'observatory'
        }
        OUTPUT_DIR_BASE = BASE_DIR / '4_cropped'
        OUTPUT_DIRS = {
            'hubble': OUTPUT_DIR_BASE / 'hubble',
            'observatory': OUTPUT_DIR_BASE / 'observatory'
        }
        MOSAIC_OUTPUT_DIR = BASE_DIR / '5_mosaics'
        MOSAIC_OUTPUT_FILE = MOSAIC_OUTPUT_DIR / 'final_mosaic.fits'
        
        start_time_target = time.time()
        
        # STEP 1: Ritaglio
        crop_success = crop_all_images(INPUT_DIRS, OUTPUT_DIRS)
        
        if not crop_success:
            print(f"\n❌ Pipeline interrotta per {BASE_DIR.name}: errore durante il ritaglio.")
            continue 
        
        crop_time = time.time() - start_time_target
        
        # STEP 2: Mosaico (opzionale)
        mosaic_success = False
        if ask_continue_to_mosaic():
            mosaic_start = time.time()
            mosaic_success = create_mosaic(OUTPUT_DIRS, MOSAIC_OUTPUT_FILE, MOSAIC_OUTPUT_DIR)
            mosaic_time = time.time() - mosaic_start
            if mosaic_success:
                print(f"\n⏱️  Tempo mosaico: {mosaic_time:.1f} secondi")

        successful_targets_to_pass.append(BASE_DIR)
        
        elapsed_target = time.time() - start_time_target
        print(f"\n⏱️  Tempo totale per {BASE_DIR.name}: {elapsed_target:.1f} secondi")
        
    elapsed_total = time.time() - start_time_total
    print("\n" + "="*70)
    print("📊 RIEPILOGO BATCH (Step 3+4)")
    print("="*70)
    print(f"   Target totali elaborati: {len(target_dirs)}")
    print(f"   Target completati (crop): {len(successful_targets_to_pass)}")
    print(f"   ⏱️ Tempo totale batch: {elapsed_total:.1f} secondi")

    if not successful_targets_to_pass:
        print("\nNessun target completato. Uscita.")
        return

    if ask_continue_to_step3(successful_targets_to_pass):
        try:
            # Usa il percorso assoluto definito in alto
            step3_script = SCRIPTS_DIR / 'step3_analizzapatch.py'
            
            if step3_script.exists():
                print(f"\n🚀 Avvio Step 5+6 in loop per {len(successful_targets_to_pass)} target...")
                for BASE_DIR in successful_targets_to_pass:
                    print(f"\n--- Avvio per {BASE_DIR.name} ---")
                    subprocess.run([sys.executable, str(step3_script), str(BASE_DIR)])
                    print(f"--- Completato {BASE_DIR.name} ---")
            else:
                print(f"\n⚠️  Script {step3_script.name} non trovato nella directory {SCRIPTS_DIR}")
        except Exception as e:
            print(f"\n⚠️  Impossibile avviare automaticamente {step3_script.name}: {e}")
    else:
        print("\n👋 Arrivederci!")

if __name__ == "__main__":
    main()