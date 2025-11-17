"""
STEP 3+4: RITAGLIO IMMAGINI REGISTRATE E CREAZIONE MOSAICO
1. Ritaglia tutte le immagini registrate alle dimensioni dell'immagine più piccola
2. Crea un mosaico (media) da tutte le immagini ritagliate
CON MENU DI PROSEGUIMENTO TRA RITAGLIO E MOSAICO

MODIFICATO:
- Aggiunto menu per avviare step3_analizzapatch.py (passando BASE_DIR)
- Aggiunto menu iniziale per selezionare la cartella del target (se avviato da solo).
- Accetta BASE_DIR come argomento da riga di comando (se avviato da step1).
- Tutti i percorsi sono ora relativi alla cartella del target selezionata (BASE_DIR).
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
import subprocess # Aggiunto per coerenza

warnings.filterwarnings('ignore')

# ============================================================================
# NUOVA FUNZIONE: SELEZIONE CARTELLA TARGET (DA STEP 1)
# ============================================================================

# Percorso radice da cui cercare le cartelle dei target
# Deve essere lo stesso di step1_wcsregister.py
ROOT_DATA_DIR = Path(r'F:\Super Revolt Gaia\SuperResolution\data')

def select_target_directory():
    """
    Mostra un menu per selezionare la cartella del target da cui derivare BASE_DIR.
    """
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET".center(70))
    print("📂"*35)
    print(f"\nScansione sottocartelle in: {ROOT_DATA_DIR}")

    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir()]
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile leggere la cartella {ROOT_DATA_DIR}")
        print(f"   Dettagli: {e}")
        return None

    if not subdirs:
        print(f"\n❌ ERRORE: Nessuna sottocartella trovata in {ROOT_DATA_DIR}")
        print("   Assicurati di aver creato cartelle come 'M33', 'M42', ecc.")
        return None

    print("\nCartelle target disponibili:")
    for i, dir_path in enumerate(subdirs):
        print(f"   {i+1}: {dir_path.name}")

    while True:
        print("\n" + "─"*70)
        try:
            choice_str = input(f"👉 Seleziona un numero (1-{len(subdirs)}) o 'q' per uscire: ").strip()

            if choice_str.lower() == 'q':
                print("👋 Uscita.")
                return None

            choice = int(choice_str) - 1
            if 0 <= choice < len(subdirs):
                selected_dir = subdirs[choice]
                print(f"\n✅ Cartella selezionata: {selected_dir.name}")
                print(f"   Percorso completo: {selected_dir}")
                return selected_dir  # Ritorna l'oggetto Path
            else:
                print(f"❌ Scelta non valida. Inserisci un numero tra 1 e {len(subdirs)}.")
        except ValueError:
            print("❌ Input non valido. Inserisci un numero.")
        except Exception as e:
            print(f"❌ Errore: {e}")
            return None

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
                # Cerca l'HDU con i dati
                data_hdu = None
                for hdu in hdul:
                    if hdu.data is not None and len(hdu.data.shape) >= 2:
                        data_hdu = hdu
                        break
                
                if data_hdu is None:
                    print(f"\n⚠️  ATTENZIONE: Nessun dato immagine trovato in {filepath}. Saltato.")
                    continue

                # Gestisce dati 3D (es. cubi) prendendo il primo slice 2D
                if len(data_hdu.data.shape) == 3:
                    shape = data_hdu.data.shape[1:3] # Assumendo (Z, Y, X)
                else:
                    shape = data_hdu.data.shape # (Y, X)

                height, width = shape
                
                if height < min_height or width < min_width:
                    if height < min_height:
                        min_height = height
                    if width < min_width:
                        min_width = width
                    smallest_file = filepath
                    
        except Exception as e:
            print(f"\n⚠️  ATTENZIONE: Impossibile leggere {filepath}: {e}")
            continue
    
    if smallest_file is None:
        print("\n❌ ERRORE: Impossibile determinare le dimensioni minime. Nessun file valido.")
        return None, None

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
            # Trova l'HDU con i dati
            data_hdu = None
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    data_hdu = hdu
                    break
            
            if data_hdu is None:
                print(f"\n❌ ERRORE: Nessun dato immagine in {input_path.name}")
                return False

            data = data_hdu.data
            header = data_hdu.header
            
            # Gestisce dati 3D (es. cubi) prendendo il primo slice 2D
            if len(data.shape) == 3:
                data = data[0] # Prende il primo piano
                print(f"\n⚠️  ATTENZIONE: {input_path.name} è 3D, ritagliato primo piano.")
            
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
            if 'NAXIS3' in header: # Rimuovi asse 3 se esisteva
                del header['NAXIS3']
            header['NAXIS'] = 2
            
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


def crop_all_images(INPUT_DIRS, OUTPUT_DIRS):
    """
    Esegue il ritaglio di tutte le immagini.
    Args:
        INPUT_DIRS (dict): Dizionario con i percorsi di input
        OUTPUT_DIRS (dict): Dizionario con i percorsi di output
    """
    print("\n" + "✂️ "*35)
    print("STEP 3: RITAGLIO IMMAGINI REGISTRATE".center(70))
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
        if not input_dir.exists():
            print(f"\n⚠️  ATTENZIONE: Directory di input non trovata: {input_dir}")
            print(f"   La categoria '{category}' verrà saltata.")
            continue
            
        files = glob.glob(str(input_dir / '*.fits'))
        files.extend(glob.glob(str(input_dir / '*.fit')))
        
        for f in files:
            all_files.append(f)
            file_mapping[f] = category
        
        print(f"\n   {category}: {len(files)} file trovati")
    
    if not all_files:
        print(f"\n❌ ERRORE: Nessun file FITS trovato nelle cartelle di input.")
        print("   Assicurati di aver eseguito 'step1_wcsregister.py' prima.")
        return False
    
    print(f"\n✅ Totale: {len(all_files)} file da ritagliare")
    
    # 2. Trova le dimensioni minime
    min_height, min_width = find_smallest_dimensions(all_files)
    
    if min_height is None or min_width is None:
        print(f"\n❌ ERRORE: Impossibile procedere senza dimensioni valide.")
        return False

    # Conferma dimensioni
    print(f"\n🔍 Le immagini verranno ritagliate a: {min_width} x {min_height} pixel")
    
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

def create_mosaic(INPUT_DIRS_CROPPED, MOSAIC_OUTPUT_FILE, MOSAIC_OUTPUT_DIR):
    """
    Crea il mosaico.
    Args:
        INPUT_DIRS_CROPPED (dict): Cartelle input (output del ritaglio)
        MOSAIC_OUTPUT_FILE (Path): File FITS di output
        MOSAIC_OUTPUT_DIR (Path): Cartella di output
    """
    print("\n" + "🖼️ "*35)
    print("STEP 4: CREAZIONE MOSAICO DA IMMAGINI RITAGLIATE".center(70))
    print("🖼️ "*35)
    
    # Crea cartella output
    MOSAIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Cartelle con immagini ritagliate
    INPUT_DIRS = [
        INPUT_DIRS_CROPPED['hubble'],
        INPUT_DIRS_CROPPED['observatory']
    ]
    
    print(f"\n📂 Cartelle di input:")
    for d in INPUT_DIRS:
        print(f"   - {d}")
    print(f"\n📂 File di output:")
    print(f"   - {MOSAIC_OUTPUT_FILE}")
    
    # 1. Trova tutti i file FITS ritagliati
    all_files = []
    for d in INPUT_DIRS:
        if not d.exists():
            print(f"\n⚠️  ATTENZIONE: Directory input mosaico non trovata: {d}")
            continue
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
    template_header['HISTORY'] = 'Mosaico creato da step2_croppedmosaico.py'
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
# MENU DI PROSEGUIMENTO
# ============================================================================

def ask_continue_to_mosaic():
    """Chiede all'utente se vuole proseguire con la creazione del mosaico."""
    print("\n" + "="*70)
    print("🎯 RITAGLIO COMPLETATO!")
    print("="*70)
    
    print("\n📋 OPZIONI:")
    print("   1️⃣  Continua con creazione Mosaico")
    print("   2️⃣  Salta Mosaico (passa a Step 5+6)")
    
    while True:
        print("\n" + "─"*70)
        choice = input("👉 Vuoi creare il Mosaico ora? [S/n, default=S]: ").strip().lower()
        
        if choice == '' or choice == 's' or choice == 'si' or choice == 'y' or choice == 'yes':
            print("\n✅ Proseguimento con creazione Mosaico...")
            return True
        elif choice == 'n' or choice == 'no':
            print("\n✅ Mosaico saltato.")
            return False
        else:
            print("❌ Scelta non valida. Inserisci S per Sì o N per No.")


def ask_continue_to_step3(BASE_DIR):
    """
    Chiede all'utente se vuole proseguire con Step 3 (Analisi Patches).
    Args:
        BASE_DIR (Path): Il percorso base del target da passare allo script successivo.
    """
    print("\n" + "="*70)
    print("🎯 FASI PRECEDENTI COMPLETATE")
    print("="*70)
    
    print("\n📋 OPZIONI PROSSIMO STEP:")
    print("   1️⃣  Continua con Step 5+6 (Analisi e Patches)")
    print("   2️⃣  Termina qui")
    
    while True:
        print("\n" + "─"*70)
        choice = input("👉 Vuoi continuare con 'step3_analizzapatch.py'? [S/n, default=S]: ").strip().lower()
        
        if choice == '' or choice == 's' or choice == 'si' or choice == 'y' or choice == 'yes':
            print("\n✅ Avvio Step 5+6 (step3_analizzapatch.py)...")
            return True
        elif choice == 'n' or choice == 'no':
            print("\n✅ Pipeline completata")
            print("   Per eseguire Step 5+6 in seguito, esegui 'step3_analizzapatch.py'")
            return False
        else:
            print("❌ Scelta non valida. Inserisci S per Sì o N per No.")


# ============================================================================
# MAIN (MODIFICATA)
# ============================================================================

def main():
    """Funzione principale che esegue ritaglio e creazione mosaico."""
    
    BASE_DIR = None
    
    # --- NUOVO: GESTIONE BASE_DIR ---
    if len(sys.argv) > 1:
        # Se avviato da step1, prende BASE_DIR dall'argomento
        BASE_DIR = Path(sys.argv[1])
        print(f"🚀 Avviato da script precedente. Target: {BASE_DIR.name}")
    else:
        # Se avviato da solo, mostra il menu
        BASE_DIR = select_target_directory()
    
    if BASE_DIR is None:
        print("Nessun target selezionato. Uscita.")
        return
    # --- FINE NUOVO ---

    print("\n" + "="*70)
    print("PIPELINE: RITAGLIO IMMAGINI + CREAZIONE MOSAICO".center(70))
    print(f"TARGET: {BASE_DIR.name}".center(70))
    print("="*70)
    
    # --- NUOVO: Definizione percorsi ---
    # Input: cartelle con immagini registrate da step1
    INPUT_DIRS = {
        'hubble': BASE_DIR / '3_registered_native' / 'hubble',
        'observatory': BASE_DIR / '3_registered_native' / 'observatory'
    }
    
    # Output: cartelle per le immagini ritagliate
    OUTPUT_DIR_BASE = BASE_DIR / '4_cropped'
    OUTPUT_DIRS = {
        'hubble': OUTPUT_DIR_BASE / 'hubble',
        'observatory': OUTPUT_DIR_BASE / 'observatory'
    }
    
    # Output: cartella e file per il mosaico finale
    MOSAIC_OUTPUT_DIR = BASE_DIR / '5_mosaics'
    MOSAIC_OUTPUT_FILE = MOSAIC_OUTPUT_DIR / 'final_mosaic.fits'
    # --- FINE NUOVO ---
    
    start_time = time.time()
    
    # STEP 1: Ritaglio delle immagini
    # Passa i percorsi come argomenti
    crop_success = crop_all_images(INPUT_DIRS, OUTPUT_DIRS)
    
    if not crop_success:
        print("\n❌ Pipeline interrotta: errore during il ritaglio.")
        return
    
    crop_time = time.time() - start_time
    print(f"\n⏱️  Tempo ritaglio: {crop_time:.1f} secondi")
    
    # Variabili per il riepilogo
    mosaic_success = False
    mosaic_time = 0.0

    # MENU: Chiedi se continuare con il mosaico
    if ask_continue_to_mosaic():
        print("\n\n")
        # STEP 2: Creazione del mosaico
        mosaic_start = time.time()
        # Passa i percorsi come argomenti
        mosaic_success = create_mosaic(OUTPUT_DIRS, MOSAIC_OUTPUT_FILE, MOSAIC_OUTPUT_DIR)
        mosaic_time = time.time() - mosaic_start
        
        if mosaic_success:
            print(f"\n⏱️  Tempo mosaico: {mosaic_time:.1f} secondi")
        else:
            print("\n❌ Errore during la creazione del mosaico.")
            # Non usciamo, continuiamo a chiedere per Step 3
    
    # Riepilogo finale
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    if crop_success:
        print("✅ FASE DI RITAGLIO (Step 3) COMPLETATA!")
    if mosaic_success:
        print("✅ FASE MOSAICO (Step 4) COMPLETATA!")
    else:
        print("⚠️  FASE MOSAICO (Step 4) SALTATA O FALLITA.")
    print("="*70)
    print(f"   ⏱️  Tempo ritaglio: {crop_time:.1f} secondi")
    if mosaic_success:
        print(f"   ⏱️  Tempo mosaico: {mosaic_time:.1f} secondi")
    print(f"   ⏱️  Tempo totale: {elapsed:.1f} secondi ({elapsed/60:.1f} minuti)")
    
    # MENU: Chiedi se avviare Step 3 (viene chiesto in ogni caso)
    # Passa BASE_DIR alla funzione
    if ask_continue_to_step3(BASE_DIR):
        # Prova ad avviare step3_analizzapatch.py
        try:
            script_dir = Path(__file__).parent
            step3_script = script_dir / 'step3_analizzapatch.py'
            
            if step3_script.exists():
                print(f"\n🚀 Avvio {step3_script.name}...")
                # Usa sys.executable per assicurare lo stesso interprete python
                # --- MODIFICA: Passa BASE_DIR come argomento ---
                subprocess.run([sys.executable, str(step3_script), str(BASE_DIR)])
            else:
                print(f"\n⚠️  Script {step3_script.name} non trovato nella directory {script_dir}")
                print(f"   Eseguilo manualmente quando pronto")
        except Exception as e:
            print(f"\n⚠️  Impossibile avviare automaticamente {step3_script.name}: {e}")
            print(f"   Eseguilo manualmente: python {step3_script.name}")
    else:
        print("\n👋 Arrivederci!")


if __name__ == "__main__":
    main()