"""
STEP 6: CREAZIONE MOSAICO FINALE
Crea un mosaico (media) da tutte le immagini ritagliate.

INPUT: Cartella 'img_cropped' (output di step5)
OUTPUT: Mosaico finale in 'mosaics/'
"""

import os
import glob
import time
from datetime import datetime
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

# Input: cartella con immagini ritagliate da step5
INPUT_DIR = os.path.join(BASE_DIR, 'img_cropped')

# Output: cartella per i mosaici
OUTPUT_DIR = os.path.join(BASE_DIR, 'mosaics')

# ============================================================================
# FUNZIONI MENU INTERATTIVO
# ============================================================================

def list_available_sources():
    """Lista le fonti disponibili nell'output di step5."""
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


def find_common_objects():
    """Trova oggetti presenti in più fonti."""
    sources = list_available_sources()
    
    if len(sources) < 2:
        return []
    
    # Dizionario: oggetto -> lista di fonti che lo contengono
    object_sources = {}
    
    for source in sources:
        objects = list_available_objects(source)
        for obj_name, img_count in objects:
            if obj_name not in object_sources:
                object_sources[obj_name] = []
            object_sources[obj_name].append((source, img_count))
    
    # Trova oggetti presenti in più fonti
    common_objects = []
    for obj_name, sources_list in object_sources.items():
        if len(sources_list) >= 2:
            total_images = sum(count for _, count in sources_list)
            common_objects.append((obj_name, sources_list, total_images))
    
    return sorted(common_objects, key=lambda x: x[0])


def interactive_menu():
    """Menu interattivo per selezionare fonte e oggetto."""
    print("\n" + "=" * 70)
    print("🖼️  SELEZIONE IMMAGINI PER MOSAICO".center(70))
    print("=" * 70)
    
    # === STEP 1: Selezione Modalità ===
    print("\n🎯 MODALITÀ DI CREAZIONE MOSAICO:")
    print("-" * 70)
    print("   1. Singola fonte (es. solo Hubble o solo Local)")
    print("   2. Multi-fonte (combina Hubble + Local per stesso oggetto)")
    print("-" * 70)
    
    while True:
        try:
            mode_choice = input("\n➤ Scegli modalità [1-2]: ").strip()
            
            if mode_choice == '1':
                return interactive_menu_single_source()
            elif mode_choice == '2':
                return interactive_menu_multi_source()
            else:
                print("⚠️  Inserisci 1 o 2")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Operazione annullata.")
            return None, None, None


def interactive_menu_single_source():
    """Menu per modalità singola fonte."""
    print("\n" + "=" * 70)
    print("📂 MODALITÀ: SINGOLA FONTE")
    print("=" * 70)
    
    sources = list_available_sources()
    
    if not sources:
        print("\n❌ Nessuna fonte trovata in data/img_cropped/")
        print("   Esegui prima: python scripts/step5_crop.py")
        return None, None, None
    
    print("\n📂 FONTI DISPONIBILI:")
    print("-" * 70)
    for i, source in enumerate(sources, 1):
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
            return None, None, None
    
    print(f"\n✓ Fonte selezionata: {selected_source}")
    
    # === Selezione Oggetto ===
    objects = list_available_objects(selected_source)
    
    if not objects:
        print(f"\n❌ Nessun oggetto con immagini FITS in {selected_source}/")
        return None, None, None
    
    print(f"\n🎯 OGGETTI DISPONIBILI ({selected_source}):")
    print("-" * 70)
    for i, (obj_name, img_count) in enumerate(objects, 1):
        print(f"   {i}. {obj_name:<20} ({img_count} immagini)")
    print(f"   {len(objects)+1}. TUTTI (crea mosaico per ogni oggetto)")
    print("-" * 70)
    
    # Input oggetto
    while True:
        try:
            choice = input(f"\n➤ Scegli oggetto [1-{len(objects)+1}]: ").strip()
            obj_idx = int(choice) - 1
            
            if obj_idx == len(objects):
                selected_object = None
                print(f"\n✓ Creerò mosaici per TUTTI gli oggetti di {selected_source}")
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
            return None, None, None
    
    # === RIEPILOGO ===
    print("\n" + "=" * 70)
    print("📋 RIEPILOGO SELEZIONE")
    print("=" * 70)
    print(f"   Modalità: Singola fonte")
    print(f"   Fonte: {selected_source}")
    
    if selected_object:
        print(f"   Oggetto: {selected_object}")
    else:
        print(f"   Oggetti: TUTTI ({len(objects)} oggetti)")
    
    print(f"   Output: {OUTPUT_DIR}/")
    print("=" * 70)
    
    # Conferma
    confirm = input("\n➤ Confermi e procedi? [S/n]: ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ Operazione annullata.")
        return None, None, None
    
    return 'single', selected_source, selected_object


def interactive_menu_multi_source():
    """Menu per modalità multi-fonte (Hubble + Local)."""
    print("\n" + "=" * 70)
    print("🔀 MODALITÀ: MULTI-FONTE (Hubble + Local)")
    print("=" * 70)
    
    common_objects = find_common_objects()
    
    if not common_objects:
        print("\n❌ Nessun oggetto trovato in più fonti!")
        print("   Per usare questa modalità, devi avere lo stesso oggetto")
        print("   sia in 'hubble' che in 'local' (o altre fonti).")
        return None, None, None
    
    print(f"\n🎯 OGGETTI DISPONIBILI IN PIÙ FONTI:")
    print("-" * 70)
    for i, (obj_name, sources_list, total_images) in enumerate(common_objects, 1):
        sources_str = ", ".join([f"{src}({cnt})" for src, cnt in sources_list])
        print(f"   {i}. {obj_name:<20} → {sources_str} = {total_images} tot")
    print(f"   {len(common_objects)+1}. TUTTI (crea mosaico combinato per ogni oggetto)")
    print("-" * 70)
    
    # Input oggetto
    while True:
        try:
            choice = input(f"\n➤ Scegli oggetto [1-{len(common_objects)+1}]: ").strip()
            obj_idx = int(choice) - 1
            
            if obj_idx == len(common_objects):
                selected_object = None
                print(f"\n✓ Creerò mosaici combinati per TUTTI gli oggetti comuni")
                break
            elif 0 <= obj_idx < len(common_objects):
                selected_object = common_objects[obj_idx][0]
                sources_list = common_objects[obj_idx][1]
                total_images = common_objects[obj_idx][2]
                
                sources_str = " + ".join([src for src, _ in sources_list])
                print(f"\n✓ Oggetto selezionato: {selected_object}")
                print(f"   Fonti da combinare: {sources_str} ({total_images} immagini totali)")
                break
            else:
                print(f"⚠️  Inserisci un numero tra 1 e {len(common_objects)+1}")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Operazione annullata.")
            return None, None, None
    
    # === RIEPILOGO ===
    print("\n" + "=" * 70)
    print("📋 RIEPILOGO SELEZIONE")
    print("=" * 70)
    print(f"   Modalità: Multi-fonte")
    
    if selected_object:
        print(f"   Oggetto: {selected_object}")
        obj_data = [x for x in common_objects if x[0] == selected_object][0]
        for src, cnt in obj_data[1]:
            print(f"   - {src}: {cnt} immagini")
    else:
        print(f"   Oggetti: TUTTI ({len(common_objects)} oggetti comuni)")
    
    print(f"   Output: {OUTPUT_DIR}/")
    print("=" * 70)
    
    # Conferma
    confirm = input("\n➤ Confermi e procedi? [S/n]: ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ Operazione annullata.")
        return None, None, None
    
    return 'multi', None, selected_object


# ============================================================================
# FUNZIONI CREAZIONE MOSAICO
# ============================================================================

def create_mosaic_for_object(source, obj_name):
    """
    Crea un mosaico per un singolo oggetto da una singola fonte.
    
    Args:
        source (str): Nome della fonte (es. 'hubble', 'local')
        obj_name (str): Nome dell'oggetto
        
    Returns:
        str: Percorso del file mosaico creato, o None se fallito
    """
    input_dir = os.path.join(INPUT_DIR, source, obj_name)
    
    print(f"\n{'─'*70}")
    print(f"🖼️  Creazione mosaico: {source}/{obj_name}")
    print(f"{'─'*70}")
    print(f"   Input: {input_dir}")
    
    # Trova tutti i file FITS
    all_files = glob.glob(os.path.join(input_dir, '*.fits'))
    all_files.extend(glob.glob(os.path.join(input_dir, '*.fit')))
    
    if not all_files:
        print(f"   ⚠️  Nessun file FITS trovato")
        return None
    
    print(f"   Trovati: {len(all_files)} file da combinare")
    
    # Leggi template dal primo file
    try:
        with fits.open(all_files[0]) as hdul:
            template_header = hdul[0].header.copy()
            shape = hdul[0].data.shape
            
        print(f"   Dimensioni: {shape[1]} x {shape[0]} pixel")
        
    except Exception as e:
        print(f"   ❌ Errore leggendo {os.path.basename(all_files[0])}: {e}")
        return None
    
    # Inizializza array per combinazione
    total_flux = np.zeros(shape, dtype=np.float64)
    n_pixels = np.zeros(shape, dtype=np.int32)
    
    # Combina immagini
    print("\n   🔄 Combinazione immagini...")
    
    for filepath in tqdm(all_files, desc="     Combinazione", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                img_data = hdul[0].data
                
                # Verifica dimensioni
                if img_data.shape != shape:
                    print(f"\n     ⚠️  {os.path.basename(filepath)}: dimensioni diverse. Saltato.")
                    continue
                
                # Trova pixel validi (non NaN e non zero)
                valid_mask = np.isfinite(img_data) & (img_data != 0)
                
                # Sostituisci NaN con 0
                img_data_clean = np.nan_to_num(img_data, nan=0.0, copy=False)
                
                # Aggiungi ai totali
                total_flux += img_data_clean
                n_pixels[valid_mask] += 1
                
        except Exception as e:
            print(f"\n     ⚠️  Errore con {os.path.basename(filepath)}: {e}")
            continue
    
    # Calcola media
    print("\n   🧮 Calcolo media...")
    
    mosaic_data = np.full(shape, np.nan, dtype=np.float32)
    valid_stack = n_pixels > 0
    
    if valid_stack.sum() == 0:
        print("   ❌ Nessun pixel valido nel mosaico!")
        return None
    
    mosaic_data[valid_stack] = (total_flux[valid_stack] / n_pixels[valid_stack]).astype(np.float32)
    
    # Statistiche
    coverage = (valid_stack.sum() / mosaic_data.size) * 100
    valid_data = mosaic_data[valid_stack]
    
    print(f"\n   📊 Statistiche:")
    print(f"      Coverage: {coverage:.1f}%")
    print(f"      Min: {np.min(valid_data):.2e}")
    print(f"      Max: {np.max(valid_data):.2e}")
    print(f"      Media: {np.mean(valid_data):.2e}")
    print(f"      Immagini sovrapposte (max): {np.max(n_pixels)}")
    
    # Prepara output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"mosaic_{source}_{obj_name}_{len(all_files)}img_{timestamp}.fits"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Aggiorna header
    template_header['OBJECT'] = obj_name
    template_header['SOURCE'] = source
    template_header['HISTORY'] = 'Mosaic created by step6_mosaic.py'
    template_header['NCOMBINE'] = (len(all_files), 'Number of images combined')
    template_header['COMBMETH'] = ('mean', 'Combination method')
    template_header['COVERAGE'] = (coverage, 'Coverage percentage')
    template_header['MOSDATE'] = datetime.now().isoformat()
    template_header['MAXSTACK'] = (int(np.max(n_pixels)), 'Max stacked images per pixel')
    template_header['MOSMODE'] = ('single-source', 'Mosaic creation mode')
    
    # Salva
    print(f"\n   💾 Salvataggio: {output_filename}")
    
    try:
        fits.PrimaryHDU(data=mosaic_data, header=template_header).writeto(
            output_path,
            overwrite=True,
            output_verify='silentfix'
        )
        
        print(f"   ✅ Mosaico salvato: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"   ❌ Errore salvando mosaico: {e}")
        return None


def create_mosaic_multi_source(obj_name):
    """
    Crea un mosaico combinando tutte le fonti disponibili per un oggetto.
    
    Args:
        obj_name (str): Nome dell'oggetto
        
    Returns:
        str: Percorso del file mosaico creato, o None se fallito
    """
    print(f"\n{'─'*70}")
    print(f"🔀 Creazione mosaico MULTI-FONTE: {obj_name}")
    print(f"{'─'*70}")
    
    # Trova tutte le fonti che hanno questo oggetto
    sources = list_available_sources()
    available_sources = []
    
    for source in sources:
        obj_dir = os.path.join(INPUT_DIR, source, obj_name)
        if os.path.exists(obj_dir):
            files = glob.glob(os.path.join(obj_dir, '*.fits'))
            files.extend(glob.glob(os.path.join(obj_dir, '*.fit')))
            if files:
                available_sources.append((source, len(files)))
    
    if not available_sources:
        print(f"   ⚠️  Nessuna fonte trovata per {obj_name}")
        return None
    
    print(f"   Fonti trovate:")
    for src, count in available_sources:
        print(f"      - {src}: {count} immagini")
    
    # Raccogli tutti i file da tutte le fonti
    all_files = []
    source_info = {}  # Per tracciare quante immagini da ogni fonte
    
    for source, _ in available_sources:
        obj_dir = os.path.join(INPUT_DIR, source, obj_name)
        source_files = glob.glob(os.path.join(obj_dir, '*.fits'))
        source_files.extend(glob.glob(os.path.join(obj_dir, '*.fit')))
        
        all_files.extend(source_files)
        source_info[source] = len(source_files)
    
    print(f"\n   Totale file da combinare: {len(all_files)}")
    
    # Leggi template dal primo file
    try:
        with fits.open(all_files[0]) as hdul:
            template_header = hdul[0].header.copy()
            shape = hdul[0].data.shape
            
        print(f"   Dimensioni: {shape[1]} x {shape[0]} pixel")
        
    except Exception as e:
        print(f"   ❌ Errore leggendo {os.path.basename(all_files[0])}: {e}")
        return None
    
    # Inizializza array per combinazione
    total_flux = np.zeros(shape, dtype=np.float64)
    n_pixels = np.zeros(shape, dtype=np.int32)
    
    # Combina immagini da TUTTE le fonti
    print("\n   🔄 Combinazione immagini da tutte le fonti...")
    
    for filepath in tqdm(all_files, desc="     Combinazione", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                img_data = hdul[0].data
                
                # Verifica dimensioni
                if img_data.shape != shape:
                    print(f"\n     ⚠️  {os.path.basename(filepath)}: dimensioni diverse. Saltato.")
                    continue
                
                # Trova pixel validi (non NaN e non zero)
                valid_mask = np.isfinite(img_data) & (img_data != 0)
                
                # Sostituisci NaN con 0
                img_data_clean = np.nan_to_num(img_data, nan=0.0, copy=False)
                
                # Aggiungi ai totali
                total_flux += img_data_clean
                n_pixels[valid_mask] += 1
                
        except Exception as e:
            print(f"\n     ⚠️  Errore con {os.path.basename(filepath)}: {e}")
            continue
    
    # Calcola media
    print("\n   🧮 Calcolo media...")
    
    mosaic_data = np.full(shape, np.nan, dtype=np.float32)
    valid_stack = n_pixels > 0
    
    if valid_stack.sum() == 0:
        print("   ❌ Nessun pixel valido nel mosaico!")
        return None
    
    mosaic_data[valid_stack] = (total_flux[valid_stack] / n_pixels[valid_stack]).astype(np.float32)
    
    # Statistiche
    coverage = (valid_stack.sum() / mosaic_data.size) * 100
    valid_data = mosaic_data[valid_stack]
    
    print(f"\n   📊 Statistiche:")
    print(f"      Coverage: {coverage:.1f}%")
    print(f"      Min: {np.min(valid_data):.2e}")
    print(f"      Max: {np.max(valid_data):.2e}")
    print(f"      Media: {np.mean(valid_data):.2e}")
    print(f"      Immagini sovrapposte (max): {np.max(n_pixels)}")
    
    # Prepara output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sources_str = "+".join([src for src, _ in available_sources])
    output_filename = f"mosaic_MULTI_{sources_str}_{obj_name}_{len(all_files)}img_{timestamp}.fits"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Aggiorna header
    template_header['OBJECT'] = obj_name
    template_header['HISTORY'] = 'Multi-source mosaic created by step6_mosaic.py'
    template_header['NCOMBINE'] = (len(all_files), 'Number of images combined')
    template_header['COMBMETH'] = ('mean', 'Combination method')
    template_header['COVERAGE'] = (coverage, 'Coverage percentage')
    template_header['MOSDATE'] = datetime.now().isoformat()
    template_header['MAXSTACK'] = (int(np.max(n_pixels)), 'Max stacked images per pixel')
    template_header['MOSMODE'] = ('multi-source', 'Mosaic creation mode')
    template_header['NSOURCES'] = (len(available_sources), 'Number of sources combined')
    
    # Aggiungi info per ogni fonte
    for i, (src, count) in enumerate(available_sources, 1):
        template_header[f'SOURCE{i}'] = (src, f'Source {i} name')
        template_header[f'NSRC{i}'] = (count, f'Images from source {i}')
    
    # Salva
    print(f"\n   💾 Salvataggio: {output_filename}")
    
    try:
        fits.PrimaryHDU(data=mosaic_data, header=template_header).writeto(
            output_path,
            overwrite=True,
            output_verify='silentfix'
        )
        
        print(f"   ✅ Mosaico multi-fonte salvato: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"   ❌ Errore salvando mosaico: {e}")
        return None


def create_mosaics_for_all_objects(source):
    """Crea mosaici per tutti gli oggetti di una fonte (modalità single-source)."""
    objects = list_available_objects(source)
    
    print(f"\n🔄 Creazione mosaici per {len(objects)} oggetti...")
    
    created_mosaics = []
    failed_objects = []
    
    for obj_idx, (obj_name, img_count) in enumerate(objects, 1):
        print(f"\n[{obj_idx}/{len(objects)}]")
        
        mosaic_path = create_mosaic_for_object(source, obj_name)
        
        if mosaic_path:
            created_mosaics.append(os.path.basename(mosaic_path))
        else:
            failed_objects.append(obj_name)
    
    return created_mosaics, failed_objects


def create_mosaics_multi_all():
    """Crea mosaici multi-fonte per tutti gli oggetti comuni."""
    common_objects = find_common_objects()
    
    print(f"\n🔄 Creazione mosaici multi-fonte per {len(common_objects)} oggetti...")
    
    created_mosaics = []
    failed_objects = []
    
    for obj_idx, (obj_name, sources_list, total_images) in enumerate(common_objects, 1):
        print(f"\n[{obj_idx}/{len(common_objects)}]")
        
        mosaic_path = create_mosaic_multi_source(obj_name)
        
        if mosaic_path:
            created_mosaics.append(os.path.basename(mosaic_path))
        else:
            failed_objects.append(obj_name)
    
    return created_mosaics, failed_objects


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Funzione principale."""
    print("\n" + "🖼️ "*35)
    print("STEP 6: CREAZIONE MOSAICO FINALE".center(70))
    print("🖼️ "*35)
    
    # Menu interattivo
    result = interactive_menu()
    
    if result[0] is None:
        print("\n❌ Nessuna selezione effettuata.")
        return
    
    mode, source_or_none, selected_object = result
    
    start_time = time.time()
    
    # === MODALITÀ SINGLE-SOURCE ===
    if mode == 'single':
        if selected_object:
            # Singolo oggetto, singola fonte
            mosaic_path = create_mosaic_for_object(source_or_none, selected_object)
            
            # Riepilogo
            print("\n" + "=" * 70)
            print("📊 RIEPILOGO")
            print("=" * 70)
            print(f"   Modalità: Singola fonte")
            print(f"   Fonte: {source_or_none}")
            print(f"   Oggetto: {selected_object}")
            
            if mosaic_path:
                print(f"   ✅ Mosaico creato: {os.path.basename(mosaic_path)}")
                print(f"   Path completo: {mosaic_path}")
            else:
                print(f"   ❌ Creazione mosaico fallita")
        else:
            # Tutti gli oggetti, singola fonte
            created, failed = create_mosaics_for_all_objects(source_or_none)
            
            # Riepilogo
            print("\n" + "=" * 70)
            print("📊 RIEPILOGO TOTALE")
            print("=" * 70)
            print(f"   Modalità: Singola fonte")
            print(f"   Fonte: {source_or_none}")
            print(f"   Mosaici creati: {len(created)}")
            
            if created:
                print(f"\n   ✅ Mosaici salvati in: {OUTPUT_DIR}/")
                for mosaic in created[:5]:  # Mostra solo primi 5
                    print(f"      - {mosaic}")
                if len(created) > 5:
                    print(f"      ... e altri {len(created)-5} mosaici")
            
            if failed:
                print(f"\n   ⚠️  Oggetti falliti ({len(failed)}):")
                for obj in failed:
                    print(f"      - {obj}")
    
    # === MODALITÀ MULTI-SOURCE ===
    elif mode == 'multi':
        if selected_object:
            # Singolo oggetto, multi-fonte
            mosaic_path = create_mosaic_multi_source(selected_object)
            
            # Riepilogo
            print("\n" + "=" * 70)
            print("📊 RIEPILOGO")
            print("=" * 70)
            print(f"   Modalità: Multi-fonte")
            print(f"   Oggetto: {selected_object}")
            
            if mosaic_path:
                print(f"   ✅ Mosaico combinato creato: {os.path.basename(mosaic_path)}")
                print(f"   Path completo: {mosaic_path}")
            else:
                print(f"   ❌ Creazione mosaico fallita")
        else:
            # Tutti gli oggetti comuni, multi-fonte
            created, failed = create_mosaics_multi_all()
            
            # Riepilogo
            print("\n" + "=" * 70)
            print("📊 RIEPILOGO TOTALE")
            print("=" * 70)
            print(f"   Modalità: Multi-fonte")
            print(f"   Mosaici combinati creati: {len(created)}")
            
            if created:
                print(f"\n   ✅ Mosaici salvati in: {OUTPUT_DIR}/")
                for mosaic in created[:5]:  # Mostra solo primi 5
                    print(f"      - {mosaic}")
                if len(created) > 5:
                    print(f"      ... e altri {len(created)-5} mosaici")
            
            if failed:
                print(f"\n   ⚠️  Oggetti falliti ({len(failed)}):")
                for obj in failed:
                    print(f"      - {obj}")
    
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Tempo totale: {elapsed:.1f} secondi")
    print(f"\n✅ STEP 6 COMPLETATO!")
    print(f"\n   📁 Mosaici salvati in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()