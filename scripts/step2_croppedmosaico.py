"""
STEP 2: MOSAICO DA IMMAGINI REGISTRATE (CON CROP INTELLIGENTE)

Combina immagini registrate in un mosaico finale, con crop automatico
delle zone vuote. Supporta immagini sparse e con sovrapposizione.

VERSIONE CORRETTA:
✅ Gestione robusta pixel validi (NaN, 0, inf)
✅ Calcolo media pesata per overlap
✅ Crop automatico zone vuote
✅ Normalizzazione percentile-based
✅ Preview PNG automatico
✅ Compatibilità multi-target
"""

import os
import sys
import glob
import time
import logging
import gc
import numpy as np
from astropy.io import fits
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# --- CONFIGURAZIONE PERCORSI ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"

# Normalizzazione
PERCENTILE_LOW = 0.5    # 0.5% inferiore
PERCENTILE_HIGH = 99.5  # 99.5% superiore

def setup_logging():
    """Configura il sistema di logging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'mosaic_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def select_target_for_mosaic(logger):
    """
    Mostra menu per selezionare il target da cui creare il mosaico.
    """
    print("\n" + "=" * 80)
    print("🎯 SELEZIONE TARGET PER MOSAICO".center(80))
    print("=" * 80)
    
    # Cerca target con immagini registrate
    available_targets = []
    for item in os.listdir(ROOT_DATA_DIR):
        item_path = ROOT_DATA_DIR / item
        if item_path.is_dir() and not item.startswith('.'):
            # Cerca cartella 3_registered_native/hubble
            registered_path = item_path / "3_registered_native" / "hubble"
            
            if registered_path.exists():
                # Conta file registrati
                register_files = list(registered_path.glob("register_*.fits"))
                if register_files:
                    available_targets.append({
                        'name': item,
                        'input_dir': registered_path,
                        'count': len(register_files)
                    })
    
    if not available_targets:
        print(f"\n❌ ERRORE: Nessun target con immagini registrate trovato")
        print("   Esegui prima step1_wcsregister.py")
        logger.error("Nessun target disponibile per mosaico")
        return None, None, None
    
    # Ordina per nome
    available_targets.sort(key=lambda x: x['name'])
    
    print("\nTarget disponibili:")
    for i, target in enumerate(available_targets, 1):
        print(f"  [{i}] {target['name']:<15} ({target['count']} immagini registrate)")
    
    print(f"  [0] Esci")
    
    while True:
        try:
            choice = input(f"\n👉 Seleziona un target (0-{len(available_targets)}): ")
            choice = int(choice)
            
            if choice == 0:
                print("\n👋 Operazione annullata.")
                logger.info("Utente ha annullato la selezione.")
                return None, None, None
            
            if 1 <= choice <= len(available_targets):
                selected = available_targets[choice - 1]
                
                # Costruisci path output
                output_dir = ROOT_DATA_DIR / selected['name'] / "4_mosaic"
                
                print(f"\n✅ Target selezionato: {selected['name']}")
                print(f"   Input: {selected['input_dir'].relative_to(ROOT_DATA_DIR)}")
                print(f"   Output: {output_dir.relative_to(ROOT_DATA_DIR)}")
                
                logger.info(f"Target selezionato: {selected['name']}")
                logger.info(f"Input: {selected['input_dir']}")
                logger.info(f"Output: {output_dir}")
                
                return selected['name'], selected['input_dir'], output_dir
            else:
                print(f"⚠️ Scelta non valida. Inserisci un numero tra 0 e {len(available_targets)}.")
                
        except ValueError:
            print("⚠️ Input non valido. Inserisci un numero.")
        except KeyboardInterrupt:
            print("\n\n👋 Operazione annullata.")
            logger.info("Utente ha interrotto la selezione (Ctrl+C).")
            return None, None, None

def find_bounding_box(count_array):
    """
    Trova il bounding box dell'area con dati validi.
    Restituisce (y_min, y_max, x_min, x_max).
    """
    # Trova dove ci sono dati
    rows_with_data = np.any(count_array > 0, axis=1)
    cols_with_data = np.any(count_array > 0, axis=0)
    
    if not np.any(rows_with_data) or not np.any(cols_with_data):
        # Nessun dato trovato
        return None
    
    y_min = np.argmax(rows_with_data)
    y_max = len(rows_with_data) - np.argmax(rows_with_data[::-1])
    
    x_min = np.argmax(cols_with_data)
    x_max = len(cols_with_data) - np.argmax(cols_with_data[::-1])
    
    return (y_min, y_max, x_min, x_max)

def create_mosaic(target_name, input_dir, output_dir, logger):
    """
    Crea mosaico da immagini registrate con crop intelligente.
    """
    logger.info("=" * 60)
    logger.info("CREAZIONE MOSAICO")
    logger.info("=" * 60)

    print("\n" + "=" * 70)
    print("🖼️ CREAZIONE MOSAICO".center(70))
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Cerca file registrati
    fits_files = sorted(input_dir.glob("register_*.fits"))
    
    if not fits_files:
        logger.error(f"Nessun file register_*.fits trovato in {input_dir}")
        print(f"\n❌ ERRORE: Nessun file registrato trovato")
        return None
    
    logger.info(f"Trovati {len(fits_files)} file registrati")
    print(f"\n✓ Trovati {len(fits_files)} file registrati")
    
    # Mostra alcuni file
    print(f"\n📄 File da combinare:")
    for i, f in enumerate(fits_files[:5]):
        print(f"   {i+1}. {f.name}")
    if len(fits_files) > 5:
        print(f"   ... e altri {len(fits_files)-5} file")

    # Carica riferimento per shape e WCS
    try:
        with fits.open(fits_files[0]) as hdul:
            ref_header = hdul[0].header.copy()
            final_shape = hdul[0].data.shape
        
        logger.info(f"Dimensioni canvas: {final_shape}")
        print(f"\n✓ Canvas iniziale: {final_shape[1]}x{final_shape[0]} pixel")
        
        # Stima memoria
        memory_gb = (final_shape[0] * final_shape[1] * 4 * 2) / (1024**3)
        print(f"✓ Memoria stimata: ~{memory_gb:.1f} GB")
            
    except Exception as e:
        logger.error(f"Errore caricamento riferimento: {e}")
        print(f"\n❌ ERRORE: Impossibile caricare file di riferimento")
        return None

    # Inizializza accumulatori
    sum_array = np.zeros(final_shape, dtype=np.float64)
    count_array = np.zeros(final_shape, dtype=np.int32)
    
    print(f"\n🔄 Combinazione di {len(fits_files)} immagini...")
    logger.info("Avvio combinazione...")
    
    valid_count = 0
    
    with tqdm(total=len(fits_files), desc="Combinazione", unit="img") as pbar:
        for filepath in fits_files:
            try:
                with fits.open(filepath) as hdul:
                    data = hdul[0].data
                    
                    if data.shape != final_shape:
                        logger.warning(f"Shape diverso: {filepath.name} ({data.shape} vs {final_shape})")
                        continue
                    
                    # Trova pixel validi (gestisce NaN, 0, inf)
                    valid_mask = (
                        np.isfinite(data) & 
                        (data != 0) & 
                        ~np.isnan(data) & 
                        ~np.isinf(data)
                    )
                    
                    n_valid = valid_mask.sum()
                    coverage = (n_valid / data.size) * 100
                    
                    if coverage < 0.01:
                        logger.warning(f"Copertura troppo bassa ({coverage:.4f}%): {filepath.name}")
                        continue
                    
                    # Aggiungi ai pixel validi
                    sum_array[valid_mask] += data[valid_mask]
                    count_array[valid_mask] += 1
                    
                    valid_count += 1
                    logger.debug(f"✓ {filepath.name}: {n_valid:,} pixel ({coverage:.3f}%)")
                    pbar.set_description(f"✓ {valid_count} processate")
                    
            except Exception as e:
                logger.error(f"Errore {filepath.name}: {e}")
                pbar.set_description(f"❌ Errore: {filepath.name}")
            
            pbar.update(1)
            
            # Garbage collection periodico
            if (pbar.n % 5) == 0:
                gc.collect()
    
    if valid_count == 0:
        logger.error("Nessuna immagine valida processata!")
        print(f"\n❌ ERRORE: Nessuna immagine valida")
        return None
    
    logger.info(f"Processate {valid_count} immagini valide")
    print(f"\n✓ Processate: {valid_count}/{len(fits_files)} immagini")

    # Calcola mosaico finale (media dove overlap)
    print(f"\n🧮 Calcolo mosaico finale...")
    
    with np.errstate(divide='ignore', invalid='ignore'):
        final_mosaic = np.divide(
            sum_array, 
            count_array, 
            out=np.full_like(sum_array, np.nan, dtype=np.float32),
            where=count_array > 0
        )
    
    # Trova bounding box con dati
    print(f"\n✂️ Crop zone vuote...")
    bbox = find_bounding_box(count_array)
    
    if bbox is None:
        logger.error("Nessun pixel con dati trovato!")
        print(f"\n❌ ERRORE: Nessun pixel con dati")
        return None
    
    y_min, y_max, x_min, x_max = bbox
    
    # Crop il mosaico
    final_mosaic_cropped = final_mosaic[y_min:y_max, x_min:x_max]
    count_array_cropped = count_array[y_min:y_max, x_min:x_max]
    
    logger.info(f"Crop: ({y_min}:{y_max}, {x_min}:{x_max})")
    logger.info(f"Dimensioni finali: {final_mosaic_cropped.shape}")
    
    print(f"✓ Crop applicato:")
    print(f"   Da: {final_shape[1]}x{final_shape[0]} pixel")
    print(f"   A:  {final_mosaic_cropped.shape[1]}x{final_mosaic_cropped.shape[0]} pixel")
    print(f"   Riduzione: {(1 - final_mosaic_cropped.size/final_shape[0]/final_shape[1])*100:.1f}%")
    
    # Statistiche
    valid_pixels = np.isfinite(final_mosaic_cropped) & (final_mosaic_cropped != 0)
    coverage_total = (valid_pixels.sum() / final_mosaic_cropped.size) * 100
    
    # Distribuzione sovrapposizioni
    unique_counts = np.unique(count_array_cropped[count_array_cropped > 0])
    
    logger.info("Statistiche mosaico:")
    logger.info(f"  Pixel totali: {final_mosaic_cropped.size:,}")
    logger.info(f"  Pixel con dati: {valid_pixels.sum():,} ({coverage_total:.2f}%)")
    
    if valid_pixels.sum() > 0:
        valid_data = final_mosaic_cropped[valid_pixels]
        logger.info(f"  Range valori: {np.min(valid_data):.2e} - {np.max(valid_data):.2e}")
        
        print(f"\n📊 STATISTICHE MOSAICO:")
        print(f"   Pixel totali: {final_mosaic_cropped.size:,}")
        print(f"   Pixel con dati: {valid_pixels.sum():,} ({coverage_total:.2f}%)")
        print(f"   Min: {np.min(valid_data):.2e}")
        print(f"   Max: {np.max(valid_data):.2e}")
        print(f"   Media: {np.mean(valid_data):.2e}")
        print(f"   Mediana: {np.median(valid_data):.2e}")
        
        print(f"\n📈 DISTRIBUZIONE SOVRAPPOSIZIONI:")
        for count in sorted(unique_counts):
            n_pixels = (count_array_cropped == count).sum()
            percent = (n_pixels / valid_pixels.sum()) * 100
            print(f"   {count} immagini: {n_pixels:,} pixel ({percent:.1f}%)")
    else:
        print(f"\n⚠️ Nessun pixel valido nel mosaico finale!")
        return None
    
    # Aggiorna header con WCS croppato
    ref_header['NAXIS1'] = final_mosaic_cropped.shape[1]
    ref_header['NAXIS2'] = final_mosaic_cropped.shape[0]
    
    # Aggiorna CRPIX per il crop
    if 'CRPIX1' in ref_header:
        ref_header['CRPIX1'] -= x_min
    if 'CRPIX2' in ref_header:
        ref_header['CRPIX2'] -= y_min
    
    # Metadati mosaico
    ref_header['NIMAGES'] = (valid_count, 'Number of images in mosaic')
    ref_header['COVERAGE'] = (coverage_total, 'Percentage with data')
    ref_header['COMBMODE'] = ('mean', 'Combination mode')
    ref_header['CROPPED'] = (True, 'Empty areas cropped')
    ref_header['CROPBOX'] = (f"{y_min},{y_max},{x_min},{x_max}", 'Crop bounding box')
    ref_header['DATE'] = datetime.now().isoformat()
    ref_header['CREATOR'] = 'AstroMosaic.py v2.0'
    ref_header['TARGET'] = target_name
    
    # Normalizzazione
    print(f"\n🎨 Normalizzazione dati...")
    
    if valid_pixels.sum() > 100:
        valid_data = final_mosaic_cropped[valid_pixels]
        
        # Percentili per rimozione outlier
        p_low = np.percentile(valid_data, PERCENTILE_LOW)
        p_high = np.percentile(valid_data, PERCENTILE_HIGH)
        
        logger.info(f"Range normalizzazione: {p_low:.2e} - {p_high:.2e}")
        print(f"   Range: {p_low:.2e} - {p_high:.2e}")
        
        # Clip e normalizza
        final_mosaic_normalized = np.clip(final_mosaic_cropped, p_low, p_high)
        
        if p_high > p_low:
            final_mosaic_normalized = (final_mosaic_normalized - p_low) / (p_high - p_low)
            final_mosaic_normalized = final_mosaic_normalized * 65535.0
        else:
            logger.warning("Range troppo piccolo per normalizzazione")
            final_mosaic_normalized = final_mosaic_cropped
        
        # Ripristina 0 dove non c'erano dati
        final_mosaic_normalized[~valid_pixels] = 0.0
        
    else:
        logger.warning("Troppo pochi pixel per normalizzazione")
        final_mosaic_normalized = final_mosaic_cropped
        final_mosaic_normalized[~valid_pixels] = 0.0
    
    logger.info("Normalizzazione completata")
    
    # Salvataggio
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f'mosaic_{target_name}_{valid_count}img_{timestamp}'
    output_file = output_dir / f'{base_name}.fits'
    
    logger.info(f"Salvataggio: {output_file}")
    print(f"\n💾 Salvataggio mosaico...")
    
    try:
        fits.PrimaryHDU(
            final_mosaic_normalized.astype(np.float32), 
            header=ref_header
        ).writeto(output_file, overwrite=True)
        
        logger.info("Salvato con successo")
        
        # Preview PNG
        try:
            import matplotlib.pyplot as plt
            
            preview_file = output_dir / f'{base_name}_preview.png'
            
            plt.figure(figsize=(12, 12))
            plt.imshow(final_mosaic_normalized, cmap='gray', origin='lower')
            plt.title(f'{target_name} Mosaic\n{valid_count} images combined')
            plt.colorbar(label='Intensity')
            plt.tight_layout()
            plt.savefig(preview_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Preview PNG salvato: {preview_file.name}")
            
        except ImportError:
            logger.info("Matplotlib non disponibile - skip preview")
        except Exception as e:
            logger.warning(f"Errore creazione preview: {e}")
        
    except Exception as e:
        logger.error(f"Errore salvataggio: {e}")
        print(f"\n❌ ERRORE: Salvataggio fallito")
        return None

    print("\n" + "=" * 70)
    print("✅ MOSAICO COMPLETATO".center(70))
    print("=" * 70)
    print(f"\n📁 File salvato:")
    print(f"   {output_file.relative_to(PROJECT_ROOT)}")
    print(f"\n📊 Risultati:")
    print(f"   Immagini combinate: {valid_count}")
    print(f"   Copertura totale: {coverage_total:.2f}%")
    print(f"   Dimensioni finali: {final_mosaic_cropped.shape[1]}x{final_mosaic_cropped.shape[0]} px")
    print("\n" + "=" * 70)
    
    logger.info("Mosaico completato con successo")
    return output_file

def run_mosaic_pipeline():
    """Esegue la pipeline di creazione mosaico."""
    start_time = time.time()
    logger = setup_logging()
    
    print("\n" + "=" * 80)
    print("🚀 PIPELINE MOSAICO: COMBINAZIONE IMMAGINI REGISTRATE".center(80))
    print("=" * 80)
    
    # Selezione target
    target_name, input_dir, output_dir = select_target_for_mosaic(logger)
    
    if target_name is None:
        return
    
    print(f"\n📂 Configurazione:")
    print(f"   Target: {target_name}")
    print(f"   Input: {input_dir.relative_to(ROOT_DATA_DIR)}")
    print(f"   Output: {output_dir.relative_to(ROOT_DATA_DIR)}")
    print("=" * 80)
    
    # Crea mosaico
    output_file = create_mosaic(target_name, input_dir, output_dir, logger)
    
    # Riepilogo
    elapsed = time.time() - start_time
    
    print(f"\n⏱️ Tempo totale: {elapsed:.1f} secondi")
    
    if output_file:
        print(f"\n✅ SUCCESSO!")
        print(f"\n🎯 PROSSIMO PASSO:")
        print(f"   - Apri il mosaico in DS9 o PixInsight")
        print(f"   - Procedi con elaborazione finale (stretch, color balance)")
    else:
        print(f"\n❌ ERRORE nella creazione del mosaico")

if __name__ == "__main__":
    run_mosaic_pipeline()