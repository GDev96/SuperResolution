"""
STEP 4: MOSAICO PER IMMAGINI SPARSE (NO STACK)
Per immagini HST che coprono zone diverse senza sovrapposizione.
Combina le immagini SENZA richiedere overlap - riempie i buchi con NaN.
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

# --- CONFIGURAZIONE PERCORSI ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ✅ PATH CORRETTI per la tua struttura
INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_register_4')    # Da img_register_4 (se esiste)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_preprocessed')  # A img_preprocessed
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Fallback se img_register_4 non esiste ancora
FALLBACK_INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_plate_2')  # Se non hai ancora registrato

def setup_logging():
    """Configura il sistema di logging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'mosaic_sparse_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_mosaic_sparse():
    """
    Crea mosaico da immagini sparse (NON sovrapposte).
    Prende il valore medio dove le immagini si sovrappongono,
    mantiene i singoli valori dove non c'è overlap.
    """
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("MOSAICO PER IMMAGINI SPARSE")
    logger.info("=" * 60)

    print("=" * 70)
    print("🖼️ MOSAICO IMMAGINI SPARSE".center(70))
    print("=" * 70)
    print("\nℹ️  Modalità: combina immagini sparse senza richiedere overlap")
    
    # ✅ VERIFICA ESISTENZA DIRECTORY INPUT
    current_input_dir = INPUT_DIR
    if not os.path.exists(INPUT_DIR):
        print(f"\n⚠️  Directory {INPUT_DIR} non trovata")
        if os.path.exists(FALLBACK_INPUT_DIR):
            current_input_dir = FALLBACK_INPUT_DIR
            print(f"✓ Uso fallback: {FALLBACK_INPUT_DIR}")
        else:
            print(f"❌ Neppure {FALLBACK_INPUT_DIR} esiste!")
            return None
    
    print(f"\n📁 Directory input: {os.path.abspath(current_input_dir)}")
    print(f"📁 Directory output: {os.path.abspath(OUTPUT_DIR)}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ✅ CERCA PATTERN DI FILE DIVERSI
    # Cerca prima file registrati, poi file originali HST
    patterns_to_try = [
        os.path.join(current_input_dir, 'register_*.fits'),
        os.path.join(current_input_dir, 'lowcov_*.fits'),
        os.path.join(current_input_dir, 'plate_*.fits'),
        os.path.join(current_input_dir, 'hst_*.fits'),
        os.path.join(current_input_dir, '*.fits')
    ]
    
    fits_files = []
    pattern_used = None
    
    for pattern in patterns_to_try:
        fits_files = sorted(glob.glob(pattern))
        if fits_files:
            pattern_used = pattern
            break
    
    if not fits_files:
        error_msg = f"Nessuna immagine trovata in {current_input_dir}"
        logger.error(error_msg)
        print(f"\n❌ ERRORE: {error_msg}")
        print(f"\n💡 Pattern cercati:")
        for pattern in patterns_to_try:
            print(f"   - {pattern}")
        return None

    logger.info(f"Trovati {len(fits_files)} file con pattern: {pattern_used}")
    print(f"\n✓ Trovati {len(fits_files)} file")
    print(f"   Pattern usato: {os.path.basename(pattern_used)}")
    
    # Mostra alcuni file di esempio
    print(f"\n📄 File trovati:")
    for i, f in enumerate(fits_files[:5]):
        print(f"   {i+1}. {os.path.basename(f)}")
    if len(fits_files) > 5:
        print(f"   ... e altri {len(fits_files)-5} file")

    # Load reference for shape and WCS
    try:
        with fits.open(fits_files[0]) as hdul:
            # ✅ TROVA HDU CON DATI (gestisce file multi-estensione HST)
            data_hdu = None
            for hdu_idx, hdu in enumerate(hdul):
                if hdu.data is not None and len(hdu.data.shape) == 2:
                    data_hdu = hdu
                    data_hdu_idx = hdu_idx
                    break
            
            if data_hdu is None:
                raise ValueError("Nessun HDU con dati 2D trovato")
            
            ref_header = data_hdu.header.copy()
            final_shape = data_hdu.data.shape
            
        logger.info(f"Dimensioni canvas: {final_shape}")
        logger.info(f"HDU dati: {data_hdu_idx}")
        print(f"✓ Canvas: {final_shape[1]}x{final_shape[0]} pixel")
        print(f"✓ HDU dati: {data_hdu_idx}")
        
        # ✅ MEMORIA STIMATA
        memory_gb = (final_shape[0] * final_shape[1] * 4 * 2) / (1024**3)  # *2 per sum+count arrays
        print(f"✓ Memoria stimata: ~{memory_gb:.1f} GB")
            
    except Exception as e:
        logger.error(f"Errore caricamento riferimento: {e}")
        print(f"\n❌ ERRORE: Impossibile caricare riferimento")
        print(f"   File: {fits_files[0]}")
        print(f"   Errore: {e}")
        return None

    # Initialize accumulators
    sum_array = np.zeros(final_shape, dtype=np.float64)
    count_array = np.zeros(final_shape, dtype=np.int32)
    
    print(f"\n🔄 Combinazione di {len(fits_files)} immagini sparse...")
    logger.info("Avvio combinazione...")
    
    valid_count = 0
    
    with tqdm(total=len(fits_files), desc="Combinazione", unit="img") as pbar:
        for filename in fits_files:
            try:
                with fits.open(filename) as hdul:
                    # ✅ TROVA HDU CON DATI
                    data_hdu = None
                    for hdu in hdul:
                        if hdu.data is not None and len(hdu.data.shape) == 2:
                            data_hdu = hdu
                            break
                    
                    if data_hdu is None:
                        logger.warning(f"No data HDU: {os.path.basename(filename)}")
                        continue
                    
                    data = data_hdu.data
                    
                    if data.shape != final_shape:
                        logger.warning(f"Shape diverso: {os.path.basename(filename)} ({data.shape} vs {final_shape})")
                        continue
                    
                    # ✅ TROVA PIXEL VALIDI (gestisce diversi tipi di "vuoto")
                    valid_mask = (
                        np.isfinite(data) & 
                        (data != 0) & 
                        ~np.isnan(data) & 
                        ~np.isinf(data)
                    )
                    
                    n_valid = valid_mask.sum()
                    coverage = (n_valid / data.size) * 100
                    
                    if coverage < 0.01:  # Soglia molto bassa
                        logger.warning(f"Copertura troppo bassa ({coverage:.4f}%): {os.path.basename(filename)}")
                        continue
                    
                    # Aggiungi ai pixel validi
                    sum_array[valid_mask] += data[valid_mask]
                    count_array[valid_mask] += 1
                    
                    valid_count += 1
                    logger.debug(f"✓ {os.path.basename(filename)}: {n_valid:,} pixel ({coverage:.3f}%)")
                    pbar.set_description(f"✓ {valid_count} processate")
                    
            except Exception as e:
                logger.error(f"Errore {os.path.basename(filename)}: {e}")
                pbar.set_description(f"❌ Errore: {os.path.basename(filename)}")
            
            pbar.update(1)
            
            # ✅ GARBAGE COLLECTION PERIODICO
            if (pbar.n % 5) == 0:
                gc.collect()
    
    if valid_count == 0:
        logger.error("Nessuna immagine valida processata!")
        print(f"\n❌ ERRORE: Nessuna immagine valida")
        return None
    
    logger.info(f"Processate {valid_count} immagini valide")
    print(f"\n✓ Processate: {valid_count}/{len(fits_files)} immagini")

    # Calculate final mosaic (mean where overlap, single value where not)
    print(f"\n🧮 Calcolo mosaico finale...")
    
    # ✅ CALCOLO SICURO DELLA MEDIA
    with np.errstate(divide='ignore', invalid='ignore'):
        final_mosaic = np.divide(
            sum_array, 
            count_array, 
            out=np.full_like(sum_array, np.nan, dtype=np.float32),
            where=count_array > 0
        )
    
    # Statistics
    valid_pixels = np.isfinite(final_mosaic) & (final_mosaic != 0)
    coverage_total = (valid_pixels.sum() / final_mosaic.size) * 100
    
    # Coverage map statistics
    unique_counts = np.unique(count_array[count_array > 0])
    
    logger.info("Statistiche mosaico:")
    logger.info(f"  Pixel totali: {final_mosaic.size:,}")
    logger.info(f"  Pixel con dati: {valid_pixels.sum():,} ({coverage_total:.2f}%)")
    
    if valid_pixels.sum() > 0:
        valid_data = final_mosaic[valid_pixels]
        logger.info(f"  Range valori: {np.min(valid_data):.2e} - {np.max(valid_data):.2e}")
        
        print(f"\n📊 STATISTICHE MOSAICO:")
        print(f"   Pixel totali: {final_mosaic.size:,}")
        print(f"   Pixel con dati: {valid_pixels.sum():,} ({coverage_total:.2f}%)")
        print(f"   Min: {np.min(valid_data):.2e}")
        print(f"   Max: {np.max(valid_data):.2e}")
        print(f"   Media: {np.mean(valid_data):.2e}")
        print(f"   Mediana: {np.median(valid_data):.2e}")
        
        print(f"\n📈 DISTRIBUZIONE SOVRAPPOSIZIONI:")
        for count in sorted(unique_counts):
            n_pixels = (count_array == count).sum()
            percent = (n_pixels / valid_pixels.sum()) * 100
            print(f"   {count} immagini: {n_pixels:,} pixel ({percent:.1f}%)")
    else:
        print(f"\n⚠️  Nessun pixel valido nel mosaico finale!")
        return None
    
    # Update header
    ref_header['NIMAGES'] = (valid_count, 'Number of images in mosaic')
    ref_header['COVERAGE'] = (coverage_total, 'Percentage of canvas with data')
    ref_header['COMBMODE'] = ('sparse', 'Sparse mosaic mode')
    ref_header['DATE'] = datetime.now().isoformat()
    ref_header['CREATOR'] = 'AstroMosaic.py v1.1'
    ref_header['INPUTDIR'] = os.path.basename(current_input_dir)
    ref_header['PATTERN'] = os.path.basename(pattern_used)
    
    # ✅ NORMALIZZAZIONE MIGLIORATA
    print(f"\n🎨 Normalizzazione dati...")
    
    if valid_pixels.sum() > 100:  # Abbastanza pixel per statistiche affidabili
        valid_data = final_mosaic[valid_pixels]
        
        # Usa percentili per rimozione outlier robusta
        p_low = np.percentile(valid_data, 0.5)   # 0.5% percentile
        p_high = np.percentile(valid_data, 99.5)  # 99.5% percentile
        
        logger.info(f"Range normalizzazione: {p_low:.2e} - {p_high:.2e}")
        print(f"   Range: {p_low:.2e} - {p_high:.2e}")
        
        # Clip e normalizza
        final_mosaic_normalized = np.clip(final_mosaic, p_low, p_high)
        
        # Normalizza a 0-1, poi scala a range appropriato
        if p_high > p_low:
            final_mosaic_normalized = (final_mosaic_normalized - p_low) / (p_high - p_low)
            
            # Scala a 16-bit (0-65535) per massima dinamica
            final_mosaic_normalized = final_mosaic_normalized * 65535.0
        else:
            logger.warning("Range di valori troppo piccolo per normalizzazione")
            final_mosaic_normalized = final_mosaic
        
        # Ripristina NaN dove non c'erano dati
        final_mosaic_normalized[~valid_pixels] = 0.0  # Usa 0 invece di NaN per compatibilità
        
    else:
        logger.warning("Troppo pochi pixel validi per normalizzazione")
        final_mosaic_normalized = final_mosaic
        final_mosaic_normalized[~valid_pixels] = 0.0
    
    logger.info("Normalizzazione completata")
    
    # ✅ SALVATAGGIO CON NOME DESCRITTIVO
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f'mosaic_M42_HST_{valid_count}img_{timestamp}'
    output_file = os.path.join(OUTPUT_DIR, f'{base_name}.fits')
    
    logger.info(f"Salvataggio: {output_file}")
    print(f"\n💾 Salvataggio mosaico...")
    
    try:
        # Salva come float32 per compatibilità
        fits.PrimaryHDU(
            final_mosaic_normalized.astype(np.float32), 
            header=ref_header
        ).writeto(output_file, overwrite=True)
        
        logger.info("Salvato con successo")
        
        # ✅ CREA ANCHE VERSIONE PNG PER PREVIEW (opzionale)
        try:
            import matplotlib.pyplot as plt
            
            preview_file = os.path.join(OUTPUT_DIR, f'{base_name}_preview.png')
            
            plt.figure(figsize=(12, 12))
            plt.imshow(final_mosaic_normalized, cmap='gray', origin='lower')
            plt.title(f'M42 Mosaic - HST F656N (Hα)\n{valid_count} images combined')
            plt.colorbar(label='Intensity')
            plt.tight_layout()
            plt.savefig(preview_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Preview PNG salvato: {os.path.basename(preview_file)}")
            
        except ImportError:
            logger.info("Matplotlib non disponibile - skip preview PNG")
        except Exception as e:
            logger.warning(f"Errore creazione preview: {e}")
        
    except Exception as e:
        logger.error(f"Errore salvataggio: {e}")
        print(f"\n❌ ERRORE: Salvataggio fallito")
        print(f"   Errore: {e}")
        return None

    print("\n" + "=" * 70)
    print("✅ MOSAICO COMPLETATO".center(70))
    print("=" * 70)
    print(f"\n📁 File salvato:")
    print(f"   {os.path.abspath(output_file)}")
    print(f"\n📊 Risultati:")
    print(f"   Immagini combinate: {valid_count}")
    print(f"   Copertura totale: {coverage_total:.2f}%")
    print(f"   Dimensioni: {final_shape[1]}x{final_shape[0]} pixel")
    print(f"\n💡 NOTE:")
    print(f"   - Zone con 1 immagine: valore originale")
    print(f"   - Zone con overlap: media pesata")
    print(f"   - Zone vuote: 0 (nero)")
    print("\n" + "=" * 70)
    
    logger.info("Mosaico completato con successo")
    return output_file

if __name__ == "__main__":
    start_time = time.time()
    
    print("🔭 MOSAICO IMMAGINI HST SPARSE")
    print(f"Versione ottimizzata per gestire diversi formati di file")
    
    output = create_mosaic_sparse()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Tempo totale: {elapsed:.1f} secondi")
    
    if output:
        print(f"✅ Mosaico creato con successo!")
    else:
        print(f"❌ Errore nella creazione del mosaico")