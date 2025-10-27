"""
HST MOSAIC CREATOR - Script di Stacking con Edge Blending
Versione migliorata con fusione graduale dei bordi e gestione intelligente
"""

import os
import sys
import glob
import time 
import logging
import gc
import numpy as np
from scipy import ndimage
from astropy.io import fits
from astropy.wcs import WCS
from datetime import datetime
from tqdm import tqdm

# --- CONFIGURAZIONE PERCORSI ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ✅ PATH CORRETTI per la tua struttura
INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_register_4')   # Da img_register_4
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_preprocessed') # A img_preprocessed
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# --- CONFIGURAZIONE BLENDING ---
FEATHER_RADIUS = 100  # Pixel per la fusione graduale dei bordi (aumentato)
SIGMA_CLIP_THRESHOLD = 3.0  # Soglia per sigma clipping
USE_WEIGHTED_STACK = True  # Usa pesi basati sulla distanza dai bordi
MIN_COVERAGE = 0.01  # Copertura minima per includere immagini (0.01%)

def setup_logging():
    """Configura il sistema di logging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'mosaic_enhanced_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Log system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Numpy version: {np.__version__}")
    logger.info(f"Feather radius: {FEATHER_RADIUS} pixels")
    logger.info(f"Weighted stacking: {USE_WEIGHTED_STACK}")
    logger.info(f"Sigma clipping: {SIGMA_CLIP_THRESHOLD}")
    logger.info(f"Min coverage: {MIN_COVERAGE}%")
    
    return logger

def create_weight_map(data_shape, feather_radius=FEATHER_RADIUS):
    """
    Crea una mappa dei pesi per il blending graduale.
    I pixel vicini ai bordi hanno peso minore.
    """
    height, width = data_shape
    
    # Crea griglia di coordinate
    y, x = np.mgrid[0:height, 0:width]
    
    # Calcola distanza dai bordi
    dist_from_edges = np.minimum(
        np.minimum(x, width - 1 - x),
        np.minimum(y, height - 1 - y)
    )
    
    # Crea mappa pesi con transizione graduale
    if feather_radius > 0:
        weights = np.clip(dist_from_edges / feather_radius, 0, 1)
        # Applica funzione smooth (coseno per transizione più morbida)
        weights = 0.5 * (1 + np.cos(np.pi * (1 - weights)))
    else:
        weights = np.ones_like(dist_from_edges, dtype=np.float32)
    
    return weights.astype(np.float32)

def sigma_clip_combine(data_stack, weights_stack, sigma=SIGMA_CLIP_THRESHOLD):
    """
    Combina le immagini con sigma clipping per rimuovere outlier.
    """
    if data_stack.shape[0] < 3:
        # Troppo poche immagini per sigma clipping, usa media pesata
        weight_sum = np.sum(weights_stack, axis=0)
        valid_mask = weight_sum > 0
        
        result = np.zeros(data_stack.shape[1:], dtype=np.float32)
        numerator = np.sum(data_stack * weights_stack, axis=0)
        result[valid_mask] = numerator[valid_mask] / weight_sum[valid_mask]
        
        return result
    
    # Calcola mediana e deviazione standard robusta
    median = np.median(data_stack, axis=0)
    mad = np.median(np.abs(data_stack - median[np.newaxis, :, :]), axis=0)
    sigma_est = 1.4826 * mad  # Stima robusta della deviazione standard
    
    # Evita divisione per zero
    sigma_est = np.where(sigma_est == 0, np.nanstd(data_stack, axis=0), sigma_est)
    
    # Crea maschera per outlier
    diff = np.abs(data_stack - median[np.newaxis, :, :])
    outlier_mask = diff > (sigma * sigma_est[np.newaxis, :, :])
    
    # Azzera pesi per gli outlier
    weights_clipped = weights_stack.copy()
    weights_clipped[outlier_mask] = 0
    
    # Media pesata escludendo outlier
    weight_sum = np.sum(weights_clipped, axis=0)
    
    # Evita divisione per zero
    valid_mask = weight_sum > 0
    result = np.zeros_like(median, dtype=np.float32)
    
    numerator = np.sum(data_stack * weights_clipped, axis=0)
    result[valid_mask] = numerator[valid_mask] / weight_sum[valid_mask]
    
    return result

def stack_mosaic_enhanced(combine_function='weighted_mean'):
    """Esegue lo stacking migliorato con edge blending e gestione intelligente."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("HST MOSAIC CREATOR - STACKING ENHANCED")
    logger.info("=" * 60)

    print("=" * 70)
    print("🔭 HST MOSAIC CREATOR - STACKING MIGLIORATO".center(70))
    print("=" * 70)
    print(f"Input:  {os.path.abspath(INPUT_DIR)}")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Edge blending: {FEATHER_RADIUS} pixel")
    print(f"Sigma clipping: {SIGMA_CLIP_THRESHOLD}")

    # Setup directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find input files
    register_pattern = os.path.join(INPUT_DIR, 'register_*.fits')
    lowcov_pattern = os.path.join(INPUT_DIR, 'lowcov_*.fits')
    
    register_files = sorted(glob.glob(register_pattern))
    lowcov_files = sorted(glob.glob(lowcov_pattern))
    
    print(f"\n📊 FILE TROVATI:")
    print(f"   Copertura normale: {len(register_files)}")
    print(f"   Copertura bassa: {len(lowcov_files)}")
    
    if not register_files and not lowcov_files:
        msg = f"Nessuna immagine registrata trovata in {INPUT_DIR}"
        logger.error(msg)
        print(f"\n❌ ERRORE: {msg}")
        print("💡 Esegui prima AstroRegister.py")
        return None
    
    # Chiedi all'utente cosa includere
    include_lowcov = False
    if lowcov_files:
        try:
            choice = input("\nIncludere file a bassa copertura? (y/N): ").strip().lower()
            include_lowcov = choice in ['y', 'yes', 'si', 's']
        except:
            include_lowcov = False
    
    # Seleziona file da usare
    if include_lowcov:
        files = register_files + lowcov_files
        print(f"✓ Usando tutti i {len(files)} file")
    else:
        files = register_files
        print(f"✓ Usando solo {len(files)} file a copertura normale")
    
    logger.info(f"Trovate {len(files)} immagini registrate")

    # Load reference image for WCS and shape
    logger.info("Caricamento immagine di riferimento...")
    try:
        with fits.open(files[0]) as hdul:
            ref_data = hdul[0].data
            ref_header = hdul[0].header
            
            if ref_data is None:
                raise ValueError("Nessun dato nell'immagine di riferimento")
                
            final_shape = ref_data.shape
            logger.info(f"Dimensioni riferimento: {final_shape}")
            print(f"✓ Dimensioni canvas: {final_shape[1]}x{final_shape[0]} pixel")
            
            # Extract WCS info
            wcs = WCS(ref_header)
            if wcs.has_celestial:
                ra, dec = wcs.wcs.crval
                logger.info(f"Centro WCS: RA={ra:.3f}°, DEC={dec:.3f}°")
                print(f"✓ Centro: RA={ra:.3f}°, DEC={dec:.3f}°")
                
                # Calcola campo di vista
                try:
                    pixel_scale = ref_header.get('REGCOVER', 0.1)  # Fallback
                    if 'CD1_1' in ref_header:
                        pixel_scale_deg = abs(float(ref_header['CD1_1']))
                        pixel_scale_arcsec = pixel_scale_deg * 3600
                    else:
                        pixel_scale_arcsec = 0.1  # Default Hubble
                    
                    fov_arcmin_x = (final_shape[1] * pixel_scale_arcsec) / 60
                    fov_arcmin_y = (final_shape[0] * pixel_scale_arcsec) / 60
                    print(f"✓ Campo di vista: {fov_arcmin_x:.1f}' x {fov_arcmin_y:.1f}'")
                except:
                    pass
    
    except Exception as e:
        logger.error(f"Errore caricamento riferimento: {e}", exc_info=True)
        print(f"\n❌ ERRORE: Impossibile caricare immagine di riferimento")
        return None

    # Carica tutte le immagini in memoria (se possibile)
    logger.info("Precaricamento immagini...")
    print(f"\n📥 Caricamento di {len(files)} immagini...")
    
    data_stack = []
    weight_stack = []
    metadata_list = []
    
    with tqdm(total=len(files), desc="Caricamento", unit="img") as pbar:
        for filename in files:
            try:
                with fits.open(filename) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                    
                    if data.shape != final_shape:
                        logger.warning(f"Skip {os.path.basename(filename)}: shape mismatch")
                        continue
                    
                    # Verifica validità dati
                    finite_mask = np.isfinite(data)
                    nonzero_mask = data != 0
                    valid_pixels = finite_mask & nonzero_mask
                    
                    n_valid = np.sum(valid_pixels)
                    coverage = (n_valid / data.size) * 100
                    
                    if coverage < MIN_COVERAGE:
                        logger.warning(f"Skip {os.path.basename(filename)}: copertura {coverage:.3f}%")
                        continue
                    
                    # Crea mappa pesi per questa immagine
                    if USE_WEIGHTED_STACK:
                        # Combina pesi geometrici con validità dati
                        geometric_weights = create_weight_map(data.shape, FEATHER_RADIUS)
                        data_weights = valid_pixels.astype(np.float32)
                        combined_weights = geometric_weights * data_weights
                        
                        # Normalizza i pesi per evitare bias verso immagini più grandi
                        weight_sum = np.sum(combined_weights)
                        if weight_sum > 0:
                            combined_weights = combined_weights / weight_sum * n_valid
                    else:
                        combined_weights = valid_pixels.astype(np.float32)
                    
                    # Sostituisci NaN/inf con zero
                    clean_data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    data_stack.append(clean_data)
                    weight_stack.append(combined_weights)
                    
                    # Salva metadati
                    metadata_list.append({
                        'filename': os.path.basename(filename),
                        'coverage': coverage,
                        'regcover': header.get('REGCOVER', coverage),
                        'valid_pixels': n_valid
                    })
                    
                    logger.debug(f"Caricato {os.path.basename(filename)}: {coverage:.2f}% copertura, {n_valid:,} pixel")
                    
            except Exception as e:
                logger.error(f"Errore caricamento {os.path.basename(filename)}: {e}")
                
            pbar.update(1)
    
    if len(data_stack) == 0:
        logger.error("Nessuna immagine valida caricata!")
        print("\n❌ ERRORE: Nessuna immagine valida!")
        return None
    
    # Converti in array numpy
    logger.info("Conversione in array numpy...")
    data_stack = np.array(data_stack, dtype=np.float32)
    weight_stack = np.array(weight_stack, dtype=np.float32)
    
    print(f"\n✓ Caricate {len(data_stack)} immagini valide")
    print(f"📏 Shape stack: {data_stack.shape}")
    print(f"💾 Memoria usata: ~{data_stack.nbytes / 1024**3:.1f} GB")
    
    # Combina le immagini
    logger.info(f"Combinazione con metodo: {combine_function}")
    print(f"\n🔄 Combinazione immagini con {combine_function}...")
    
    if combine_function == 'weighted_mean':
        # Media pesata con sigma clipping
        print("   Metodo: Media pesata con sigma clipping e edge blending")
        final_mosaic = sigma_clip_combine(data_stack, weight_stack, SIGMA_CLIP_THRESHOLD)
        
    elif combine_function == 'simple_mean':
        # Media semplice con pesi
        print("   Metodo: Media pesata semplice con edge blending")
        weight_sum = np.sum(weight_stack, axis=0)
        valid_mask = weight_sum > 0
        
        final_mosaic = np.zeros(final_shape, dtype=np.float32)
        numerator = np.sum(data_stack * weight_stack, axis=0)
        final_mosaic[valid_mask] = numerator[valid_mask] / weight_sum[valid_mask]
        
    elif combine_function == 'median':
        # Mediana (più robusto ma più lento)
        print("   Metodo: Mediana robusta")
        # Usa solo pixel con peso > 0 per la mediana
        masked_data = np.where(weight_stack > 0, data_stack, np.nan)
        final_mosaic = np.nanmedian(masked_data, axis=0)
        final_mosaic = np.nan_to_num(final_mosaic, nan=0.0)
        
    else:
        # Default: weighted mean
        final_mosaic = sigma_clip_combine(data_stack, weight_stack, SIGMA_CLIP_THRESHOLD)
    
    # Cleanup memoria
    del data_stack, weight_stack
    gc.collect()
    
    # Calcola statistiche finali
    valid_pixels_final = np.sum(final_mosaic != 0)
    coverage_final = (valid_pixels_final / final_mosaic.size) * 100
    
    # Update header con metadati migliorati
    ref_header['NCOMBINE'] = (len(metadata_list), 'Number of images combined')
    ref_header['COMBFUNC'] = (combine_function, 'Combination method')
    ref_header['FEATHER'] = (FEATHER_RADIUS, 'Edge feathering radius (pixels)')
    ref_header['SIGCLIP'] = (SIGMA_CLIP_THRESHOLD, 'Sigma clipping threshold')
    ref_header['WEIGHTED'] = (USE_WEIGHTED_STACK, 'Used weighted stacking')
    ref_header['MINCOVER'] = (MIN_COVERAGE, 'Minimum coverage threshold (%)')
    ref_header['DATE'] = datetime.now().isoformat()
    ref_header['CREATOR'] = 'AstroMosaic.py v2.1 Enhanced'
    
    # Aggiungi statistiche finali
    ref_header['TOTPIX'] = final_mosaic.size
    ref_header['VALIDPIX'] = valid_pixels_final
    ref_header['COVERAGE'] = (coverage_final, 'Final mosaic coverage (%)')
    ref_header['NIMGUSED'] = len(metadata_list)
    
    # Aggiungi statistiche per immagine
    total_input_pixels = sum(meta['valid_pixels'] for meta in metadata_list)
    ref_header['INPIXELS'] = (total_input_pixels, 'Total input valid pixels')
    
    # Save mosaic
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(OUTPUT_DIR, f'mosaic_enhanced_{timestamp}.fits')
    logger.info(f"Salvataggio mosaico migliorato: {output_file}")
    
    fits.PrimaryHDU(final_mosaic.astype(np.float32), 
                    header=ref_header).writeto(output_file, overwrite=True)

    # Print summary
    print(f"\n📊 RIEPILOGO INTEGRAZIONE MIGLIORATA:")
    print(f"   Immagini combinate: {len(metadata_list)}")
    print(f"   Metodo: {combine_function}")
    print(f"   Edge feathering: {FEATHER_RADIUS} pixel")
    print(f"   Sigma clipping: {SIGMA_CLIP_THRESHOLD}")
    print(f"   Copertura finale: {coverage_final:.2f}%")
    print(f"   Pixel validi: {valid_pixels_final:,} / {final_mosaic.size:,}")
    print(f"   Pixel input totali: {total_input_pixels:,}")
    print(f"   Compressione: {final_mosaic.size / total_input_pixels:.1f}x")

    logger.info("Stacking migliorato completato")
    return output_file

if __name__ == "__main__":
    start_time = time.time()
    
    print("\n🔧 METODI DI COMBINAZIONE DISPONIBILI:")
    print("1. weighted_mean (consigliato) - Media pesata con sigma clipping e edge blending")
    print("2. simple_mean - Media pesata semplice con edge blending")
    print("3. median - Mediana robusta (più lento ma elimina outlier)")
    
    try:
        choice = input("\nScegli metodo (1-3, default=1): ").strip()
        if choice == "2":
            method = "simple_mean"
        elif choice == "3":
            method = "median"
        else:
            method = "weighted_mean"
    except:
        method = "weighted_mean"
    
    print(f"✓ Usando metodo: {method}")
    
    output = stack_mosaic_enhanced(combine_function=method)
    
    if output:
        print(f"\n✅ MOSAICO MIGLIORATO COMPLETATO")
        print(f"📁 Salvato in: {output}")
        print(f"\n💡 MIGLIORAMENTI APPLICATI:")
        print(f"   ✓ Bordi sfumati con feathering di {FEATHER_RADIUS} pixel")
        print(f"   ✓ Sigma clipping per rimuovere outlier")
        print(f"   ✓ Pesi basati su distanza dai bordi")
        print(f"   ✓ Gestione intelligente copertura bassa")
        print(f"   ✓ Normalizzazione pesi per uniformità")
    else:
        print(f"\n❌ ERRORE nella creazione del mosaico")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Tempo totale: {elapsed:.1f} secondi")