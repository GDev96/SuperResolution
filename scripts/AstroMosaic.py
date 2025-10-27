from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
import numpy as np
import glob
import os
import gc
import time
import sys
import logging
from datetime import datetime

# --- CONFIGURAZIONE PERCORSI ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_register_4')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'img_preprocessed')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# --- CONFIGURAZIONE LOGGING ---
def setup_logging():
    """
    Configura il sistema di logging per salvare i log nella cartella logs.
    """
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

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, 
                       length=50, fill='█', elapsed_time=None):
    """
    Stampa una progress bar nella console
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    time_str = ""
    if elapsed_time is not None:
        if elapsed_time < 60:
            time_str = f" | Tempo: {elapsed_time:.1f}s"
        else:
            mins = int(elapsed_time // 60)
            secs = elapsed_time % 60
            time_str = f" | Tempo: {mins}m {secs:.0f}s"
        
        if iteration > 0:
            avg_time = elapsed_time / iteration
            remaining = avg_time * (total - iteration)
            if remaining < 60:
                time_str += f" | Rimanente: {remaining:.1f}s"
            else:
                mins = int(remaining // 60)
                secs = remaining % 60
                time_str += f" | Rimanente: {mins}m {secs:.0f}s"
    
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}{time_str}')
    sys.stdout.flush()
    
    if iteration == total:
        print()

def create_feather_mask(shape, feather_pixels=50):
    """Crea una maschera con bordi sfumati per il blending"""
    height, width = shape
    mask = np.ones(shape, dtype=np.float32)
    for i in range(feather_pixels):
        weight = (i + 1) / feather_pixels
        mask[i, :] *= weight
        mask[-(i+1), :] *= weight
        mask[:, i] *= weight
        mask[:, -(i+1)] *= weight
    return mask

def create_mosaic(input_pattern=None, output_file=None, combine_function='mean', 
                  feather_pixels=50, downsample_factor=1, max_dimension=None):
    """
    Crea un mosaico da immagini FITS con coordinate WCS
    
    Parametri:
    - input_pattern: pattern per trovare i file (opzionale, usa INPUT_DIR se None)
    - output_file: percorso del file di output (opzionale, usa OUTPUT_DIR se None)
    - combine_function: metodo ('mean', 'sum', 'min', 'max', 'first', 'last')
    - feather_pixels: pixel per sfumamento bordi
    - downsample_factor: riduce dimensioni di questo fattore (2 = metà dimensioni)
    - max_dimension: dimensione massima in pixel (es. 8000)
    """
    logger = setup_logging()
    
    # Configura percorsi se non specificati
    if input_pattern is None:
        input_pattern = os.path.join(INPUT_DIR, "*.fits")
    if output_file is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(OUTPUT_DIR, f"mosaic_{timestamp}.fits")
    
    logger.info("=" * 70)
    logger.info("HST MOSAIC CREATOR - OTTIMIZZATO")
    logger.info("=" * 70)
    logger.info(f"Input pattern: {input_pattern}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Combine method: {combine_function}")
    logger.info(f"Feather pixels: {feather_pixels}")
    logger.info(f"Downsample factor: {downsample_factor}")
    logger.info(f"Max dimension: {max_dimension}")

    print("=" * 70)
    print("HST MOSAIC CREATOR - OTTIMIZZATO".center(70))
    print("=" * 70)

    # Trova i file
    print("\n[1/5] Ricerca file...")
    start_time = time.time()
    files = sorted(glob.glob(input_pattern))
    if len(files) == 0:
        error_msg = f"Nessun file trovato con pattern '{input_pattern}'"
        print(f"\n❌ ERRORE: {error_msg}")
        logger.error(error_msg)
        return None

    print(f"✓ Trovati {len(files)} file in {time.time()-start_time:.2f}s")
    logger.info(f"Trovati {len(files)} file")
    
    for i, f in enumerate(files[:5]):
        print(f"  {i+1:2d}. {os.path.basename(f)}")
    if len(files) > 5:
        print(f"  ... e altri {len(files)-5} file")

    # Carica le immagini
    print(f"\n[2/5] Caricamento {len(files)} immagini...")
    hdus = []
    failed_files = []
    load_start = time.time()
    
    for i, filepath in enumerate(files):
        try:
            hdu = fits.open(filepath)[0]
            wcs = WCS(hdu.header)
            if not wcs.has_celestial:
                failed_files.append(filepath)
                logger.warning(f"File senza coordinate celestiali: {os.path.basename(filepath)}")
                continue
            hdus.append(hdu)
            
            # Progress bar
            print_progress_bar(i + 1, len(files), 
                             prefix='Caricamento:', 
                             suffix='Completato',
                             elapsed_time=time.time() - load_start)
        except Exception as e:
            failed_files.append(filepath)
            logger.error(f"Errore caricamento {os.path.basename(filepath)}: {e}")
            continue

    if len(hdus) == 0:
        error_msg = "Nessuna immagine valida caricata!"
        print(f"\n❌ ERRORE: {error_msg}")
        logger.error(error_msg)
        return None

    print(f"✓ {len(hdus)}/{len(files)} immagini caricate con successo")
    logger.info(f"Immagini caricate: {len(hdus)}/{len(files)}")
    
    if failed_files:
        print(f"⚠️  {len(failed_files)} file saltati")
        logger.warning(f"File saltati: {len(failed_files)}")

    # Calcola WCS ottimale
    print(f"\n[3/5] Calcolo WCS ottimale...")
    wcs_start = time.time()
    
    all_wcs = []
    for i, hdu in enumerate(hdus):
        all_wcs.append(WCS(hdu.header).celestial)
        if (i + 1) % 10 == 0 or i == len(hdus) - 1:
            print_progress_bar(i + 1, len(hdus), 
                             prefix='Analisi WCS:', 
                             suffix='Completato',
                             elapsed_time=time.time() - wcs_start)
    
    reference_wcs, shape_out = find_optimal_celestial_wcs(all_wcs)
    
    original_shape = shape_out
    print(f"\n✓ WCS calcolato in {time.time()-wcs_start:.2f}s")
    print(f"  Dimensioni originali: {shape_out[0]}x{shape_out[1]} pixel")
    print(f"  Dimensioni stimate: {shape_out[0]*shape_out[1]/1e6:.1f} megapixel")
    print(f"  Memoria stimata: {shape_out[0]*shape_out[1]*4/1e9:.2f} GB per immagine")
    
    logger.info(f"WCS calcolato - Dimensioni: {shape_out[0]}x{shape_out[1]}")

    # Applica downsampling o max_dimension
    if max_dimension is not None:
        max_side = max(shape_out)
        if max_side > max_dimension:
            downsample_factor = max_side / max_dimension
            print(f"\n⚠️  Dimensioni troppo grandi! Applicato downsample automatico: {downsample_factor:.2f}x")
            logger.info(f"Downsample automatico applicato: {downsample_factor:.2f}x")
    
    if downsample_factor > 1:
        shape_out = (int(shape_out[0] / downsample_factor), 
                     int(shape_out[1] / downsample_factor))
        # Aggiusta il WCS per il downsampling
        reference_wcs.wcs.cdelt *= downsample_factor
        reference_wcs.wcs.crpix /= downsample_factor
        print(f"  Dimensioni ridotte: {shape_out[0]}x{shape_out[1]} pixel")
        print(f"  Fattore riduzione: {downsample_factor:.2f}x")

    print(f"\n  Dimensioni finali output: {shape_out[0]}x{shape_out[1]} pixel")
    print(f"  Memoria totale richiesta: ~{shape_out[0]*shape_out[1]*4*2/1e9:.2f} GB")
    
    logger.info(f"Dimensioni finali: {shape_out[0]}x{shape_out[1]}")

    # Crea il mosaico
    print(f"\n[4/5] Creazione mosaico")
    print(f"  Metodo: {combine_function}")
    print(f"  Feather: {feather_pixels} pixel")
    print(f"  Immagini da elaborare: {len(hdus)}")

    try:
        # Inizializza accumulatori
        sum_images = np.zeros(shape_out, dtype=np.float64)
        sum_masks = np.zeros(shape_out, dtype=np.float64)
        count = 0
        
        mosaic_start = time.time()

        # Processa un'immagine alla volta
        for i, hdu in enumerate(hdus):
            try:
                # Reproietta l'immagine
                reprojected_data, _ = reproject_interp(hdu, reference_wcs, shape_out=shape_out)
                
                # Crea maschera e pulisci dati
                mask = create_feather_mask(reprojected_data.shape, feather_pixels=feather_pixels)
                data_clean = np.nan_to_num(reprojected_data, nan=0.0)
                
                # Accumula
                sum_images += data_clean * mask
                sum_masks += mask
                count += 1
                
                # Libera memoria
                del reprojected_data, mask, data_clean
                gc.collect()
                
                # Progress bar
                filename = getattr(hdu, '_file', None)
                if filename and hasattr(filename, 'name'):
                    display_name = os.path.basename(filename.name)[:30]
                else:
                    display_name = f"Image {i+1}"
                
                print_progress_bar(i + 1, len(hdus), 
                                 prefix='Elaborazione:', 
                                 suffix=display_name,
                                 elapsed_time=time.time() - mosaic_start)
                
            except Exception as e:
                filename = getattr(hdu, '_file', None)
                if filename and hasattr(filename, 'name'):
                    display_name = os.path.basename(filename.name)
                else:
                    display_name = f"Image {i+1}"
                print(f"\n    ⚠️  Saltata {display_name}: {e}")
                logger.error(f"Errore elaborazione {display_name}: {e}")
                continue

        if count == 0:
            error_msg = "Nessuna immagine elaborata con successo!"
            print(f"\n❌ ERRORE: {error_msg}")
            logger.error(error_msg)
            return None

        print(f"\n✓ {count} immagini elaborate in {time.time()-mosaic_start:.2f}s")
        logger.info(f"Immagini elaborate: {count}")
        
        # Normalizzazione
        print("\n[5/5] Normalizzazione e salvataggio...")
        norm_start = time.time()
        
        print("  ⏳ Normalizzazione dati...", end='', flush=True)
        mosaic = np.divide(sum_images, sum_masks, 
                          out=np.zeros_like(sum_images), 
                          where=sum_masks != 0)
        print(f" ✓ ({time.time()-norm_start:.2f}s)")

        # Crea directory di output se non esiste
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Crea header
        print("  ⏳ Creazione header FITS...", end='', flush=True)
        header = reference_wcs.to_header()
        header['COMMENT'] = 'Mosaic created from registered images'
        header['NIMAGES'] = count
        header['COMBFUNC'] = combine_function
        header['FEATHERPX'] = feather_pixels
        header['DOWNSAMP'] = downsample_factor
        header['ORIGSHP1'] = original_shape[0]
        header['ORIGSHP2'] = original_shape[1]
        header['CREATOR'] = 'AstroMosaic - SuperResolution Pipeline'
        header['DATE'] = datetime.now().isoformat()
        print(" ✓")

        # Salva file
        print(f"  ⏳ Salvataggio file ({mosaic.nbytes/1e9:.2f} GB)...", end='', flush=True)
        save_start = time.time()
        hdu_out = fits.PrimaryHDU(data=mosaic.astype(np.float32), header=header)
        hdu_out.writeto(output_file, overwrite=True)
        print(f" ✓ ({time.time()-save_start:.2f}s)")
        
        # Statistiche finali
        valid_pixels = np.sum(mosaic != 0)
        print(f"\n✓ Normalizzazione completata in {time.time()-norm_start:.2f}s")
        print(f"\nStatistiche finali:")
        print(f"  File output: {output_file}")
        print(f"  Dimensioni: {mosaic.shape[1]}x{mosaic.shape[0]} pixel")
        print(f"  Pixel totali: {mosaic.size:,}")
        print(f"  Pixel validi: {valid_pixels:,}")
        print(f"  Copertura: {valid_pixels/mosaic.size*100:.1f}%")
        if valid_pixels > 0:
            print(f"  Min/Max valori: {np.min(mosaic[mosaic!=0]):.2e} / {np.max(mosaic):.2e}")
        print(f"  Dimensione file: {mosaic.nbytes/1e6:.2f} MB")

        logger.info(f"Mosaico completato - Dimensioni: {mosaic.shape}")
        logger.info(f"File salvato: {output_file}")
        logger.info(f"Pixel validi: {valid_pixels:,}/{mosaic.size:,} ({valid_pixels/mosaic.size*100:.1f}%)")

        # Chiudi i file
        for hdu in hdus:
            if hasattr(hdu, '_file') and hdu._file is not None:
                hdu._file.close()

        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("COMPLETATO CON SUCCESSO!".center(70))
        print(f"Tempo totale: {total_time/60:.1f} minuti ({total_time:.0f} secondi)".center(70))
        print("=" * 70)

        logger.info(f"Tempo totale di esecuzione: {total_time:.2f} secondi")
        logger.info("ELABORAZIONE TERMINATA CON SUCCESSO")

        return output_file

    except Exception as e:
        error_msg = f"Errore durante la creazione del mosaico: {e}"
        print(f"\n\n❌ ERRORE: {error_msg}")
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        logger.error(traceback.format_exc())
        return None


def diagnose_fits_files(input_pattern=None):
    """Diagnostica le immagini FITS"""
    if input_pattern is None:
        input_pattern = os.path.join(INPUT_DIR, "*.fits")
    
    print("=" * 70)
    print("DIAGNOSTICA FILE FITS".center(70))
    print("=" * 70)
    
    files = sorted(glob.glob(input_pattern))
    print(f"\nTrovati {len(files)} file\n")
    
    start_time = time.time()
    for idx, f in enumerate(files):
        try:
            hdul = fits.open(f)
            chosen = None
            for h in hdul:
                if h.data is None:
                    continue
                w = WCS(h.header)
                if w.has_celestial:
                    chosen = h
                    break
            
            if chosen is None:
                print(f"{os.path.basename(f):40s} → ❌ NO HDU con dati + WCS")
                hdul.close()
                continue
            
            shape = chosen.data.shape
            crval = chosen.header.get('CRVAL1'), chosen.header.get('CRVAL2')
            crpix = chosen.header.get('CRPIX1'), chosen.header.get('CRPIX2')
            print(f"{os.path.basename(f):40s} shape={shape} CRVAL={crval} CRPIX={crpix}")
            hdul.close()
            
        except Exception as e:
            print(f"{os.path.basename(f):40s} → ❌ ERRORE: {e}")
        
        # Progress bar
        print_progress_bar(idx + 1, len(files), 
                         prefix='\nProgresso:', 
                         suffix='Completato',
                         elapsed_time=time.time() - start_time)


if __name__ == "__main__":
    # ===== CONFIGURAZIONE =====
    
    # I percorsi vengono ora configurati automaticamente dalle costanti in cima al file
    # ma puoi sovrascriverli qui se necessario
    
    # input_pattern = os.path.join(INPUT_DIR, "*.fits")  # Usa default
    # output_file = None  # Genera automaticamente con timestamp
    
    # Parametri del mosaico
    combine_method = 'mean'
    feather_pixels = 50
    
    # OPZIONI PER GESTIRE GRANDI DIMENSIONI:
    
    # OPZIONE 1: Ridimensiona automaticamente se troppo grande
    max_dimension = 8000  # Dimensione massima in pixel (None per disabilitare)
    
    # OPZIONE 2: Downsample fisso (2 = metà dimensioni, 4 = un quarto, ecc.)
    downsample_factor = 1  # 1 = nessun downsample
    
    # SUGGERIMENTI:
    # - Per molte immagini, prova max_dimension=8000 o 10000
    # - Se hai ancora errori di memoria, riduci a 6000 o 4000
    # - Oppure usa downsample_factor=2 per dimezzare le dimensioni
    
    print("CONFIGURAZIONE:")
    print(f"  Input Directory:  {INPUT_DIR}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Log Directory:    {LOG_DIR}")
    print(f"  Metodo:           {combine_method}")
    print(f"  Max dimension:    {max_dimension if max_dimension else 'Nessun limite'}")
    print(f"  Downsample:       {downsample_factor}x\n")
    
    # Decommentare per diagnostica
    # diagnose_fits_files()
    
    # Creazione mosaico
    result = create_mosaic(
        combine_function=combine_method,
        feather_pixels=feather_pixels,
        downsample_factor=downsample_factor,
        max_dimension=max_dimension
    )
    
    if result:
        print(f"\n🎉 Mosaico completato: {result}")