"""
STEP 2: CREAZIONE MOSAICO (UNIFICATO) - ORDINE FINALE
Permette di scegliere:
1. Prima la Cartella Target (su quali dati lavorare).
2. Poi la Modalità Mosaico (WCS o CROP+STACK).

INPUT: Cartelle '3_registered_native/hubble' e '3_registered_native/observatory'
OUTPUT: 
  - File '5_mosaics/final_mosaic_wcs.fits' o 'final_mosaic_stack.fits'
  - Cartelle '4_cropped/hubble' e '4_cropped/observatory' (solo per metodo CROP)
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
from astropy.wcs import WCS
from tqdm import tqdm
import warnings
import subprocess

# Rimuove alcuni warning di Astropy che possono essere fastidiosi
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE PATH DINAMICI (UNIVERSALI)
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
    log_filename = LOG_DIR_ROOT / f'mosaic_full_{timestamp}.log'
    
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

def select_target_directory(logger):
    """Mostra un menu per selezionare una o TUTTE le cartelle target."""
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET".center(70))
    print("📂"*35)
    print(f"\nScansione sottocartelle in: {ROOT_DATA_DIR}")

    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    except Exception as e:
        logger.error(f"ERRORE: Impossibile leggere la cartella {ROOT_DATA_DIR}: {e}")
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

def select_mosaic_mode(target_name):
    """Mostra un menu semplice per selezionare la modalità di creazione del mosaico."""
    print("\n--------------------------------------------------")
    print(f"MODALITÀ MOSAICO per: {target_name}")
    print("--------------------------------------------------")
    print("Scegli il metodo di mosaico da eseguire:")
    print("   1: HUBBLE WCS (Allineamento basato su coordinate, gestisce dimensioni diverse)")
    print("   2: OSSERVATORY CROP + STACK (Ritaglia alla dimensione minima e fa la media)")
    print("   3: ENTRAMBI i metodi")

    while True:
        print("\n" + "─"*50)
        try:
            choice_str = input(f"👉 Seleziona un numero (1-3) o 'q' per uscire: ").strip()

            if choice_str.lower() == 'q':
                return [] 

            choice = int(choice_str)

            if choice == 1:
                print("Metodo WCS selezionato.")
                return ['wcs']
            elif choice == 2:
                print("Metodo CROP + STACK selezionato.")
                return ['crop']
            elif choice == 3:
                print("Entrambi i metodi selezionati.")
                return ['wcs', 'crop']
            else:
                print("❌ Scelta non valida.")
        except ValueError:
            print("❌ Input non valido.")

def ask_continue_to_next_step(successful_targets, logger):
    """Chiede se proseguire con il prossimo script."""
    print("\n" + "="*70)
    print("🎯 ELABORAZIONE MOSAICO COMPLETATA!")
    print("="*70)
    print("\n📋 OPZIONI:")
    print("   1️⃣  Continua con Step 3 (Analisi Patch - step3_analizzapatch.py)")
    print("   2️⃣  Termina qui")
    
    next_script_name = 'Dataset_step3_analizzapatch.py'
    
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
# FUNZIONI - MOSAICO CON WCS (Da Dataset_step2_mosaico_42.py)
# ============================================================================

def calculate_common_grid(image_infos):
    """Calcola una griglia comune che copre tutte le immagini usando WCS."""
    print("\n🔍 Calcolo griglia comune...")
    
    ra_min = float('inf')
    ra_max = float('-inf')
    dec_min = float('inf')
    dec_max = float('-inf')
    
    pixel_scales = []
    
    for info in image_infos:
        wcs = info['wcs']
        shape = info['shape']
        ny, nx = shape
        
        corners_pix = np.array([
            [0, 0], [nx-1, 0], [0, ny-1], [nx-1, ny-1]
        ])
        
        try:
            corners_world = wcs.pixel_to_world(corners_pix[:, 0], corners_pix[:, 1])
            
            for coord in corners_world:
                ra = coord.ra.deg
                dec = coord.dec.deg
                ra_min = min(ra_min, ra)
                ra_max = max(ra_max, ra)
                dec_min = min(dec_min, dec)
                dec_max = max(dec_max, dec)
        except:
            continue
        
        try:
            if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
                cd = wcs.wcs.cd
                pixel_scale = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
            else:
                pixel_scale = abs(wcs.wcs.cdelt[0])
            pixel_scales.append(pixel_scale)
        except:
            continue
    
    if ra_max - ra_min > 180:
        print("   ⚠️  Rilevato wraparound RA vicino a 0°/360°")
    
    ra_center = (ra_min + ra_max) / 2.0
    dec_center = (dec_min + dec_max) / 2.0
    
    ra_span = ra_max - ra_min
    dec_span = dec_max - dec_min
    
    target_pixel_scale = min(pixel_scales) if pixel_scales else 0.0001  # deg/px
    
    nx_out = int(np.ceil(abs(ra_span / np.cos(np.radians(dec_center))) / target_pixel_scale))
    ny_out = int(np.ceil(abs(dec_span) / target_pixel_scale))
    
    max_dim = max(nx_out, ny_out)
    nx_out = max_dim
    ny_out = max_dim
    
    MAX_DIM = 20000
    if max_dim > MAX_DIM:
        print(f"   ⚠️  Limitazione dimensioni: {max_dim} -> {MAX_DIM}")
        nx_out = MAX_DIM
        ny_out = MAX_DIM
        target_pixel_scale *= (max_dim / MAX_DIM)
    
    print(f"   Centro: RA={ra_center:.4f}°, DEC={dec_center:.4f}°")
    print(f"   Span: RA={ra_span:.4f}° ({ra_span*60:.1f}'), DEC={dec_span:.4f}° ({dec_span*60:.1f}')")
    print(f"   Pixel scale: {target_pixel_scale*3600:.4f}\"/px")
    print(f"   Dimensioni output: {nx_out} x {ny_out} pixel ⬜ (QUADRATA)")
    
    output_wcs = WCS(naxis=2)
    output_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    output_wcs.wcs.crval = [ra_center, dec_center]
    output_wcs.wcs.crpix = [nx_out / 2.0, ny_out / 2.0]
    output_wcs.wcs.cdelt = [-target_pixel_scale, target_pixel_scale]
    output_wcs.wcs.radesys = 'ICRS'
    output_wcs.wcs.equinox = 2000.0
    
    return output_wcs, (ny_out, nx_out)


def create_mosaic_wcs(base_dir, logger):
    """Crea il mosaico usando WCS per allineare immagini di dimensioni diverse."""
    print("\n" + "🖼️ "*35)
    print(f"MOSAICO WCS: {base_dir.name}".center(70))
    print("🖼️ "*35)
    
    input_dirs = [
        base_dir / '3_registered_native' / 'hubble',
        base_dir / '3_registered_native' / 'observatory'
    ]
    
    mosaic_output_dir = base_dir / '5_mosaics'
    mosaic_output_file = mosaic_output_dir / 'final_mosaic_wcs.fits'
    
    mosaic_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📂 Caricamento immagini...")
    all_files = []
    for d in input_dirs:
        if d.exists():
            all_files.extend(list(d.glob('*.fits')) + list(d.glob('*.fit')))
    
    if not all_files:
        print(f"\n❌ ERRORE: Nessun file FITS trovato in {base_dir.name}/3_registered_native.")
        logger.error(f"WCS: Nessun file FITS trovato per {base_dir.name}")
        return False
    
    print(f"✅ Trovati {len(all_files)} file FITS")
    
    image_infos = []
    for filepath in tqdm(all_files, desc="Lettura WCS", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                data = None
                header = None
                for hdu in hdul:
                    if hdu.data is not None and hdu.data.size > 0:
                        data = hdu.data
                        header = hdu.header
                        break
                
                if data is None:
                    continue
                
                if len(data.shape) == 3:
                    data = data[0]
                
                wcs = WCS(header)
                if not wcs.has_celestial:
                    logger.warning(f"WCS: {filepath.name}: WCS non valido, saltato")
                    continue
                
                image_infos.append({
                    'file': filepath,
                    'data': data,
                    'header': header,
                    'wcs': wcs,
                    'shape': data.shape
                })
                
        except Exception as e:
            logger.error(f"WCS: Errore leggendo {filepath.name}: {e}")
            continue
    
    if not image_infos:
        print(f"\n❌ ERRORE: Nessuna immagine con WCS valido trovata.")
        logger.error(f"WCS: Nessuna immagine con WCS valido per {base_dir.name}")
        return False
    
    print(f"✅ {len(image_infos)} immagini con WCS valido")
    
    try:
        output_wcs, output_shape = calculate_common_grid(image_infos)
    except Exception as e:
        print(f"\n❌ ERRORE: Fallimento calcolo griglia comune: {e}")
        logger.error(f"WCS: Fallimento calcolo griglia comune per {base_dir.name}: {e}")
        return False
    
    print("\n🔄 Creazione mosaico...")
    mosaic_data = np.zeros(output_shape, dtype=np.float64)
    mosaic_weight = np.zeros(output_shape, dtype=np.float64)
    
    for info in tqdm(image_infos, desc="Proiezione", unit="immagine"):
        try:
            data = info['data']
            wcs_in = info['wcs']
            shape_in = info['shape']
            
            ny_in, nx_in = shape_in
            
            y_in, x_in = np.mgrid[0:ny_in, 0:nx_in]
            
            coords_world = wcs_in.pixel_to_world(x_in.ravel(), y_in.ravel())
            
            x_out, y_out = output_wcs.world_to_pixel(coords_world)
            
            x_out = np.round(x_out).astype(int)
            y_out = np.round(y_out).astype(int)
            
            x_out = x_out.reshape(shape_in)
            y_out = y_out.reshape(shape_in)
            
            valid = (x_out >= 0) & (x_out < output_shape[1]) & \
                    (y_out >= 0) & (y_out < output_shape[0]) & \
                    ~np.isnan(data)
            
            y_valid = y_out[valid]
            x_valid = x_out[valid]
            data_valid = data[valid]
            
            np.add.at(mosaic_data, (y_valid, x_valid), data_valid)
            np.add.at(mosaic_weight, (y_valid, x_valid), 1.0)
            
        except Exception as e:
            logger.error(f"WCS: Errore proiettando {info['file'].name}: {e}")
            continue
    
    print("\n🧮 Calcolo della media finale...")
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mosaic_final = np.where(mosaic_weight > 0, 
                                mosaic_data / mosaic_weight, 
                                np.nan)
    
    valid_pixels = np.sum(~np.isnan(mosaic_final))
    total_pixels = mosaic_final.size
    coverage = (valid_pixels / total_pixels) * 100
    
    print(f"📊 Statistiche mosaico WCS:")
    print(f"   Dimensioni: {output_shape[1]} x {output_shape[0]} pixel")
    print(f"   Pixel validi: {valid_pixels}")
    print(f"   Copertura: {coverage:.1f}%")
    print(f"   Immagini combinate: {len(image_infos)}")
    logger.info(f"WCS: Completato {base_dir.name} - Dim: {output_shape[1]}x{output_shape[0]}, Copertura: {coverage:.1f}%")
    
    print(f"\n💾 Salvataggio mosaico WCS...")
    
    output_header = output_wcs.to_header()
    output_header['HISTORY'] = 'Mosaico creato usando WCS'
    output_header['METHOD'] = ('WCS_ALIGN_STACK', 'Metodo di allineamento e combinazione')
    output_header['NCOMBINE'] = (len(image_infos), 'Numero di immagini combinate')
    output_header['COVERAGE'] = (coverage, 'Percentuale copertura')
    output_header['NPIXVAL'] = (valid_pixels, 'Numero pixel validi')
    
    try:
        fits.PrimaryHDU(data=mosaic_final.astype(np.float32), 
                       header=output_header).writeto(mosaic_output_file, overwrite=True)
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile salvare {mosaic_output_file.name}: {e}")
        logger.error(f"WCS: Impossibile salvare {mosaic_output_file.name}: {e}")
        return False
    
    print(f"\n✅ MOSAICO WCS COMPLETATO: {mosaic_output_file}")
    return True

# ============================================================================
# FUNZIONI - RITAGLIO E MOSAICO (Da Dataset_step2_mosaico_1_33.py)
# ============================================================================

def find_smallest_dimensions(all_files, logger):
    """Trova le dimensioni dell'immagine più piccola tra tutti i file."""
    print("\n🔍 Ricerca dimensioni minime...")
    
    min_height = float('inf')
    min_width = float('inf')
    smallest_file = None
    
    for filepath in tqdm(all_files, desc="Scansione", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                data = None
                for hdu in hdul:
                    if hdu.data is not None and hdu.data.size > 0:
                        data = hdu.data
                        break

                if data is None:
                    continue

                if len(data.shape) == 3:
                    height, width = data[0].shape
                else:
                    height, width = data.shape
                
                if height < min_height or width < min_width:
                    if height < min_height: min_height = height
                    if width < min_width: min_width = width
                    smallest_file = filepath
                    
        except Exception as e:
            logger.warning(f"CROP: Impossibile leggere {filepath}: {e}")
            continue
    
    if min_height == float('inf') or min_width == float('inf'):
        print("\n❌ ERRORE: Impossibile determinare le dimensioni minime.")
        return None, None
        
    print(f"\n✅ Dimensioni minime trovate: {min_width} x {min_height} pixel")
    if smallest_file:
        print(f"   File più piccolo: {Path(smallest_file).name}")
    return min_height, min_width


def crop_image(input_path, output_path, target_height, target_width, logger):
    """Ritaglia un'immagine FITS alle dimensioni target (centrato)."""
    try:
        with fits.open(input_path) as hdul:
            data = None
            header = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.size > 0:
                    data = hdu.data
                    header = hdu.header.copy()
                    break

            if data is None:
                return False

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
        logger.error(f"CROP: ERRORE nel ritaglio di {input_path.name}: {e}")
        return False


def crop_all_images_for_target(base_dir, logger):
    """Esegue il ritaglio di tutte le immagini per un target specifico."""
    print("\n" + "✂️ "*35)
    print(f"RITAGLIO (CROP): {base_dir.name}".center(70))
    print("✂️ "*35)
    
    input_dirs = {
        'hubble': base_dir / '3_registered_native' / 'hubble',
        'observatory': base_dir / '3_registered_native' / 'observatory'
    }
    
    output_dir_base = base_dir / '4_cropped'
    output_dirs = {
        'hubble': output_dir_base / 'hubble',
        'observatory': output_dir_base / 'observatory'
    }
    
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    all_files = []
    file_mapping = {}
    
    for category, input_dir in input_dirs.items():
        files = list(input_dir.glob('*.fits')) + list(input_dir.glob('*.fit'))
        for f in files:
            all_files.append(f)
            file_mapping[f] = category
        print(f"   {category}: {len(files)} file trovati")
    
    if not all_files:
        print(f"\n❌ ERRORE: Nessun file in {base_dir.name}/3_registered_native.")
        logger.error(f"CROP: Nessun file FITS trovato per {base_dir.name}")
        return False
    
    min_height, min_width = find_smallest_dimensions(all_files, logger)
    if min_height is None:
        return False
        
    print(f"\n📐 Target: {min_width} x {min_height} pixel")
    logger.info(f"CROP: Dimensioni target per {base_dir.name}: {min_width}x{min_height}")
    
    success_count = 0
    for filepath in tqdm(all_files, desc="Ritaglio", unit="file"):
        category = file_mapping[filepath]
        if crop_image(filepath, output_dirs[category] / filepath.name, min_height, min_width, logger):
            success_count += 1
            
    if success_count == 0:
        print("\n❌ ERRORE: Nessuna immagine ritagliata con successo.")
        logger.error(f"CROP: Nessuna immagine ritagliata con successo per {base_dir.name}")
        return False

    print(f"\n✅ Ritaglio completato: {success_count} immagini salvate in 4_cropped.")
    return True


def create_mosaic_for_target(base_dir, logger):
    """Crea il mosaico (stacking) per un target specifico dalle immagini ritagliate."""
    print("\n" + "🖼️ "*35)
    print(f"MOSAICO (STACK): {base_dir.name}".center(70))
    print("🖼️ "*35)
    
    output_dir_base = base_dir / '4_cropped'
    all_files = list((output_dir_base / 'hubble').glob('*.fits')) + \
                list((output_dir_base / 'observatory').glob('*.fits'))
        
    if not all_files:
        print(f"\n❌ ERRORE: Nessun file FITS ritagliato trovato in 4_cropped.")
        logger.error(f"STACK: Nessun file FITS ritagliato trovato per {base_dir.name}")
        return False
    
    try:
        with fits.open(all_files[0]) as hdul:
            data = None
            header = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.size > 0:
                    data = hdu.data
                    header = hdu.header.copy()
                    break
            
            if data is None:
                raise ValueError("Nessun dato valido nel primo file.")
                
            template_header = header
            shape = data.shape if len(data.shape) == 2 else data[0].shape
            
    except Exception as e:
        print(f"❌ Errore lettura primo file ritagliato: {e}")
        logger.error(f"STACK: Errore lettura primo file ritagliato per {base_dir.name}: {e}")
        return False
        
    total_flux = np.zeros(shape, dtype=np.float64)
    n_pixels = np.zeros(shape, dtype=np.int32)
    
    print(f"\n📐 Dimensione mosaico: {shape[1]} x {shape[0]} pixel")
    print("🔄 Combinazione immagini...")
    
    for filepath in tqdm(all_files, desc="Stacking", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                d = hdul[0].data
                if len(d.shape) == 3: d = d[0]
                if d.shape != shape: continue
                
                valid = ~np.isnan(d)
                d_clean = np.nan_to_num(d, nan=0.0)
                
                total_flux += d_clean
                n_pixels[valid] += 1
        except Exception as e:
            logger.warning(f"STACK: Errore stacking {filepath.name}: {e}")
            continue
            
    print("\n🧮 Calcolo media...")
    
    mosaic_data = np.full(shape, np.nan, dtype=np.float32)
    valid_stack = n_pixels > 0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mosaic_data[valid_stack] = (total_flux[valid_stack] / n_pixels[valid_stack]).astype(np.float32)
    
    valid_pixels = np.sum(valid_stack)
    total_pixels = mosaic_data.size
    coverage = (valid_pixels / total_pixels) * 100
    
    print(f"📊 Statistiche mosaico STACK:")
    print(f"   Pixel validi: {valid_pixels}")
    print(f"   Copertura: {coverage:.1f}%")
    print(f"   Immagini combinate: {len(all_files)}")
    logger.info(f"STACK: Completato {base_dir.name} - Dim: {shape[1]}x{shape[0]}, Copertura: {coverage:.1f}%")
    
    mosaic_out = base_dir / '5_mosaics'
    mosaic_out.mkdir(parents=True, exist_ok=True)
    final_path = mosaic_out / 'final_mosaic_stack.fits'
    
    template_header['HISTORY'] = 'Mosaic created by Crop and Stack'
    template_header['METHOD'] = ('CROP_STACK_AVG', 'Metodo di allineamento e combinazione')
    template_header['NCOMBINE'] = (len(all_files), 'Numero di immagini combinate')
    template_header['COVERAGE'] = (coverage, 'Percentuale copertura')
    template_header['NPIXVAL'] = (valid_pixels, 'Numero pixel validi')
    
    try:
        fits.PrimaryHDU(data=mosaic_data, header=template_header).writeto(final_path, overwrite=True)
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile salvare {final_path.name}: {e}")
        logger.error(f"STACK: Impossibile salvare {final_path.name}: {e}")
        return False
    
    print(f"\n✅ MOSAICO STACK SALVATO: {final_path}")
    return True

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Funzione principale.
    
    Sequenza:
    1. Setup Logging
    2. Selezione Cartella Target (Batch/Singolo)
    3. Selezione Modalità Mosaico (WCS/CROP)
    4. Esecuzione del processo di mosaico
    5. Riepilogo e transizione allo Step 3/5
    """
    logger = setup_logging()
    
    # 1. Selezione Target (SU QUALI DATI FARLO)
    target_dirs = select_target_directory(logger)
    if not target_dirs:
        logger.info("Uscita su richiesta utente (selezione target).")
        return

    # 2. Selezione Modalità Mosaico (COSA FARE)
    # Mostra la modalità una sola volta per tutti i target selezionati.
    target_name_for_menu = target_dirs[0].name if len(target_dirs) == 1 else "TUTTI I TARGET"
    
    mosaic_modes = select_mosaic_mode(target_name_for_menu)
    if not mosaic_modes:
        logger.info("Uscita su richiesta utente (selezione modalità mosaico).")
        return
    
    logger.info(f"Inizio batch su {len(target_dirs)} target con modalità: {', '.join(mosaic_modes)}")
    
    print("\n" + "="*70)
    print("PIPELINE: CREAZIONE MOSAICO (BATCH)".center(70))
    print("="*70)
    
    start_time_total = time.time()
    successful_targets = []
    failed_targets = []
    
    # 3. Esecuzione del processo per ogni target
    for base_dir in target_dirs:
        print("\n" + "🚀"*35)
        print(f"ELABORAZIONE TARGET: {base_dir.name}".center(70))
        print("🚀"*35)
        
        target_success = True
        
        # Esegue MOSAICO WCS
        if 'wcs' in mosaic_modes:
            if not create_mosaic_wcs(base_dir, logger):
                target_success = False
        
        # Esegue CROP e MOSAICO STACK
        if 'crop' in mosaic_modes:
            if crop_all_images_for_target(base_dir, logger):
                if not create_mosaic_for_target(base_dir, logger):
                    target_success = False
            else:
                target_success = False
        
        if target_success:
            successful_targets.append(base_dir)
            logger.info(f"Target completato con successo: {base_dir.name}")
        else:
            print(f"\n❌ Target {base_dir.name} fallito.")
            failed_targets.append(base_dir)
            logger.error(f"Target fallito: {base_dir.name}")


    # 4. Riepilogo finale
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

    # 5. Transizione al prossimo step
    if ask_continue_to_next_step(successful_targets, logger):
        try:
            next_script = SCRIPTS_DIR / 'Dataset_step3_analizzapatch.py'
            if next_script.exists():
                print(f"\n🚀 Avvio Step 3/5 (Analisi Patch) per {len(successful_targets)} target...")
                for base_dir in successful_targets:
                    print(f"\n--- Avvio per {base_dir.name} ---")
                    subprocess.run([sys.executable, str(next_script), str(base_dir.resolve())])
            else:
                print(f"\n⚠️  Script {next_script.name} non trovato in {SCRIPTS_DIR}")
        except Exception as e:
            print(f"❌ Errore avvio script successivo: {e}")
            logger.error(f"Errore avvio script successivo: {e}")

if __name__ == "__main__":
    main()