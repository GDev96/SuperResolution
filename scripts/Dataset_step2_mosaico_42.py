"""
STEP 2: CREAZIONE MOSAICO USANDO WCS
Crea un mosaico usando le coordinate WCS per allineare immagini di dimensioni diverse

INPUT: Cartelle '3_registered_native/hubble' e '3_registered_native/observatory'
OUTPUT: 
  - File '5_mosaics/final_mosaic.fits' con il mosaico finale
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
    log_filename = LOG_DIR_ROOT / f'mosaic_{timestamp}.log'
    
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
    print("🎯 MOSAICO COMPLETATO!")
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
# FUNZIONI - MOSAICO CON WCS
# ============================================================================

def calculate_common_grid(image_infos):
    """
    Calcola una griglia comune che copre tutte le immagini usando WCS.
    
    Returns:
        tuple: (output_wcs, output_shape)
    """
    print("\n🔍 Calcolo griglia comune...")
    
    # Trova i limiti del campo totale
    ra_min = float('inf')
    ra_max = float('-inf')
    dec_min = float('inf')
    dec_max = float('-inf')
    
    pixel_scales = []
    
    for info in image_infos:
        wcs = info['wcs']
        shape = info['shape']
        ny, nx = shape
        
        # Coordinate dei 4 angoli
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
        
        # Calcola pixel scale
        try:
            if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
                cd = wcs.wcs.cd
                pixel_scale = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
            else:
                pixel_scale = abs(wcs.wcs.cdelt[0])
            pixel_scales.append(pixel_scale)
        except:
            continue
    
    # Gestione wraparound RA
    if ra_max - ra_min > 180:
        print("   ⚠️  Rilevato wraparound RA vicino a 0°/360°")
    
    # Centro del campo
    ra_center = (ra_min + ra_max) / 2.0
    dec_center = (dec_min + dec_max) / 2.0
    
    # Dimensioni campo
    ra_span = ra_max - ra_min
    dec_span = dec_max - dec_min
    
    # Usa il pixel scale più fine (minore) per preservare i dettagli
    target_pixel_scale = min(pixel_scales) if pixel_scales else 0.0001  # deg/px
    
    # Calcola dimensioni output
    nx_out = int(np.ceil(abs(ra_span / np.cos(np.radians(dec_center))) / target_pixel_scale))
    ny_out = int(np.ceil(abs(dec_span) / target_pixel_scale))
    
    # TELA QUADRATA: usa la dimensione maggiore per entrambi i lati
    max_dim = max(nx_out, ny_out)
    nx_out = max_dim
    ny_out = max_dim
    
    # Limita dimensioni massime
    MAX_DIM = 20000
    if max_dim > MAX_DIM:
        nx_out = MAX_DIM
        ny_out = MAX_DIM
        target_pixel_scale *= (max_dim / MAX_DIM)
    
    print(f"   Centro: RA={ra_center:.4f}°, DEC={dec_center:.4f}°")
    print(f"   Span: RA={ra_span:.4f}° ({ra_span*60:.1f}'), DEC={dec_span:.4f}° ({dec_span*60:.1f}')")
    print(f"   Pixel scale: {target_pixel_scale*3600:.4f}\"/px")
    print(f"   Dimensioni output: {nx_out} x {ny_out} pixel ⬜ (QUADRATA)")
    
    # Crea WCS output
    output_wcs = WCS(naxis=2)
    output_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    output_wcs.wcs.crval = [ra_center, dec_center]
    output_wcs.wcs.crpix = [nx_out / 2.0, ny_out / 2.0]
    output_wcs.wcs.cdelt = [-target_pixel_scale, target_pixel_scale]
    output_wcs.wcs.radesys = 'ICRS'
    output_wcs.wcs.equinox = 2000.0
    
    return output_wcs, (ny_out, nx_out)


def create_mosaic_wcs(base_dir):
    """Crea il mosaico usando WCS per allineare immagini di dimensioni diverse."""
    print("\n" + "🖼️ "*35)
    print(f"MOSAICO WCS: {base_dir.name}".center(70))
    print("🖼️ "*35)
    
    # Path
    input_dirs = [
        base_dir / '3_registered_native' / 'hubble',
        base_dir / '3_registered_native' / 'observatory'
    ]
    
    mosaic_output_dir = base_dir / '5_mosaics'
    mosaic_output_file = mosaic_output_dir / 'final_mosaic.fits'
    
    # Crea cartella output
    mosaic_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Trova tutti i file FITS e carica info WCS
    print("\n📂 Caricamento immagini...")
    all_files = []
    for d in input_dirs:
        if d.exists():
            all_files.extend(list(d.glob('*.fits')) + list(d.glob('*.fit')))
    
    if not all_files:
        print(f"\n❌ ERRORE: Nessun file FITS trovato.")
        return False
    
    print(f"✅ Trovati {len(all_files)} file FITS")
    
    # Carica informazioni WCS
    image_infos = []
    for filepath in tqdm(all_files, desc="Lettura WCS", unit="file"):
        try:
            with fits.open(filepath) as hdul:
                # Cerca HDU con dati
                data = None
                header = None
                for hdu in hdul:
                    if hdu.data is not None and hdu.data.size > 0:
                        data = hdu.data
                        header = hdu.header
                        break
                
                if data is None:
                    continue
                
                # Gestisci 3D
                if len(data.shape) == 3:
                    data = data[0]
                
                # Verifica WCS
                wcs = WCS(header)
                if not wcs.has_celestial:
                    print(f"\n⚠️  {filepath.name}: WCS non valido, saltato")
                    continue
                
                image_infos.append({
                    'file': filepath,
                    'data': data,
                    'header': header,
                    'wcs': wcs,
                    'shape': data.shape
                })
                
        except Exception as e:
            print(f"\n⚠️  Errore leggendo {filepath.name}: {e}")
            continue
    
    if not image_infos:
        print(f"\n❌ ERRORE: Nessuna immagine con WCS valido trovata.")
        return False
    
    print(f"✅ {len(image_infos)} immagini con WCS valido")
    
    # 2. Calcola griglia comune
    output_wcs, output_shape = calculate_common_grid(image_infos)
    
    # 3. Crea array per mosaico
    print("\n🔄 Creazione mosaico...")
    mosaic_data = np.zeros(output_shape, dtype=np.float64)
    mosaic_weight = np.zeros(output_shape, dtype=np.float64)
    
    # 4. Proietta ogni immagine sulla griglia comune
    for info in tqdm(image_infos, desc="Proiezione", unit="immagine"):
        try:
            data = info['data']
            wcs_in = info['wcs']
            shape_in = info['shape']
            
            # Per ogni pixel dell'immagine input, trova dove cade nell'output
            ny_in, nx_in = shape_in
            
            # Crea griglia di coordinate pixel input
            y_in, x_in = np.mgrid[0:ny_in, 0:nx_in]
            
            # Converti in coordinate mondo
            try:
                coords_world = wcs_in.pixel_to_world(x_in.ravel(), y_in.ravel())
                
                # Converti in coordinate pixel output
                x_out, y_out = output_wcs.world_to_pixel(coords_world)
                
                # Arrotonda a pixel interi
                x_out = np.round(x_out).astype(int)
                y_out = np.round(y_out).astype(int)
                
                # Reshape
                x_out = x_out.reshape(shape_in)
                y_out = y_out.reshape(shape_in)
                
                # Filtra pixel validi (dentro i bounds dell'output)
                valid = (x_out >= 0) & (x_out < output_shape[1]) & \
                        (y_out >= 0) & (y_out < output_shape[0]) & \
                        ~np.isnan(data)
                
                # Aggiungi al mosaico
                y_valid = y_out[valid]
                x_valid = x_out[valid]
                data_valid = data[valid]
                
                mosaic_data[y_valid, x_valid] += data_valid
                mosaic_weight[y_valid, x_valid] += 1.0
                
            except Exception as e:
                print(f"\n⚠️  Errore proiettando {info['file'].name}: {e}")
                continue
                
        except Exception as e:
            print(f"\n⚠️  Errore processando {info['file'].name}: {e}")
            continue
    
    # 5. Calcola media
    print("\n🧮 Calcolo della media finale...")
    
    # Evita divisione per zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mosaic_final = np.where(mosaic_weight > 0, 
                                mosaic_data / mosaic_weight, 
                                np.nan)
    
    # Statistiche
    valid_pixels = np.sum(~np.isnan(mosaic_final))
    total_pixels = mosaic_final.size
    coverage = (valid_pixels / total_pixels) * 100
    
    print(f"📊 Statistiche mosaico:")
    print(f"   Dimensioni: {output_shape[1]} x {output_shape[0]} pixel")
    print(f"   Pixel validi: {valid_pixels}")
    print(f"   Copertura: {coverage:.1f}%")
    print(f"   Immagini combinate: {len(image_infos)}")
    
    # 6. Salva
    print(f"\n💾 Salvataggio mosaico...")
    
    # Crea header dal WCS output
    output_header = output_wcs.to_header()
    output_header['HISTORY'] = 'Mosaico creato usando WCS'
    output_header['NCOMBINE'] = (len(image_infos), 'Numero di immagini combinate')
    output_header['COVERAGE'] = (coverage, 'Percentuale copertura')
    output_header['NPIXVAL'] = (valid_pixels, 'Numero pixel validi')
    
    try:
        fits.PrimaryHDU(data=mosaic_final.astype(np.float32), 
                       header=output_header).writeto(mosaic_output_file, overwrite=True)
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile salvare: {e}")
        return False
    
    print(f"\n✅ MOSAICO COMPLETATO: {mosaic_output_file}")
    return True

# ============================================================================
# MAIN
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
    print("PIPELINE: CREAZIONE MOSAICO WCS (BATCH)".center(70))
    print("="*70)
    
    start_time_total = time.time()
    successful_targets = []
    failed_targets = []
    
    for base_dir in target_dirs:
        print("\n" + "🚀"*35)
        print(f"ELABORAZIONE TARGET: {base_dir.name}".center(70))
        print("🚀"*35)
        
        # Crea mosaico
        if not create_mosaic_wcs(base_dir):
            print(f"\n❌ Target {base_dir.name} fallito.")
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
                print(f"\n🚀 Avvio Step 3 per {len(successful_targets)} target...")
                for base_dir in successful_targets:
                    print(f"\n--- Avvio per {base_dir.name} ---")
                    subprocess.run([sys.executable, str(next_script), str(base_dir)])
            else:
                print(f"\n⚠️  Script {next_script.name} non trovato in {SCRIPTS_DIR}")
        except Exception as e:
            print(f"❌ Errore avvio script successivo: {e}")

if __name__ == "__main__":
    main()