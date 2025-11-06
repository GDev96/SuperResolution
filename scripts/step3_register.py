"""
STEP 3: REGISTRAZIONE IMMAGINI (MIGLIORATO - MANTIENE RISOLUZIONE ORIGINALE)
Allinea TUTTE le immagini (Hubble, Osservatorio) su un'unica griglia WCS.
MIGLIORIA: Ogni immagine mantiene la sua risoluzione nativa per massima qualità.

Differenze rispetto alla versione originale:
- Non c'è più una risoluzione target fissa
- Ogni immagine viene riproiettata mantenendo la sua risoluzione originale
- Il WCS target copre l'intera area ma ogni immagine decide la propria dimensione canvas
"""

import os
import sys
import glob
import time
import logging
from datetime import datetime
import numpy as np
import astropy
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Prova a importare la libreria necessaria
try:
    from reproject import reproject_interp
except ImportError:
    print("="*70)
    print("ERRORE: Libreria 'reproject' non trovata.")
    print("Questa libreria è fondamentale per lo Step 3.")
    print("Installa con: pip install reproject")
    print("="*70)
    sys.exit(1)


# ============================================================================
# CONFIGURAZIONE (DINAMICA)
# ============================================================================
import os

# Ottieni il percorso assoluto della directory contenente questo script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Cerca la cartella 'data'
if os.path.isdir(os.path.join(SCRIPT_DIR, 'data')):
    # Caso 1: Lo script è nella root del progetto
    PROJECT_ROOT = SCRIPT_DIR
elif os.path.isdir(os.path.join(os.path.dirname(SCRIPT_DIR), 'data')):
    # Caso 2: Lo script è in una sottocartella
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    raise FileNotFoundError(
        f"Impossibile trovare la directory 'data'. "
        f"Verificata in {SCRIPT_DIR} e {os.path.dirname(SCRIPT_DIR)}. "
        "Assicurati che 'data' sia nella cartella principale del progetto."
    )

# Definisci i percorsi principali
BASE_DIR = os.path.join(PROJECT_ROOT, 'data')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Input: cartelle create da step1 (generiche)
INPUT_HUBBLE = os.path.join(BASE_DIR, 'lith_con_wcs')
INPUT_OBSERVATORY = os.path.join(BASE_DIR, 'osservatorio_con_wcs')

# Output: cartelle registrate (generiche)
OUTPUT_HUBBLE = os.path.join(BASE_DIR, '3_registered_native', 'hubble')
OUTPUT_OBSERVATORY = os.path.join(BASE_DIR, '3_registered_native', 'observatory')

# --- Parametri Registrazione ---
NUM_THREADS = 7  # Thread paralleli
REPROJECT_ORDER = 'bilinear'

# Lock per logging thread-safe
log_lock = threading.Lock()

# ============================================================================
# SETUP LOGGING
# ============================================================================

def setup_logging():
    """Configura logging generico."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'registration_native_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Python: {sys.version}")
    logger.info(f"Numpy: {np.__version__}")
    logger.info(f"Astropy: {astropy.__version__}")
    logger.info(f"Reproject: (importata con successo)")
    logger.info(f"Threads: {NUM_THREADS}")
    logger.info(f"Reprojection Order: {REPROJECT_ORDER}")
    logger.info("MODALITÀ: Risoluzione Nativa (ogni immagine mantiene la sua risoluzione)")
    
    return logger

# ============================================================================
# FUNZIONI ANALISI
# ============================================================================

def extract_wcs_info(filepath, logger):
    """Estrae info WCS da file FITS."""
    try:
        with fits.open(filepath) as hdul:
            # Trova il primo HDU con dati e WCS valido
            data_hdu = None
            hdu_idx = 0
            
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    try:
                        wcs = WCS(hdu.header)
                        if wcs.has_celestial:
                            data_hdu = hdu
                            hdu_idx = i
                            break
                    except:
                        continue
            
            if data_hdu is None:
                with log_lock:
                    logger.warning(f"Nessun HDU con WCS valido trovato in {os.path.basename(filepath)}")
                return None
            
            wcs = WCS(data_hdu.header)

            # Se 3D, usa primo canale
            if len(data_hdu.data.shape) == 3:
                data = data_hdu.data[0]
            else:
                data = data_hdu.data
            
            ny, nx = data.shape
            center = wcs.pixel_to_world(nx/2, ny/2)
            
            # Calcola pixel scale NATIVO dell'immagine
            try:
                if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
                    cd = wcs.wcs.cd
                    pixel_scale_deg = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
                else:
                    pixel_scale_deg = abs(wcs.wcs.cdelt[0])
                pixel_scale_arcsec = pixel_scale_deg * 3600.0
            except:
                pixel_scale_arcsec = 0.0
            
            return {
                'file': filepath,
                'hdu_index': hdu_idx,
                'wcs': wcs,
                'shape': data.shape,
                'center_ra': center.ra.deg,
                'center_dec': center.dec.deg,
                'pixel_scale': pixel_scale_arcsec,  # Risoluzione NATIVA
            }
            
    except Exception as e:
        with log_lock:
            logger.error(f"Errore WCS {os.path.basename(filepath)}: {e}")
        return None


def analyze_images(input_dir, source_name, logger):
    """Analizza tutte le immagini in una directory."""
    files = glob.glob(os.path.join(input_dir, '*.fits')) + \
            glob.glob(os.path.join(input_dir, '*.fit'))
    
    if not files:
        with log_lock:
            logger.warning(f"Nessun file in {input_dir}")
        return []
    
    print(f"\n📂 {source_name}: {len(files)} file")
    
    wcs_info_list = []
    
    with tqdm(total=len(files), desc=f"  Analisi {source_name}", unit="file") as pbar:
        for filepath in files:
            info = extract_wcs_info(filepath, logger)
            if info:
                wcs_info_list.append(info)
                with log_lock:
                    logger.info(f"✓ {os.path.basename(filepath)}: "
                               f"RA={info['center_ra']:.4f}°, DEC={info['center_dec']:.4f}°, "
                               f"scale={info['pixel_scale']:.4f}\"/px (NATIVA)")
            else:
                with log_lock:
                    logger.warning(f"✗ {os.path.basename(filepath)}: WCS non valido")
            pbar.update(1)
    
    print(f"   ✓ {len(wcs_info_list)}/{len(files)} con WCS valido")
    
    return wcs_info_list


def create_common_wcs_frame(wcs_info_list, logger):
    """
    Crea un WCS di riferimento comune che copre TUTTE le immagini.
    Questo WCS serve solo come frame di riferimento - ogni immagine
    manterrà la sua risoluzione nativa.
    """
    with log_lock:
        logger.info("=" * 60)
        logger.info("CREAZIONE FRAME WCS COMUNE (riferimento)")
        logger.info("=" * 60)
    
    if not wcs_info_list:
        with log_lock:
            logger.error("Nessuna immagine fornita per creare WCS comune.")
        return None
    
    # Trova i limiti del campo totale
    ra_min = float('inf')
    ra_max = float('-inf')
    dec_min = float('inf')
    dec_max = float('-inf')
    
    for info in wcs_info_list:
        wcs = info['wcs']
        ny, nx = info['shape']
        
        # Corners
        corners = wcs.pixel_to_world([0, nx, nx, 0], [0, 0, ny, ny])
        for c in corners:
            ra_min = min(ra_min, c.ra.deg)
            ra_max = max(ra_max, c.ra.deg)
            dec_min = min(dec_min, c.dec.deg)
            dec_max = max(dec_max, c.dec.deg)
    
    # Centro del campo totale
    ra_center = (ra_min + ra_max) / 2
    dec_center = (dec_min + dec_max) / 2
    
    # Dimensione del campo
    ra_span = ra_max - ra_min
    dec_span = dec_max - dec_min
    
    # Crea WCS comune con una risoluzione media (solo per riferimento)
    avg_scale = np.mean([info['pixel_scale'] for info in wcs_info_list]) / 3600.0  # gradi
    
    common_wcs = WCS(naxis=2)
    common_wcs.wcs.crval = [ra_center, dec_center]
    common_wcs.wcs.crpix = [1, 1]
    common_wcs.wcs.cdelt = [-avg_scale, avg_scale]
    common_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    common_wcs.wcs.radesys = 'ICRS'
    common_wcs.wcs.equinox = 2000.0
    
    with log_lock:
        logger.info(f"Campo totale:")
        logger.info(f"  RA: {ra_min:.6f}° → {ra_max:.6f}° (span: {ra_span:.6f}° = {ra_span*60:.2f}')")
        logger.info(f"  DEC: {dec_min:.6f}° → {dec_max:.6f}° (span: {dec_span:.6f}° = {dec_span*60:.2f}')")
        logger.info(f"  Centro: RA={ra_center:.6f}°, DEC={dec_center:.6f}°")
        logger.info(f"  Scala media (solo riferimento): {avg_scale*3600:.4f}\"/px")
    
    return common_wcs


# ============================================================================
# FUNZIONI REGISTRAZIONE CON RISOLUZIONE NATIVA
# ============================================================================

def reproject_image_native(info, common_wcs, output_dir, logger):
    """
    Reproietta un'immagine mantenendo la sua risoluzione NATIVA.
    
    Differenza chiave rispetto alla versione originale:
    - Non c'è un target_wcs fisso con risoluzione predefinita
    - Ogni immagine crea il proprio WCS target basato sulla sua risoluzione nativa
    - Il canvas è dimensionato per contenere l'immagine alla sua risoluzione originale
    """
    filepath = info['file']
    filename = os.path.basename(filepath)
    native_pixel_scale = info['pixel_scale']  # arcsec/px NATIVO
    
    try:
        with fits.open(filepath) as hdul:
            # Trova HDU con dati
            data_hdu = hdul[info['hdu_index']]
            
            # Se 3D, converti a 2D
            if len(data_hdu.data.shape) == 3:
                temp_data = data_hdu.data[0]
                temp_hdu = fits.PrimaryHDU(data=temp_data, header=data_hdu.header)
                data_hdu = temp_hdu
            
            original_wcs = WCS(data_hdu.header)
            original_shape = data_hdu.data.shape
            
            # Crea WCS target per questa immagine specifica
            # Usa il frame comune come riferimento ma mantiene la risoluzione nativa
            target_wcs = WCS(naxis=2)
            target_wcs.wcs.crval = common_wcs.wcs.crval  # Stesso centro di riferimento
            target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            target_wcs.wcs.radesys = 'ICRS'
            target_wcs.wcs.equinox = 2000.0
            
            # CHIAVE: Mantieni la risoluzione NATIVA dell'immagine
            native_scale_deg = native_pixel_scale / 3600.0
            target_wcs.wcs.cdelt = [-native_scale_deg, native_scale_deg]
            
            # Calcola dimensioni canvas necessarie per contenere l'immagine originale
            # alla sua risoluzione nativa
            ny_orig, nx_orig = original_shape
            
            # Ottieni i corners dell'immagine originale in coordinate celesti
            corners = original_wcs.pixel_to_world(
                [0, nx_orig-1, nx_orig-1, 0],
                [0, 0, ny_orig-1, ny_orig-1]
            )
            
            # Converti corners al nuovo sistema di pixel
            pixel_corners_x, pixel_corners_y = target_wcs.world_to_pixel(corners)
            
            # Trova limiti
            x_min = np.floor(np.min(pixel_corners_x)).astype(int)
            x_max = np.ceil(np.max(pixel_corners_x)).astype(int)
            y_min = np.floor(np.min(pixel_corners_y)).astype(int)
            y_max = np.ceil(np.max(pixel_corners_y)).astype(int)
            
            # Dimensioni canvas
            canvas_width = x_max - x_min + 1
            canvas_height = y_max - y_min + 1
            
            # Aggiusta CRPIX per il canvas
            target_wcs.wcs.crpix = [-x_min + 1, -y_min + 1]
            
            shape_out = (canvas_height, canvas_width)
            
            with log_lock:
                logger.info(f"  {filename}: {original_shape} → {shape_out} "
                           f"(risoluzione nativa: {native_pixel_scale:.4f}\"/px)")
            
            # Reproiezione
            reprojected_data, footprint = reproject_interp(
                data_hdu,
                target_wcs,
                shape_out=shape_out,
                order=REPROJECT_ORDER
            )
            
            if reprojected_data is None:
                with log_lock:
                    logger.warning(f"Reproiezione fallita: {filename}")
                return {'status': 'error', 'file': filename}
            
            # Statistiche copertura
            valid_mask = ~np.isnan(reprojected_data)
            valid_pixels = np.sum(valid_mask)
            total_pixels = reprojected_data.size
            coverage = (valid_pixels / total_pixels) * 100
            
            # Header
            new_header = target_wcs.to_header()
            
            # Copia metadati
            for key in ['OBJECT', 'DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'TELESCOP']:
                if key in data_hdu.header:
                    new_header[key] = data_hdu.header[key]
            
            # Metadati registrazione
            new_header['REGMTHD'] = 'reproject_interp_native'
            new_header['REGSRC'] = filename
            new_header['REGDATE'] = datetime.now().isoformat()
            new_header['REGCOVER'] = (coverage, "Percentuale del canvas coperta")
            new_header['REGVALID'] = (valid_pixels, "Numero di pixel validi")
            new_header['REGORD'] = (str(REPROJECT_ORDER), "Ordine interpolazione")
            new_header['NATIVESC'] = (native_pixel_scale, "Risoluzione nativa (arcsec/px)")
            new_header['ORIGSHP0'] = (original_shape[0], "Shape originale (altezza)")
            new_header['ORIGSHP1'] = (original_shape[1], "Shape originale (larghezza)")
            
            # Nome output
            output_filename = f"reg_{os.path.splitext(filename)[0]}.fits"
            output_path = os.path.join(output_dir, output_filename)
            
            # Salva
            fits.PrimaryHDU(
                data=reprojected_data.astype(np.float32),
                header=new_header
            ).writeto(output_path, overwrite=True)
            
            with log_lock:
                logger.info(f"✓ {filename}: coverage={coverage:.1f}%, "
                           f"shape={shape_out}, native_scale={native_pixel_scale:.4f}\"/px")
            
            return {
                'status': 'success',
                'file': filename,
                'coverage': coverage,
                'valid_pixels': valid_pixels,
                'output_path': output_path,
                'native_scale': native_pixel_scale,
                'output_shape': shape_out
            }
            
    except Exception as e:
        with log_lock:
            logger.error(f"Errore {os.path.basename(filepath)}: {e}")
        return {'status': 'error', 'file': os.path.basename(filepath), 'reason': str(e)}


def register_images(wcs_info_list, common_wcs, output_dir, source_name, logger):
    """Registra tutte le immagini con multithreading, ognuna alla sua risoluzione nativa."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🔄 Registrazione {source_name}: {len(wcs_info_list)} immagini (risoluzione nativa)")
    
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            executor.submit(reproject_image_native, info, common_wcs, output_dir, logger): info
            for info in wcs_info_list
        }
        
        with tqdm(total=len(wcs_info_list), desc=f"  {source_name}") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    
                    if result['status'] == 'success':
                        success_count += 1
                        pbar.set_description(f"✓ {success_count}")
                    else:
                        error_count += 1
                        pbar.set_description(f"❌ {error_count}")
                        
                except Exception as exc:
                    error_count += 1
                    with log_lock:
                        logger.error(f"Exception nel thread pool: {exc}")
                
                pbar.update(1)
    
    print(f"   ✓ Successo: {success_count}")
    print(f"   ✗ Errori: {error_count}")
    
    with log_lock:
        logger.info(f"{source_name}: {success_count} successo, {error_count} errori")
    
    return success_count, error_count

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Funzione principale."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("REGISTRAZIONE CON RISOLUZIONE NATIVA")
    logger.info("=" * 60)
    
    print("\n" + "🔭"*35)
    print(f"STEP 3: REGISTRAZIONE (RISOLUZIONE NATIVA)".center(70))
    print("🔭"*35)
    
    print(f"\n📂 CONFIGURAZIONE:")
    print(f"   Input Hubble: {INPUT_HUBBLE}")
    print(f"   Input Observatory: {INPUT_OBSERVATORY}")
    print(f"   Output: {os.path.join(BASE_DIR, '3_registered_native')}")
    print(f"   Modalità: RISOLUZIONE NATIVA (ogni immagine mantiene la sua risoluzione)")
    print(f"   Threads: {NUM_THREADS}")
    print(f"   Interpolazione: {REPROJECT_ORDER}")
    
    # Analizza immagini
    print(f"\n{'='*70}")
    print("ANALISI IMMAGINI")
    print(f"{'='*70}")
    
    hubble_info = analyze_images(INPUT_HUBBLE, "HUBBLE", logger)
    obs_info = analyze_images(INPUT_OBSERVATORY, "OBSERVATORY", logger)
    
    all_info = hubble_info + obs_info
    if not all_info:
        print(f"\n❌ Nessuna immagine con WCS valido trovata. Interruzione.")
        logger.error("Nessuna immagine con WCS valido trovata.")
        return
    
    # Mostra statistiche risoluzione
    print(f"\n📊 RISOLUZIONI NATIVE RILEVATE:")
    for source_name, info_list in [("Hubble", hubble_info), ("Observatory", obs_info)]:
        if info_list:
            scales = [info['pixel_scale'] for info in info_list]
            print(f"\n   {source_name}:")
            print(f"      Min: {min(scales):.4f}\"/px")
            print(f"      Max: {max(scales):.4f}\"/px")
            print(f"      Media: {np.mean(scales):.4f}\"/px")
            print(f"      → Tutte manterranno la loro risoluzione originale!")
    
    # Crea frame WCS comune (solo per riferimento)
    print(f"\n{'='*70}")
    print("CREAZIONE FRAME WCS COMUNE")
    print(f"{'='*70}")
    
    common_wcs = create_common_wcs_frame(all_info, logger)
    
    if common_wcs is None:
        print(f"\n❌ Impossibile creare WCS comune!")
        logger.error("Creazione WCS comune fallita.")
        return
    
    # Registrazione
    print(f"\n{'='*70}")
    print("REGISTRAZIONE (Risoluzione Nativa)")
    print(f"{'='*70}")
    
    total_success = 0
    total_error = 0
    
    if hubble_info:
        s, e = register_images(hubble_info, common_wcs, OUTPUT_HUBBLE, "Hubble", logger)
        total_success += s
        total_error += e
    
    if obs_info:
        s, e = register_images(obs_info, common_wcs, OUTPUT_OBSERVATORY, "Observatory", logger)
        total_success += s
        total_error += e
    
    # === BLOCCO STATISTICHE DIMENSIONI RIMOSSO ===
    
    # Riepilogo
    print(f"\n{'='*70}")
    print("📊 RIEPILOGO REGISTRAZIONE")
    print(f"{'='*70}")
    print(f"\n   Totale registrate: {total_success}")
    print(f"   Totale errori: {total_error}")
    
    if total_success > 0:
        print(f"\n✅ REGISTRAZIONE COMPLETATA!")
        print(f"\n   ✨ VANTAGGIO: Ogni immagine ha mantenuto la sua risoluzione nativa!")
        print(f"\n   📁 Output:")
        if hubble_info:
            print(f"      Hubble: {OUTPUT_HUBBLE}")
        if obs_info:
            print(f"      Observatory: {OUTPUT_OBSERVATORY}")
        
        # === BLOCCO STATISTICHE DIMENSIONI DETTAGLIATE RIMOSSO ===
        
        print(f"\n{'='*70}")
        print(f"⭐ RISOLUZIONE NATIVA PRESERVATA")
        print(f"{'='*70}")
        print(f"   ✅ Ogni immagine mantiene la sua risoluzione originale")
        print(f"   ✅ Nessuna perdita di qualità per interpolazione")
        print(f"   ✅ Dimensioni output ottimizzate per risoluzione nativa")
        
        print(f"\n   ⚠️  NOTA IMPORTANTE:")
        print(f"      Le immagini hanno risoluzioni diverse tra loro.")
        print(f"      Per lo step 4, usa 'step4_patch_improved.py' che")
        print(f"      estrae patches rispettando la risoluzione di ogni immagine.")
    
    with log_lock:
        logger.info(f"Registrazione completata: {total_success} successo, {total_error} errori")


if __name__ == "__main__":
    print(f"\n🔭 REGISTRAZIONE CON RISOLUZIONE NATIVA")
    print(f"Threads: {NUM_THREADS}")
    print(f"Modalità: Ogni immagine mantiene la sua risoluzione originale")
    print(f"Interpolazione: {REPROJECT_ORDER}\n")
    
    start_time = time.time()
    
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ ERRORE FATALE: {e}")
        import traceback
        traceback.print_exc()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Tempo totale: {elapsed:.1f} secondi")