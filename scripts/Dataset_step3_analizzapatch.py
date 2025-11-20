"""
STEP 5: ESTRAZIONE PATCH - VERSIONE OTTIMIZZATA
Features:
- Menu interattivo per selezione target (singolo o batch)
- Multithreading avanzato per estrazione e matching
- Ottimizzato per RTX 2060 (6GB VRAM) + 64GB RAM
- Dimensioni target: 512x512 (Hubble HR) e 34x34 (Observatory LR)
- Parallelizzazione completa del matching
- Progress tracking dettagliato
- Gestione memoria ottimizzata
- Automazione da step precedente con sys.argv

INPUT: Cartelle '4_cropped/hubble' e '4_cropped/observatory'
OUTPUT: Cartella '6_patches_from_cropped' con:
  - hubble_patches/ (512x512px)
  - observatory_patches/ (34x34px)
  - paired_patches_folders/ (coppie matched)
  - extraction_metadata.json

MODIFICATO: Gestione path assoluti e sys.argv per automazione da Step 4.
"""

import sys
import time
import json
import logging
import shutil
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from datetime import datetime
import multiprocessing as mp
import subprocess

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE PATH DINAMICI (ASSOLUTI - Identici a Step 2)
# ============================================================================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR_ROOT = ROOT_DATA_DIR / "logs"
SCRIPTS_DIR = CURRENT_SCRIPT_DIR

print(f"📂 Project Root: {PROJECT_ROOT}")
print(f"📂 Data Dir:     {ROOT_DATA_DIR}")

# ============================================================================
# PARAMETRI PATCH - OTTIMIZZATI PER RTX 2060 (6GB VRAM) + 64GB RAM
# ============================================================================
# Con RTX 2060 (6GB VRAM): 512x512 consente batch_size 4-6 durante training
# Rapporto 15:1 → 34 × 15 = 510 ≈ 512
TARGET_SIZE_HUBBLE = 512    # 512x512 ottimale per 6GB VRAM
TARGET_SIZE_GROUND = 34     # 34x34 per rapporto 15:1
OVERLAP_PERCENT = 25        # Overlap tra patch adiacenti
MATCH_THRESHOLD = 0.5 / 60.0  # Soglia matching in gradi (0.5 arcmin)

# Multithreading/Multiprocessing - AGGRESSIVO per 64GB RAM
NUM_WORKERS = mp.cpu_count()  # USA TUTTI I CORE (hai 64GB RAM)
IO_WORKERS = 24  # I/O parallelo MOLTO aggressivo per 64GB RAM
USE_PROCESS_POOL = True  # Usa ProcessPoolExecutor invece di Thread per sfruttare RAM

print(f"⚙️  Config: Hubble {TARGET_SIZE_HUBBLE}px, Ground {TARGET_SIZE_GROUND}px")
print(f"⚙️  Workers: {NUM_WORKERS} (extraction - MULTIPROCESS), {IO_WORKERS} (I/O)")
print(f"⚙️  RAM Mode: AGGRESSIVE (64GB disponibili)")

# ============================================================================
# SETUP LOGGING
# ============================================================================
def setup_logging(base_dir_name="batch"):
    """Configura logging dettagliato."""
    LOG_DIR_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR_ROOT / f'patch_extraction_{base_dir_name}_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ============================================================================
# FUNZIONI MENU E SELEZIONE (DA STEP 2)
# ============================================================================
def select_target_directory():
    """Mostra un menu per selezionare una o TUTTE le cartelle target."""
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET".center(70))
    print("📂"*35)
    print(f"\nScansione sottocartelle in: {ROOT_DATA_DIR}")

    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() 
                   if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
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
    print("🎯 STEP 5 (Estrazione Patch) COMPLETATO!")
    print("="*70)
    print("\n📋 OPZIONI:")
    print("   1️⃣  Continua con Step 6 (Verifica Dataset - se esiste)")
    print("   2️⃣  Termina qui")
    
    while True:
        print("\n" + "─"*70)
        choice = input(f"👉 Vuoi continuare con il prossimo step? [S/n, default=S]: ").strip().lower()
        if choice in ('', 's', 'si', 'y', 'yes'):
            return True
        elif choice in ('n', 'no'):
            return False
        else:
            print("❌ Scelta non valida.")

# ============================================================================
# FUNZIONI ESTRAZIONE PATCH
# ============================================================================
def calculate_patch_size_for_target(wcs, target_size_pixels):
    """
    Calcola la dimensione della patch in base al WCS.
    Usa sempre dimensione fissa ottimizzata per training.
    """
    # Usa sempre la dimensione target fissa
    patch_size = target_size_pixels
    
    # Arrotonda a multiplo di 8 per efficienza CNN
    patch_size = ((patch_size + 7) // 8) * 8
    
    return patch_size

def extract_patches_from_file(file_path, output_dir, prefix, target_size, overlap_percent):
    """
    Estrae patch da un singolo file FITS.
    
    Args:
        file_path: Path al file FITS
        output_dir: Directory output per le patch
        prefix: Prefisso nome file ('hr' o 'lr')
        target_size: Dimensione target della patch in pixel (512 o 34)
        overlap_percent: Percentuale di overlap tra patch (25%)
    
    Returns:
        List di dict con metadata delle patch estratte
    """
    patches = []
    
    try:
        with fits.open(file_path, memmap=True) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()
            
            # Gestione array 3D (prendi primo layer)
            if len(data.shape) == 3:
                data = data[0]
            
            # Carica WCS
            wcs = WCS(header)
            if not wcs.has_celestial:
                logging.warning(f"File {file_path.name} non ha coordinate celesti valide")
                return []
            
            # Usa dimensione fissa
            patch_size = calculate_patch_size_for_target(wcs, target_size)
            
            # Calcola step per overlap
            step = int(patch_size * (1 - overlap_percent / 100))
            
            ny, nx = data.shape
            
            # Estrai patch con sliding window
            patch_count = 0
            for y in range(0, ny - patch_size + 1, step):
                for x in range(0, nx - patch_size + 1, step):
                    # Estrai patch
                    patch_data = data[y:y+patch_size, x:x+patch_size].copy()
                    
                    # Filtra patch con troppi NaN
                    if np.isnan(patch_data).mean() > 0.5:
                        continue
                    
                    # Filtra patch con varianza troppo bassa (zone vuote)
                    valid_data = patch_data[~np.isnan(patch_data)]
                    if len(valid_data) > 0 and np.std(valid_data) < 1e-6:
                        continue
                    
                    # Calcola coordinate celesti del centro
                    try:
                        center_coord = wcs.pixel_to_world(x + patch_size/2, y + patch_size/2)
                        ra, dec = center_coord.ra.deg, center_coord.dec.deg
                    except:
                        continue
                    
                    # Crea nome univoco
                    patch_name = f"{prefix}_{file_path.stem}_p{patch_count:04d}.fits"
                    patch_path = output_dir / patch_name
                    
                    # Aggiorna header con nuove dimensioni
                    new_header = header.copy()
                    new_header['NAXIS1'] = patch_size
                    new_header['NAXIS2'] = patch_size
                    
                    # Aggiorna WCS reference pixel
                    if 'CRPIX1' in new_header:
                        new_header['CRPIX1'] = new_header['CRPIX1'] - x
                    if 'CRPIX2' in new_header:
                        new_header['CRPIX2'] = new_header['CRPIX2'] - y
                    
                    # Salva patch
                    fits.PrimaryHDU(data=patch_data, header=new_header).writeto(
                        patch_path, overwrite=True
                    )
                    
                    # Aggiungi metadata
                    patches.append({
                        'file': patch_name,
                        'path': str(patch_path),
                        'ra': ra,
                        'dec': dec,
                        'size': patch_size,
                        'source': file_path.name
                    })
                    
                    patch_count += 1
            
        return patches
        
    except Exception as e:
        logging.error(f"Errore nell'estrazione patch da {file_path.name}: {e}")
        return []

# ============================================================================
# FUNZIONI MATCHING PARALLELO CON KDTREE
# ============================================================================
def build_coordinate_array(patches):
    """Costruisce array numpy di coordinate per KDTree."""
    coords = np.array([[p['ra'], p['dec']] for p in patches])
    return coords

def find_matches_chunk(args):
    """
    Trova match per un chunk di patch Hubble usando KDTree.
    Ottimizzato per multiprocessing con 64GB RAM.
    """
    hubble_chunk, observatory_patches, obs_coords, match_threshold = args
    
    try:
        from scipy.spatial import cKDTree
        
        # Costruisci KDTree per ricerca veloce
        tree = cKDTree(obs_coords)
        
        chunk_matches = []
        
        for h_patch in hubble_chunk:
            h_coord = np.array([h_patch['ra'], h_patch['dec']])
            
            # Trova il punto più vicino usando KDTree (MOLTO più veloce)
            dist, idx = tree.query(h_coord, k=1)
            
            # Verifica soglia
            if dist < match_threshold:
                best_match = observatory_patches[idx]
                chunk_matches.append((h_patch, best_match, dist))
        
        return chunk_matches
        
    except ImportError:
        # Fallback senza scipy (più lento)
        from astropy.coordinates import SkyCoord
        chunk_matches = []
        
        for h_patch in hubble_chunk:
            h_coord = SkyCoord(h_patch['ra'], h_patch['dec'], unit='deg')
            
            best_match = None
            min_distance = float('inf')
            
            for obs_patch in observatory_patches:
                o_coord = SkyCoord(obs_patch['ra'], obs_patch['dec'], unit='deg')
                distance = h_coord.separation(o_coord).deg
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = obs_patch
            
            if min_distance < match_threshold and best_match:
                chunk_matches.append((h_patch, best_match, min_distance))
        
        return chunk_matches

def parallel_matching(hubble_patches, observatory_patches, match_threshold, num_workers):
    """
    Esegue matching parallelo tra patch Hubble e Observatory.
    USA MULTIPROCESSING + KDTREE per massima velocità con 64GB RAM.
    """
    print(f"   🧮 Preparazione coordinate per ricerca ottimizzata...")
    
    # Costruisci array coordinate per KDTree
    obs_coords = build_coordinate_array(observatory_patches)
    
    # Dividi hubble_patches in chunks per multiprocessing
    chunk_size = max(1, len(hubble_patches) // (num_workers * 4))  # 4x chunks per worker
    hubble_chunks = [
        hubble_patches[i:i+chunk_size] 
        for i in range(0, len(hubble_patches), chunk_size)
    ]
    
    print(f"   🔗 Matching parallelo: {len(hubble_chunks)} chunks su {num_workers} processi")
    print(f"   💾 Uso RAM stimato: ~{len(hubble_patches) * 0.001:.1f}MB per coordinates")
    
    # Prepara argomenti per workers
    worker_args = [
        (chunk, observatory_patches, obs_coords, match_threshold)
        for chunk in hubble_chunks
    ]
    
    matches = []
    
    # USA PROCESSPOOL per matching (CPU-intensive)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(find_matches_chunk, args) for args in worker_args]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="🔗 Matching"):
            chunk_matches = future.result()
            matches.extend(chunk_matches)
    
    return matches

def copy_patch_pair(args):
    """Copia una coppia di patch nella cartella pair."""
    pair_id, h_patch, o_patch, pairs_dir = args
    
    try:
        pair_dir = pairs_dir / f"pair_{pair_id:05d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(h_patch['path'], pair_dir / h_patch['file'])
        shutil.copy2(o_patch['path'], pair_dir / o_patch['file'])
        
        return True
    except Exception as e:
        logging.error(f"Errore nella copia coppia {pair_id}: {e}")
        return False

# ============================================================================
# PIPELINE PRINCIPALE
# ============================================================================
def process_single_target(base_dir, logger):
    """
    Processa un singolo target.
    
    Returns:
        True se successo, False altrimenti
    """
    print("\n" + "🎯"*35)
    print(f"ESTRAZIONE PATCH: {base_dir.name}".center(70))
    print("🎯"*35)
    
    logger.info(f"Inizio estrazione per {base_dir.name}")
    logger.info(f"Workers: {NUM_WORKERS}, IO Workers: {IO_WORKERS}")
    logger.info(f"Target size - Hubble: {TARGET_SIZE_HUBBLE}px, Ground: {TARGET_SIZE_GROUND}px")
    
    # ========================================
    # PATH INPUT/OUTPUT
    # ========================================
    input_hubble = base_dir / '4_cropped' / 'hubble'
    input_observatory = base_dir / '4_cropped' / 'observatory'
    
    output_base = base_dir / '6_patches_from_cropped'
    output_hubble = output_base / 'hubble_patches'
    output_observatory = output_base / 'observatory_patches'
    pairs_dir = output_base / 'paired_patches_folders'
    
    # Verifica input
    if not input_hubble.exists() or not input_observatory.exists():
        print(f"❌ Cartelle input mancanti in {base_dir.name}/4_cropped/")
        logger.error(f"Input mancante per {base_dir.name}")
        return False
    
    # Crea directory output
    output_hubble.mkdir(parents=True, exist_ok=True)
    output_observatory.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)
    
    # Trova file
    hubble_files = sorted(list(input_hubble.glob('*.fits')) + list(input_hubble.glob('*.fit')))
    observatory_files = sorted(list(input_observatory.glob('*.fits')) + list(input_observatory.glob('*.fit')))
    
    print(f"\n📂 File trovati:")
    print(f"   Hubble: {len(hubble_files)} file")
    print(f"   Observatory: {len(observatory_files)} file")
    logger.info(f"Hubble: {len(hubble_files)}, Observatory: {len(observatory_files)}")
    
    if not hubble_files or not observatory_files:
        print(f"❌ Nessun file FITS trovato.")
        return False
    
    # ========================================
    # ESTRAZIONE PATCH HUBBLE
    # ========================================
    print(f"\n🔵 Estrazione patch HUBBLE ({TARGET_SIZE_HUBBLE}x{TARGET_SIZE_HUBBLE}px)...")
    print(f"   💪 Multiprocessing: {NUM_WORKERS} processi paralleli")
    start_time = time.time()
    
    hubble_patches = []
    
    # USA PROCESSPOOL per sfruttare tutti i core + RAM
    PoolExecutor = ProcessPoolExecutor if USE_PROCESS_POOL else ThreadPoolExecutor
    
    with PoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(
                extract_patches_from_file,
                f, output_hubble, 'hr', TARGET_SIZE_HUBBLE, OVERLAP_PERCENT
            )
            for f in hubble_files
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="💫 Hubble"):
            patches = future.result()
            hubble_patches.extend(patches)
    
    hubble_time = time.time() - start_time
    print(f"   ✅ Estratte {len(hubble_patches)} patch in {hubble_time:.1f}s")
    logger.info(f"Hubble patches: {len(hubble_patches)} in {hubble_time:.1f}s")
    
    # ========================================
    # ESTRAZIONE PATCH OBSERVATORY
    # ========================================
    print(f"\n🟢 Estrazione patch OBSERVATORY ({TARGET_SIZE_GROUND}x{TARGET_SIZE_GROUND}px)...")
    print(f"   💪 Multiprocessing: {NUM_WORKERS} processi paralleli")
    start_time = time.time()
    
    observatory_patches = []
    with PoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(
                extract_patches_from_file,
                f, output_observatory, 'lr', TARGET_SIZE_GROUND, OVERLAP_PERCENT
            )
            for f in observatory_files
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="🔭 Observatory"):
            patches = future.result()
            observatory_patches.extend(patches)
    
    obs_time = time.time() - start_time
    print(f"   ✅ Estratte {len(observatory_patches)} patch in {obs_time:.1f}s")
    logger.info(f"Observatory patches: {len(observatory_patches)} in {obs_time:.1f}s")
    
    if not hubble_patches or not observatory_patches:
        print(f"\n❌ Nessuna patch estratta per {base_dir.name}.")
        return False
    
    # ========================================
    # MATCHING PARALLELO
    # ========================================
    print(f"\n🔗 Matching patch (threshold: {MATCH_THRESHOLD*60:.2f} arcmin)...")
    start_time = time.time()
    
    matches = parallel_matching(
        hubble_patches, 
        observatory_patches, 
        MATCH_THRESHOLD, 
        NUM_WORKERS
    )
    
    match_time = time.time() - start_time
    print(f"   ✅ Trovate {len(matches)} coppie in {match_time:.1f}s")
    logger.info(f"Matches: {len(matches)} in {match_time:.1f}s")
    
    if not matches:
        print(f"\n⚠️  Nessuna coppia trovata per {base_dir.name}.")
        return False
    
    # ========================================
    # SALVATAGGIO COPPIE
    # ========================================
    print(f"\n💾 Salvataggio {len(matches)} coppie...")
    start_time = time.time()
    
    copy_args = [
        (idx, h_patch, o_patch, pairs_dir)
        for idx, (h_patch, o_patch, dist) in enumerate(matches)
    ]
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as executor:
        futures = [executor.submit(copy_patch_pair, args) for args in copy_args]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="📁 Copia"):
            if future.result():
                success_count += 1
    
    copy_time = time.time() - start_time
    
    # ========================================
    # SALVA METADATA
    # ========================================
    metadata = {
        'target': base_dir.name,
        'timestamp': datetime.now().isoformat(),
        'hardware': {
            'gpu': 'RTX 2060 (6GB VRAM)',
            'ram': '64GB',
            'optimization': 'aggressive_ram_usage',
            'multiprocessing': USE_PROCESS_POOL
        },
        'config': {
            'hubble_target_size': TARGET_SIZE_HUBBLE,
            'observatory_target_size': TARGET_SIZE_GROUND,
            'scale_ratio': TARGET_SIZE_HUBBLE / TARGET_SIZE_GROUND,
            'overlap_percent': OVERLAP_PERCENT,
            'match_threshold_arcmin': MATCH_THRESHOLD * 60,
            'num_workers': NUM_WORKERS,
            'io_workers': IO_WORKERS,
            'matching_workers': NUM_WORKERS * 2,
            'ram_mode': 'aggressive'
        },
        'stats': {
            'hubble_files': len(hubble_files),
            'observatory_files': len(observatory_files),
            'hubble_patches': len(hubble_patches),
            'observatory_patches': len(observatory_patches),
            'matched_pairs': len(matches),
            'saved_pairs': success_count
        },
        'timing': {
            'hubble_extraction_sec': round(hubble_time, 2),
            'observatory_extraction_sec': round(obs_time, 2),
            'matching_sec': round(match_time, 2),
            'copy_sec': round(copy_time, 2),
            'total_sec': round(hubble_time + obs_time + match_time + copy_time, 2)
        },
        'training_recommendations': {
            'batch_size': '4-6 (optimal for 6GB VRAM)',
            'mixed_precision': True,
            'gradient_accumulation': 'optional (if batch_size<4)'
        }
    }
    
    metadata_file = output_base / 'extraction_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # ========================================
    # SUMMARY
    # ========================================
    total_time = hubble_time + obs_time + match_time + copy_time
    
    print(f"\n✅ {base_dir.name}: {success_count} coppie salvate in {total_time:.1f}s")
    print(f"   📁 Output: {pairs_dir}")
    
    logger.info(f"Completato {base_dir.name}: {success_count} coppie in {total_time:.1f}s")
    
    return True

# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main entry point."""
    
    # ========================================
    # GESTIONE INPUT (MANUALE O AUTOMATICA)
    # ========================================
    if len(sys.argv) > 1:
        # Modalità automatica (da step precedente)
        input_path = Path(sys.argv[1]).resolve()
        if input_path.exists():
            print(f"\n🤖 Modalità Automatica: Target ricevuto {input_path.name}")
            target_dirs = [input_path]
        else:
            print(f"❌ Errore: Path fornito non valido: {input_path}")
            return
    else:
        # Modalità manuale (menu interattivo)
        target_dirs = select_target_directory()
        if not target_dirs:
            print("\n👋 Operazione annullata.")
            return
    
    # Setup logging
    logger = setup_logging(target_dirs[0].name if len(target_dirs) == 1 else "batch")
    
    # ========================================
    # PROCESSING LOOP
    # ========================================
    print("\n" + "="*80)
    print("PIPELINE: ESTRAZIONE PATCH (64GB RAM AGGRESSIVE MODE)".center(80))
    print("="*80)
    
    logger.info(f"Inizio batch su {len(target_dirs)} target")
    
    successful_targets = []
    failed_targets = []
    
    for base_dir in target_dirs:
        try:
            if process_single_target(base_dir, logger):
                successful_targets.append(base_dir)
            else:
                failed_targets.append(base_dir)
        except Exception as e:
            print(f"\n❌ Errore critico in {base_dir.name}: {e}")
            logger.error(f"Errore critico in {base_dir.name}: {e}")
            failed_targets.append(base_dir)
    
    # ========================================
    # SUMMARY FINALE
    # ========================================
    print("\n" + "="*80)
    print("📊 RIEPILOGO FINALE".center(80))
    print("="*80)
    print(f"\n✅ Target completati:  {len(successful_targets)}/{len(target_dirs)}")
    if successful_targets:
        for t in successful_targets:
            print(f"   • {t.name}")
    
    if failed_targets:
        print(f"\n❌ Target falliti: {len(failed_targets)}")
        for t in failed_targets:
            print(f"   • {t.name}")
    
    print("="*80)
    
    if not successful_targets:
        print("\n❌ Nessun target completato.")
        return
    
    # ========================================
    # CONTINUA CON PROSSIMO STEP?
    # ========================================
    if ask_continue_to_next_step():
        try:
            next_script = SCRIPTS_DIR / 'Modello_2_prepare_data.py'
            if next_script.exists():
                print(f"\n🚀 Avvio preparazione dataset...")
                # Passa alla preparazione dataset del modello
                subprocess.run([sys.executable, str(next_script)])
            else:
                print(f"⚠️  Script successivo non trovato: {next_script}")
                print(f"   Puoi procedere manualmente con:")
                print(f"   cd {SCRIPTS_DIR.parent / 'Modello'}")
                print(f"   python Modello_2_prepare_data.py")
        except Exception as e:
            print(f"❌ Errore avvio script successivo: {e}")
    else:
        print("\n👋 Pipeline completata!")
        print(f"\n📝 PROSSIMI PASSI:")
        print(f"   1. Verifica le coppie generate in 6_patches_from_cropped/")
        print(f"   2. Lancia Modello_2_prepare_data.py per creare train/val/test splits")
        print(f"   3. Training con batch_size 4-6 (ottimale per RTX 2060)")

if __name__ == "__main__":
    main()