"""
STEP 3: CALCOLO DIMENSIONI PATCH (POST-REGISTRAZIONE)
Analizza le immagini registrate (da step2) per determinare dimensioni ottimali delle patch.
Considera TUTTE le fonti (local + hubble) per ogni oggetto.

FLUSSO:
Step 1 → Step 3 → Step 2 → Step 4
         (registrazione) (calcolo patch) (estrazione patch)
"""

import os
import glob
import json
import logging
from datetime import datetime
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from tqdm import tqdm
from collections import defaultdict

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
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Input: immagini registrate da step2
INPUT_REGISTERED_DIR = os.path.join(BASE_DIR, 'img_register')

# Output: configurazione patch
OUTPUT_CONFIG_DIR = os.path.join(BASE_DIR, 'patch_configs')

# ============================================================================
# MENU INTERATTIVO
# ============================================================================

def list_available_objects():
    """Lista tutti gli oggetti disponibili (unione di tutte le fonti)."""
    objects = set()
    
    if not os.path.exists(INPUT_REGISTERED_DIR):
        return []
    
    # Cerca in tutte le fonti (local, hubble)
    for source in os.listdir(INPUT_REGISTERED_DIR):
        source_path = os.path.join(INPUT_REGISTERED_DIR, source)
        if os.path.isdir(source_path):
            for obj_name in os.listdir(source_path):
                obj_path = os.path.join(source_path, obj_name)
                if os.path.isdir(obj_path):
                    # Verifica che ci siano FITS
                    fits_count = len(glob.glob(os.path.join(obj_path, '*.fits')) + 
                                   glob.glob(os.path.join(obj_path, '*.fit')))
                    if fits_count > 0:
                        objects.add(obj_name)
    
    return sorted(list(objects))


def count_images_for_object(obj_name):
    """Conta immagini per oggetto in tutte le fonti."""
    total = 0
    sources_info = {}
    
    for source in ['local', 'hubble']:
        source_path = os.path.join(INPUT_REGISTERED_DIR, source, obj_name)
        if os.path.exists(source_path):
            count = len(glob.glob(os.path.join(source_path, '*.fits')) + 
                       glob.glob(os.path.join(source_path, '*.fit')))
            if count > 0:
                sources_info[source] = count
                total += count
    
    return total, sources_info


def interactive_menu():
    """Menu interattivo per selezionare oggetto."""
    print("\n" + "=" * 70)
    print("🎯 SELEZIONE OGGETTO PER CALCOLO PATCH".center(70))
    print("=" * 70)
    
    objects = list_available_objects()
    
    if not objects:
        print("\n❌ Nessun oggetto trovato in data/img_register/")
        print("   Esegui prima: python scripts/step2_register.py")
        return None
    
    print("\n📦 OGGETTI DISPONIBILI:")
    print("-" * 70)
    for i, obj_name in enumerate(objects, 1):
        total, sources = count_images_for_object(obj_name)
        sources_str = ", ".join([f"{src}: {cnt}" for src, cnt in sources.items()])
        print(f"   {i}. {obj_name:<20} ({total} immagini: {sources_str})")
    print(f"   {len(objects)+1}. TUTTI (calcola per tutti gli oggetti)")
    print("-" * 70)
    
    # Input oggetto
    while True:
        try:
            choice = input(f"\n➤ Scegli oggetto [1-{len(objects)+1}]: ").strip()
            obj_idx = int(choice) - 1
            
            if obj_idx == len(objects):
                # Tutti gli oggetti
                selected_object = None
                print(f"\n✓ Calcolerò patch per TUTTI gli oggetti")
                break
            elif 0 <= obj_idx < len(objects):
                selected_object = objects[obj_idx]
                total, sources = count_images_for_object(selected_object)
                print(f"\n✓ Oggetto selezionato: {selected_object}")
                print(f"   Immagini: {total} ({', '.join([f'{s}: {c}' for s, c in sources.items()])})")
                break
            else:
                print(f"⚠️  Inserisci un numero tra 1 e {len(objects)+1}")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Operazione annullata.")
            return None
    
    # Riepilogo
    print("\n" + "=" * 70)
    print("📋 RIEPILOGO SELEZIONE")
    print("=" * 70)
    
    if selected_object:
        print(f"   Oggetto: {selected_object}")
        print(f"   Input: data/img_register/{{fonte}}/{selected_object}/")
        print(f"   Output: data/patch_configs/{selected_object}_patch_config.json")
    else:
        print(f"   Oggetti: TUTTI ({len(objects)} oggetti)")
        print(f"   Input: data/img_register/{{fonte}}/{{oggetto}}/")
        print(f"   Output: data/patch_configs/{{oggetto}}_patch_config.json")
    
    print("=" * 70)
    
    # Conferma
    confirm = input("\n➤ Confermi e procedi? [S/n]: ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ Operazione annullata.")
        return None
    
    return selected_object


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(object_name=None):
    """Setup logging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if object_name:
        log_filename = os.path.join(LOG_DIR, f'step3_patches_{object_name}_{timestamp}.log')
    else:
        log_filename = os.path.join(LOG_DIR, f'step3_patches_all_{timestamp}.log')
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"LOG FILE: {log_filename}")
    logger.info("=" * 80)
    
    return logger


# ============================================================================
# ANALISI IMMAGINI
# ============================================================================

def analyze_registered_images(obj_name, logger):
    """
    Analizza tutte le immagini registrate per un oggetto (tutte le fonti).
    
    Returns:
        dict: Statistiche aggregate di tutte le fonti
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"📊 ANALISI IMMAGINI: {obj_name}")
    logger.info("=" * 80)
    
    all_images = []
    sources_analyzed = []
    images_by_source = defaultdict(list)
    
    # Analizza tutte le fonti
    for source in ['local', 'hubble']:
        source_dir = os.path.join(INPUT_REGISTERED_DIR, source, obj_name)
        
        if not os.path.exists(source_dir):
            logger.info(f"⚠️  Fonte '{source}' non trovata per {obj_name}")
            continue
        
        logger.info(f"\n{'─'*80}")
        logger.info(f"📂 FONTE: {source.upper()}")
        logger.info(f"{'─'*80}")
        logger.info(f"Directory: {source_dir}")
        
        # Trova file FITS
        fits_files = (glob.glob(os.path.join(source_dir, '*.fits')) + 
                     glob.glob(os.path.join(source_dir, '*.fit')))
        
        if not fits_files:
            logger.info(f"⚠️  Nessun file FITS")
            continue
        
        logger.info(f"Trovati {len(fits_files)} file FITS\n")
        
        sources_analyzed.append(source)
        
        # Console header per lista immagini
        print(f"\n{'─'*90}")
        print(f"📂 FONTE: {source.upper()} ({len(fits_files)} immagini)")
        print(f"{'─'*90}")
        print(f"{'#':<4} {'Filename':<45} {'Width':<8} {'Height':<8} {'Scale':<10}")
        print(f"{'─'*90}")
        
        logger.info(f"{'#':<4} {'Filename':<50} {'Dimensioni':<15} {'Scale (\"/px)':<12}")
        logger.info("─" * 85)
        
        # Analizza ciascuna immagine
        for idx, filepath in enumerate(sorted(fits_files), 1):
            try:
                with fits.open(filepath, mode='readonly') as hdul:
                    header = hdul[0].header
                    data = hdul[0].data
                    
                    if data is None:
                        continue
                    
                    # Estrai informazioni
                    wcs = WCS(header)
                    if not wcs.has_celestial:
                        continue
                    
                    shape = data.shape  # (height, width)
                    height, width = shape
                    
                    # Pixel scale
                    try:
                        pixel_scale = header.get('NATIVESC', None)  # Da registrazione
                        if pixel_scale is None:
                            cd = wcs.wcs.cd
                            pixel_scale = np.sqrt(cd[0,0]**2 + cd[0,1]**2) * 3600
                    except:
                        pixel_scale = abs(wcs.wcs.cdelt[0]) * 3600
                    
                    # Coverage
                    coverage = header.get('REGCOVER', 100.0)
                    
                    filename = os.path.basename(filepath)
                    
                    # Info immagine
                    img_info = {
                        'file': filename,
                        'source': source,
                        'shape': shape,
                        'width': width,
                        'height': height,
                        'pixel_scale': pixel_scale,
                        'coverage': coverage,
                        'total_pixels': width * height
                    }
                    
                    all_images.append(img_info)
                    images_by_source[source].append(img_info)
                    
                    # Output console
                    print(f"{idx:<4} {filename:<45} {width:<8} {height:<8} {pixel_scale:<10.4f}")
                    
                    # Output log
                    logger.info(f"{idx:<4} {filename:<50} {width:>5}×{height:<5}px   {pixel_scale:>8.4f}")
                    
            except Exception as e:
                logger.warning(f"✗ Errore leggendo {os.path.basename(filepath)}: {e}")
                continue
        
        # Statistiche per fonte
        if images_by_source[source]:
            widths = [img['width'] for img in images_by_source[source]]
            heights = [img['height'] for img in images_by_source[source]]
            scales = [img['pixel_scale'] for img in images_by_source[source]]
            
            print(f"{'─'*90}")
            print(f"📊 STATISTICHE {source.upper()}:")
            print(f"   Numero immagini: {len(images_by_source[source])}")
            print(f"   Dimensioni:")
            print(f"      Width:  min={min(widths)}px, max={max(widths)}px, media={np.mean(widths):.0f}px")
            print(f"      Height: min={min(heights)}px, max={max(heights)}px, media={np.mean(heights):.0f}px")
            print(f"   Risoluzione:")
            print(f"      Min: {min(scales):.4f}\"/px, Max: {max(scales):.4f}\"/px, Media: {np.mean(scales):.4f}\"/px")
            
            logger.info("")
            logger.info(f"📊 STATISTICHE {source.upper()}:")
            logger.info(f"   Totale immagini: {len(images_by_source[source])}")
            logger.info(f"   Width:  {min(widths)} - {max(widths)} px (media: {np.mean(widths):.0f})")
            logger.info(f"   Height: {min(heights)} - {max(heights)} px (media: {np.mean(heights):.0f})")
            logger.info(f"   Scale:  {min(scales):.4f} - {max(scales):.4f} \"/px (media: {np.mean(scales):.4f})")
    
    if not all_images:
        logger.warning(f"❌ Nessuna immagine valida per {obj_name}")
        return None
    
    # Calcola statistiche aggregate TOTALI
    heights = [img['height'] for img in all_images]
    widths = [img['width'] for img in all_images]
    pixel_scales = [img['pixel_scale'] for img in all_images]
    coverages = [img['coverage'] for img in all_images]
    
    stats = {
        'object': obj_name,
        'sources': sources_analyzed,
        'num_images': len(all_images),
        'images_per_source': {
            src: len(images_by_source[src]) for src in sources_analyzed
        },
        'images_list': all_images,  # Lista completa per riferimento
        'shape_stats': {
            'min_height': min(heights),
            'max_height': max(heights),
            'avg_height': np.mean(heights),
            'median_height': np.median(heights),
            'min_width': min(widths),
            'max_width': max(widths),
            'avg_width': np.mean(widths),
            'median_width': np.median(widths)
        },
        'pixel_scale_stats': {
            'min': min(pixel_scales),
            'max': max(pixel_scales),
            'avg': np.mean(pixel_scales),
            'median': np.median(pixel_scales),
            'std': np.std(pixel_scales)
        },
        'coverage_stats': {
            'min': min(coverages),
            'max': max(coverages),
            'avg': np.mean(coverages)
        }
    }
    
    # Log statistiche aggregate
    print(f"\n{'='*90}")
    print(f"📈 STATISTICHE AGGREGATE TOTALI")
    print(f"{'='*90}")
    print(f"Oggetto: {obj_name}")
    print(f"Fonti: {', '.join(sources_analyzed)}")
    print(f"Totale immagini: {stats['num_images']}")
    for src, count in stats['images_per_source'].items():
        print(f"   • {src}: {count} immagini")
    print(f"\n📐 DIMENSIONI COMPLESSIVE:")
    print(f"   Width:  {stats['shape_stats']['min_width']} - {stats['shape_stats']['max_width']} px")
    print(f"           Media: {stats['shape_stats']['avg_width']:.0f}px, Mediana: {stats['shape_stats']['median_width']:.0f}px")
    print(f"   Height: {stats['shape_stats']['min_height']} - {stats['shape_stats']['max_height']} px")
    print(f"           Media: {stats['shape_stats']['avg_height']:.0f}px, Mediana: {stats['shape_stats']['median_height']:.0f}px")
    print(f"\n📏 RISOLUZIONE COMPLESSIVA:")
    print(f"   Min:    {stats['pixel_scale_stats']['min']:.4f}\"/px")
    print(f"   Max:    {stats['pixel_scale_stats']['max']:.4f}\"/px")
    print(f"   Media:  {stats['pixel_scale_stats']['avg']:.4f}\"/px")
    print(f"   Mediana: {stats['pixel_scale_stats']['median']:.4f}\"/px")
    print(f"   Std:    {stats['pixel_scale_stats']['std']:.4f}\"/px")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("📈 STATISTICHE AGGREGATE TOTALI")
    logger.info("=" * 80)
    logger.info(f"Oggetto: {obj_name}")
    logger.info(f"Fonti analizzate: {', '.join(sources_analyzed)}")
    logger.info(f"Totale immagini: {stats['num_images']}")
    for src, count in stats['images_per_source'].items():
        logger.info(f"  • {src}: {count} immagini")
    
    logger.info("")
    logger.info("📐 DIMENSIONI:")
    logger.info(f"  Width:  {stats['shape_stats']['min_width']} - {stats['shape_stats']['max_width']} px")
    logger.info(f"          Media: {stats['shape_stats']['avg_width']:.0f}, Mediana: {stats['shape_stats']['median_width']:.0f}")
    logger.info(f"  Height: {stats['shape_stats']['min_height']} - {stats['shape_stats']['max_height']} px")
    logger.info(f"          Media: {stats['shape_stats']['avg_height']:.0f}, Mediana: {stats['shape_stats']['median_height']:.0f}")
    
    logger.info("")
    logger.info("📏 RISOLUZIONE:")
    logger.info(f"  Min:    {stats['pixel_scale_stats']['min']:.4f}\"/px")
    logger.info(f"  Max:    {stats['pixel_scale_stats']['max']:.4f}\"/px")
    logger.info(f"  Media:  {stats['pixel_scale_stats']['avg']:.4f}\"/px")
    logger.info(f"  Mediana: {stats['pixel_scale_stats']['median']:.4f}\"/px")
    logger.info(f"  Std:    {stats['pixel_scale_stats']['std']:.4f}\"/px")
    
    logger.info("")
    logger.info("📊 COVERAGE:")
    logger.info(f"  Min:    {stats['coverage_stats']['min']:.1f}%")
    logger.info(f"  Max:    {stats['coverage_stats']['max']:.1f}%")
    logger.info(f"  Media:  {stats['coverage_stats']['avg']:.1f}%")
    
    return stats


# ============================================================================
# CALCOLO DIMENSIONI PATCH
# ============================================================================

def calculate_patch_dimensions(stats, logger):
    """
    Calcola dimensioni ottimali delle patch basandosi sulle statistiche.
    Propone multiple opzioni basate su percentuali dell'immagine.
    
    Args:
        stats: Statistiche aggregate
        logger: Logger
    
    Returns:
        dict: Configurazione patch
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("🧮 CALCOLO DIMENSIONI PATCH")
    logger.info("=" * 80)
    
    # Usa mediana come riferimento (più robusta agli outlier)
    median_height = stats['shape_stats']['median_height']
    median_width = stats['shape_stats']['median_width']
    avg_scale = stats['pixel_scale_stats']['avg']
    
    logger.info(f"Dimensioni mediane immagine: {median_width:.0f}×{median_height:.0f}px")
    logger.info(f"Risoluzione media: {avg_scale:.4f}\"/px")
    
    # Calcola opzioni patch basate su frazioni dell'immagine
    options = []
    fractions = [1/4, 1/8, 1/16]  # 1/4, 1/8, 1/16 dell'area
    
    print(f"\n{'='*90}")
    print(f"🧮 OPZIONI DIMENSIONI PATCH")
    print(f"{'='*90}")
    print(f"Basate su dimensioni mediane: {median_width:.0f}×{median_height:.0f}px\n")
    print(f"{'Opzione':<10} {'Frazione':<12} {'Patch Size':<15} {'Overlap':<12} {'Stride':<12} {'FOV':<15} {'Patch/Img':<12}")
    print(f"{'─'*90}")
    
    logger.info("")
    logger.info("📦 OPZIONI PATCH CALCOLATE:")
    logger.info(f"{'Opzione':<10} {'Frazione':<12} {'Size':<12} {'Overlap':<10} {'Stride':<10} {'FOV (arcmin)':<15} {'Patch/Img':<12}")
    logger.info("─" * 85)
    
    for frac in fractions:
        # Area target
        target_area = (median_width * median_height) * frac
        target_size = int(np.sqrt(target_area))
        
        # Arrotonda a multiplo di 16 (per CNN)
        patch_size = max(128, (target_size // 16) * 16)
        
        # Limiti ragionevoli
        patch_size = min(2048, max(128, patch_size))
        
        # Overlap: 25%
        overlap = patch_size // 4
        stride = patch_size - overlap
        
        # FOV angolare
        fov_arcmin = (patch_size * avg_scale) / 60.0
        fov_arcsec = patch_size * avg_scale
        
        # Stima patch per immagine mediana
        patches_x = int(np.ceil((median_width - patch_size) / stride) + 1) if median_width >= patch_size else 0
        patches_y = int(np.ceil((median_height - patch_size) / stride) + 1) if median_height >= patch_size else 0
        patches_per_img = patches_x * patches_y
        
        option = {
            'fraction': frac,
            'patch_size': patch_size,
            'overlap': overlap,
            'stride': stride,
            'fov_arcmin': fov_arcmin,
            'fov_arcsec': fov_arcsec,
            'patches_per_image': patches_per_img,
            'total_patches': patches_per_img * stats['num_images']
        }
        options.append(option)
        
        # Console output
        frac_str = f"1/{int(1/frac)}"
        print(f"{len(options):<10} {frac_str:<12} {patch_size}×{patch_size}px{'':<4} {overlap}px{'':<5} {stride}px{'':<5} {fov_arcmin:.2f}'{'':<10} {patches_per_img}")
        
        # Log output
        logger.info(f"{len(options):<10} {frac_str:<12} {patch_size}px{'':<6} {overlap}px{'':<4} {stride}px{'':<4} {fov_arcmin:.2f}{'':<11} {patches_per_img}")
    
    print(f"{'─'*90}")
    print(f"\n💡 RACCOMANDAZIONE:")
    print(f"   • Opzione 2 (1/8): Bilancio tra dettaglio e numero patch")
    print(f"   • Opzione 3 (1/16): Massimo dettaglio, più patch")
    
    logger.info("")
    logger.info("💡 Raccomandazione: Opzione 2 (1/8) per bilanciamento ottimale")
    
    # Usa opzione centrale (1/8) come default
    default_option = options[1] if len(options) > 1 else options[0]
    
    # Configurazione finale
    config = {
        'object': stats['object'],
        'sources': stats['sources'],
        'created_date': datetime.now().isoformat(),
        'num_images_analyzed': stats['num_images'],
        'images_per_source': stats['images_per_source'],
        'statistics': {
            'median_image_size': {
                'width': int(median_width),
                'height': int(median_height)
            },
            'avg_image_size': {
                'width': int(stats['shape_stats']['avg_width']),
                'height': int(stats['shape_stats']['avg_height'])
            },
            'size_range': {
                'min_width': int(stats['shape_stats']['min_width']),
                'max_width': int(stats['shape_stats']['max_width']),
                'min_height': int(stats['shape_stats']['min_height']),
                'max_height': int(stats['shape_stats']['max_height'])
            },
            'pixel_scale': {
                'avg': float(stats['pixel_scale_stats']['avg']),
                'median': float(stats['pixel_scale_stats']['median']),
                'min': float(stats['pixel_scale_stats']['min']),
                'max': float(stats['pixel_scale_stats']['max'])
            }
        },
        'patch_options': options,
        'recommended_config': {
            'patch_size': default_option['patch_size'],
            'overlap': default_option['overlap'],
            'stride': default_option['stride'],
            'min_coverage': 80.0  # % minimo di pixel validi
        },
        'estimated_patches': {
            'per_image': default_option['patches_per_image'],
            'total': default_option['total_patches']
        }
    }
    
    return config


# ============================================================================
# SALVATAGGIO CONFIGURAZIONE
# ============================================================================

def save_patch_config(config, logger):
    """Salva configurazione patch su file JSON."""
    os.makedirs(OUTPUT_CONFIG_DIR, exist_ok=True)
    
    obj_name = config['object']
    config_file = os.path.join(OUTPUT_CONFIG_DIR, f'{obj_name}_patch_config.json')
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info("")
    logger.info(f"💾 Configurazione salvata: {config_file}")
    
    return config_file


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Funzione principale."""
    print("=" * 70)
    print("🧮 STEP 2: CALCOLO DIMENSIONI PATCH".center(70))
    print("=" * 70)
    
    # Menu interattivo
    selected_object = interactive_menu()
    
    if selected_object is None:
        print("\n❌ Nessuna selezione effettuata.")
        return
    
    # Setup logging
    logger = setup_logging(selected_object)
    logger.info("=" * 80)
    logger.info(f"STEP 2: CALCOLO PATCH" + (f" - {selected_object}" if selected_object else " - TUTTI"))
    logger.info("=" * 80)
    logger.info(f"Data/Ora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    if selected_object:
        # Singolo oggetto
        print(f"\n🔍 Analisi immagini per: {selected_object}")
        
        stats = analyze_registered_images(selected_object, logger)
        
        if not stats:
            print(f"\n❌ Nessuna immagine valida per {selected_object}")
            return
        
        # Calcola patch
        config = calculate_patch_dimensions(stats, logger)
        
        # Salva
        config_file = save_patch_config(config, logger)
        
        # Riepilogo
        recommended = config['recommended_config']
        print(f"\n{'='*90}")
        print(f"📊 CONFIGURAZIONE SALVATA")
        print(f"{'='*90}")
        print(f"   Oggetto: {selected_object}")
        print(f"   File: {config_file}")
        print(f"\n   📦 Configurazione raccomandata (1/8 area):")
        print(f"      Patch size: {recommended['patch_size']}×{recommended['patch_size']}px")
        print(f"      Overlap: {recommended['overlap']}px ({(recommended['overlap']/recommended['patch_size'])*100:.0f}%)")
        print(f"      Stride: {recommended['stride']}px")
        print(f"      Min coverage: {recommended['min_coverage']}%")
        print(f"\n   📊 Patch stimate: ~{config['estimated_patches']['total']}")
        
    else:
        # TUTTI gli oggetti
        objects = list_available_objects()
        
        print(f"\n🔄 Calcolo patch per {len(objects)} oggetti...")
        
        results = []
        
        for obj_idx, obj_name in enumerate(objects, 1):
            print(f"\n{'='*90}")
            print(f"📦 OGGETTO {obj_idx}/{len(objects)}: {obj_name}")
            print(f"{'='*90}")
            
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"OGGETTO {obj_idx}/{len(objects)}: {obj_name}")
            logger.info("=" * 80)
            
            stats = analyze_registered_images(obj_name, logger)
            
            if not stats:
                print(f"   ⚠️  Nessuna immagine valida")
                continue
            
            config = calculate_patch_dimensions(stats, logger)
            config_file = save_patch_config(config, logger)
            
            results.append({
                'object': obj_name,
                'config_file': config_file,
                'num_images': config['num_images_analyzed'],
                'patch_size': config['recommended_config']['patch_size'],
                'estimated_patches': config['estimated_patches']['total']
            })
            
            print(f"\n   ✓ Configurazione salvata")
        
        # Riepilogo totale
        print(f"\n{'='*90}")
        print(f"📊 RIEPILOGO TOTALE")
        print(f"{'='*90}")
        print(f"   Oggetti processati: {len(results)}")
        
        total_images = sum(r['num_images'] for r in results)
        total_patches = sum(r['estimated_patches'] for r in results)
        
        print(f"   Totale immagini: {total_images}")
        print(f"   Totale patch stimate: ~{total_patches}")
        print(f"\n   📁 Config salvate in: {OUTPUT_CONFIG_DIR}/")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 RIEPILOGO FINALE")
        logger.info("=" * 80)
        logger.info(f"Oggetti processati: {len(results)}")
        logger.info(f"Totale immagini: {total_images}")
        logger.info(f"Totale patch stimate: ~{total_patches}")
    
    logger.info("")
    logger.info(f"Data/Ora fine: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    print(f"\n✅ STEP 2 COMPLETATO!")
    print(f"\n   ➡️  Prossimo passo: python scripts/step4_extract_patches.py")
    print(f"       (estrae patch usando la configurazione calcolata)")


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️  Tempo totale: {elapsed:.2f}s")