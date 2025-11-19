"""
Diagnostica completa di TUTTI i target per identificare problemi di allineamento e dati.
Analizza:
1. File registrati (3_registered_native)
2. Mosaici finali (5_mosaics)
3. Allineamento WCS tra tile
4. Metadati e trasformazioni coordinate
FIX: Aggiunta comparazione side-by-side Hubble vs Observatory
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

ROOT_DATA = Path(r"c:\Users\fratt\Documents\GitHub\SuperResolution\data")
OUTPUT_DIR = Path(__file__).parent  # Salva plot nella cartella diagnose_scripts

# ============================================================================
# ANALISI WCS E ALLINEAMENTO
# ============================================================================

def extract_wcs_corners(filepath):
    """Estrae le coordinate degli angoli di un'immagine."""
    try:
        with fits.open(filepath) as hdul:
            header = hdul[0].header
            data = hdul[0].data
            
            if data is None:
                return None
            
            if len(data.shape) == 3:
                ny, nx = data.shape[1:3]
            else:
                ny, nx = data.shape
            
            wcs = WCS(header)
            
            if not wcs.has_celestial:
                return None
            
            corners_pix = np.array([
                [0, 0], [nx-1, 0], [0, ny-1], [nx-1, ny-1], [nx/2, ny/2]
            ])
            
            corners_world = wcs.pixel_to_world(corners_pix[:, 0], corners_pix[:, 1])
            
            corners = {
                'bl': (corners_world[0].ra.deg, corners_world[0].dec.deg),
                'br': (corners_world[1].ra.deg, corners_world[1].dec.deg),
                'tl': (corners_world[2].ra.deg, corners_world[2].dec.deg),
                'tr': (corners_world[3].ra.deg, corners_world[3].dec.deg),
                'center': (corners_world[4].ra.deg, corners_world[4].dec.deg)
            }
            
            try:
                if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
                    cd = wcs.wcs.cd
                    pixel_scale_deg = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
                else:
                    pixel_scale_deg = abs(wcs.wcs.cdelt[0])
                pixel_scale_arcsec = pixel_scale_deg * 3600
            except:
                pixel_scale_arcsec = None
            
            all_ra = [corners['bl'][0], corners['br'][0], corners['tl'][0], corners['tr'][0]]
            all_dec = [corners['bl'][1], corners['br'][1], corners['tl'][1], corners['tr'][1]]
            
            return {
                'corners': corners,
                'bounds': {
                    'ra_min': min(all_ra),
                    'ra_max': max(all_ra),
                    'dec_min': min(all_dec),
                    'dec_max': max(all_dec),
                    'ra_span': max(all_ra) - min(all_ra),
                    'dec_span': max(all_dec) - min(all_dec)
                },
                'pixel_scale': pixel_scale_arcsec,
                'shape': (ny, nx),
                'crval': wcs.wcs.crval,
                'crpix': wcs.wcs.crpix
            }
            
    except Exception as e:
        return None

def check_alignment(wcs_info_list):
    """Verifica allineamento tra più immagini."""
    if len(wcs_info_list) < 2:
        return {'status': 'SINGLE_IMAGE', 'overlap': None}
    
    overlaps = []
    
    for i in range(len(wcs_info_list)):
        for j in range(i+1, len(wcs_info_list)):
            info_a = wcs_info_list[i]
            info_b = wcs_info_list[j]
            
            bounds_a = info_a['bounds']
            bounds_b = info_b['bounds']
            
            ra_overlap_min = max(bounds_a['ra_min'], bounds_b['ra_min'])
            ra_overlap_max = min(bounds_a['ra_max'], bounds_b['ra_max'])
            dec_overlap_min = max(bounds_a['dec_min'], bounds_b['dec_min'])
            dec_overlap_max = min(bounds_a['dec_max'], bounds_b['dec_max'])
            
            has_overlap = (ra_overlap_max > ra_overlap_min) and (dec_overlap_max > dec_overlap_min)
            
            if has_overlap:
                overlap_area = (ra_overlap_max - ra_overlap_min) * (dec_overlap_max - dec_overlap_min)
                area_a = bounds_a['ra_span'] * bounds_a['dec_span']
                area_b = bounds_b['ra_span'] * bounds_b['dec_span']
                overlap_pct = overlap_area / min(area_a, area_b) * 100
            else:
                overlap_area = 0
                overlap_pct = 0
            
            center_a = info_a['corners']['center']
            center_b = info_b['corners']['center']
            
            delta_ra = (center_b[0] - center_a[0]) * 3600
            delta_dec = (center_b[1] - center_a[1]) * 3600
            distance_arcsec = np.sqrt(delta_ra**2 + delta_dec**2)
            
            overlaps.append({
                'pair': (info_a['file'], info_b['file']),
                'has_overlap': has_overlap,
                'overlap_pct': overlap_pct,
                'distance_arcsec': distance_arcsec,
                'delta_ra_arcsec': delta_ra,
                'delta_dec_arcsec': delta_dec
            })
    
    total_overlaps = sum(1 for o in overlaps if o['has_overlap'])
    avg_distance = np.mean([o['distance_arcsec'] for o in overlaps])
    
    if total_overlaps == 0:
        status = 'NO_OVERLAP'
        diagnosis = "❌ NESSUN OVERLAP"
    elif total_overlaps < len(overlaps):
        status = 'PARTIAL_OVERLAP'
        diagnosis = "⚠️ OVERLAP PARZIALE"
    else:
        if avg_distance < 100:
            status = 'GOOD_ALIGNMENT'
            diagnosis = "✅ ALLINEAMENTO BUONO"
        else:
            status = 'POOR_ALIGNMENT'
            diagnosis = f"⚠️ ALLINEAMENTO SCARSO: {avg_distance:.1f}\""
    
    return {
        'status': status,
        'diagnosis': diagnosis,
        'overlaps': overlaps,
        'total_overlaps': total_overlaps,
        'avg_distance': avg_distance
    }

def analyze_fits_detailed(filepath):
    """Analizza un file FITS in dettaglio."""
    try:
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            
            if data is None:
                return None
            
            if len(data.shape) == 3:
                data = data[0]
            
            nan_mask = np.isnan(data)
            inf_mask = np.isinf(data)
            valid_data = data[~(nan_mask | inf_mask)]
            
            if valid_data.size == 0:
                return {'status': 'ALL_NAN', 'shape': data.shape, 'nan_pct': 100.0}
            
            stats = {
                'status': 'OK',
                'shape': data.shape,
                'dtype': str(data.dtype),
                'nan_pct': (np.sum(nan_mask) / data.size * 100),
                'inf_pct': (np.sum(inf_mask) / data.size * 100),
                'min': float(valid_data.min()),
                'max': float(valid_data.max()),
                'mean': float(valid_data.mean()),
                'median': float(np.median(valid_data)),
                'std': float(valid_data.std()),
                'p01': float(np.percentile(valid_data, 0.1)),
                'p1': float(np.percentile(valid_data, 1)),
                'p99': float(np.percentile(valid_data, 99)),
                'p999': float(np.percentile(valid_data, 99.9)),
                'zero_pct': (np.sum(valid_data == 0) / valid_data.size * 100),
                'negative_pct': (np.sum(valid_data < 0) / valid_data.size * 100)
            }
            
            if stats['min'] > 0:
                stats['dyn_range'] = stats['max'] / stats['min']
            else:
                stats['dyn_range'] = float('inf')
            
            stats['max_mean_ratio'] = stats['max'] / stats['mean'] if stats['mean'] > 0 else float('inf')
            stats['p99_median_ratio'] = stats['p99'] / stats['median'] if stats['median'] > 0 else float('inf')
            
            stats['bunit'] = header.get('BUNIT', 'N/A')
            stats['exptime'] = header.get('EXPTIME', 'N/A')
            stats['filter'] = header.get('FILTER', 'N/A')
            stats['aligned'] = header.get('ALIGNED', False)
            stats['globcenr'] = header.get('GLOBCENR', 'N/A')
            stats['globcend'] = header.get('GLOBCEND', 'N/A')
            
            unique_values = len(np.unique(valid_data[:10000]))
            stats['unique_sample'] = unique_values
            stats['likely_binary'] = (unique_values < 50)
            
            return stats
            
    except Exception as e:
        return {'status': 'ERROR', 'error': str(e)}

# ============================================================================
# DIAGNOSTICA TARGET
# ============================================================================

def diagnose_target(target_dir):
    """Diagnostica completa di un singolo target."""
    target_name = target_dir.name
    
    print(f"\n{'='*70}")
    print(f"📊 TARGET: {target_name}")
    print(f"{'='*70}")
    
    results = {
        'target': target_name,
        'hubble_wcs': [],
        'obs_wcs': [],
        'hubble_data': [],
        'obs_data': [],
        'mosaic': None,
        'alignment': None
    }
    
    # === ANALISI WCS ===
    print(f"\n{'─'*70}")
    print("🗺️  ANALISI WCS E ALLINEAMENTO")
    print(f"{'─'*70}")
    
    hubble_dir = target_dir / '3_registered_native' / 'hubble'
    obs_dir = target_dir / '3_registered_native' / 'observatory'
    
    all_wcs_info = []
    
    if hubble_dir.exists():
        hubble_files = list(hubble_dir.glob('*.fits'))
        print(f"\n📂 Hubble: {len(hubble_files)} file")
        
        for filepath in hubble_files:
            wcs_info = extract_wcs_corners(filepath)
            if wcs_info:
                wcs_info['file'] = filepath.name
                wcs_info['source'] = 'hubble'
                results['hubble_wcs'].append(wcs_info)
                all_wcs_info.append(wcs_info)
                
                print(f"\n   📄 {filepath.name}")
                print(f"      Centro: RA={wcs_info['crval'][0]:.6f}°, DEC={wcs_info['crval'][1]:.6f}°")
    
    if obs_dir.exists():
        obs_files = list(obs_dir.glob('*.fits'))
        if obs_files:
            print(f"\n📂 Observatory: {len(obs_files)} file")
            
            for filepath in obs_files[:3]:
                wcs_info = extract_wcs_corners(filepath)
                if wcs_info:
                    wcs_info['file'] = filepath.name
                    wcs_info['source'] = 'observatory'
                    results['obs_wcs'].append(wcs_info)
                    all_wcs_info.append(wcs_info)
                    
                    print(f"\n   📄 {filepath.name}")
                    print(f"      Centro: RA={wcs_info['crval'][0]:.6f}°, DEC={wcs_info['crval'][1]:.6f}°")
    
    # === VERIFICA ALLINEAMENTO ===
    if len(all_wcs_info) >= 2:
        print(f"\n{'─'*70}")
        print("🎯 VERIFICA ALLINEAMENTO")
        print(f"{'─'*70}")
        
        alignment = check_alignment(all_wcs_info)
        results['alignment'] = alignment
        
        print(f"\n{alignment['diagnosis']}")
        print(f"   Overlap totali: {alignment['total_overlaps']}/{len(alignment['overlaps'])}")
        print(f"   Distanza media: {alignment['avg_distance']:.1f} arcsec")
    
    # === ANALISI MOSAICO ===
    mosaic_path = target_dir / '5_mosaics' / 'final_mosaic.fits'
    if mosaic_path.exists():
        print(f"\n{'─'*70}")
        print("🖼️  MOSAICO FINALE")
        print(f"{'─'*70}")
        
        stats = analyze_fits_detailed(mosaic_path)
        
        if stats and stats['status'] == 'OK':
            results['mosaic'] = stats
            print(f"\n   Range: [{stats['min']:.3e}, {stats['max']:.3e}]")
            print(f"   Median: {stats['median']:.3e}")
            print(f"   {'✅ OK' if not stats['likely_binary'] else '❌ BINARIO'}")
    
    return results

# ============================================================================
# VISUALIZZAZIONE COMPARATIVA
# ============================================================================

def create_comparison_plot(results):
    """
    Crea plot comparativo completo per un target.
    Layout: 3 colonne x N righe (Hubble tiles, Observatory tiles, Mosaico finale)
    """
    target = results['target']
    target_dir = ROOT_DATA / target
    
    # Conta file disponibili
    hubble_files = list((target_dir / '3_registered_native' / 'hubble').glob('*.fits'))
    obs_files = list((target_dir / '3_registered_native' / 'observatory').glob('*.fits'))
    mosaic_file = target_dir / '5_mosaics' / 'final_mosaic.fits'
    
    has_mosaic = mosaic_file.exists()
    
    # Determina layout
    n_hubble = len(hubble_files)
    n_obs = len(obs_files)
    n_rows = max(n_hubble, n_obs, 1 if has_mosaic else 0)
    
    if n_rows == 0:
        print(f"   ⏭️  {target}: Nessun file da visualizzare")
        return
    
    # Crea figura
    fig = plt.figure(figsize=(18, 6*n_rows))
    gs = GridSpec(n_rows, 3, figure=fig, hspace=0.3, wspace=0.2)
    
    # Funzione helper per caricare e normalizzare
    def load_and_normalize(filepath):
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            if len(data.shape) == 3:
                data = data[0]
            
            valid = np.isfinite(data) & (data > 0)
            if not valid.any():
                return data, 0, 1
            
            vmin = np.percentile(data[valid], 1)
            vmax = np.percentile(data[valid], 99.9)
            return data, vmin, vmax
    
    # === COLONNA 1: HUBBLE ===
    for i, filepath in enumerate(hubble_files[:n_rows]):
        ax = fig.add_subplot(gs[i, 0])
        try:
            data, vmin, vmax = load_and_normalize(filepath)
            ax.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax, interpolation='none')
            ax.set_title(f'Hubble #{i+1}\n{filepath.stem[:40]}', fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading\n{filepath.name}', 
                   ha='center', va='center', fontsize=10, color='red')
        ax.axis('off')
    
    # Riempi celle vuote
    for i in range(n_hubble, n_rows):
        ax = fig.add_subplot(gs[i, 0])
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14, color='gray')
        ax.axis('off')
    
    # === COLONNA 2: OBSERVATORY ===
    for i, filepath in enumerate(obs_files[:n_rows]):
        ax = fig.add_subplot(gs[i, 1])
        try:
            data, vmin, vmax = load_and_normalize(filepath)
            ax.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax, interpolation='none')
            ax.set_title(f'Observatory #{i+1}\n{filepath.stem[:40]}', fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading\n{filepath.name}', 
                   ha='center', va='center', fontsize=10, color='red')
        ax.axis('off')
    
    # Riempi celle vuote
    for i in range(n_obs, n_rows):
        ax = fig.add_subplot(gs[i, 1])
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14, color='gray')
        ax.axis('off')
    
    # === COLONNA 3: MOSAICO (span tutte le righe) ===
    if has_mosaic:
        ax = fig.add_subplot(gs[:, 2])
        try:
            data, vmin, vmax = load_and_normalize(mosaic_file)
            
            im = ax.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax, interpolation='none')
            ax.set_title(f'Final Mosaic\n({data.shape[1]}x{data.shape[0]} px)', fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Intensity', fontsize=10)
            
            # Statistiche
            stats = results.get('mosaic')
            if stats:
                stats_text = (
                    f"Range: [{stats['min']:.2e}, {stats['max']:.2e}]\n"
                    f"Median: {stats['median']:.2e}\n"
                    f"p99/med: {stats['p99_median_ratio']:.1f}\n"
                    f"Status: {'✅ OK' if not stats['likely_binary'] else '❌ Binary'}"
                )
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading mosaic\n{str(e)}', 
                   ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')
    else:
        # Nessun mosaico: mostra messaggio
        ax = fig.add_subplot(gs[:, 2])
        ax.text(0.5, 0.5, 'No mosaic available', ha='center', va='center', 
               fontsize=14, color='gray')
        ax.axis('off')
    
    # Titolo generale - FIX: gestione alignment None
    alignment = results.get('alignment')
    if alignment and isinstance(alignment, dict):
        alignment_status = alignment.get('diagnosis', 'N/A')
    else:
        alignment_status = 'Singola immagine (nessun allineamento richiesto)'
    
    fig.suptitle(f'{target} - Comparazione Completa\nAllineamento: {alignment_status}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Salva
    output_path = OUTPUT_DIR / f'{target}_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Comparazione salvata: {output_path}")
    plt.close()

def plot_wcs_alignment(results):
    """Plot allineamento WCS."""
    target = results['target']
    all_wcs = results['hubble_wcs'] + results['obs_wcs']
    
    if len(all_wcs) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors_map = {'hubble': 'red', 'observatory': 'blue'}
    
    for wcs_info in all_wcs:
        corners = wcs_info['corners']
        source = wcs_info['source']
        color = colors_map.get(source, 'gray')
        
        ra_vals = [corners['bl'][0], corners['br'][0], corners['tr'][0], corners['tl'][0], corners['bl'][0]]
        dec_vals = [corners['bl'][1], corners['br'][1], corners['tr'][1], corners['tl'][1], corners['bl'][1]]
        
        ax.plot(ra_vals, dec_vals, color=color, linewidth=2, 
               label=source if source not in [l.get_label() for l in ax.get_lines()] else "")
        
        ax.plot(corners['center'][0], corners['center'][1], 'o', color=color, markersize=8)
        
        short_name = Path(wcs_info['file']).stem[:15]
        ax.text(corners['center'][0], corners['center'][1], short_name, 
               fontsize=8, ha='center', va='bottom', color=color)
    
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('DEC (deg)')
    ax.set_title(f'{target} - Allineamento WCS')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f'{target}_wcs_alignment.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 WCS alignment: {output_path}")
    plt.close()

def create_diagnostic_summary(all_results):
    """Riepilogo finale."""
    print(f"\n{'='*70}")
    print("📋 RIEPILOGO DIAGNOSTICA")
    print(f"{'='*70}")
    
    for result in all_results:
        target = result['target']
        print(f"\n🎯 {target}")
        
        # Conteggio file
        n_hubble = len(result['hubble_wcs'])
        n_obs = len(result['obs_wcs'])
        print(f"   File registrati: Hubble={n_hubble}, Observatory={n_obs}")
        
        # Allineamento
        if result['alignment']:
            print(f"   Allineamento: {result['alignment']['diagnosis']}")
        
        # Mosaico
        if result.get('mosaic'):
            mosaic = result['mosaic']
            status = '✅ OK' if not mosaic.get('likely_binary') else '❌ BINARIO'
            print(f"   Mosaico: {status}")

def main():
    """Funzione principale."""
    print("="*70)
    print("🔬 DIAGNOSTICA COMPLETA + COMPARAZIONE".center(70))
    print("="*70)
    
    target_dirs = [d for d in ROOT_DATA.iterdir() if d.is_dir() and d.name.startswith('M')]
    
    if not target_dirs:
        print(f"\n❌ Nessun target in {ROOT_DATA}")
        return
    
    print(f"\n📂 Target: {[d.name for d in target_dirs]}")
    
    all_results = []
    for target_dir in sorted(target_dirs):
        result = diagnose_target(target_dir)
        all_results.append(result)
        
        # Plot WCS
        if len(result['hubble_wcs']) + len(result['obs_wcs']) >= 2:
            plot_wcs_alignment(result)
        
        # Plot comparazione
        create_comparison_plot(result)
    
    create_diagnostic_summary(all_results)
    
    print(f"\n{'='*70}")
    print(f"✅ File salvati in: {OUTPUT_DIR}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()