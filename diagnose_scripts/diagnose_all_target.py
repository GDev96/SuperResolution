"""
Diagnostica completa di TUTTI i target per identificare problemi di allineamento e dati.
Analizza:
1. File registrati (3_registered_native)
2. Mosaici finali (5_mosaics)
3. Allineamento WCS tra tile
4. Metadati e trasformazioni coordinate
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

ROOT_DATA = Path(r"c:\Users\fratt\Documents\GitHub\SuperResolution\data")

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
            
            # 4 angoli + centro
            corners_pix = np.array([
                [0, 0],           # Bottom-left
                [nx-1, 0],        # Bottom-right
                [0, ny-1],        # Top-left
                [nx-1, ny-1],     # Top-right
                [nx/2, ny/2]      # Center
            ])
            
            corners_world = wcs.pixel_to_world(corners_pix[:, 0], corners_pix[:, 1])
            
            corners = {
                'bl': (corners_world[0].ra.deg, corners_world[0].dec.deg),
                'br': (corners_world[1].ra.deg, corners_world[1].dec.deg),
                'tl': (corners_world[2].ra.deg, corners_world[2].dec.deg),
                'tr': (corners_world[3].ra.deg, corners_world[3].dec.deg),
                'center': (corners_world[4].ra.deg, corners_world[4].dec.deg)
            }
            
            # Calcola pixel scale
            try:
                if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
                    cd = wcs.wcs.cd
                    pixel_scale_deg = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
                else:
                    pixel_scale_deg = abs(wcs.wcs.cdelt[0])
                pixel_scale_arcsec = pixel_scale_deg * 3600
            except:
                pixel_scale_arcsec = None
            
            # Bounds
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
    
    # Calcola overlap tra tutte le coppie
    overlaps = []
    
    for i in range(len(wcs_info_list)):
        for j in range(i+1, len(wcs_info_list)):
            info_a = wcs_info_list[i]
            info_b = wcs_info_list[j]
            
            bounds_a = info_a['bounds']
            bounds_b = info_b['bounds']
            
            # Calcola overlap
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
            
            # Distanza tra centri
            center_a = info_a['corners']['center']
            center_b = info_b['corners']['center']
            
            delta_ra = (center_b[0] - center_a[0]) * 3600  # arcsec
            delta_dec = (center_b[1] - center_a[1]) * 3600  # arcsec
            distance_arcsec = np.sqrt(delta_ra**2 + delta_dec**2)
            
            overlaps.append({
                'pair': (info_a['file'], info_b['file']),
                'has_overlap': has_overlap,
                'overlap_pct': overlap_pct,
                'distance_arcsec': distance_arcsec,
                'delta_ra_arcsec': delta_ra,
                'delta_dec_arcsec': delta_dec
            })
    
    # Analizza pattern
    total_overlaps = sum(1 for o in overlaps if o['has_overlap'])
    avg_distance = np.mean([o['distance_arcsec'] for o in overlaps])
    
    # Diagnosi
    if total_overlaps == 0:
        status = 'NO_OVERLAP'
        diagnosis = "❌ NESSUN OVERLAP: Immagini completamente separate!"
    elif total_overlaps < len(overlaps):
        status = 'PARTIAL_OVERLAP'
        diagnosis = "⚠️ OVERLAP PARZIALE: Alcune immagini non si sovrappongono"
    else:
        if avg_distance < 100:  # < 100 arcsec
            status = 'GOOD_ALIGNMENT'
            diagnosis = "✅ ALLINEAMENTO BUONO"
        else:
            status = 'POOR_ALIGNMENT'
            diagnosis = f"⚠️ ALLINEAMENTO SCARSO: Distanza media {avg_distance:.1f} arcsec"
    
    return {
        'status': status,
        'diagnosis': diagnosis,
        'overlaps': overlaps,
        'total_overlaps': total_overlaps,
        'avg_distance': avg_distance
    }

# ============================================================================
# ANALISI DATI
# ============================================================================

def analyze_fits_detailed(filepath):
    """Analizza un file FITS in dettaglio."""
    try:
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            
            # Gestisci 3D
            if data is None:
                return None
            
            if len(data.shape) == 3:
                data = data[0]
            
            # Statistiche
            nan_mask = np.isnan(data)
            inf_mask = np.isinf(data)
            valid_data = data[~(nan_mask | inf_mask)]
            
            if valid_data.size == 0:
                return {
                    'status': 'ALL_NAN',
                    'shape': data.shape,
                    'nan_pct': 100.0
                }
            
            # Calcola statistiche robuste
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
            
            # Range dinamico
            if stats['min'] > 0:
                stats['dyn_range'] = stats['max'] / stats['min']
            else:
                stats['dyn_range'] = float('inf')
            
            # Ratio importanti
            stats['max_mean_ratio'] = stats['max'] / stats['mean'] if stats['mean'] > 0 else float('inf')
            stats['p99_median_ratio'] = stats['p99'] / stats['median'] if stats['median'] > 0 else float('inf')
            
            # Header info importanti
            stats['bunit'] = header.get('BUNIT', 'N/A')
            stats['exptime'] = header.get('EXPTIME', 'N/A')
            stats['filter'] = header.get('FILTER', 'N/A')
            stats['nativesc'] = header.get('NATIVESC', 'N/A')
            stats['regcov'] = header.get('REGCOV', 'N/A')
            stats['normlow'] = header.get('NORMLOW', 'N/A')
            stats['normhigh'] = header.get('NORMHIGH', 'N/A')
            stats['aligned'] = header.get('ALIGNED', False)
            stats['globcenr'] = header.get('GLOBCENR', 'N/A')
            stats['globcend'] = header.get('GLOBCEND', 'N/A')
            
            # Controlla se è "binario" (pochi valori unici)
            unique_values = len(np.unique(valid_data[:10000]))  # Sample più grande
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
    
    # === ANALISI WCS FILE REGISTRATI ===
    print(f"\n{'─'*70}")
    print("🗺️  ANALISI WCS E ALLINEAMENTO")
    print(f"{'─'*70}")
    
    hubble_dir = target_dir / '3_registered_native' / 'hubble'
    obs_dir = target_dir / '3_registered_native' / 'observatory'
    
    all_wcs_info = []
    
    # Hubble
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
                print(f"      CRPIX: [{wcs_info['crpix'][0]:.2f}, {wcs_info['crpix'][1]:.2f}]")
                print(f"      Bounds: RA=[{wcs_info['bounds']['ra_min']:.6f}, {wcs_info['bounds']['ra_max']:.6f}]")
                print(f"              DEC=[{wcs_info['bounds']['dec_min']:.6f}, {wcs_info['bounds']['dec_max']:.6f}]")
                print(f"      Pixel Scale: {wcs_info['pixel_scale']:.4f}\"/px")
    
    # Observatory
    if obs_dir.exists():
        obs_files = list(obs_dir.glob('*.fits'))
        if obs_files:
            print(f"\n📂 Observatory: {len(obs_files)} file")
            
            for filepath in obs_files[:3]:  # Max 3 per non sovracaricare
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
        print(f"   Distanza media centri: {alignment['avg_distance']:.1f} arcsec")
        
        # Dettagli overlap
        for overlap in alignment['overlaps']:
            file_a = Path(overlap['pair'][0]).stem[:30]
            file_b = Path(overlap['pair'][1]).stem[:30]
            
            if overlap['has_overlap']:
                print(f"\n   ✓ {file_a}")
                print(f"     ↔ {file_b}")
                print(f"     Overlap: {overlap['overlap_pct']:.1f}%")
                print(f"     ΔRA: {overlap['delta_ra_arcsec']:.1f}\", ΔDEC: {overlap['delta_dec_arcsec']:.1f}\"")
            else:
                print(f"\n   ❌ {file_a}")
                print(f"      ↔ {file_b}")
                print(f"      NESSUN OVERLAP (distanza: {overlap['distance_arcsec']:.1f} arcsec)")
    
    # === ANALISI DATI ===
    print(f"\n{'─'*70}")
    print("📊 ANALISI QUALITÀ DATI")
    print(f"{'─'*70}")
    
    # Hubble data
    if hubble_dir.exists():
        print(f"\n📂 Hubble - Qualità Dati")
        for filepath in list(hubble_dir.glob('*.fits'))[:2]:  # Max 2
            stats = analyze_fits_detailed(filepath)
            
            if stats and stats['status'] == 'OK':
                results['hubble_data'].append({'file': filepath.name, 'stats': stats})
                
                print(f"\n   📄 {filepath.name}")
                print(f"      Range: [{stats['min']:.3e}, {stats['max']:.3e}]")
                print(f"      Median: {stats['median']:.3e}")
                print(f"      p99/median: {stats['p99_median_ratio']:.1f}")
                print(f"      Unique values (sample): {stats['unique_sample']}")
                print(f"      ALIGNED={stats['aligned']}, GLOBCENR={stats['globcenr']}, GLOBCEND={stats['globcend']}")
                
                if stats['likely_binary']:
                    print(f"      ⚠️  BINARIO RILEVATO!")
    
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
            print(f"   p99/median: {stats['p99_median_ratio']:.1f}")
            print(f"   Unique values: {stats['unique_sample']}")
            
            if stats['likely_binary']:
                print(f"\n   ❌ MOSAICO BINARIO!")
            else:
                print(f"\n   ✅ Mosaico OK")
    
    return results

# ============================================================================
# CONFRONTO E VISUALIZZAZIONE
# ============================================================================

def plot_wcs_alignment(results):
    """Crea plot dell'allineamento WCS."""
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
        
        # Disegna box
        ra_vals = [corners['bl'][0], corners['br'][0], corners['tr'][0], corners['tl'][0], corners['bl'][0]]
        dec_vals = [corners['bl'][1], corners['br'][1], corners['tr'][1], corners['tl'][1], corners['bl'][1]]
        
        ax.plot(ra_vals, dec_vals, color=color, linewidth=2, label=source if source not in [l.get_label() for l in ax.get_lines()] else "")
        
        # Marca centro
        ax.plot(corners['center'][0], corners['center'][1], 'o', color=color, markersize=8)
        
        # Nome file abbreviato
        short_name = Path(wcs_info['file']).stem[:15]
        ax.text(corners['center'][0], corners['center'][1], short_name, 
               fontsize=8, ha='center', va='bottom', color=color)
    
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('DEC (deg)')
    ax.set_title(f'{target} - Allineamento WCS')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # RA decresce verso destra
    
    plt.tight_layout()
    plt.savefig(f'{target}_wcs_alignment.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Plot WCS salvato: {target}_wcs_alignment.png")
    plt.close()

def create_diagnostic_summary(all_results):
    """Crea summary completo con tutti i target."""
    print(f"\n{'='*70}")
    print("📋 RIEPILOGO DIAGNOSTICA")
    print(f"{'='*70}")
    
    for result in all_results:
        target = result['target']
        
        print(f"\n🎯 {target}")
        
        # Allineamento
        if result['alignment']:
            align = result['alignment']
            print(f"   WCS: {align['diagnosis']}")
            
            if align['status'] in ['NO_OVERLAP', 'POOR_ALIGNMENT']:
                print(f"   ⚠️  PROBLEMA ALLINEAMENTO RILEVATO!")
                print(f"       → File registrati ma NON allineati nello stesso frame")
                print(f"       → Causa: reproject_image_native() crea WCS diversi per ogni file")
        
        # Dati
        mosaic = result.get('mosaic')
        if mosaic:
            if mosaic.get('likely_binary'):
                print(f"   Dati: ❌ BINARIO ({mosaic['unique_sample']} valori)")
            else:
                print(f"   Dati: ✅ OK (range dinamico corretto)")
    
    # === DIAGNOSI FINALE ===
    print(f"\n{'='*70}")
    print("💡 DIAGNOSI FINALE")
    print(f"{'='*70}")
    
    # Conta problemi
    alignment_issues = sum(1 for r in all_results if r.get('alignment') and r['alignment']['status'] in ['NO_OVERLAP', 'POOR_ALIGNMENT'])
    binary_issues = sum(1 for r in all_results if r.get('mosaic') and r['mosaic'].get('likely_binary'))
    
    if alignment_issues > 0:
        print(f"\n❌ PROBLEMA PRINCIPALE: ALLINEAMENTO WCS")
        print(f"   {alignment_issues} target con allineamento errato")
        print(f"\n🔧 CAUSA:")
        print(f"   • reproject_image_native() crea un WCS TARGET diverso per ogni immagine")
        print(f"   • Ogni file ha: target_wcs.wcs.crval = wcs_orig.wcs.crval  (linea ~688)")
        print(f"   • Risultato: file con centri diversi non si sovrappongono nel mosaico")
        print(f"\n💡 SOLUZIONE:")
        print(f"   • Implementare register_to_unified_frame() come nell'ultimo fix")
        print(f"   • Calcolare UN SOLO frame WCS globale per tutto il gruppo")
        print(f"   • Reproiettare TUTTE le immagini in quel frame condiviso")
    
    if binary_issues > 0:
        print(f"\n❌ PROBLEMA SECONDARIO: DATI BINARI")
        print(f"   {binary_issues} target con dati binarizzati")
        print(f"\n🔧 CAUSA:")
        print(f"   • Normalizzazione/denormalizzazione errata durante reproiezione")
        print(f"\n💡 SOLUZIONE:")
        print(f"   • Già implementata con fix MIN_RANGE_THRESHOLD = 0.01")
        print(f"   • Verificare che NORMLOW/NORMHIGH siano corretti nell'header")

def main():
    """Funzione principale."""
    print("="*70)
    print("🔬 DIAGNOSTICA COMPLETA: WCS + ALLINEAMENTO + DATI".center(70))
    print("="*70)
    
    # Trova tutti i target
    target_dirs = [d for d in ROOT_DATA.iterdir() if d.is_dir() and d.name.startswith('M')]
    
    if not target_dirs:
        print(f"\n❌ Nessun target trovato in {ROOT_DATA}")
        return
    
    print(f"\n📂 Trovati {len(target_dirs)} target: {[d.name for d in target_dirs]}")
    
    # Analizza tutti
    all_results = []
    for target_dir in sorted(target_dirs):
        result = diagnose_target(target_dir)
        all_results.append(result)
        
        # Plot WCS alignment
        if len(result['hubble_wcs']) + len(result['obs_wcs']) >= 2:
            plot_wcs_alignment(result)
    
    # Summary finale
    create_diagnostic_summary(all_results)
    
    print(f"\n{'='*70}")
    print("✅ DIAGNOSTICA COMPLETATA")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()