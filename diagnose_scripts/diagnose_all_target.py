"""
Diagnostica completa di TUTTI i target per identificare il problema sistematico.
Analizza i file registrati e i mosaici finali.
"""

import numpy as np
from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as colors

ROOT_DATA = Path(r"c:\Users\fratt\Documents\GitHub\SuperResolution\data")

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
            stats['max_median_ratio'] = stats['max'] / stats['median'] if stats['median'] > 0 else float('inf')
            stats['p99_median_ratio'] = stats['p99'] / stats['median'] if stats['median'] > 0 else float('inf')
            
            # Header info
            stats['bunit'] = header.get('BUNIT', 'N/A')
            stats['exptime'] = header.get('EXPTIME', 'N/A')
            stats['filter'] = header.get('FILTER', 'N/A')
            
            # Controlla se è "binario" (pochi valori unici)
            unique_values = len(np.unique(valid_data[:1000]))  # Sample
            stats['unique_sample'] = unique_values
            stats['likely_binary'] = (unique_values < 10)
            
            return stats
            
    except Exception as e:
        return {'status': 'ERROR', 'error': str(e)}

def diagnose_target(target_dir):
    """Diagnostica un singolo target."""
    target_name = target_dir.name
    
    print(f"\n{'='*70}")
    print(f"📊 TARGET: {target_name}")
    print(f"{'='*70}")
    
    results = {
        'target': target_name,
        'registered': [],
        'mosaic': None
    }
    
    # Analizza file registrati
    registered_dirs = [
        target_dir / '3_registered_native' / 'hubble',
        target_dir / '3_registered_native' / 'observatory'
    ]
    
    for reg_dir in registered_dirs:
        if not reg_dir.exists():
            continue
        
        source = reg_dir.parent.name + '/' + reg_dir.name
        fits_files = list(reg_dir.glob('*.fits'))
        
        if not fits_files:
            continue
        
        print(f"\n📂 {source}: {len(fits_files)} file")
        
        for filepath in fits_files[:3]:  # Primi 3 file
            stats = analyze_fits_detailed(filepath)
            
            if stats and stats['status'] == 'OK':
                results['registered'].append({
                    'file': filepath.name,
                    'source': source,
                    'stats': stats
                })
                
                # Report
                print(f"\n   📄 {filepath.name}")
                print(f"      Shape: {stats['shape']}, Dtype: {stats['dtype']}")
                print(f"      Range: [{stats['min']:.3e}, {stats['max']:.3e}]")
                print(f"      Mean: {stats['mean']:.3e}, Median: {stats['median']:.3e}")
                print(f"      p99/median: {stats['p99_median_ratio']:.1f}")
                print(f"      NaN: {stats['nan_pct']:.2f}%")
                print(f"      Unique values (sample): {stats['unique_sample']}")
                
                if stats['likely_binary']:
                    print(f"      ⚠️  POSSIBILE PROBLEMA: Pochi valori unici (binario?)")
            
            elif stats and stats['status'] == 'ALL_NAN':
                print(f"\n   ❌ {filepath.name}: TUTTI NaN")
            else:
                print(f"\n   ❌ {filepath.name}: Errore lettura")
    
    # Analizza mosaico finale
    mosaic_path = target_dir / '5_mosaics' / 'final_mosaic.fits'
    if mosaic_path.exists():
        print(f"\n🖼️  MOSAICO FINALE")
        stats = analyze_fits_detailed(mosaic_path)
        
        if stats and stats['status'] == 'OK':
            results['mosaic'] = stats
            
            print(f"\n   Shape: {stats['shape']}, Dtype: {stats['dtype']}")
            print(f"   Range: [{stats['min']:.3e}, {stats['max']:.3e}]")
            print(f"   Mean: {stats['mean']:.3e}, Median: {stats['median']:.3e}")
            print(f"   p99/median: {stats['p99_median_ratio']:.1f}")
            print(f"   Unique values (sample): {stats['unique_sample']}")
            
            if stats['likely_binary']:
                print(f"   ⚠️  MOSAICO BINARIO RILEVATO!")
                print(f"   🔍 CAUSA POSSIBILE:")
                print(f"      - Normalizzazione errata durante registrazione")
                print(f"      - Conversione tipo dati con perdita informazione")
                print(f"      - Stacking con algoritmo errato")
    
    return results

def compare_all_targets(all_results):
    """Confronta tutti i target per trovare pattern."""
    print(f"\n{'='*70}")
    print("🔬 CONFRONTO COMPLETO TRA TARGET")
    print(f"{'='*70}")
    
    # Trova target funzionante (M82) e problematici
    working_target = None
    broken_targets = []
    
    for result in all_results:
        if result['mosaic']:
            if not result['mosaic'].get('likely_binary', False):
                working_target = result
                print(f"\n✅ TARGET FUNZIONANTE: {result['target']}")
            else:
                broken_targets.append(result)
                print(f"\n❌ TARGET PROBLEMATICO: {result['target']}")
    
    if working_target and broken_targets:
        print(f"\n{'='*70}")
        print("💡 ANALISI DIFFERENZE")
        print(f"{'='*70}")
        
        wt_mosaic = working_target['mosaic']
        
        print(f"\n📊 {working_target['target']} (FUNZIONANTE):")
        print(f"   Range: [{wt_mosaic['min']:.3e}, {wt_mosaic['max']:.3e}]")
        print(f"   p99/median: {wt_mosaic['p99_median_ratio']:.1f}")
        print(f"   Unique values: {wt_mosaic['unique_sample']}")
        
        for bt in broken_targets:
            bt_mosaic = bt['mosaic']
            print(f"\n📊 {bt['target']} (PROBLEMATICO):")
            print(f"   Range: [{bt_mosaic['min']:.3e}, {bt_mosaic['max']:.3e}]")
            print(f"   p99/median: {bt_mosaic['p99_median_ratio']:.1f}")
            print(f"   Unique values: {bt_mosaic['unique_sample']}")
            
            # Analizza file registrati
            if bt['registered']:
                print(f"\n   File registrati analizzati:")
                for reg in bt['registered'][:2]:
                    reg_stats = reg['stats']
                    print(f"      - {reg['file']}")
                    print(f"        Range: [{reg_stats['min']:.3e}, {reg_stats['max']:.3e}]")
                    print(f"        p99/median: {reg_stats['p99_median_ratio']:.1f}")
                    print(f"        Unique: {reg_stats['unique_sample']}")
                    
                    if reg_stats['likely_binary']:
                        print(f"        ⚠️  GIÀ BINARIO DOPO REGISTRAZIONE!")
        
        # DIAGNOSI FINALE
        print(f"\n{'='*70}")
        print("🎯 DIAGNOSI CAUSA")
        print(f"{'='*70}")
        
        # Controlla se problema è già nei file registrati
        broken_in_registration = False
        for bt in broken_targets:
            for reg in bt['registered']:
                if reg['stats'].get('likely_binary', False):
                    broken_in_registration = True
                    break
        
        if broken_in_registration:
            print(f"\n⚠️  PROBLEMA IDENTIFICATO: File già binari DOPO registrazione (Step 2)")
            print(f"\n🔧 CAUSA PROBABILE:")
            print(f"   1. reproject_interp normalizza dati ma non de-normalizza correttamente")
            print(f"   2. Footprint > 1 crea valori fuori scala")
            print(f"   3. Conversione float32 senza preservare range originale")
            print(f"\n💡 SOLUZIONE:")
            print(f"   • Implementare normalizzazione percentile-based DENTRO reproject_image_native()")
            print(f"   • Salvare parametri normalizzazione nell'header")
            print(f"   • De-normalizzare PRIMA del salvataggio")
        else:
            print(f"\n⚠️  PROBLEMA IDENTIFICATO: Binario solo nel mosaico (Step 4)")
            print(f"\n🔧 CAUSA PROBABILE:")
            print(f"   1. Algoritmo stacking usa mediana su valori normalizzati")
            print(f"   2. Conversione tipo dati finale errata")
            print(f"\n💡 SOLUZIONE:")
            print(f"   • Verificare step2_croppedmosaico.py")
            print(f"   • Assicurare che stacking preservi range dinamico")

def create_comparison_plot(all_results):
    """Crea plot comparativo per tutti i target."""
    fig, axes = plt.subplots(len(all_results), 2, figsize=(12, 4*len(all_results)))
    
    if len(all_results) == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(all_results):
        target_name = result['target']
        
        # Plot mosaico
        if result['mosaic']:
            mosaic_path = ROOT_DATA / target_name / '5_mosaics' / 'final_mosaic.fits'
            
            try:
                with fits.open(mosaic_path) as hdul:
                    data = hdul[0].data
                    if len(data.shape) == 3:
                        data = data[0]
                    
                    # Linear scale
                    valid = data[np.isfinite(data)]
                    p1, p99 = np.percentile(valid, [1, 99])
                    
                    axes[i, 0].imshow(data, cmap='gray', vmin=p1, vmax=p99, origin='lower')
                    axes[i, 0].set_title(f'{target_name} - Mosaico (p1-p99)')
                    axes[i, 0].axis('off')
                    
                    # Histogram
                    axes[i, 1].hist(valid.flatten(), bins=200, alpha=0.7, log=True)
                    axes[i, 1].set_xlabel('Valore')
                    axes[i, 1].set_ylabel('Frequenza (log)')
                    axes[i, 1].set_title(f'{target_name} - Distribuzione')
                    axes[i, 1].grid(True, alpha=0.3)
                    
                    # Marca se binario
                    if result['mosaic'].get('likely_binary', False):
                        axes[i, 0].text(0.5, 0.95, '⚠️ BINARIO', 
                                       transform=axes[i, 0].transAxes,
                                       ha='center', va='top',
                                       color='red', fontsize=14, fontweight='bold',
                                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
                    else:
                        axes[i, 0].text(0.5, 0.95, '✅ OK', 
                                       transform=axes[i, 0].transAxes,
                                       ha='center', va='top',
                                       color='green', fontsize=14, fontweight='bold',
                                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            except Exception as e:
                axes[i, 0].text(0.5, 0.5, f'Errore: {e}', ha='center', va='center')
                axes[i, 0].axis('off')
                axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_all_targets.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Plot salvato: comparison_all_targets.png")
    plt.close()

def main():
    """Funzione principale."""
    print("="*70)
    print("🔬 DIAGNOSTICA COMPLETA TUTTI I TARGET".center(70))
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
    
    # Confronta
    compare_all_targets(all_results)
    
    # Crea plot
    create_comparison_plot(all_results)
    
    print(f"\n{'='*70}")
    print("✅ DIAGNOSTICA COMPLETATA")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()