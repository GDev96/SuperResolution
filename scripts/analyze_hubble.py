"""
Analizza le 101 immagini registrate per preparare il dataset SR
"""
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# ✅ CONFIGURAZIONE OGGETTO CELESTE
# Cambia questo valore per elaborare oggetti diversi (M42, M33, NGC2024, etc.)
TARGET_OBJECT = "M42"  # <-- MODIFICA QUI IL NOME DELL'OGGETTO

class HubbleAnalyzer:
    def __init__(self, target_object=TARGET_OBJECT, registered_dir=f'data/img_register_4/{TARGET_OBJECT}'):
        self.target_object = target_object
        self.registered_dir = Path(registered_dir)
        self.analysis_results = {}
    
    def analyze_all_images(self):
        """Analizza tutte le immagini registrate"""
        fits_files = sorted(list(self.registered_dir.glob('*.fit*')))
        
        print(f"Trovate {len(fits_files)} immagini registrate")
        
        all_info = []
        
        for fpath in tqdm(fits_files, desc="Analisi immagini"):
            info = self.analyze_single_image(fpath)
            if info:
                all_info.append(info)
        
        # Statistiche globali
        self.print_statistics(all_info)
        
        # Salva risultati
        self.save_analysis(all_info)
        
        # Visualizza footprints
        self.plot_coverage(all_info)
        
        return all_info
    
    def analyze_single_image(self, fpath):
        """Analizza singola immagine"""
        try:
            with fits.open(fpath) as hdul:
                data = hdul[0].data
                header = hdul[0].header
                
                # WCS
                try:
                    wcs = WCS(header)
                    wcs_valid = True
                    
                    # Centro
                    ny, nx = data.shape
                    center_sky = wcs.pixel_to_world(nx/2, ny/2)
                    
                    # Scala pixel
                    pixel_scale = wcs.proj_plane_pixel_scales()[0].to('arcsec').value
                    
                except:
                    wcs_valid = False
                    center_sky = None
                    pixel_scale = None
                
                # Statistiche dati
                valid_data = data[~np.isnan(data)]
                
                info = {
                    'filename': fpath.name,
                    'path': str(fpath),
                    'shape': data.shape,
                    'wcs_valid': wcs_valid,
                    'center_ra': center_sky.ra.degree if center_sky else None,
                    'center_dec': center_sky.dec.degree if center_sky else None,
                    'pixel_scale_arcsec': pixel_scale,
                    'data_min': float(np.min(valid_data)),
                    'data_max': float(np.max(valid_data)),
                    'data_mean': float(np.mean(valid_data)),
                    'data_median': float(np.median(valid_data)),
                    'data_std': float(np.std(valid_data)),
                    'n_pixels': int(valid_data.size),
                    'exposure_time': header.get('EXPTIME', None),
                    'filter': header.get('FILTER', header.get('FILTNAM1', 'Unknown'))
                }
                
                return info
                
        except Exception as e:
            print(f"Errore con {fpath.name}: {e}")
            return None
    
    def print_statistics(self, all_info):
        """Stampa statistiche globali"""
        print("\n" + "="*60)
        print("ANALISI IMMAGINI HUBBLE REGISTRATE")
        print("="*60)
        
        valid_wcs = [i for i in all_info if i['wcs_valid']]
        
        print(f"\nTotale immagini: {len(all_info)}")
        print(f"WCS validi: {len(valid_wcs)}/{len(all_info)}")
        
        if valid_wcs:
            # Dimensioni
            shapes = [i['shape'] for i in all_info]
            unique_shapes = set(shapes)
            print(f"\nDimensioni uniche:")
            for shape in sorted(unique_shapes):
                count = shapes.count(shape)
                print(f"  {shape}: {count} immagini")
            
            # Scala pixel
            scales = [i['pixel_scale_arcsec'] for i in valid_wcs]
            print(f"\nScala pixel:")
            print(f"  Media: {np.mean(scales):.4f} arcsec/pixel")
            print(f"  Range: {np.min(scales):.4f} - {np.max(scales):.4f}")
            
            # Coverage
            ras = [i['center_ra'] for i in valid_wcs]
            decs = [i['center_dec'] for i in valid_wcs]
            print(f"\nCopertura cielo:")
            print(f"  RA:  {np.min(ras):.4f}° - {np.max(ras):.4f}°")
            print(f"  DEC: {np.min(decs):.4f}° - {np.max(decs):.4f}°")
            print(f"  Span RA: {np.max(ras) - np.min(ras):.4f}°")
            print(f"  Span DEC: {np.max(decs) - np.min(decs):.4f}°")
            
            # Filtro
            filters = [i['filter'] for i in all_info]
            unique_filters = set(filters)
            print(f"\nFiltri:")
            for filt in unique_filters:
                count = filters.count(filt)
                print(f"  {filt}: {count} immagini")
            
            # Esposizioni
            exptimes = [i['exposure_time'] for i in all_info if i['exposure_time']]
            if exptimes:
                print(f"\nTempo esposizione:")
                print(f"  Range: {np.min(exptimes):.1f} - {np.max(exptimes):.1f} s")
                print(f"  Media: {np.mean(exptimes):.1f} s")
        
        print("="*60 + "\n")
    
    def save_analysis(self, all_info):
        """Salva analisi in JSON"""
        output_file = Path('results') / self.target_object / 'hubble_analysis.json'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(all_info, f, indent=2)
        
        print(f"Analisi salvata in: {output_file}")
    
    def plot_coverage(self, all_info):
        """Visualizza copertura footprints"""
        valid_wcs = [i for i in all_info if i['wcs_valid']]
        
        if not valid_wcs:
            print("Nessun WCS valido per visualizzazione")
            return
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        ras = [i['center_ra'] for i in valid_wcs]
        decs = [i['center_dec'] for i in valid_wcs]
        
        # Plot centri
        scatter = ax.scatter(ras, decs, c=range(len(valid_wcs)), 
                           cmap='viridis', alpha=0.6, s=100)
        
        # Numera alcune immagini per riferimento
        for i, info in enumerate(valid_wcs[::10]):  # Ogni 10
            ax.annotate(f"{i*10}", 
                       (info['center_ra'], info['center_dec']),
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('RA (degrees)', fontsize=12)
        ax.set_ylabel('DEC (degrees)', fontsize=12)
        ax.set_title(f'Copertura {len(valid_wcs)} Immagini Hubble Registrate\nM42 Hα', 
                    fontsize=14, fontweight='bold')
        ax.invert_xaxis()  # Convenzione astronomica
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.colorbar(scatter, label='Image Index', ax=ax)
        
        output_file = Path('results/visualizations') / self.target_object
        output_file.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file / f'{self.target_object}_coverage.png', dpi=150, bbox_inches='tight')
        print(f"Mappa coverage salvata in: {output_file / f'{self.target_object}_coverage.png'}")
        plt.close()
    
    def select_best_images(self, all_info, criteria='std', top_n=None):
        """
        Seleziona le migliori immagini per training
        
        criteria: 'std' (varianza), 'median' (SNR), 'all'
        """
        valid = [i for i in all_info if i['wcs_valid']]
        
        if criteria == 'std':
            # Più dettagli = più std
            sorted_imgs = sorted(valid, key=lambda x: x['data_std'], reverse=True)
        elif criteria == 'median':
            # SNR più alto
            sorted_imgs = sorted(valid, key=lambda x: x['data_median'], reverse=True)
        else:
            sorted_imgs = valid
        
        if top_n:
            sorted_imgs = sorted_imgs[:top_n]
        
        print(f"\nTop {len(sorted_imgs)} immagini (criterio: {criteria}):")
        for i, img in enumerate(sorted_imgs[:10], 1):
            print(f"{i:2d}. {img['filename']:30s} - "
                  f"std: {img['data_std']:.2e}, "
                  f"median: {img['data_median']:.2e}")
        
        return sorted_imgs

if __name__ == "__main__":
    print(f"🔭 Analisi immagini per oggetto: {TARGET_OBJECT}")
    analyzer = HubbleAnalyzer(target_object=TARGET_OBJECT)
    all_info = analyzer.analyze_all_images()
    
    # Opzionale: seleziona subset migliori
    # best = analyzer.select_best_images(all_info, criteria='std', top_n=80)