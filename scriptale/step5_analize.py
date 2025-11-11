#!/usr/bin/env python3
"""
STEP 5: ANALISI OVERLAP E DIMENSIONI PATCHES
Analizza le immagini con WCS per determinare:
1. Overlap tra Hubble e Osservatorio
2. Dimensioni ottimali delle patches
3. Strategia di allineamento

Input: Cartelle con immagini cropped o registrate
Script generalizzato per tutti i target.
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime
import glob

# ============================================================================
# CONFIGURAZIONE - Usa le cartelle corrette
# ============================================================================

BASE_DIR = r'F:\Super Revolt Gaia\parte 2(patch)\data'

# Opzione 1: Immagini CROPPED (consigliato per patches uniformi)
HUBBLE_DIR = Path(BASE_DIR) / '4_cropped' / 'hubble'
OBS_DIR = Path(BASE_DIR) / '4_cropped' / 'observatory'

# Opzione 2: Immagini REGISTRATE (se vuoi usare quelle non cropped)
# HUBBLE_DIR = Path(BASE_DIR) / '3_registered_native' / 'hubble'
# OBS_DIR = Path(BASE_DIR) / '3_registered_native' / 'observatory'

OUTPUT_DIR = Path(BASE_DIR) / 'analisi_overlap'

# Parametri patches
TARGET_PATCH_ARCMIN = 1.0  # Dimensione target in arcmin
PATCH_OVERLAP_PERCENT = 10  # Overlap tra patches (%)

# ============================================================================
# CLASSI
# ============================================================================

class ImageAnalyzer:
    """Analizza immagine FITS con WCS"""
    
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.data = None
        self.header = None
        self.wcs = None
        self.info = {}
        
    def load(self):
        """Carica e analizza immagine"""
        try:
            with fits.open(self.filepath) as hdul:
                # Trova HDU con dati
                data_hdu = None
                for i, hdu in enumerate(hdul):
                    if hdu.data is not None and len(hdu.data.shape) >= 2:
                        data_hdu = hdu
                        break
                
                if data_hdu is None:
                    return False
                
                self.data = data_hdu.data
                self.header = data_hdu.header
                
                # Se 3D, usa primo canale
                if len(self.data.shape) == 3:
                    self.data = self.data[0]
                
                ny, nx = self.data.shape
                
                # Info base
                self.info = {
                    'filename': self.filepath.name,
                    'shape': (ny, nx),
                    'dtype': str(self.data.dtype),
                    'size_mb': round(self.data.nbytes / (1024**2), 2),
                }
                
                # Statistiche
                valid_mask = np.isfinite(self.data)
                valid_data = self.data[valid_mask]
                
                if len(valid_data) > 0:
                    self.info['stats'] = {
                        'valid_pixels': int(valid_mask.sum()),
                        'coverage_percent': round(100 * valid_mask.sum() / self.data.size, 2),
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'mean': float(np.mean(valid_data)),
                        'median': float(np.median(valid_data)),
                    }
                
                # WCS
                try:
                    self.wcs = WCS(self.header)
                    if self.wcs.has_celestial:
                        self._analyze_wcs()
                    else:
                        self.wcs = None
                        return False
                except:
                    self.wcs = None
                    return False
                
                return True
                
        except Exception as e:
            print(f"   ✗ Errore caricamento {self.filepath.name}: {e}")
            return False
    
    def _analyze_wcs(self):
        """Analizza WCS e calcola FOV"""
        ny, nx = self.data.shape
        
        # Centro
        center = self.wcs.pixel_to_world(nx/2, ny/2)
        
        # Pixel scale
        pixel_scale = self._get_pixel_scale()
        
        # FOV dai corners
        corners = self.wcs.pixel_to_world([0, nx, nx, 0], [0, 0, ny, ny])
        ra_vals = [c.ra.deg for c in corners]
        dec_vals = [c.dec.deg for c in corners]
        
        ra_span = max(ra_vals) - min(ra_vals)
        dec_span = max(dec_vals) - min(dec_vals)
        
        self.info['wcs'] = {
            'center_ra': float(center.ra.deg),
            'center_dec': float(center.dec.deg),
            'pixel_scale_arcsec': pixel_scale,
            'pixel_scale_arcmin': pixel_scale / 60.0,
            'fov_ra_deg': ra_span,
            'fov_dec_deg': dec_span,
            'fov_ra_arcmin': ra_span * 60,
            'fov_dec_arcmin': dec_span * 60,
            'ra_range': [min(ra_vals), max(ra_vals)],
            'dec_range': [min(dec_vals), max(dec_vals)],
        }
    
    def _get_pixel_scale(self):
        """Calcola pixel scale in arcsec"""
        try:
            if hasattr(self.wcs.wcs, 'cd') and self.wcs.wcs.cd is not None:
                cd = self.wcs.wcs.cd
                pixel_scale_deg = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
            elif hasattr(self.wcs.wcs, 'cdelt'):
                pixel_scale_deg = abs(self.wcs.wcs.cdelt[0])
            else:
                p1 = self.wcs.pixel_to_world(0, 0)
                p2 = self.wcs.pixel_to_world(1, 0)
                pixel_scale_deg = p1.separation(p2).deg
            
            return pixel_scale_deg * 3600
        except:
            return None
    
    def calculate_patch_size(self, target_arcmin):
        """
        Calcola dimensione patch ottimale in pixel.
        
        Returns:
            dict con info patch
        """
        if 'wcs' not in self.info or self.info['wcs']['pixel_scale_arcsec'] is None:
            return None
        
        pixel_scale_arcsec = self.info['wcs']['pixel_scale_arcsec']
        target_arcsec = target_arcmin * 60
        
        # Dimensione in pixel
        patch_size_px = int(target_arcsec / pixel_scale_arcsec)
        
        # Arrotonda a multiplo di 8 (comune per reti neurali)
        patch_size_px = ((patch_size_px + 7) // 8) * 8
        
        # Dimensione effettiva
        actual_arcsec = patch_size_px * pixel_scale_arcsec
        actual_arcmin = actual_arcsec / 60
        
        # Numero patches
        ny, nx = self.data.shape
        
        # Senza overlap
        n_x = nx // patch_size_px
        n_y = ny // patch_size_px
        total_no_overlap = n_x * n_y
        
        # Con overlap
        step = int(patch_size_px * (1 - PATCH_OVERLAP_PERCENT/100))
        n_x_overlap = max(1, (nx - patch_size_px) // step + 1) if step > 0 else 1
        n_y_overlap = max(1, (ny - patch_size_px) // step + 1) if step > 0 else 1
        total_with_overlap = n_x_overlap * n_y_overlap
        
        return {
            'target_arcmin': target_arcmin,
            'patch_size_px': patch_size_px,
            'actual_size_arcmin': round(actual_arcmin, 4),
            'actual_size_arcsec': round(actual_arcsec, 2),
            'patches_no_overlap': {'x': n_x, 'y': n_y, 'total': total_no_overlap},
            'patches_with_overlap': {'x': n_x_overlap, 'y': n_y_overlap, 'total': total_with_overlap},
            'step_size': step,
        }


class OverlapAnalyzer:
    """Analizza overlap tra due immagini con WCS"""
    
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.overlap_info = None
    
    def calculate_overlap(self):
        """Calcola overlap tra le due immagini"""
        if self.img1.wcs is None or self.img2.wcs is None:
            return None
        
        ny1, nx1 = self.img1.data.shape
        ny2, nx2 = self.img2.data.shape
        
        # Corners img1
        corners1 = self.img1.wcs.pixel_to_world([0, nx1, nx1, 0], [0, 0, ny1, ny1])
        ra1 = [c.ra.deg for c in corners1]
        dec1 = [c.dec.deg for c in corners1]
        
        # Corners img2
        corners2 = self.img2.wcs.pixel_to_world([0, nx2, nx2, 0], [0, 0, ny2, ny2])
        ra2 = [c.ra.deg for c in corners2]
        dec2 = [c.dec.deg for c in corners2]
        
        # Overlap region
        ra_min = max(min(ra1), min(ra2))
        ra_max = min(max(ra1), max(ra2))
        dec_min = max(min(dec1), min(dec2))
        dec_max = min(max(dec1), max(dec2))
        
        # Check se c'è overlap
        if ra_min >= ra_max or dec_min >= dec_max:
            return None
        
        # Area overlap (approssimata, rettangolare)
        overlap_ra = ra_max - ra_min
        overlap_dec = dec_max - dec_min
        overlap_area_deg2 = overlap_ra * overlap_dec
        overlap_area_arcmin2 = overlap_area_deg2 * 3600  # deg² -> arcmin²
        
        # Area totali
        area1_deg2 = (max(ra1) - min(ra1)) * (max(dec1) - min(dec1))
        area2_deg2 = (max(ra2) - min(ra2)) * (max(dec2) - min(dec2))
        
        # Frazioni
        fraction_img1 = overlap_area_deg2 / area1_deg2 if area1_deg2 > 0 else 0
        fraction_img2 = overlap_area_deg2 / area2_deg2 if area2_deg2 > 0 else 0
        
        self.overlap_info = {
            'overlap_exists': True,
            'overlap_area_deg2': overlap_area_deg2,
            'overlap_area_arcmin2': overlap_area_arcmin2,
            'overlap_ra_deg': overlap_ra,
            'overlap_dec_deg': overlap_dec,
            'overlap_ra_arcmin': overlap_ra * 60,
            'overlap_dec_arcmin': overlap_dec * 60,
            'fraction_img1': fraction_img1,
            'fraction_img2': fraction_img2,
            'ra_range': [ra_min, ra_max],
            'dec_range': [dec_min, dec_max],
        }
        
        return self.overlap_info
    
    def visualize(self, output_path=None):
        """Crea visualizzazione overlap"""
        if self.overlap_info is None:
            print("   Calcola overlap prima di visualizzare")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Img1
        ny1, nx1 = self.img1.data.shape
        corners1 = self.img1.wcs.pixel_to_world([0, nx1, nx1, 0, 0], [0, 0, ny1, ny1, 0])
        ra1 = [c.ra.deg for c in corners1]
        dec1 = [c.dec.deg for c in corners1]
        ax.plot(ra1, dec1, 'b-', linewidth=2, label=f'Img1: {self.img1.filepath.name}')
        
        # Img2
        ny2, nx2 = self.img2.data.shape
        corners2 = self.img2.wcs.pixel_to_world([0, nx2, nx2, 0, 0], [0, 0, ny2, ny2, 0])
        ra2 = [c.ra.deg for c in corners2]
        dec2 = [c.dec.deg for c in corners2]
        ax.plot(ra2, dec2, 'r-', linewidth=2, label=f'Img2: {self.img2.filepath.name}')
        
        # Overlap region
        ov = self.overlap_info
        if ov['overlap_exists']:
            ra_ov = [ov['ra_range'][0], ov['ra_range'][1], ov['ra_range'][1], ov['ra_range'][0], ov['ra_range'][0]]
            dec_ov = [ov['dec_range'][0], ov['dec_range'][0], ov['dec_range'][1], ov['dec_range'][1], ov['dec_range'][0]]
            ax.fill(ra_ov, dec_ov, alpha=0.3, color='green', label='Overlap region')
        
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('DEC (deg)')
        ax.set_title('Overlap Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ Visualizzazione salvata: {output_path}")
        else:
            plt.show()
        
        plt.close()


# ============================================================================
# FUNZIONI
# ============================================================================

def load_images(directory, label):
    """Carica tutte le immagini FITS da una directory"""
    print(f"\n📂 Caricamento {label}...")
    print(f"   Directory: {directory}")
    
    if not directory.exists():
        print(f"   ✗ Directory non trovata!")
        return []
    
    # Cerca file FITS
    files = glob.glob(str(directory / '*.fits'))
    files.extend(glob.glob(str(directory / '*.fit')))
    
    if not files:
        print(f"   ✗ Nessun file FITS trovato")
        return []
    
    print(f"   Trovati {len(files)} file FITS")
    
    images = []
    for filepath in files:
        img = ImageAnalyzer(filepath)
        if img.load():
            images.append(img)
            print(f"   ✓ {img.filepath.name}")
        else:
            print(f"   ✗ {Path(filepath).name} (errore caricamento o WCS mancante)")
    
    print(f"   ✓ Caricate {len(images)} immagini con WCS valido")
    return images


def analyze_all_pairs(hubble_imgs, obs_imgs):
    """Analizza overlap per tutte le coppie possibili"""
    print(f"\n🔍 Analisi overlap tra {len(hubble_imgs)} Hubble e {len(obs_imgs)} Observatory...")
    
    results = []
    
    for h_img in hubble_imgs:
        for o_img in obs_imgs:
            analyzer = OverlapAnalyzer(h_img, o_img)
            overlap = analyzer.calculate_overlap()
            
            if overlap and overlap['overlap_exists']:
                results.append({
                    'hubble': h_img,
                    'observatory': o_img,
                    'overlap': overlap,
                    'analyzer': analyzer
                })
    
    # Ordina per area overlap (decrescente)
    results.sort(key=lambda x: x['overlap']['overlap_area_arcmin2'], reverse=True)
    
    return results


def print_patch_analysis(img, label):
    """Stampa analisi patches per un'immagine"""
    print(f"\n{'─'*70}")
    print(f"📷 {label}: {img.filepath.name}")
    print(f"{'─'*70}")
    
    print(f"   Dimensioni: {img.info['shape'][1]} × {img.info['shape'][0]} px")
    print(f"   Memoria: {img.info['size_mb']} MB")
    
    if 'wcs' in img.info:
        wcs = img.info['wcs']
        print(f"\n   WCS:")
        print(f"   Pixel scale: {wcs['pixel_scale_arcsec']:.4f} \"/px = {wcs['pixel_scale_arcmin']:.6f} '/px")
        print(f"   FOV: {wcs['fov_ra_arcmin']:.2f} × {wcs['fov_dec_arcmin']:.2f} arcmin")
        print(f"   Centro: RA={wcs['center_ra']:.6f}°, DEC={wcs['center_dec']:.6f}°")
    
    # Calcola patches per diverse dimensioni
    print(f"\n💡 DIMENSIONI PATCHES POSSIBILI:")
    
    for target_arcmin in [0.5, 1.0, 2.0, 5.0]:
        patch_info = img.calculate_patch_size(target_arcmin)
        
        if patch_info:
            print(f"\n   Target: {target_arcmin} arcmin")
            print(f"      Patch size: {patch_info['patch_size_px']} × {patch_info['patch_size_px']} px")
            print(f"      Actual size: {patch_info['actual_size_arcmin']:.4f} arcmin = {patch_info['actual_size_arcsec']:.2f} arcsec")
            print(f"      Patches (no overlap): {patch_info['patches_no_overlap']['x']} × {patch_info['patches_no_overlap']['y']} = {patch_info['patches_no_overlap']['total']}")
            print(f"      Patches ({PATCH_OVERLAP_PERCENT}% overlap): {patch_info['patches_with_overlap']['x']} × {patch_info['patches_with_overlap']['y']} = {patch_info['patches_with_overlap']['total']}")


def create_summary_report(hubble_imgs, obs_imgs, overlap_results, output_dir):
    """Crea report riassuntivo"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'target': 'all',
        'source_type': 'cropped_images',  # Specifica che usa immagini cropped
        'hubble': {
            'count': len(hubble_imgs),
            'images': []
        },
        'observatory': {
            'count': len(obs_imgs),
            'images': []
        },
        'overlap': {
            'pairs_with_overlap': len(overlap_results),
            'best_matches': []
        }
    }
    
    # Info Hubble
    for img in hubble_imgs:
        info = {'filename': img.filepath.name, 'shape': img.info['shape']}
        if 'wcs' in img.info:
            info['wcs'] = img.info['wcs']
            info['patch_1arcmin'] = img.calculate_patch_size(1.0)
        report['hubble']['images'].append(info)
    
    # Info Observatory
    for img in obs_imgs:
        info = {'filename': img.filepath.name, 'shape': img.info['shape']}
        if 'wcs' in img.info:
            info['wcs'] = img.info['wcs']
            info['patch_1arcmin'] = img.calculate_patch_size(1.0)
        report['observatory']['images'].append(info)
    
    # Best matches
    for result in overlap_results[:10]:  # Top 10
        report['overlap']['best_matches'].append({
            'hubble_file': result['hubble'].filepath.name,
            'observatory_file': result['observatory'].filepath.name,
            'overlap_area_arcmin2': result['overlap']['overlap_area_arcmin2'],
            'fraction_hubble': result['overlap']['fraction_img1'],
            'fraction_observatory': result['overlap']['fraction_img2'],
        })
    
    # Salva JSON
    output_path = output_dir / f'analisi_overlap_report.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Report salvato: {output_path}")
    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Funzione principale."""
    print("\n" + "🔭"*35)
    print(f"STEP 5: ANALISI OVERLAP E PATCHES".center(70))
    print("🔭"*35)
    
    print(f"\n📂 CONFIGURAZIONE:")
    print(f"   Hubble: {HUBBLE_DIR}")
    print(f"   Observatory: {OBS_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"\n   ℹ️  Tipo sorgente: Immagini CROPPED (tutte stesse dimensioni)")
    
    # Crea output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Carica immagini
    print(f"\n{'='*70}")
    print("CARICAMENTO IMMAGINI")
    print(f"{'='*70}")
    
    hubble_imgs = load_images(HUBBLE_DIR, "HUBBLE")
    obs_imgs = load_images(OBS_DIR, "OBSERVATORY")
    
    if not hubble_imgs:
        print(f"\n❌ Nessuna immagine Hubble caricata! (Controlla {HUBBLE_DIR})")
        return
    
    if not obs_imgs:
        print(f"\n❌ Nessuna immagine Observatory caricata! (Controlla {OBS_DIR})")
        return
    
    # Analisi patches
    print(f"\n{'='*70}")
    print("ANALISI DIMENSIONI PATCHES")
    print(f"{'='*70}")
    
    print(f"\n🛰️  HUBBLE IMAGES:")
    for img in hubble_imgs[:3]:  # Mostra prime 3
        print_patch_analysis(img, "HUBBLE")
    if len(hubble_imgs) > 3:
        print(f"\n   ℹ️  ... e altre {len(hubble_imgs) - 3} immagini Hubble")
    
    print(f"\n📡 OBSERVATORY IMAGES:")
    # Stampa solo prima immagine (spesso sono simili)
    if obs_imgs:
        print_patch_analysis(obs_imgs[0], "OBSERVATORY (esempio)")
        if len(obs_imgs) > 1:
            print(f"\n   ℹ️  Mostrata solo la prima immagine observatory (su {len(obs_imgs)} totali)")
    
    # Analizza overlap
    overlap_results = analyze_all_pairs(hubble_imgs, obs_imgs)
    
    print(f"\n{'='*70}")
    print("RISULTATI OVERLAP")
    print(f"{'='*70}")
    
    if overlap_results:
        print(f"\n✅ Trovate {len(overlap_results)} coppie con overlap!")
        
        print(f"\n🏆 TOP 5 MATCHES:")
        for i, result in enumerate(overlap_results[:5], 1):
            print(f"\n   {i}. {result['hubble'].filepath.name} ↔ {result['observatory'].filepath.name}")
            print(f"      Overlap area: {result['overlap']['overlap_area_arcmin2']:.2f} arcmin²")
            print(f"      Hubble coverage: {result['overlap']['fraction_img1']*100:.1f}%")
            print(f"      Observatory coverage: {result['overlap']['fraction_img2']*100:.1f}%")
        
        # Visualizza best match
        best = overlap_results[0]
        viz_path = OUTPUT_DIR / f'overlap_best_match.png'
        best['analyzer'].visualize(viz_path)
        print(f"\n📊 Visualizzazione salvata: {viz_path}")
        
    else:
        print(f"\n⚠️  Nessun overlap trovato tra le immagini!")
        print(f"\n   Possibili cause:")
        print(f"   - Le immagini puntano a regioni diverse")
        print(f"   - WCS non corretto (controlla output Step 3)")
        print(f"   - Coordinate non allineate")
    
    # Report finale
    create_summary_report(hubble_imgs, obs_imgs, overlap_results, OUTPUT_DIR)
    
    # RACCOMANDAZIONI
    print(f"\n{'='*70}")
    print("💡 RACCOMANDAZIONI PATCHES")
    print(f"{'='*70}")
    
    if hubble_imgs and obs_imgs:
        # Patch size consigliata
        h_patch = hubble_imgs[0].calculate_patch_size(1.0)
        o_patch = obs_imgs[0].calculate_patch_size(1.0)
        
        if h_patch and o_patch:
            print(f"\n📐 DIMENSIONI CONSIGLIATE (1 arcmin target):")
            print(f"\n   Hubble:")
            print(f"      Patch size: {h_patch['patch_size_px']} × {h_patch['patch_size_px']} px")
            print(f"      Patches totali: {h_patch['patches_with_overlap']['total']} (con {PATCH_OVERLAP_PERCENT}% overlap)")
            
            print(f"\n   Observatory:")
            print(f"      Patch size: {o_patch['patch_size_px']} × {o_patch['patch_size_px']} px")
            print(f"      Patches per immagine: {o_patch['patches_with_overlap']['total']} (con {PATCH_OVERLAP_PERCENT}% overlap)")
            print(f"      Patches totali: {o_patch['patches_with_overlap']['total'] * len(obs_imgs)}")
        else:
            print("\n   ⚠️ Impossibile calcolare dimensioni patch (problema WCS?)")

    
    print(f"\n{'='*70}")
    print(f"✅ ANALISI COMPLETATA")
    print(f"{'='*70}")
    print(f"\n📁 Tutti i file salvati in: {OUTPUT_DIR}")
    
    print(f"\n➡️  PROSSIMI PASSI:")
    if overlap_results:
        print(f"   1. ✅ Overlap confermato - procedi con estrazione patches")
        print(f"   2. Usa 'step6_patch.py' per estrarre le patches")
        print(f"   3. Le immagini cropped sono già uniformi, ottime per patches!")
    else:
        print(f"   1. ⚠️  Verifica coordinate WCS (output Step 2)")
        print(f"   2. Controlla che le immagini puntino alla stessa regione")
        print(f"   3. Potrebbe servire ricalibrazione astrometrica")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()