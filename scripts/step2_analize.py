#!/usr/bin/env python3
"""
STEP 2: ANALISI OVERLAP E DIMENSIONI PATCHES
Analizza le immagini con WCS per determinare:
1. Overlap tra Hubble e Osservatorio
2. Dimensioni ottimali delle patches
3. Strategia di allineamento

Input: Cartelle con WCS create da step1_convert_wcs.py
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
# import argparse # Rimosso
from datetime import datetime
import glob

# ============================================================================
# CONFIGURAZIONE - Usa le cartelle create da step1
# ============================================================================

BASE_DIR = r'F:\Super Revolt Gaia\SuperResolution\SuperResolution\data'
HUBBLE_DIR = Path(BASE_DIR) / 'lith_con_wcs'  # Output step1 - Hubble
OBS_DIR = Path(BASE_DIR) / 'osservatorio_con_wcs'  # Output step1 - Osservatorio
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
            'overlap_percent': PATCH_OVERLAP_PERCENT,
        }


class OverlapAnalyzer:
    """Analizza overlap tra due immagini"""
    
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.overlap_info = None
    
    def calculate_overlap(self):
        """Calcola overlap tra le due immagini"""
        if self.img1.wcs is None or self.img2.wcs is None:
            return None
        
        wcs1 = self.img1.info['wcs']
        wcs2 = self.img2.info['wcs']
        
        ra1_min, ra1_max = wcs1['ra_range']
        dec1_min, dec1_max = wcs1['dec_range']
        ra2_min, ra2_max = wcs2['ra_range']
        dec2_min, dec2_max = wcs2['dec_range']
        
        # Intersezione
        overlap_ra_min = max(ra1_min, ra2_min)
        overlap_ra_max = min(ra1_max, ra2_max)
        overlap_dec_min = max(dec1_min, dec2_min)
        overlap_dec_max = min(dec1_max, dec2_max)
        
        has_overlap = (overlap_ra_max > overlap_ra_min) and (overlap_dec_max > overlap_dec_min)
        
        if not has_overlap:
            # Calcola separazione
            center1 = SkyCoord(wcs1['center_ra'], wcs1['center_dec'], unit='deg')
            center2 = SkyCoord(wcs2['center_ra'], wcs2['center_dec'], unit='deg')
            separation = center1.separation(center2)
            
            return {
                'has_overlap': False,
                'separation_deg': separation.deg,
                'separation_arcmin': separation.arcmin,
            }
        
        # Calcola statistiche overlap
        overlap_area_deg2 = (overlap_ra_max - overlap_ra_min) * (overlap_dec_max - overlap_dec_min)
        area1 = (ra1_max - ra1_min) * (dec1_max - dec1_min)
        area2 = (ra2_max - ra2_min) * (dec2_max - dec2_min)
        
        center1 = SkyCoord(wcs1['center_ra'], wcs1['center_dec'], unit='deg')
        center2 = SkyCoord(wcs2['center_ra'], wcs2['center_dec'], unit='deg')
        separation = center1.separation(center2)
        
        self.overlap_info = {
            'has_overlap': True,
            'overlap_ra': [overlap_ra_min, overlap_ra_max],
            'overlap_dec': [overlap_dec_min, overlap_dec_max],
            'overlap_area_deg2': overlap_area_deg2,
            'overlap_area_arcmin2': overlap_area_deg2 * 3600,
            'fraction_img1': overlap_area_deg2 / area1 if area1 > 0 else 0,
            'fraction_img2': overlap_area_deg2 / area2 if area2 > 0 else 0,
            'separation_deg': separation.deg,
            'separation_arcmin': separation.arcmin,
        }
        
        return self.overlap_info
    
    def visualize(self, output_path):
        """Crea visualizzazione overlap"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Immagine 1
        wcs1 = self.img1.info['wcs']
        ny1, nx1 = self.img1.data.shape
        corners1 = self.img1.wcs.pixel_to_world([0, nx1, nx1, 0, 0], [0, 0, ny1, ny1, 0])
        ra1 = [c.ra.deg for c in corners1]
        dec1 = [c.dec.deg for c in corners1]
        
        ax.plot(ra1, dec1, 'b-', linewidth=2, label=f'Hubble: {self.img1.filepath.name}')
        ax.fill(ra1, dec1, color='blue', alpha=0.2)
        ax.plot(wcs1['center_ra'], wcs1['center_dec'], 'bo', markersize=8)
        
        # Immagine 2
        wcs2 = self.img2.info['wcs']
        ny2, nx2 = self.img2.data.shape
        corners2 = self.img2.wcs.pixel_to_world([0, nx2, nx2, 0, 0], [0, 0, ny2, ny2, 0])
        ra2 = [c.ra.deg for c in corners2]
        dec2 = [c.dec.deg for c in corners2]
        
        ax.plot(ra2, dec2, 'r-', linewidth=2, label=f'Observatory: {self.img2.filepath.name}')
        ax.fill(ra2, dec2, color='red', alpha=0.2)
        ax.plot(wcs2['center_ra'], wcs2['center_dec'], 'ro', markersize=8)
        
        # Overlap region
        if self.overlap_info and self.overlap_info['has_overlap']:
            ov_ra = self.overlap_info['overlap_ra']
            ov_dec = self.overlap_info['overlap_dec']
            
            ov_ra_poly = [ov_ra[0], ov_ra[1], ov_ra[1], ov_ra[0], ov_ra[0]]
            ov_dec_poly = [ov_dec[0], ov_dec[0], ov_dec[1], ov_dec[1], ov_dec[0]]
            
            ax.plot(ov_ra_poly, ov_dec_poly, 'g-', linewidth=3, label='Overlap Region')
            ax.fill(ov_ra_poly, ov_dec_poly, color='green', alpha=0.4)
            
            # Testo overlap
            overlap_text = (
                f"Overlap: {self.overlap_info['overlap_area_arcmin2']:.2f} arcmin²\n"
                f"Hubble coverage: {self.overlap_info['fraction_img1']*100:.1f}%\n"
                f"Observatory coverage: {self.overlap_info['fraction_img2']*100:.1f}%"
            )
            ax.text(0.02, 0.98, overlap_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                   fontsize=10)
        
        ax.set_xlabel('RA (deg)', fontsize=12)
        ax.set_ylabel('DEC (deg)', fontsize=12)
        ax.set_title(f'Field Overlap: Hubble vs Observatory', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# FUNZIONI PRINCIPALI
# ============================================================================

def load_images(directory, source_name):
    """Carica tutte le immagini da una directory"""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"   ✗ Directory non trovata: {directory}")
        return []
    
    # Trova FITS files
    fits_files = list(directory.glob('*.fits')) + list(directory.glob('*.fit'))
    
    if not fits_files:
        print(f"   ✗ Nessun file FITS in {directory}")
        return []
    
    print(f"\n   📂 {source_name}: {len(fits_files)} file trovati")
    
    # Carica
    analyzers = []
    for filepath in fits_files:
        analyzer = ImageAnalyzer(filepath)
        if analyzer.load():
            analyzers.append(analyzer)
            print(f"      ✓ {filepath.name}")
        else:
            print(f"      ✗ {filepath.name} - errore caricamento")
    
    print(f"   ✅ {len(analyzers)}/{len(fits_files)} immagini caricate con successo")
    
    return analyzers


def analyze_all_pairs(hubble_imgs, obs_imgs):
    """Analizza tutte le coppie possibili"""
    print(f"\n{'='*70}")
    print(f"ANALISI OVERLAP: {len(hubble_imgs)} × {len(obs_imgs)} = {len(hubble_imgs)*len(obs_imgs)} coppie")
    print(f"{'='*70}")
    
    results = []
    
    for h_img in hubble_imgs:
        for o_img in obs_imgs:
            analyzer = OverlapAnalyzer(h_img, o_img)
            overlap = analyzer.calculate_overlap()
            
            if overlap and overlap['has_overlap']:
                results.append({
                    'hubble': h_img,
                    'observatory': o_img,
                    'overlap': overlap,
                    'analyzer': analyzer
                })
    
    # Ordina per frazione overlap (osservatorio)
    results.sort(key=lambda x: x['overlap']['fraction_img2'], reverse=True)
    
    return results


def print_patch_analysis(img, source_name):
    """Stampa analisi patches per un'immagine"""
    print(f"\n{'─'*70}")
    print(f"📐 ANALISI PATCHES: {source_name}")
    print(f"   File: {img.filepath.name}")
    print(f"{'─'*70}")
    
    print(f"\n📊 DIMENSIONI IMMAGINE:")
    ny, nx = img.data.shape
    print(f"   Pixel: {nx} × {ny}")
    print(f"   Size: {img.info['size_mb']} MB")
    
    if 'wcs' in img.info:
        wcs = img.info['wcs']
        print(f"\n🌍 WCS:")
        print(f"   Pixel scale: {wcs['pixel_scale_arcsec']:.4f}\"/px = {wcs['pixel_scale_arcmin']:.4f}'/px")
        print(f"   FOV: {wcs['fov_ra_arcmin']:.2f}' × {wcs['fov_dec_arcmin']:.2f}'")
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
        'target': 'all', # Target generico
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
    # Rimosso input interattivo
    # TARGET_NAME = input("Inserisci il nome del target (es. M33, M31): ")
    # ...
    # TARGET_NAME = TARGET_NAME.strip()
    
    print("\n" + "🔭"*35)
    print(f"STEP 2: ANALISI OVERLAP E PATCHES (Tutti i target)".center(70))
    print("🔭"*35)
    
    print(f"\n📂 CONFIGURAZIONE:")
    print(f"   Hubble (LITH): {HUBBLE_DIR}")
    print(f"   Observatory: {OBS_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    
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
    for img in hubble_imgs:
        print_patch_analysis(img, "HUBBLE")
    
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
        print(f"   - WCS non corretto (controlla output Step 1)")
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
        print(f"   1. ✅ Overlap confermato - procedi con allineamento")
        print(f"   2. Usa 'step3_register.py'")
        print(f"   3. Estrai patches dalla regione di overlap")
    else:
        print(f"   1. ⚠️  Verifica coordinate WCS (output Step 1)")
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