#!/usr/bin/env python3
"""
STEP 5-6 UNIFICATO: ANALISI OVERLAP E ESTRAZIONE PATCHES
Script unificato che permette di:
1. Eseguire solo l'analisi overlap e dimensioni patches (Step 5)
2. Eseguire solo l'estrazione patches (Step 6)
3. Eseguire entrambi in sequenza (Analisi + Patch)

Mantiene tutte le funzionalità e output degli script originali.
"""

import os
import glob
import time
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import shutil  # <-- MODIFICA: Aggiunto import

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE GLOBALE
# ============================================================================

BASE_DIR = r'F:\Super Revolt Gaia\parte 2(patch)\data'

# Directory per Step 5 (Analisi)
HUBBLE_DIR_ANALYSIS = Path(BASE_DIR) / '4_cropped' / 'hubble'
OBS_DIR_ANALYSIS = Path(BASE_DIR) / '4_cropped' / 'observatory'
OUTPUT_DIR_ANALYSIS = Path(BASE_DIR) / 'analisi_overlap'

# Directory per Step 6 (Patch) - Input selezionabile
INPUT_CROPPED_HUBBLE = Path(BASE_DIR) / '4_cropped' / 'hubble'
INPUT_CROPPED_OBSERVATORY = Path(BASE_DIR) / '4_cropped' / 'observatory'
INPUT_REGISTERED_HUBBLE = Path(BASE_DIR) / '3_registered_native' / 'hubble'
INPUT_REGISTERED_OBSERVATORY = Path(BASE_DIR) / '3_registered_native' / 'observatory'

# Directory per riepilogo pipeline
INPUT_ORIG_HUBBLE = Path(BASE_DIR) / 'img_lights_1'
INPUT_ORIG_OBSERVATORY = Path(BASE_DIR) / 'local_raw'
INPUT_WCS_HUBBLE = Path(BASE_DIR) / 'lith_con_wcs'
INPUT_WCS_OBSERVATORY = Path(BASE_DIR) / 'osservatorio_con_wcs'

LOG_DIR = Path(r'F:\Super Revolt Gaia\logs')

# Variabili globali per Step 6
INPUT_HUBBLE = None
INPUT_OBSERVATORY = None
OUTPUT_DIR_PATCHES = None
OUTPUT_HUBBLE_PATCHES = None
OUTPUT_OBS_PATCHES = None

# Parametri Step 5
TARGET_PATCH_ARCMIN_ANALYSIS = 1.0
PATCH_OVERLAP_PERCENT_ANALYSIS = 10

# Parametri Step 6
TARGET_FOV_ARCMIN = 1
OVERLAP_PERCENT = 25
MIN_VALID_PERCENT = 50
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================================
# MENU PRINCIPALE
# ============================================================================

def main_menu():
    """Menu principale per scegliere quale operazione eseguire."""
    print("\n" + "🌟"*35)
    print("STEP 5-6 UNIFICATO: ANALISI & PATCHES".center(70))
    print("🌟"*35)
    
    print(f"\n📋 SCEGLI OPERAZIONE:")
    print(f"\n1️⃣  SOLO ANALISI")
    print(f"   • Analizza overlap tra immagini Hubble e Observatory")
    print(f"   • Calcola dimensioni ottimali patches")
    print(f"   • Genera report e visualizzazioni")
    
    print(f"\n2️⃣  SOLO ESTRAZIONE PATCHES")
    print(f"   • Estrae patches dalle immagini registrate")
    print(f"   • Mantiene risoluzione nativa")
    print(f"   • Crea dataset splits (train/val/test)")
    
    print(f"\n3️⃣  ENTRAMBI (Step 5 + Step 6)")
    print(f"   • Esegue prima l'analisi completa")
    print(f"   • Poi estrae le patches")
    print(f"   • Pipeline completa end-to-end")
    
    while True:
        print(f"\n" + "─"*70)
        choice = input("👉 Scegli opzione [1/2/3, default=3]: ").strip()
        
        if choice == '1':
            print(f"\n✅ Selezionato: SOLO ANALISI")
            return 'analysis'
        elif choice == '2':
            print(f"\n✅ Selezionato: SOLO ESTRAZIONE PATCHES")
            return 'patches'
        elif choice == '' or choice == '3':
            print(f"\n✅ Selezionato: ENTRAMBI (Analisi + Patches)")
            return 'both'
        else:
            print(f"❌ Scelta non valida. Inserisci 1, 2 o 3.")


# ============================================================================
# STEP 5: ANALISI OVERLAP E DIMENSIONI PATCHES
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
        """Calcola dimensione patch ottimale in pixel."""
        if 'wcs' not in self.info or self.info['wcs']['pixel_scale_arcsec'] is None:
            return None
        
        pixel_scale_arcsec = self.info['wcs']['pixel_scale_arcsec']
        target_arcsec = target_arcmin * 60
        
        # Dimensione in pixel
        patch_size_px = int(target_arcsec / pixel_scale_arcsec)
        
        # Arrotonda a multiplo di 8
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
        step = int(patch_size_px * (1 - PATCH_OVERLAP_PERCENT_ANALYSIS/100))
        n_x_overlap = max(1, (nx - patch_size_px) // step + 1) if step > 0 else 1
        n_y_overlap = max(1, (ny - patch_size_px) // step + 1) if step > 0 else 1
        total_with_overlap = n_x_overlap * n_y_overlap
        
        return {
            'target_arcmin': target_arcmin,
            'patch_size_px': patch_size_px,
            'actual_size_arcmin': actual_arcmin,
            'actual_size_arcsec': actual_arcsec,
            'patches_no_overlap': {
                'x': n_x,
                'y': n_y,
                'total': total_no_overlap
            },
            'patches_with_overlap': {
                'x': n_x_overlap,
                'y': n_y_overlap,
                'total': total_with_overlap
            }
        }


class OverlapAnalyzer:
    """Analizza overlap tra due immagini"""
    
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.overlap_info = None
        
    def calculate_overlap(self):
        """Calcola regione di overlap"""
        if self.img1.wcs is None or self.img2.wcs is None:
            return None
        
        # Bounding boxes
        wcs1 = self.img1.info['wcs']
        wcs2 = self.img2.info['wcs']
        
        # Overlap RA
        ra1_min, ra1_max = wcs1['ra_range']
        ra2_min, ra2_max = wcs2['ra_range']
        
        overlap_ra_min = max(ra1_min, ra2_min)
        overlap_ra_max = min(ra1_max, ra2_max)
        
        # Overlap DEC
        dec1_min, dec1_max = wcs1['dec_range']
        dec2_min, dec2_max = wcs2['dec_range']
        
        overlap_dec_min = max(dec1_min, dec2_min)
        overlap_dec_max = min(dec1_max, dec2_max)
        
        # Verifica overlap
        if overlap_ra_max <= overlap_ra_min or overlap_dec_max <= overlap_dec_min:
            return None
        
        # Area overlap
        overlap_ra_span = overlap_ra_max - overlap_ra_min
        overlap_dec_span = overlap_dec_max - overlap_dec_min
        overlap_area_deg2 = overlap_ra_span * overlap_dec_span
        overlap_area_arcmin2 = overlap_area_deg2 * 3600
        
        # Frazioni
        area1 = wcs1['fov_ra_deg'] * wcs1['fov_dec_deg']
        area2 = wcs2['fov_ra_deg'] * wcs2['fov_dec_deg']
        
        self.overlap_info = {
            'overlap_ra_range': [overlap_ra_min, overlap_ra_max],
            'overlap_dec_range': [overlap_dec_min, overlap_dec_max],
            'overlap_ra_deg': overlap_ra_span,
            'overlap_dec_deg': overlap_dec_span,
            'overlap_area_deg2': overlap_area_deg2,
            'overlap_area_arcmin2': overlap_area_arcmin2,
            'fraction_img1': overlap_area_deg2 / area1 if area1 > 0 else 0,
            'fraction_img2': overlap_area_deg2 / area2 if area2 > 0 else 0,
        }
        
        return self.overlap_info
    
    def visualize(self, output_path):
        """Crea visualizzazione overlap"""
        if self.overlap_info is None:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Immagine 1
        wcs1 = self.img1.info['wcs']
        ra1 = wcs1['ra_range']
        dec1 = wcs1['dec_range']
        ax.add_patch(plt.Rectangle(
            (ra1[0], dec1[0]), ra1[1]-ra1[0], dec1[1]-dec1[0],
            fill=False, edgecolor='blue', linewidth=2, label=self.img1.filepath.name
        ))
        
        # Immagine 2
        wcs2 = self.img2.info['wcs']
        ra2 = wcs2['ra_range']
        dec2 = wcs2['dec_range']
        ax.add_patch(plt.Rectangle(
            (ra2[0], dec2[0]), ra2[1]-ra2[0], dec2[1]-dec2[0],
            fill=False, edgecolor='red', linewidth=2, label=self.img2.filepath.name
        ))
        
        # Overlap
        ov = self.overlap_info
        ax.add_patch(plt.Rectangle(
            (ov['overlap_ra_range'][0], ov['overlap_dec_range'][0]),
            ov['overlap_ra_deg'], ov['overlap_dec_deg'],
            fill=True, facecolor='green', alpha=0.3, edgecolor='green', 
            linewidth=2, label='Overlap'
        ))
        
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('DEC (deg)')
        ax.set_title('Image Overlap Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def load_images_analysis(directory, label):
    """Carica immagini per analisi"""
    print(f"\n🔍 Caricamento {label}...")
    
    files = list(directory.glob('*.fits')) + list(directory.glob('*.fit'))
    
    if not files:
        print(f"   ⚠️  Nessun file trovato in {directory}")
        return []
    
    images = []
    for fpath in files:
        img = ImageAnalyzer(fpath)
        if img.load():
            images.append(img)
            print(f"   ✓ {img.filepath.name}: {img.info['shape']}")
        else:
            print(f"   ✗ {img.filepath.name}: errore caricamento")
    
    print(f"   Totale {label}: {len(images)}/{len(files)} immagini caricate")
    return images


def analyze_all_pairs(hubble_imgs, obs_imgs):
    """Analizza overlap per tutte le coppie"""
    print(f"\n🔗 Analisi overlap tra {len(hubble_imgs)} Hubble e {len(obs_imgs)} Observatory...")
    
    results = []
    
    for h_img in hubble_imgs:
        for o_img in obs_imgs:
            analyzer = OverlapAnalyzer(h_img, o_img)
            overlap = analyzer.calculate_overlap()
            
            if overlap and overlap['overlap_area_arcmin2'] > 0.1:
                results.append({
                    'hubble': h_img,
                    'observatory': o_img,
                    'overlap': overlap,
                    'analyzer': analyzer
                })
    
    # Ordina per area overlap
    results.sort(key=lambda x: x['overlap']['overlap_area_arcmin2'], reverse=True)
    
    return results


def print_patch_analysis(img, label):
    """Stampa analisi patches per un'immagine"""
    print(f"\n📊 {label}: {img.filepath.name}")
    print(f"   Dimensioni: {img.info['shape'][1]} × {img.info['shape'][0]} px")
    
    if 'wcs' in img.info:
        wcs = img.info['wcs']
        print(f"   Pixel scale: {wcs['pixel_scale_arcsec']:.4f} arcsec/px")
        print(f"   FOV: {wcs['fov_ra_arcmin']:.2f} × {wcs['fov_dec_arcmin']:.2f} arcmin")
        print(f"   Centro: RA={wcs['center_ra']:.6f}°, DEC={wcs['center_dec']:.6f}°")
    
    print(f"\n💡 DIMENSIONI PATCHES POSSIBILI:")
    
    for target_arcmin in [0.5, 1.0, 2.0, 5.0]:
        patch_info = img.calculate_patch_size(target_arcmin)
        
        if patch_info:
            print(f"\n   Target: {target_arcmin} arcmin")
            print(f"      Patch size: {patch_info['patch_size_px']} × {patch_info['patch_size_px']} px")
            print(f"      Actual size: {patch_info['actual_size_arcmin']:.4f} arcmin")
            print(f"      Patches (no overlap): {patch_info['patches_no_overlap']['total']}")
            print(f"      Patches ({PATCH_OVERLAP_PERCENT_ANALYSIS}% overlap): {patch_info['patches_with_overlap']['total']}")


def create_summary_report(hubble_imgs, obs_imgs, overlap_results, output_dir):
    """Crea report riassuntivo"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'target': 'all',
        'source_type': 'cropped_images',
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
    for result in overlap_results[:10]:
        report['overlap']['best_matches'].append({
            'hubble_file': result['hubble'].filepath.name,
            'observatory_file': result['observatory'].filepath.name,
            'overlap_area_arcmin2': result['overlap']['overlap_area_arcmin2'],
            'fraction_hubble': result['overlap']['fraction_img1'],
            'fraction_observatory': result['overlap']['fraction_img2'],
        })
    
    # Salva JSON
    output_path = output_dir / 'analisi_overlap_report.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Report salvato: {output_path}")
    return output_path


def run_analysis():
    """Esegue Step 5: Analisi Overlap"""
    print("\n" + "🔭"*35)
    print(f"STEP 5: ANALISI OVERLAP E PATCHES".center(70))
    print("🔭"*35)
    
    print(f"\n📂 CONFIGURAZIONE:")
    print(f"   Hubble: {HUBBLE_DIR_ANALYSIS}")
    print(f"   Observatory: {OBS_DIR_ANALYSIS}")
    print(f"   Output: {OUTPUT_DIR_ANALYSIS}")
    
    # Crea output dir
    OUTPUT_DIR_ANALYSIS.mkdir(parents=True, exist_ok=True)
    
    # Carica immagini
    print(f"\n{'='*70}")
    print("CARICAMENTO IMMAGINI")
    print(f"{'='*70}")
    
    hubble_imgs = load_images_analysis(HUBBLE_DIR_ANALYSIS, "HUBBLE")
    obs_imgs = load_images_analysis(OBS_DIR_ANALYSIS, "OBSERVATORY")
    
    if not hubble_imgs:
        print(f"\n❌ Nessuna immagine Hubble caricata!")
        return False
    
    if not obs_imgs:
        print(f"\n❌ Nessuna immagine Observatory caricata!")
        return False
    
    # Analisi patches
    print(f"\n{'='*70}")
    print("ANALISI DIMENSIONI PATCHES")
    print(f"{'='*70}")
    
    print(f"\n🛰️  HUBBLE IMAGES:")
    for img in hubble_imgs[:3]:
        print_patch_analysis(img, "HUBBLE")
    if len(hubble_imgs) > 3:
        print(f"\n   ℹ️  ... e altre {len(hubble_imgs) - 3} immagini Hubble")
    
    print(f"\n📡 OBSERVATORY IMAGES:")
    if obs_imgs:
        print_patch_analysis(obs_imgs[0], "OBSERVATORY (esempio)")
        if len(obs_imgs) > 1:
            print(f"\n   ℹ️  Mostrata solo la prima immagine observatory")
    
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
        viz_path = OUTPUT_DIR_ANALYSIS / 'overlap_best_match.png'
        best['analyzer'].visualize(viz_path)
        print(f"\n📊 Visualizzazione salvata: {viz_path}")
    else:
        print(f"\n⚠️  Nessun overlap trovato!")
    
    # Report finale
    create_summary_report(hubble_imgs, obs_imgs, overlap_results, OUTPUT_DIR_ANALYSIS)
    
    # Raccomandazioni
    print(f"\n{'='*70}")
    print("💡 RACCOMANDAZIONI PATCHES")
    print(f"{'='*70}")
    
    if hubble_imgs and obs_imgs:
        h_patch = hubble_imgs[0].calculate_patch_size(1.0)
        o_patch = obs_imgs[0].calculate_patch_size(1.0)
        
        if h_patch and o_patch:
            print(f"\n📐 DIMENSIONI CONSIGLIATE (1 arcmin target):")
            print(f"\n   Hubble:")
            print(f"      Patch size: {h_patch['patch_size_px']} × {h_patch['patch_size_px']} px")
            print(f"      Patches totali: {h_patch['patches_with_overlap']['total']}")
            print(f"\n   Observatory:")
            print(f"      Patch size: {o_patch['patch_size_px']} × {o_patch['patch_size_px']} px")
            print(f"      Patches totali: {o_patch['patches_with_overlap']['total'] * len(obs_imgs)}")
    
    print(f"\n{'='*70}")
    print(f"✅ ANALISI COMPLETATA")
    print(f"{'='*70}")
    print(f"\n📁 Output salvato in: {OUTPUT_DIR_ANALYSIS}")
    
    return True


# ============================================================================
# STEP 6: ESTRAZIONE PATCHES
# ============================================================================

def select_input_type():
    """Menu per scegliere tipo input patches"""
    global INPUT_HUBBLE, INPUT_OBSERVATORY, OUTPUT_DIR_PATCHES
    global OUTPUT_HUBBLE_PATCHES, OUTPUT_OBS_PATCHES
    
    print("\n" + "🎯"*35)
    print("SELEZIONE TIPO IMMAGINI INPUT".center(70))
    print("🎯"*35)
    
    print(f"\n📋 OPZIONI DISPONIBILI:")
    print(f"\n1️⃣  IMMAGINI CROPPED (CONSIGLIATO)")
    print(f"   ✓ Tutte le immagini hanno le stesse dimensioni")
    print(f"   📁 Input: {INPUT_CROPPED_HUBBLE.parent}")
    
    print(f"\n2️⃣  IMMAGINI REGISTERED")
    print(f"   • Mantiene dimensioni originali")
    print(f"   📁 Input: {INPUT_REGISTERED_HUBBLE.parent}")
    
    while True:
        print(f"\n" + "─"*70)
        choice = input("👉 Scegli opzione [1/2, default=1]: ").strip()
        
        if choice == '' or choice == '1':
            INPUT_HUBBLE = INPUT_CROPPED_HUBBLE
            INPUT_OBSERVATORY = INPUT_CROPPED_OBSERVATORY
            OUTPUT_DIR_PATCHES = Path(BASE_DIR) / '6_patches_from_cropped'
            source_type = "cropped"
            print(f"\n✅ Selezionato: IMMAGINI CROPPED")
            break
        elif choice == '2':
            INPUT_HUBBLE = INPUT_REGISTERED_HUBBLE
            INPUT_OBSERVATORY = INPUT_REGISTERED_OBSERVATORY
            OUTPUT_DIR_PATCHES = Path(BASE_DIR) / '6_patches_from_registered'
            source_type = "registered"
            print(f"\n✅ Selezionato: IMMAGINI REGISTERED")
            break
        else:
            print(f"❌ Scelta non valida. Inserisci 1 o 2.")
    
    OUTPUT_HUBBLE_PATCHES = OUTPUT_DIR_PATCHES / 'hubble_native'
    OUTPUT_OBS_PATCHES = OUTPUT_DIR_PATCHES / 'observatory_native'
    
    print(f"\n🔍 Verifica directory...")
    
    if INPUT_HUBBLE.exists():
        hubble_files = list(INPUT_HUBBLE.glob('*.fits')) + list(INPUT_HUBBLE.glob('*.fit'))
        print(f"   ✓ Hubble: {len(hubble_files)} file trovati")
    else:
        print(f"   ⚠️  Directory Hubble non trovata: {INPUT_HUBBLE}")
    
    if INPUT_OBSERVATORY.exists():
        obs_files = list(INPUT_OBSERVATORY.glob('*.fits')) + list(INPUT_OBSERVATORY.glob('*.fit'))
        print(f"   ✓ Observatory: {len(obs_files)} file trovati")
    else:
        print(f"   ⚠️  Directory Observatory non trovata: {INPUT_OBSERVATORY}")
    
    print(f"\n📂 Output patches: {OUTPUT_DIR_PATCHES}")
    
    return source_type


def setup_logging():
    """Configura logging"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'patch_extraction_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def extract_patches_from_image(fits_path, output_dir, source_label, logger):
    """Estrae patches da una singola immagine"""
    try:
        with fits.open(fits_path) as hdul:
            data_hdu = None
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    data_hdu = hdu
                    break
            
            if data_hdu is None:
                return []
            
            data = data_hdu.data
            header = data_hdu.header
            
            if len(data.shape) == 3:
                data = data[0]
            
            # WCS
            try:
                wcs = WCS(header)
                if not wcs.has_celestial:
                    return []
            except:
                return []
            
            # Pixel scale
            try:
                if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
                    cd = wcs.wcs.cd
                    pixel_scale_deg = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
                elif hasattr(wcs.wcs, 'cdelt'):
                    pixel_scale_deg = abs(wcs.wcs.cdelt[0])
                else:
                    p1 = wcs.pixel_to_world(0, 0)
                    p2 = wcs.pixel_to_world(1, 0)
                    pixel_scale_deg = p1.separation(p2).deg
                
                pixel_scale_arcsec = pixel_scale_deg * 3600
            except:
                return []
            
            # Dimensione patch in pixel
            target_arcsec = TARGET_FOV_ARCMIN * 60
            patch_size_px = int(target_arcsec / pixel_scale_arcsec)
            patch_size_px = ((patch_size_px + 7) // 8) * 8
            
            # Step con overlap
            step = int(patch_size_px * (1 - OVERLAP_PERCENT/100))
            
            ny, nx = data.shape
            patches_info = []
            patch_idx = 0
            
            y = 0
            while y + patch_size_px <= ny:
                x = 0
                while x + patch_size_px <= nx:
                    # Estrai patch
                    patch_data = data[y:y+patch_size_px, x:x+patch_size_px]
                    
                    # Verifica validità
                    valid_mask = np.isfinite(patch_data)
                    valid_percent = 100 * valid_mask.sum() / patch_data.size
                    
                    if valid_percent >= MIN_VALID_PERCENT:
                        # Coordinate centro patch
                        center_x = x + patch_size_px // 2
                        center_y = y + patch_size_px // 2
                        center_coord = wcs.pixel_to_world(center_x, center_y)
                        
                        # Salva patch
                        filename = f'{source_label}_{Path(fits_path).stem}_p{patch_idx:04d}.fits'
                        output_path = output_dir / filename
                        
                        # Crea nuovo header
                        patch_header = header.copy()
                        patch_header['NAXIS1'] = patch_size_px
                        patch_header['NAXIS2'] = patch_size_px
                        
                        # Salva
                        patch_hdu = fits.PrimaryHDU(patch_data, header=patch_header)
                        patch_hdu.writeto(output_path, overwrite=True)
                        
                        patches_info.append({
                            'filename': filename,
                            'source_file': Path(fits_path).name,
                            'patch_index': patch_idx,
                            'position_px': (x, y),
                            'patch_size_px': patch_size_px,
                            'center_ra': center_coord.ra.deg,
                            'center_dec': center_coord.dec.deg,
                            'pixel_scale_arcsec': pixel_scale_arcsec,
                            'valid_percent': valid_percent,
                        })
                        
                        patch_idx += 1
                    
                    x += step
                y += step
            
            return patches_info
            
    except Exception as e:
        logger.error(f"Errore in {fits_path}: {e}")
        return []


def extract_all_patches(input_dir, output_dir, label, logger):
    """Estrae patches da tutte le immagini"""
    print(f"\n🔪 Estrazione patches {label.upper()}...")
    
    files = list(input_dir.glob('*.fits')) + list(input_dir.glob('*.fit'))
    
    if not files:
        print(f"   ⚠️  Nessun file in {input_dir}")
        return []
    
    all_patches = []
    
    for fpath in tqdm(files, desc=f"   {label}", ncols=70):
        patches = extract_patches_from_image(fpath, output_dir, label, logger)
        all_patches.extend(patches)
    
    print(f"   ✓ Estratte {len(all_patches)} patches da {len(files)} immagini")
    
    # Salva metadata
    metadata_file = output_dir / f'{label}_patches_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_patches': len(all_patches),
            'num_source_images': len(files),
            'patches': all_patches
        }, f, indent=2)
    
    return all_patches


# ============================================================================
# MODIFICA: Funzione create_patch_pairs aggiornata
# ============================================================================

def create_patch_pairs(hubble_patches, obs_patches, output_dir, logger):
    """Crea coppie di patches basate su prossimità spaziale
    E SALVA OGNI COPPIA IN UNA CARTELLA DEDICATA.
    """
    print(f"\n🔗 Creazione coppie patches...")
    
    # --- INIZIO MODIFICA ---
    
    # Definisci le directory sorgente da cui copiare le patch
    hubble_patch_dir = output_dir / 'hubble_native'
    obs_patch_dir = output_dir / 'observatory_native'
    
    # Definisci la directory di output principale per le cartelle delle coppie
    pairs_folders_dir = output_dir / 'paired_patches_folders'
    pairs_folders_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"   📂 Creazione cartelle coppie in: {pairs_folders_dir}")
    
    # --- FINE MODIFICA ---
    
    threshold_arcmin = 0.5
    pairs = []
    
    for h_patch in tqdm(hubble_patches, desc="   Pairing", ncols=70):
        h_coord = SkyCoord(h_patch['center_ra']*u.deg, h_patch['center_dec']*u.deg)
        
        best_dist = float('inf')
        best_obs = None
        
        for o_patch in obs_patches:
            o_coord = SkyCoord(o_patch['center_ra']*u.deg, o_patch['center_dec']*u.deg)
            dist_arcmin = h_coord.separation(o_coord).arcmin
            
            if dist_arcmin < best_dist:
                best_dist = dist_arcmin
                best_obs = o_patch
        
        if best_dist < threshold_arcmin and best_obs:
            
            # --- INIZIO MODIFICA ---
            
            # Definisci la cartella specifica per questa coppia
            pair_index = len(pairs)
            pair_folder_name = f'pair_{pair_index:05d}' # (es. pair_00001)
            pair_dest_dir = pairs_folders_dir / pair_folder_name
            pair_dest_dir.mkdir(exist_ok=True)
            
            # Nomi file
            h_filename = h_patch['filename']
            o_filename = best_obs['filename']
            
            # Percorsi sorgente
            h_src_path = hubble_patch_dir / h_filename
            o_src_path = obs_patch_dir / o_filename
            
            # Copia i file FITS nella loro nuova cartella
            try:
                shutil.copy2(h_src_path, pair_dest_dir / h_filename)
                shutil.copy2(o_src_path, pair_dest_dir / o_filename)
            except FileNotFoundError as e:
                logger.warning(f"File non trovato durante copia coppia: {e}")
            except Exception as e:
                logger.error(f"Errore copiando {pair_folder_name}: {e}")
                
            # --- FINE MODIFICA ---
                
            pairs.append({
                'hubble_patch': h_patch['filename'],
                'observatory_patch': best_obs['filename'],
                'separation_arcmin': best_dist
            })
    
    # Salva pairs (questa parte resta invariata)
    pairs_file = output_dir / 'patch_pairs.json'
    with open(pairs_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_pairs': len(pairs),
            'threshold_arcmin': threshold_arcmin,
            'pairs': pairs
        }, f, indent=2)
    
    print(f"   ✓ {len(pairs)} coppie create (file JSON)")
    print(f"   ✓ {len(pairs)} cartelle coppie copiate in {pairs_folders_dir}")
    return pairs

# ============================================================================

def create_dataset_split(patches, output_dir, logger):
    """Crea split train/val/test"""
    if not patches:
        return None
    
    np.random.shuffle(patches)
    
    n = len(patches)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    
    splits = {
        'train': patches[:n_train],
        'val': patches[n_train:n_train+n_val],
        'test': patches[n_train+n_val:]
    }
    
    # Salva split
    split_file = output_dir / 'dataset_split.json'
    with open(split_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'train_ratio': TRAIN_RATIO,
            'val_ratio': VAL_RATIO,
            'test_ratio': TEST_RATIO,
            'splits': {k: [p['filename'] for p in v] for k, v in splits.items()}
        }, f, indent=2)
    
    return splits


def run_patches():
    """Esegue Step 6: Estrazione Patches"""
    source_type = select_input_type()
    logger = setup_logging()
    
    print("\n" + "✂️ "*35)
    print(f"STEP 6: ESTRAZIONE PATCHES".center(70))
    print("✂️ "*35)
    
    print(f"\n📋 CONFIGURAZIONE:")
    print(f"   Tipo input: {source_type.upper()}")
    print(f"   FOV target: {TARGET_FOV_ARCMIN} arcmin")
    print(f"   Overlap: {OVERLAP_PERCENT}%")
    
    print(f"\n📂 INPUT:")
    print(f"   Hubble: {INPUT_HUBBLE}")
    print(f"   Observatory: {INPUT_OBSERVATORY}")
    
    print(f"\n📂 OUTPUT:")
    print(f"   {OUTPUT_DIR_PATCHES}")
    
    # Crea directories
    OUTPUT_HUBBLE_PATCHES.mkdir(parents=True, exist_ok=True)
    OUTPUT_OBS_PATCHES.mkdir(parents=True, exist_ok=True)
    
    # Estrazione
    print(f"\n{'='*70}")
    print("ESTRAZIONE PATCHES")
    print(f"{'='*70}")
    
    hubble_patches = []
    obs_patches = []
    
    if INPUT_HUBBLE.exists():
        hubble_patches = extract_all_patches(INPUT_HUBBLE, OUTPUT_HUBBLE_PATCHES, 'hubble', logger)
    
    if INPUT_OBSERVATORY.exists():
        obs_patches = extract_all_patches(INPUT_OBSERVATORY, OUTPUT_OBS_PATCHES, 'observatory', logger)
    
    # Riepilogo
    print(f"\n{'='*70}")
    print("📊 RIEPILOGO ESTRAZIONE")
    print(f"{'='*70}")
    print(f"\n   Hubble patches: {len(hubble_patches)}")
    print(f"   Observatory patches: {len(obs_patches)}")
    print(f"   TOTALE: {len(hubble_patches) + len(obs_patches)}")
    
    # Crea coppie
    pairs = []
    if hubble_patches and obs_patches:
        pairs = create_patch_pairs(hubble_patches, obs_patches, OUTPUT_DIR_PATCHES, logger)
    
    # Split dataset
    print(f"\n{'='*70}")
    print("SPLIT DATASET")
    print(f"{'='*70}")
    
    if hubble_patches:
        print(f"\n📊 Hubble:")
        hubble_splits = create_dataset_split(hubble_patches, OUTPUT_HUBBLE_PATCHES, logger)
        if hubble_splits:
            print(f"   Train: {len(hubble_splits['train'])}")
            print(f"   Val: {len(hubble_splits['val'])}")
            print(f"   Test: {len(hubble_splits['test'])}")
    
    if obs_patches:
        print(f"\n📊 Observatory:")
        obs_splits = create_dataset_split(obs_patches, OUTPUT_OBS_PATCHES, logger)
        if obs_splits:
            print(f"   Train: {len(obs_splits['train'])}")
            print(f"   Val: {len(obs_splits['val'])}")
            print(f"   Test: {len(obs_splits['test'])}")
    
    # Metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'source_type': source_type,
        'target_fov_arcmin': TARGET_FOV_ARCMIN,
        'overlap_percent': OVERLAP_PERCENT,
        'hubble': {'num_patches': len(hubble_patches)},
        'observatory': {'num_patches': len(obs_patches)},
        'pairs': {'num_pairs': len(pairs)},
    }
    
    metadata_file = OUTPUT_DIR_PATCHES / 'dataset_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ ESTRAZIONE COMPLETATA")
    print(f"{'='*70}")
    print(f"\n📁 Output: {OUTPUT_DIR_PATCHES}")
    
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main unificato"""
    start_time = time.time()
    
    # Menu principale
    mode = main_menu()
    
    success = True
    
    if mode == 'analysis':
        # Solo analisi
        success = run_analysis()
        
    elif mode == 'patches':
        # Solo patches
        success = run_patches()
        
    elif mode == 'both':
        # Prima analisi
        print("\n" + "="*70)
        print("FASE 1: ANALISI OVERLAP")
        print("="*70)
        success_analysis = run_analysis()
        
        if success_analysis:
            # Poi patches
            print("\n" + "="*70)
            print("FASE 2: ESTRAZIONE PATCHES")
            print("="*70)
            success_patches = run_patches()
            success = success_analysis and success_patches
        else:
            print(f"\n⚠️  Analisi fallita, skip estrazione patches")
            success = False
    
    # Summary finale
    elapsed = time.time() - start_time
    
    print(f"\n" + "="*70)
    print("COMPLETAMENTO PIPELINE")
    print("="*70)
    
    if success:
        print(f"\n✅ Pipeline completata con successo!")
    else:
        print(f"\n⚠️  Pipeline completata con alcuni problemi")
    
    print(f"\n⏱️  Tempo totale: {elapsed:.1f} secondi ({elapsed/60:.1f} minuti)")
    
    print(f"\n📁 OUTPUT DIRECTORIES:")
    if mode in ['analysis', 'both']:
        print(f"   Analisi: {OUTPUT_DIR_ANALYSIS}")
    if mode in ['patches', 'both']:
        print(f"   Patches: {OUTPUT_DIR_PATCHES}")
        # Mostra anche la nuova cartella delle coppie
        if OUTPUT_DIR_PATCHES:
             print(f"   Coppie:  {OUTPUT_DIR_PATCHES / 'paired_patches_folders'}")
    
    print(f"\n{'='*70}")
    print("GRAZIE PER AVER USATO LO SCRIPT UNIFICATO!")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()