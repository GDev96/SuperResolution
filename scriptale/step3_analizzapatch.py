#!/usr/bin/env python3
"""
STEP 5-6 UNIFICATO: ANALISI OVERLAP E ESTRAZIONE PATCHES

MODIFICATO:
- Aggiunto menu iniziale per selezionare la cartella del target (se avviato da solo).
- AGGIUNTO: Opzione 'Processa TUTTI' nel menu di selezione.
- AGGIUNTO: Loop principale in main() per processare uno o più target.
- Accetta BASE_DIR come argomento da riga di comando (da step2).
- AGGIUNTO: Multi-threading per l'estrazione patches (Step 6).
- AGGIUNTO: Multi-threading per l'accoppiamento patches (Step 6).
- AGGIUNTO: Sostituzione dei file .json di metadati patch/coppie con report .md.
- AGGIUNTO: Sostituzione di 'analisi_overlap_report.json' (Step 5) con un report .md.
- AGGIUNTO: Menu di prosecuzione dopo la modalità "SOLO ANALISI".
- CORRETTO: Typo 'Clock' in 'run_patches'.
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
import shutil
import sys 
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

LOG_DIR_ROOT = Path(r'F:\Super Revolt Gaia\logs')
TARGET_PATCH_ARCMIN_ANALYSIS = 1.0
PATCH_OVERLAP_PERCENT_ANALYSIS = 10
TARGET_FOV_ARCMIN = 1
OVERLAP_PERCENT = 25
MIN_VALID_PERCENT = 50
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_THREADS = 7

# ============================================================================
# FUNZIONE SELEZIONE TARGET (MODIFICATA PER "TUTTI")
# ============================================================================

ROOT_DATA_DIR = Path(r'F:\Super Revolt Gaia\SuperResolution\data')

def select_target_directory():
    """
    Mostra un menu per selezionare una o TUTTE le cartelle target.
    Restituisce: Elenco di Path (uno o più) o elenco vuoto.
    """
    print("\n" + "📂"*35); print("SELEZIONE CARTELLA TARGET".center(70)); print("📂"*35)
    print(f"\nScansione sottocartelle in: {ROOT_DATA_DIR}")

    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir()]
    except Exception as e:
        print(f"\n❌ ERRORE: Impossibile leggere la cartella {ROOT_DATA_DIR}: {e}"); return []
    if not subdirs:
        print(f"\n❌ ERRORE: Nessuna sottocartella trovata in {ROOT_DATA_DIR}"); return []

    print("\nCartelle target disponibili:")
    print(f"   0: ✨ Processa TUTTI i {len(subdirs)} target")
    print("   " + "─"*30)
    for i, dir_path in enumerate(subdirs): print(f"   {i+1}: {dir_path.name}")

    while True:
        print("\n" + "─"*70)
        try:
            choice_str = input(f"👉 Seleziona un numero (0-{len(subdirs)}) o 'q' per uscire: ").strip()
            if choice_str.lower() == 'q': print("👋 Uscita."); return []
            choice = int(choice_str)
            if choice == 0:
                print(f"\n✅ Selezionati TUTTI i {len(subdirs)} target."); return subdirs
            choice_idx = choice - 1
            if 0 <= choice_idx < len(subdirs):
                selected_dir = subdirs[choice_idx]
                print(f"\n✅ Cartella selezionata: {selected_dir.name}"); return [selected_dir]
            else:
                print(f"❌ Scelta non valida. Inserisci un numero tra 0 e {len(subdirs)}.")
        except ValueError: print("❌ Input non valido. Inserisci un numero.")
        except Exception as e: print(f"❌ Errore: {e}"); return []

# ============================================================================
# MENU PRINCIPALE E DI PROSEGUIMENTO (MODIFICATI)
# ============================================================================

def main_menu():
    print("\n" + "🌟"*35); print("STEP 5-6 UNIFICATO: ANALISI & PATCHES".center(70)); print("🌟"*35)
    print(f"\n📋 SCEGLI OPERAZIONE (verrà applicata a tutti i target selezionati):")
    print(f"\n1️⃣  SOLO ANALISI (Step 5)")
    print(f"\n2️⃣  SOLO ESTRAZIONE PATCHES (Step 6)")
    print(f"\n3️⃣  ENTRAMBI (Step 5 + Step 6)")
    while True:
        print(f"\n" + "─"*70); choice = input("👉 Scegli opzione [1/2/3, default=3]: ").strip()
        if choice == '1': print(f"\n✅ Selezionato: SOLO ANALISI"); return 'analysis'
        elif choice == '2': print(f"\n✅ Selezionato: SOLO ESTRAZIONE PATCHES"); return 'patches'
        elif choice in ('', '3'): print(f"\n✅ Selezionato: ENTRAMBI (Analisi + Patches)"); return 'both'
        else: print(f"❌ Scelta non valida. Inserisci 1, 2 o 3.")

def ask_continue_to_patches(target_list):
    """
    Chiede all'utente se vuole proseguire con l'estrazione Patches (Step 6).
    """
    print("\n" + "="*70); print("🎯 ANALISI (Step 5) COMPLETATA"); print("="*70)
    print("\n📋 OPZIONI PROSSIMO STEP:\n   1️⃣  Continua con Estrazione Patches (Step 6)\n   2️⃣  Termina qui")
    
    if len(target_list) > 1:
        prompt_msg = f"per i {len(target_list)} target analizzati?"
    else:
        prompt_msg = f"per {target_list[0].name}?"

    while True:
        print("\n" + "─"*70)
        choice = input(f"👉 Vuoi continuare con l'estrazione delle Patches {prompt_msg} [S/n, default=S]: ").strip().lower()
        if choice in ('', 's', 'si', 'y', 'yes'):
            print("\n✅ Proseguimento con Estrazione Patches...")
            return True
        elif choice in ('n', 'no'):
            print("\n✅ Terminato dopo l'analisi.")
            return False
        else:
            print("❌ Scelta non valida. Inserisci S per Sì o N per No.")

# ============================================================================
# STEP 5: ANALISI (MODIFICATO PER REPORT MD)
# ============================================================================

class ImageAnalyzer:
    def __init__(self, filepath):
        self.filepath = Path(filepath); self.data = None; self.header = None; self.wcs = None; self.info = {}
    def load(self):
        try:
            with fits.open(self.filepath) as hdul:
                data_hdu = None
                for i, hdu in enumerate(hdul):
                    if hdu.data is not None and len(hdu.data.shape) >= 2: data_hdu = hdu; break
                if data_hdu is None: return False
                self.data = data_hdu.data; self.header = data_hdu.header
                if len(self.data.shape) == 3: self.data = self.data[0]
                ny, nx = self.data.shape
                self.info = { 'filename': self.filepath.name, 'shape': (ny, nx), 'dtype': str(self.data.dtype), 'size_mb': round(self.data.nbytes / (1024**2), 2) }
                valid_mask = np.isfinite(self.data); valid_data = self.data[valid_mask]
                if len(valid_data) > 0:
                    self.info['stats'] = { 'valid_pixels': int(valid_mask.sum()), 'coverage_percent': round(100 * valid_mask.sum() / self.data.size, 2),
                                           'min': float(np.min(valid_data)), 'max': float(np.max(valid_data)), 'mean': float(np.mean(valid_data)), 'median': float(np.median(valid_data)) }
                try:
                    self.wcs = WCS(self.header)
                    if self.wcs.has_celestial: self._analyze_wcs()
                    else: self.wcs = None; return False
                except: self.wcs = None; return False
                return True
        except Exception as e: return False
    def _analyze_wcs(self):
        ny, nx = self.data.shape; center = self.wcs.pixel_to_world(nx/2, ny/2); pixel_scale = self._get_pixel_scale()
        corners = self.wcs.pixel_to_world([0, nx, nx, 0], [0, 0, ny, ny]); ra_vals = [c.ra.deg for c in corners]; dec_vals = [c.dec.deg for c in corners]
        ra_span, dec_span = max(ra_vals) - min(ra_vals), max(dec_vals) - min(dec_vals)
        self.info['wcs'] = { 'center_ra': float(center.ra.deg), 'center_dec': float(center.dec.deg), 'pixel_scale_arcsec': pixel_scale, 'pixel_scale_arcmin': pixel_scale / 60.0,
                             'fov_ra_deg': ra_span, 'fov_dec_deg': dec_span, 'fov_ra_arcmin': ra_span * 60, 'fov_dec_arcmin': dec_span * 60,
                             'ra_range': [min(ra_vals), max(ra_vals)], 'dec_range': [min(dec_vals), max(dec_vals)] }
    def _get_pixel_scale(self):
        try:
            if hasattr(self.wcs.wcs, 'cd') and self.wcs.wcs.cd is not None: pixel_scale_deg = np.sqrt(self.wcs.wcs.cd[0,0]**2 + self.wcs.wcs.cd[0,1]**2)
            elif hasattr(self.wcs.wcs, 'cdelt'): pixel_scale_deg = abs(self.wcs.wcs.cdelt[0])
            else: p1 = self.wcs.pixel_to_world(0, 0); p2 = self.wcs.pixel_to_world(1, 0); pixel_scale_deg = p1.separation(p2).deg
            return pixel_scale_deg * 3600
        except: return None
    def calculate_patch_size(self, target_arcmin):
        if 'wcs' not in self.info or self.info['wcs']['pixel_scale_arcsec'] is None: return None
        pixel_scale_arcsec = self.info['wcs']['pixel_scale_arcsec']; target_arcsec = target_arcmin * 60
        patch_size_px = int(target_arcsec / pixel_scale_arcsec); patch_size_px = ((patch_size_px + 7) // 8) * 8
        actual_arcsec = patch_size_px * pixel_scale_arcsec; actual_arcmin = actual_arcsec / 60; ny, nx = self.data.shape
        n_x, n_y, total_no_overlap = nx // patch_size_px, ny // patch_size_px, (nx // patch_size_px) * (ny // patch_size_px)
        step = int(patch_size_px * (1 - PATCH_OVERLAP_PERCENT_ANALYSIS/100))
        n_x_overlap, n_y_overlap = (max(1, (nx - patch_size_px) // step + 1) if step > 0 else 1), (max(1, (ny - patch_size_px) // step + 1) if step > 0 else 1)
        total_with_overlap = n_x_overlap * n_y_overlap
        return { 'target_arcmin': target_arcmin, 'patch_size_px': patch_size_px, 'actual_size_arcmin': actual_arcmin, 'actual_size_arcsec': actual_arcsec,
                 'patches_no_overlap': {'x': n_x, 'y': n_y, 'total': total_no_overlap}, 'patches_with_overlap': {'x': n_x_overlap, 'y': n_y_overlap, 'total': total_with_overlap} }

class OverlapAnalyzer:
    def __init__(self, img1, img2): self.img1 = img1; self.img2 = img2; self.overlap_info = None
    def calculate_overlap(self):
        if self.img1.wcs is None or self.img2.wcs is None: return None
        wcs1, wcs2 = self.img1.info['wcs'], self.img2.info['wcs']
        ra1_min, ra1_max = wcs1['ra_range']; ra2_min, ra2_max = wcs2['ra_range']
        overlap_ra_min, overlap_ra_max = max(ra1_min, ra2_min), min(ra1_max, ra2_max)
        dec1_min, dec1_max = wcs1['dec_range']; dec2_min, dec2_max = wcs2['dec_range']
        overlap_dec_min, overlap_dec_max = max(dec1_min, dec2_min), min(dec1_max, dec2_max)
        if overlap_ra_max <= overlap_ra_min or overlap_dec_max <= overlap_dec_min: return None
        overlap_ra_span, overlap_dec_span = overlap_ra_max - overlap_ra_min, overlap_dec_max - overlap_dec_min
        overlap_area_deg2 = overlap_ra_span * overlap_dec_span; overlap_area_arcmin2 = overlap_area_deg2 * 3600
        area1, area2 = wcs1['fov_ra_deg'] * wcs1['fov_dec_deg'], wcs2['fov_ra_deg'] * wcs2['fov_dec_deg']
        self.overlap_info = { 'overlap_ra_range': [overlap_ra_min, overlap_ra_max], 'overlap_dec_range': [overlap_dec_min, overlap_dec_max], 'overlap_ra_deg': overlap_ra_span, 'overlap_dec_deg': overlap_dec_span,
                              'overlap_area_deg2': overlap_area_deg2, 'overlap_area_arcmin2': overlap_area_arcmin2, 'fraction_img1': overlap_area_deg2 / area1 if area1 > 0 else 0, 'fraction_img2': overlap_area_deg2 / area2 if area2 > 0 else 0 }
        return self.overlap_info
    def visualize(self, output_path):
        if self.overlap_info is None: return
        fig, ax = plt.subplots(figsize=(10, 8))
        wcs1 = self.img1.info['wcs']; ra1, dec1 = wcs1['ra_range'], wcs1['dec_range']
        ax.add_patch(plt.Rectangle((ra1[0], dec1[0]), ra1[1]-ra1[0], dec1[1]-dec1[0], fill=False, edgecolor='blue', linewidth=2, label=self.img1.filepath.name))
        wcs2 = self.img2.info['wcs']; ra2, dec2 = wcs2['ra_range'], wcs2['dec_range']
        ax.add_patch(plt.Rectangle((ra2[0], dec2[0]), ra2[1]-ra2[0], dec2[1]-dec2[0], fill=False, edgecolor='red', linewidth=2, label=self.img2.filepath.name))
        ov = self.overlap_info
        ax.add_patch(plt.Rectangle((ov['overlap_ra_range'][0], ov['overlap_dec_range'][0]), ov['overlap_ra_deg'], ov['overlap_dec_deg'], fill=True, facecolor='green', alpha=0.3, edgecolor='green', linewidth=2, label='Overlap'))
        ax.set_xlabel('RA (deg)'); ax.set_ylabel('DEC (deg)'); ax.set_title('Image Overlap Analysis'); ax.legend(); ax.grid(True, alpha=0.3); ax.invert_xaxis()
        plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close()

def load_images_analysis(directory, label):
    print(f"\n🔍 Caricamento {label}..."); files = list(directory.glob('*.fits')) + list(directory.glob('*.fit'))
    if not files: print(f"   ⚠️  Nessun file trovato in {directory}"); return []
    images = []
    for fpath in files:
        img = ImageAnalyzer(fpath)
        if img.load(): images.append(img); print(f"   ✓ {img.filepath.name}: {img.info['shape']}")
        else: print(f"   ✗ {img.filepath.name}: errore caricamento")
    print(f"   Totale {label}: {len(images)}/{len(files)} immagini caricate"); return images

def analyze_all_pairs(hubble_imgs, obs_imgs):
    print(f"\n🔗 Analisi overlap tra {len(hubble_imgs)} Hubble e {len(obs_imgs)} Observatory..."); results = []
    for h_img in hubble_imgs:
        for o_img in obs_imgs:
            analyzer = OverlapAnalyzer(h_img, o_img); overlap = analyzer.calculate_overlap()
            if overlap and overlap['overlap_area_arcmin2'] > 0.1:
                results.append({ 'hubble': h_img, 'observatory': o_img, 'overlap': overlap, 'analyzer': analyzer })
    results.sort(key=lambda x: x['overlap']['overlap_area_arcmin2'], reverse=True); return results

def print_patch_analysis(img, label):
    print(f"\n📊 {label}: {img.filepath.name}\n   Dimensioni: {img.info['shape'][1]} × {img.info['shape'][0]} px")
    if 'wcs' in img.info:
        wcs = img.info['wcs']
        print(f"   Pixel scale: {wcs['pixel_scale_arcsec']:.4f} arcsec/px\n   FOV: {wcs['fov_ra_arcmin']:.2f} × {wcs['fov_dec_arcmin']:.2f} arcmin\n   Centro: RA={wcs['center_ra']:.6f}°, DEC={wcs['center_dec']:.6f}°")
    print(f"\n💡 DIMENSIONI PATCHES POSSIBILI:")
    for target_arcmin in [0.5, 1.0, 2.0, 5.0]:
        patch_info = img.calculate_patch_size(target_arcmin)
        if patch_info:
            print(f"\n   Target: {target_arcmin} arcmin\n      Patch size: {patch_info['patch_size_px']} × {patch_info['patch_size_px']} px\n      Actual size: {patch_info['actual_size_arcmin']:.4f} arcmin"
                  f"\n      Patches (no overlap): {patch_info['patches_no_overlap']['total']}\n      Patches ({PATCH_OVERLAP_PERCENT_ANALYSIS}% overlap): {patch_info['patches_with_overlap']['total']}")

# --- NUOVA FUNZIONE: REPORT MD PER ANALISI (STEP 5) ---
def create_summary_report_md(hubble_imgs, obs_imgs, overlap_results, output_dir):
    """Crea report riassuntivo dell'ANALISI (Step 5) in Markdown."""
    output_path = output_dir / 'analisi_overlap_report.md'
    print(f"\n   ✍️  Scrittura report MD: {output_path.name}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Report Analisi Overlap e Patches (Step 5)\n\n")
            f.write(f"**Report generato il:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"--- \n")
            
            # --- Sezione Hubble ---
            f.write(f"## 🛰️ Immagini Hubble Analizzate ({len(hubble_imgs)})\n\n")
            f.write("| Immagine | Dimensioni (px) | Pixel Scale (\"/px) | FOV (arcmin) | Centro (RA, Dec) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for img in hubble_imgs:
                wcs_info = img.info.get('wcs', {})
                shape = img.info.get('shape', ('N/A', 'N/A'))
                scale = f"{wcs_info.get('pixel_scale_arcsec', 0):.4f}"
                fov = f"{wcs_info.get('fov_ra_arcmin', 0):.2f} x {wcs_info.get('fov_dec_arcmin', 0):.2f}"
                center = f"{wcs_info.get('center_ra', 0):.5f}, {wcs_info.get('center_dec', 0):.5f}"
                f.write(f"| `{img.filepath.name}` | {shape[1]}x{shape[0]} | {scale} | {fov} | {center} |\n")
            
            # --- Sezione Observatory ---
            f.write(f"\n## 📡 Immagini Observatory Analizzate ({len(obs_imgs)})\n\n")
            f.write("| Immagine | Dimensioni (px) | Pixel Scale (\"/px) | FOV (arcmin) | Centro (RA, Dec) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for img in obs_imgs:
                wcs_info = img.info.get('wcs', {})
                shape = img.info.get('shape', ('N/A', 'N/A'))
                scale = f"{wcs_info.get('pixel_scale_arcsec', 0):.4f}"
                fov = f"{wcs_info.get('fov_ra_arcmin', 0):.2f} x {wcs_info.get('fov_dec_arcmin', 0):.2f}"
                center = f"{wcs_info.get('center_ra', 0):.5f}, {wcs_info.get('center_dec', 0):.5f}"
                f.write(f"| `{img.filepath.name}` | {shape[1]}x{shape[0]} | {scale} | {fov} | {center} |\n")

            # --- Sezione Overlap ---
            f.write(f"\n---\n")
            f.write(f"## 🔗 Risultati Overlap\n\n")
            f.write(f"**Coppie con overlap trovate:** {len(overlap_results)}\n\n")
            if overlap_results:
                f.write(f"**Top 5 Matches (per area):**\n")
                f.write("| Immagine Hubble | Immagine Observatory | Area Overlap (arcmin²) | Copertura Hubble | Copertura Observatory |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- |\n")
                for result in overlap_results[:5]:
                    h_file = result['hubble'].filepath.name
                    o_file = result['observatory'].filepath.name
                    area = f"{result['overlap']['overlap_area_arcmin2']:.2f}"
                    cov_h = f"{result['overlap']['fraction_img1']*100:.1f}%"
                    cov_o = f"{result['overlap']['fraction_img2']*100:.1f}%"
                    f.write(f"| `{h_file}` | `{o_file}` | {area} | {cov_h} | {cov_o} |\n")
            
            # --- Sezione Raccomandazioni Patch ---
            f.write(f"\n---\n")
            f.write(f"## 💡 Raccomandazioni Patch (per 1.0 arcmin)\n\n")
            if hubble_imgs:
                h_patch = hubble_imgs[0].calculate_patch_size(1.0)
                if h_patch:
                    f.write(f"**Hubble (esempio):**\n")
                    f.write(f"- Dimensione Patch: `{h_patch['patch_size_px']} x {h_patch['patch_size_px']} px`\n")
                    f.write(f"- Dimensione Reale: `{h_patch['actual_size_arcmin']:.4f} arcmin`\n")
                    f.write(f"- N. Patches (con {PATCH_OVERLAP_PERCENT_ANALYSIS}% overlap): `{h_patch['patches_with_overlap']['total']}`\n\n")
            if obs_imgs:
                o_patch = obs_imgs[0].calculate_patch_size(1.0)
                if o_patch:
                    f.write(f"**Observatory (esempio):**\n")
                    f.write(f"- Dimensione Patch: `{o_patch['patch_size_px']} x {o_patch['patch_size_px']} px`\n")
                    f.write(f"- Dimensione Reale: `{o_patch['actual_size_arcmin']:.4f} arcmin`\n")
                    f.write(f"- N. Patches (con {PATCH_OVERLAP_PERCENT_ANALYSIS}% overlap): `{o_patch['patches_with_overlap']['total']}` (per immagine)\n\n")
    except Exception as e:
        print(f"   ❌ ERRORE: Impossibile scrivere il report MD {output_path.name}: {e}")

# --- FUNZIONE RUN_ANALYSIS (MODIFICATA) ---
def run_analysis(HUBBLE_DIR_ANALYSIS, OBS_DIR_ANALYSIS, OUTPUT_DIR_ANALYSIS):
    print("\n" + "🔭"*35); print(f"STEP 5: ANALISI OVERLAP E PATCHES".center(70)); print("🔭"*35)
    print(f"\n📂 CONFIGURAZIONE:\n   Hubble: {HUBBLE_DIR_ANALYSIS}\n   Observatory: {OBS_DIR_ANALYSIS}\n   Output: {OUTPUT_DIR_ANALYSIS}")
    OUTPUT_DIR_ANALYSIS.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\nCARICAMENTO IMMAGINI\n{'='*70}")
    hubble_imgs = load_images_analysis(HUBBLE_DIR_ANALYSIS, "HUBBLE")
    obs_imgs = load_images_analysis(OBS_DIR_ANALYSIS, "OBSERVATORY")
    if not hubble_imgs: print(f"\n❌ Nessuna immagine Hubble caricata!"); return False
    if not obs_imgs: print(f"\n❌ Nessuna immagine Observatory caricata!"); return False
    print(f"\n{'='*70}\nANALISI DIMENSIONI PATCHES\n{'='*70}")
    print(f"\n🛰️  HUBBLE IMAGES:"); [print_patch_analysis(img, "HUBBLE") for img in hubble_imgs[:3]]
    if len(hubble_imgs) > 3: print(f"\n   ℹ️  ... e altre {len(hubble_imgs) - 3} immagini Hubble")
    print(f"\n📡 OBSERVATORY IMAGES:")
    if obs_imgs: print_patch_analysis(obs_imgs[0], "OBSERVATORY (esempio)")
    if len(obs_imgs) > 1: print(f"\n   ℹ️  Mostrata solo la prima immagine observatory")
    overlap_results = analyze_all_pairs(hubble_imgs, obs_imgs)
    print(f"\n{'='*70}\nRISULTATI OVERLAP\n{'='*70}")
    if overlap_results:
        print(f"\n✅ Trovate {len(overlap_results)} coppie con overlap!")
        print(f"\n🏆 TOP 5 MATCHES:")
        for i, result in enumerate(overlap_results[:5], 1):
            print(f"\n   {i}. {result['hubble'].filepath.name} ↔ {result['observatory'].filepath.name}\n      Overlap area: {result['overlap']['overlap_area_arcmin2']:.2f} arcmin²"
                  f"\n      Hubble coverage: {result['overlap']['fraction_img1']*100:.1f}%\n      Observatory coverage: {result['overlap']['fraction_img2']*100:.1f}%")
        best = overlap_results[0]; viz_path = OUTPUT_DIR_ANALYSIS / 'overlap_best_match.png'; best['analyzer'].visualize(viz_path)
        print(f"\n📊 Visualizzazione salvata: {viz_path}")
    else: print(f"\n⚠️  Nessun overlap trovato!")
    
    # --- MODIFICA: Chiama la nuova funzione MD ---
    create_summary_report_md(hubble_imgs, obs_imgs, overlap_results, OUTPUT_DIR_ANALYSIS)
    # --- FINE MODIFICA ---
    
    print(f"\n{'='*70}\n💡 RACCOMANDAZIONI PATCHES\n{'='*70}")
    if hubble_imgs and obs_imgs:
        h_patch, o_patch = hubble_imgs[0].calculate_patch_size(1.0), obs_imgs[0].calculate_patch_size(1.0)
        if h_patch and o_patch:
            print(f"\n📐 DIMENSIONI CONSIGLIATE (1 arcmin target):")
            print(f"\n   Hubble:\n      Patch size: {h_patch['patch_size_px']} × {h_patch['patch_size_px']} px\n      Patches totali: {h_patch['patches_with_overlap']['total']}")
            print(f"\n   Observatory:\n      Patch size: {o_patch['patch_size_px']} × {o_patch['patch_size_px']} px\n      Patches totali: {o_patch['patches_with_overlap']['total'] * len(obs_imgs)}")
    print(f"\n{'='*70}\n✅ ANALISI (Step 5) COMPLETATA\n{'='*70}\n📁 Output salvato in: {OUTPUT_DIR_ANALYSIS}")
    return True

# ============================================================================
# STEP 6: ESTRAZIONE PATCHES
# ============================================================================

def select_input_type(BASE_DIR, INPUT_CROPPED_HUBBLE, INPUT_CROPPED_OBSERVATORY, INPUT_REGISTERED_HUBBLE, INPUT_REGISTERED_OBSERVATORY):
    print("\n" + "🎯"*35); print("SELEZIONE TIPO IMMAGINI INPUT (PER STEP 6)".center(70)); print("🎯"*35)
    print(f"\n📋 OPZIONI DISPONIBILI:\n\n1️⃣  IMMAGINI CROPPED (CONSIGLIATO)\n   ✓ Tutte le immagini hanno le stesse dimensioni\n   📁 Input: {INPUT_CROPPED_HUBBLE.parent}")
    print(f"\n2️⃣  IMMAGINI REGISTERED\n   • Mantiene dimensioni originali\n   📁 Input: {INPUT_REGISTERED_HUBBLE.parent}")
    INPUT_HUBBLE, INPUT_OBSERVATORY, OUTPUT_DIR_PATCHES, source_type = None, None, None, ""
    while True:
        print(f"\n" + "─"*70); choice = input("👉 Scegli opzione [1/2, default=1]: ").strip()
        if choice == '' or choice == '1':
            INPUT_HUBBLE, INPUT_OBSERVATORY, OUTPUT_DIR_PATCHES, source_type = INPUT_CROPPED_HUBBLE, INPUT_CROPPED_OBSERVATORY, BASE_DIR / '6_patches_from_cropped', "cropped"
            print(f"\n✅ Selezionato: IMMAGINI CROPPED"); break
        elif choice == '2':
            INPUT_HUBBLE, INPUT_OBSERVATORY, OUTPUT_DIR_PATCHES, source_type = INPUT_REGISTERED_HUBBLE, INPUT_REGISTERED_OBSERVATORY, BASE_DIR / '6_patches_from_registered', "registered"
            print(f"\n✅ Selezionato: IMMAGINI REGISTERED"); break
        else: print(f"❌ Scelta non valida. Inserisci 1 o 2.")
    OUTPUT_HUBBLE_PATCHES, OUTPUT_OBS_PATCHES = OUTPUT_DIR_PATCHES / 'hubble_native', OUTPUT_DIR_PATCHES / 'observatory_native'
    print(f"\n🔍 Verifica directory...")
    if INPUT_HUBBLE.exists(): print(f"   ✓ Hubble: {len(list(INPUT_HUBBLE.glob('*.fits')) + list(INPUT_HUBBLE.glob('*.fit')))} file trovati")
    else: print(f"   ⚠️  Directory Hubble non trovata: {INPUT_HUBBLE}")
    if INPUT_OBSERVATORY.exists(): print(f"   ✓ Observatory: {len(list(INPUT_OBSERVATORY.glob('*.fits')) + list(INPUT_OBSERVATORY.glob('*.fit')))} file trovati")
    else: print(f"   ⚠️  Directory Observatory non trovata: {INPUT_OBSERVATORY}")
    print(f"\n📂 Output patches: {OUTPUT_DIR_PATCHES}")
    return (source_type, INPUT_HUBBLE, INPUT_OBSERVATORY, OUTPUT_DIR_PATCHES, OUTPUT_HUBBLE_PATCHES, OUTPUT_OBS_PATCHES)

def setup_logging():
    LOG_DIR_ROOT.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR_ROOT / f'patch_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()])
    return logging.getLogger(__name__)

def extract_patches_from_image(fits_path, output_dir, source_label, logger):
    try:
        with fits.open(fits_path) as hdul:
            data_hdu = None
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) >= 2: data_hdu = hdu; break
            if data_hdu is None: return []
            data, header = data_hdu.data, data_hdu.header
            if len(data.shape) == 3: data = data[0]
            try: wcs = WCS(header); assert wcs.has_celestial
            except: return []
            try:
                if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None: pixel_scale_deg = np.sqrt(wcs.wcs.cd[0,0]**2 + wcs.wcs.cd[0,1]**2)
                elif hasattr(wcs.wcs, 'cdelt'): pixel_scale_deg = abs(wcs.wcs.cdelt[0])
                else: p1 = wcs.pixel_to_world(0, 0); p2 = wcs.pixel_to_world(1, 0); pixel_scale_deg = p1.separation(p2).deg
                pixel_scale_arcsec = pixel_scale_deg * 3600
            except: return []
            target_arcsec = TARGET_FOV_ARCMIN * 60
            patch_size_px = int(target_arcsec / pixel_scale_arcsec); patch_size_px = ((patch_size_px + 7) // 8) * 8
            step = int(patch_size_px * (1 - OVERLAP_PERCENT/100))
            ny, nx = data.shape; patches_info = []; patch_idx = 0; y = 0
            while y + patch_size_px <= ny:
                x = 0
                while x + patch_size_px <= nx:
                    patch_data = data[y:y+patch_size_px, x:x+patch_size_px]
                    valid_mask = np.isfinite(patch_data); valid_percent = 100 * valid_mask.sum() / patch_data.size
                    if valid_percent >= MIN_VALID_PERCENT:
                        center_x, center_y = x + patch_size_px // 2, y + patch_size_px // 2
                        center_coord = wcs.pixel_to_world(center_x, center_y)
                        filename = f'{source_label}_{Path(fits_path).stem}_p{patch_idx:04d}.fits'
                        output_path = output_dir / filename
                        patch_header = header.copy(); patch_header['NAXIS1'] = patch_size_px; patch_header['NAXIS2'] = patch_size_px
                        patch_hdu = fits.PrimaryHDU(patch_data, header=patch_header); patch_hdu.writeto(output_path, overwrite=True)
                        patches_info.append({ 'filename': filename, 'source_file': Path(fits_path).name, 'patch_index': patch_idx, 'position_px': (x, y), 'patch_size_px': patch_size_px,
                                              'center_ra': center_coord.ra.deg, 'center_dec': center_coord.dec.deg, 'pixel_scale_arcsec': pixel_scale_arcsec, 'valid_percent': valid_percent })
                        patch_idx += 1
                    x += step
                y += step
            return patches_info
    except Exception as e:
        logger.error(f"Errore in {fits_path.name}: {e}"); return []

# ============================================================================
# FUNZIONI REPORT MD (PER STEP 6)
# ============================================================================

def save_patch_metadata_md(patches_info, output_path, label):
    """Salva i metadati delle patch in un file Markdown leggibile."""
    print(f"   ✍️  Scrittura report MD: {output_path.name}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Report Metadati Patches: {label}\n\n")
            f.write(f"**Totale patches estratte:** {len(patches_info)}\n")
            f.write(f"**Report generato il:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("| Immagine Sorgente | Nome Patch | Posizione (X, Y) | Dimensione (px) | % Valida | Centro RA (deg) | Centro Dec (deg) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for patch in patches_info:
                pos = patch.get('position_px', ('N/A', 'N/A'))
                size = patch.get('patch_size_px', 'N/A')
                valid = f"{patch.get('valid_percent', 0):.2f}%"
                ra = f"{patch.get('center_ra', 0):.6f}"
                dec = f"{patch.get('center_dec', 0):.6f}"
                f.write(f"| {patch.get('source_file', 'N/A')} | `{patch.get('filename', 'N/A')}` | ({pos[0]}, {pos[1]}) | {size}x{size} | {valid} | {ra} | {dec} |\n")
    except Exception as e:
        print(f"   ❌ ERRORE: Impossibile scrivere il report MD {output_path.name}: {e}")

def save_pairs_metadata_md(pairs_info, output_path):
    """Salva i metadati delle coppie in un file Markdown leggibile."""
    print(f"   ✍️  Scrittura report MD: {output_path.name}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Report Accoppiamento Patches\n\n")
            f.write(f"**Totale coppie create:** {len(pairs_info)}\n")
            f.write(f"**Report generato il:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("| ID Coppia (Folder) | Patch Hubble | Patch Observatory | Distanza (arcmin) |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for i, pair in enumerate(pairs_info):
                pair_id = f"`pair_{i:05d}`"
                h_patch = f"`{pair.get('hubble_patch', 'N/A')}`"
                o_patch = f"`{pair.get('observatory_patch', 'N/A')}`"
                dist = f"{pair.get('separation_arcmin', 0):.4f}"
                f.write(f"| {pair_id} | {h_patch} | {o_patch} | {dist} |\n")
    except Exception as e:
        print(f"   ❌ ERRORE: Impossibile scrivere il report MD {output_path.name}: {e}")

def save_summary_metadata_md(metadata, output_path):
    """Salva il report MD di riepilogo del dataset."""
    print(f"   ✍️  Scrittura report MD: {output_path.name}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Report Riepilogo Dataset\n\n")
            f.write(f"**Generato il:** {metadata.get('timestamp', 'N/A')}\n\n")
            f.write("## Configurazione\n")
            f.write(f"- **Tipo Immagini Sorgente:** `{metadata.get('source_type', 'N/A')}`\n")
            f.write(f"- **FOV Target:** {metadata.get('target_fov_arcmin', 'N/A')} arcmin\n")
            f.write(f"- **Overlap:** {metadata.get('overlap_percent', 'N/A')}%\n")
            f.write("\n## Riepilogo Patch\n")
            f.write(f"- **Patches Hubble Estratte:** {metadata.get('hubble', {}).get('num_patches', 0)}\n")
            f.write(f"- **Patches Observatory Estratte:** {metadata.get('observatory', {}).get('num_patches', 0)}\n")
            f.write(f"- **Coppie Create:** {metadata.get('pairs', {}).get('num_pairs', 0)}\n")
    except Exception as e:
        print(f"   ❌ ERRORE: Impossibile scrivere il report MD {output_path.name}: {e}")

# ============================================================================
# FUNZIONI PATCH MODIFICATE (per chiamare i report MD)
# ============================================================================

def extract_all_patches(input_dir, output_dir, label, logger):
    """Estrae patches da tutte le immagini (con multithreading)"""
    print(f"\n🔪 Estrazione patches {label.upper()} (con {NUM_THREADS} threads)...")
    files = list(input_dir.glob('*.fits')) + list(input_dir.glob('*.fit'))
    if not files: print(f"   ⚠️  Nessun file in {input_dir}"); return []
    all_patches = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(extract_patches_from_image, fpath, output_dir, label, logger): fpath for fpath in files}
        with tqdm(total=len(files), desc=f"   {label}", ncols=70) as pbar:
            for future in as_completed(futures):
                fpath = futures[future]
                try:
                    patches_from_file = future.result()
                    if patches_from_file: all_patches.extend(patches_from_file)
                except Exception as e:
                    logger.error(f"Errore nel thread processando {fpath.name}: {e}")
                pbar.update(1)
    print(f"   ✓ Estratte {len(all_patches)} patches da {len(files)} immagini")
    metadata_file_md = output_dir / f'{label}_patches_report.md'
    save_patch_metadata_md(all_patches, metadata_file_md, label.upper())
    return all_patches


def find_best_match_for_hubble_patch(h_patch, obs_patches_list, threshold_arcmin):
    """Task per un singolo thread: Trova la migliore patch observatory"""
    h_coord = SkyCoord(h_patch['center_ra']*u.deg, h_patch['center_dec']*u.deg)
    best_dist = float('inf')
    best_obs = None
    for o_patch in obs_patches_list:
        o_coord = SkyCoord(o_patch['center_ra']*u.deg, o_patch['center_dec']*u.deg)
        dist_arcmin = h_coord.separation(o_coord).arcmin
        if dist_arcmin < best_dist:
            best_dist = dist_arcmin
            best_obs = o_patch
    if best_dist < threshold_arcmin and best_obs:
        return (h_patch, best_obs, best_dist)
    return None


def create_patch_pairs(hubble_patches, obs_patches, 
                       OUTPUT_DIR_PATCHES, 
                       OUTPUT_HUBBLE_PATCHES, 
                       OUTPUT_OBS_PATCHES, 
                       logger):
    """Crea coppie di patches (con multi-threading) e salva in cartelle dedicate."""
    print(f"\n🔗 Creazione coppie patches (con {NUM_THREADS} threads)...")
    hubble_patch_dir = OUTPUT_HUBBLE_PATCHES
    obs_patch_dir = OUTPUT_OBS_PATCHES
    pairs_folders_dir = OUTPUT_DIR_PATCHES / 'paired_patches_folders'
    pairs_folders_dir.mkdir(parents=True, exist_ok=True)
    print(f"   📂 Creazione cartelle coppie in: {pairs_folders_dir}")
    threshold_arcmin = 0.5
    raw_pairs_results = []
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = { executor.submit(find_best_match_for_hubble_patch, h_patch, obs_patches, threshold_arcmin): h_patch for h_patch in hubble_patches }
        print("   (Fase 1: Calcolo abbinamenti in parallelo...)")
        with tqdm(total=len(hubble_patches), desc="   Pairing (CPU)", ncols=70) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result: raw_pairs_results.append(result)
                pbar.update(1)

    print(f"\n   (Fase 2: Copia file e creazione cartelle...)")
    raw_pairs_results.sort(key=lambda x: x[0]['filename'])
    final_pairs_metadata = []
    
    for i, (h_patch, best_obs, best_dist) in enumerate(tqdm(raw_pairs_results, desc="   Copia (I/O)", ncols=70)):
        pair_folder_name = f'pair_{i:05d}'
        pair_dest_dir = pairs_folders_dir / pair_folder_name
        pair_dest_dir.mkdir(exist_ok=True)
        h_filename, o_filename = h_patch['filename'], best_obs['filename']
        h_src_path, o_src_path = hubble_patch_dir / h_filename, obs_patch_dir / o_filename
        try:
            shutil.copy2(h_src_path, pair_dest_dir / h_filename)
            shutil.copy2(o_src_path, pair_dest_dir / o_filename)
        except FileNotFoundError as e: logger.warning(f"File non trovato durante copia coppia: {e}")
        except Exception as e: logger.error(f"Errore copiando {pair_folder_name}: {e}")
        final_pairs_metadata.append({ 'hubble_patch': h_filename, 'observatory_patch': o_filename, 'separation_arcmin': best_dist })
    
    pairs_file_md = OUTPUT_DIR_PATCHES / 'patch_pairs_report.md'
    save_pairs_metadata_md(final_pairs_metadata, pairs_file_md)
    
    num_pairs_found = len(final_pairs_metadata)
    print(f"   ✓ {num_pairs_found} coppie create (report MD)")
    print(f"   ✓ {num_pairs_found} cartelle coppie copiate in {pairs_folders_dir}")
    return final_pairs_metadata


def create_dataset_split(patches, output_dir, logger):
    """Crea split train/val/test (MANTIENE JSON PERCHÉ È FUNZIONALE)"""
    if not patches: return None
    np.random.shuffle(patches); n = len(patches); n_train, n_val = int(n * TRAIN_RATIO), int(n * VAL_RATIO)
    splits = { 'train': patches[:n_train], 'val': patches[n_train:n_train+n_val], 'test': patches[n_train+n_val:] }
    split_file = output_dir / 'dataset_split.json'
    print(f"   ✍️  Scrittura split JSON: {split_file.name}")
    with open(split_file, 'w') as f:
        json.dump({ 'timestamp': datetime.now().isoformat(), 'train_ratio': TRAIN_RATIO, 'val_ratio': VAL_RATIO, 'test_ratio': TEST_RATIO,
                    'splits': {k: [p['filename'] for p in v] for k, v in splits.items()} }, f, indent=2)
    return splits


def run_patches(BASE_DIR, INPUT_CROPPED_HUBBLE, INPUT_CROPPED_OBSERVATORY, INPUT_REGISTERED_HUBBLE, INPUT_REGISTERED_OBSERVATORY):
    (source_type, INPUT_HUBBLE, INPUT_OBSERVATORY, OUTPUT_DIR_PATCHES, OUTPUT_HUBBLE_PATCHES, OUTPUT_OBS_PATCHES) = select_input_type(
         BASE_DIR, INPUT_CROPPED_HUBBLE, INPUT_CROPPED_OBSERVATORY, INPUT_REGISTERED_HUBBLE, INPUT_REGISTERED_OBSERVATORY)
    logger = setup_logging()
    print("\n" + "✂️ "*35); print(f"STEP 6: ESTRAZIONE PATCHES".center(70)); print("✂️ "*35)
    print(f"\n📋 CONFIGURAZIONE:\n   Tipo input: {source_type.upper()}\n   FOV target: {TARGET_FOV_ARCMIN} arcmin\n   Overlap: {OVERLAP_PERCENT}%")
    print(f"\n📂 INPUT:\n   Hubble: {INPUT_HUBBLE}\n   Observatory: {INPUT_OBSERVATORY}")
    print(f"\n📂 OUTPUT:\n   {OUTPUT_DIR_PATCHES}")
    OUTPUT_HUBBLE_PATCHES.mkdir(parents=True, exist_ok=True); OUTPUT_OBS_PATCHES.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\nESTRAZIONE PATCHES\n{'='*70}")
    hubble_patches, obs_patches = [], []
    if INPUT_HUBBLE.exists(): hubble_patches = extract_all_patches(INPUT_HUBBLE, OUTPUT_HUBBLE_PATCHES, 'hubble', logger)
    if INPUT_OBSERVATORY.exists(): obs_patches = extract_all_patches(INPUT_OBSERVATORY, OUTPUT_OBS_PATCHES, 'observatory', logger)
    
    # --- CORREZIONE TYPO ---
    print(f"\n{'='*70}\n📊 RIEPILOGO ESTRAZIONE\n{'='*70}")
    # --- FINE CORREZIONE ---
    
    print(f"\n   Hubble patches: {len(hubble_patches)}\n   Observatory patches: {len(obs_patches)}\n   TOTALE: {len(hubble_patches) + len(obs_patches)}")
    pairs = []
    if hubble_patches and obs_patches:
        pairs = create_patch_pairs(hubble_patches, obs_patches, OUTPUT_DIR_PATCHES, OUTPUT_HUBBLE_PATCHES, OUTPUT_OBS_PATCHES, logger)
    print(f"\n{'='*70}\nSPLIT DATASET\n{'='*70}")
    if hubble_patches:
        print(f"\n📊 Hubble:"); hubble_splits = create_dataset_split(hubble_patches, OUTPUT_HUBBLE_PATCHES, logger)
        if hubble_splits: print(f"   Train: {len(hubble_splits['train'])}\n   Val: {len(hubble_splits['val'])}\n   Test: {len(hubble_splits['test'])}")
    if obs_patches:
        print(f"\n📊 Observatory:"); obs_splits = create_dataset_split(obs_patches, OUTPUT_OBS_PATCHES, logger)
        if obs_splits: print(f"   Train: {len(obs_splits['train'])}\n   Val: {len(obs_splits['val'])}\n   Test: {len(obs_splits['test'])}")
    
    metadata = { 'timestamp': datetime.now().isoformat(), 'source_type': source_type, 'target_fov_arcmin': TARGET_FOV_ARCMIN, 'overlap_percent': OVERLAP_PERCENT,
                 'hubble': {'num_patches': len(hubble_patches)}, 'observatory': {'num_patches': len(obs_patches)}, 'pairs': {'num_pairs': len(pairs)} }
    
    metadata_file_md = OUTPUT_DIR_PATCHES / 'dataset_summary_report.md'
    save_summary_metadata_md(metadata, metadata_file_md)

    print(f"\n{'='*70}\n✅ ESTRAZIONE (Step 6) COMPLETATA\n{'='*70}\n📁 Output: {OUTPUT_DIR_PATCHES}")
    return True

# ============================================================================
# MAIN (MODIFICATA PER LOOP E MENU PROSECUZIONE)
# ============================================================================

def main():
    """Main unificato"""
    
    target_dirs = []
    if len(sys.argv) > 1:
        target_dirs = [Path(sys.argv[1])]
        print(f"🚀 Avviato da script precedente. Target: {target_dirs[0].name}")
    else:
        target_dirs = select_target_directory()
    
    if not target_dirs:
        print("Nessun target selezionato. Uscita."); return

    if len(target_dirs) > 1:
        print(f"Modalità Batch: {len(target_dirs)} target".center(70))
    print("="*70)

    start_time_total = time.time()
    mode = main_menu()

    # Liste per tenere traccia di cosa fare
    targets_to_analyze = []
    targets_to_patch = []
    
    if mode == 'analysis':
        targets_to_analyze = list(target_dirs)
    elif mode == 'patches':
        targets_to_patch = list(target_dirs)
    elif mode == 'both':
        targets_to_analyze = list(target_dirs)
        targets_to_patch = list(target_dirs) # Verranno filtrati se l'analisi fallisce
    
    successful_analysis_targets = []
    failed_analysis_targets = []

    # --- FASE 1: ANALISI (se richiesta) ---
    if targets_to_analyze:
        print("\n" + "="*70); print("INIZIO FASE 1: ANALISI OVERLAP (Step 5)"); print("="*70)
        for BASE_DIR in targets_to_analyze:
            print("\n" + "🚀"*35); print(f"TARGET ANALISI: {BASE_DIR.name}".center(70)); print("🚀"*35)
            start_time_target = time.time()
            HUBBLE_DIR_ANALYSIS = BASE_DIR / '4_cropped' / 'hubble'
            OBS_DIR_ANALYSIS = BASE_DIR / '4_cropped' / 'observatory'
            OUTPUT_DIR_ANALYSIS = BASE_DIR / '5_analisi_overlap'
            try:
                success = run_analysis(HUBBLE_DIR_ANALYSIS, OBS_DIR_ANALYSIS, OUTPUT_DIR_ANALYSIS)
                if success:
                    successful_analysis_targets.append(BASE_DIR)
                else:
                    failed_analysis_targets.append(BASE_DIR)
                    if mode == 'both' and BASE_DIR in targets_to_patch: # Se 'both', non fare patch
                        targets_to_patch.remove(BASE_DIR)
            except Exception as e:
                print(f"\n❌ ERRORE CRITICO (Analisi) su {BASE_DIR.name}: {e}")
                import traceback; traceback.print_exc()
                failed_analysis_targets.append(BASE_DIR)
                if mode == 'both' and BASE_DIR in targets_to_patch:
                    targets_to_patch.remove(BASE_DIR)
            elapsed_target = time.time() - start_time_target
            print(f"⏱️  Tempo analisi per {BASE_DIR.name}: {elapsed_target:.1f} secondi")

    # --- MENU INTERMEDIO (se 'analysis' e successo) ---
    if mode == 'analysis' and successful_analysis_targets:
        if ask_continue_to_patches(successful_analysis_targets):
            # Utente ha detto sì, aggiungi i target alla lista patch
            targets_to_patch = successful_analysis_targets
        else:
            targets_to_patch = [] # Utente ha detto no

    # --- FASE 2: PATCHES (se richiesta) ---
    successful_patch_targets = []
    failed_patch_targets = []
    
    if targets_to_patch:
        print("\n" + "="*70); print("INIZIO FASE 2: ESTRAZIONE PATCHES (Step 6)"); print("="*70)
        for BASE_DIR in targets_to_patch:
            print("\n" + "🚀"*35); print(f"TARGET PATCH: {BASE_DIR.name}".center(70)); print("🚀"*35)
            start_time_target = time.time()
            INPUT_CROPPED_HUBBLE = BASE_DIR / '4_cropped' / 'hubble'
            INPUT_CROPPED_OBSERVATORY = BASE_DIR / '4_cropped' / 'observatory'
            INPUT_REGISTERED_HUBBLE = BASE_DIR / '3_registered_native' / 'hubble'
            INPUT_REGISTERED_OBSERVATORY = BASE_DIR / '3_registered_native' / 'observatory'
            try:
                success = run_patches(BASE_DIR, INPUT_CROPPED_HUBBLE, INPUT_CROPPED_OBSERVATORY, INPUT_REGISTERED_HUBBLE, INPUT_REGISTERED_OBSERVATORY)
                if success:
                    successful_patch_targets.append(BASE_DIR)
                else:
                    failed_patch_targets.append(BASE_DIR)
            except Exception as e:
                print(f"\n❌ ERRORE CRITICO (Patches) su {BASE_DIR.name}: {e}")
                import traceback; traceback.print_exc()
                failed_patch_targets.append(BASE_DIR)
            elapsed_target = time.time() - start_time_target
            print(f"⏱️  Tempo patches per {BASE_DIR.name}: {elapsed_target:.1f} secondi")

    # --- RIEPILOGO FINALE ---
    elapsed_total = time.time() - start_time_total
    print(f"\n" + "="*70); print("COMPLETAMENTO PIPELINE (Step 5-6)"); print("="*70)
    print(f"   Target totali selezionati: {len(target_dirs)}")
    if targets_to_analyze:
        print(f"   Analisi (Step 5): {len(successful_analysis_targets)} success, {len(failed_analysis_targets)} fail")
    if targets_to_patch:
        print(f"   Patches (Step 6): {len(successful_patch_targets)} success, {len(failed_patch_targets)} fail")
    print(f"\n   ⏱️  Tempo totale batch: {elapsed_total:.1f} secondi ({elapsed_total/60:.1f} minuti)")
    print(f"\n{'='*70}\nGRAZIE PER AVER USATO LO SCRIPT UNIFICATO!\n{'='*70}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()