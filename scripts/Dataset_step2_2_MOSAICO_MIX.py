"""
STEP 2.5 (ADVANCED): CREAZIONE MOSAICO OSSERVATORIO SU GRIGLIA HUBBLE
------------------------------------------------------------------------
Questo script proietta ogni singola immagine dell'Osservatorio direttamente 
sulla griglia ad alta risoluzione del mosaico Hubble (WCS Master).

VANTAGGI:
- Minore perdita di qualità (una sola interpolazione).
- Allineamento perfetto nativo sulla geometria di Hubble.

INPUT: 
  - Master: [base_dir]/5_mosaics/final_mosaic_hubble.fits (WCS Target) 
            (In alternativa: final_mosaic.fits se il primo non esiste)
  - Files:  [base_dir]/3_registered_native/observatory/*.fits (Files Sorgente)

OUTPUT:
  - [base_dir]/5_mosaics/aligned_ready_for_crop/aligned_observatory.fits
  - [base_dir]/5_mosaics/aligned_ready_for_crop/aligned_hubble.fits (Copia HR Master)
------------------------------------------------------------------------
"""

import os
import sys
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from tqdm import tqdm
import warnings

# Sopprimi gli avvisi di Astropy e reproject
warnings.filterwarnings('ignore')

# ================= CONFIGURAZIONE PATH UNIVERSALI =================
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# ================= FUNZIONI AUSILIARIE =================

def select_target_directory():
    """Permette all'utente di selezionare la cartella target dei dati."""
    print("\n" + "📂"*35)
    print("SELEZIONE CARTELLA TARGET (Super Mosaico)".center(70))
    print("📂"*35)
    
    subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs', '__pycache__']]
    if not subdirs:
        print(f"❌ ERRORE: Nessuna cartella target trovata in {ROOT_DATA_DIR}")
        return None
        
    for i, d in enumerate(subdirs):
        print(f"   {i+1}: {d.name}")
        
    try:
        choice = int(input(f"\n👉 Seleziona (1-{len(subdirs)}): ").strip())
        if 0 < choice <= len(subdirs):
            return subdirs[choice-1]
    except ValueError:
        print("❌ Scelta non valida.")
        pass
        
    return None

def reprojection_core(data_in: np.ndarray, wcs_in: WCS, wcs_out: WCS, shape_out: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Esegue la riproiezione di un singolo file sulla griglia target."""
    
    # Assicura che i dati siano 2D
    if data_in.ndim > 2:
        data_in = data_in[0]
        
    # Sostituisci NaN con zero per la riproiezione e l'accumulo
    data_clean = np.nan_to_num(data_in)
    
    # Esegue la riproiezione bilineare
    reprojected, footprint = reproject_interp(
        (data_clean, wcs_in),
        wcs_out,
        shape_out=shape_out,
        order='bilinear'
    )
    
    # Gestisci eventuali NaN residui dopo la riproiezione
    reprojected = np.nan_to_num(reprojected)
    footprint = np.nan_to_num(footprint)
    
    return reprojected.astype(np.float32), footprint.astype(np.float32)

# ================= FUNZIONE PRINCIPALE =================

def create_super_mosaic(base_dir: Path):
    """Crea il mosaico dell'Osservatorio allineato sulla griglia Hubble."""
    
    # 1. Definizione Path
    mosaics_dir = base_dir / '5_mosaics'
    input_obs_dir = base_dir / '3_registered_native' / 'observatory'
    output_dir = mosaics_dir / 'aligned_ready_for_crop'
    
    # Cerca il file Master WCS con fallback
    master_hubble_1 = mosaics_dir / 'final_mosaic_hubble.fits'
    master_hubble_2 = mosaics_dir / 'final_mosaic.fits' # Fallback
    master_path = None
    
    if master_hubble_1.exists():
        master_path = master_hubble_1
    elif master_hubble_2.exists():
        print(f"⚠️ ATTENZIONE: Master '{master_hubble_1.name}' non trovato. Uso '{master_hubble_2.name}' come fallback.")
        master_path = master_hubble_2
    
    if not master_path:
        print(f"❌ ERRORE: Manca il mosaico Master WCS ({master_hubble_1.name} o {master_hubble_2.name}) in {mosaics_dir}")
        return

    output_file = output_dir / 'aligned_observatory.fits'
    
    # Controlli iniziali
    if not input_obs_dir.exists():
        print(f"❌ ERRORE: Cartella immagini observatory non trovata: {input_obs_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. CARICA MASTER WCS (HUBBLE)
    print(f"\n📡 Caricamento Master WCS da: {master_path.name}")
    try:
        with fits.open(master_path) as h:
            h_data = h[0].data
            h_header = h[0].header
            
            # Assicurati che i dati siano 2D per estrarre shape
            if h_data.ndim > 2: 
                h_data = h_data[0]
            if h_data.ndim != 2: 
                raise ValueError(f"I dati master non sono 2D dopo la pulizia (Shape: {h_data.shape}).")
                
            h_wcs = WCS(h_header)
            h_shape = h_data.shape
            
            if not h_wcs.has_celestial:
                print("❌ ERRORE: Il file Master non contiene WCS valido.")
                return

        print(f"   Target Grid: {h_shape[1]} x {h_shape[0]} pixel (Alta Risoluzione)")
        
        # Copia di sicurezza del master Hubble (richiesta da Step 3)
        hubble_copy = output_dir / 'aligned_hubble.fits'
        if not hubble_copy.exists():
            print("   Copia di sicurezza del master Hubble...")
            # Riscrive il file usando solo i dati 2D puliti per evitare problemi
            fits.PrimaryHDU(data=h_data.astype(np.float32), header=h_header).writeto(hubble_copy, overwrite=True)

    except Exception as e:
        print(f"❌ ERRORE nella lettura del Master WCS: {e}")
        return


    # 3. PREPARA ACCUMULATORI
    print("   Allocazione memoria per il Super Mosaico...")
    try:
        # Usiamo float64 (double) per massima precisione durante l'accumulo di somme e pesi
        mosaic_sum = np.zeros(h_shape, dtype=np.float64) 
        mosaic_weights = np.zeros(h_shape, dtype=np.float64)
    except MemoryError:
        print("❌ ERRORE: Memoria insufficiente per questa griglia! Prova a ridurre la dimensione del mosaico Master.")
        return

    # 4. LISTA FILES SORGENTE
    obs_files = sorted(list(input_obs_dir.glob("*.fits")) + list(input_obs_dir.glob("*.fit")))
    if not obs_files:
        print("❌ Nessun file FITS trovato nella cartella observatory.")
        return
    
    print(f"\n🚀 Inizio fusione di {len(obs_files)} immagini Observatory sulla griglia Hubble...")

    # 5. LOOP DI PROIEZIONE
    success_count = 0
    with tqdm(total=len(obs_files), desc="Projecting", unit="img") as pbar:
        for f in obs_files:
            try:
                with fits.open(f) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                    
                    if data is None or data.size == 0: continue
                    
                    wcs_in = WCS(header)
                    if not wcs_in.has_celestial: continue
                    
                    # Proiezione diretta
                    reprojected, footprint = reprojection_core(data, wcs_in, h_wcs, h_shape)

                    # Somma accumulativa (float64)
                    mosaic_sum += reprojected
                    mosaic_weights += footprint
                    
                    success_count += 1

            except Exception:
                # print(f"Err su {f.name}: {e}") # Decommenta per debug
                pass
            
            pbar.update(1)

    if success_count == 0:
        print("\n❌ NESSUN FILE È STATO PROIETTATO CON SUCCESSO. Verificare i file WCS.")
        return

    # 6. NORMALIZZAZIONE (MEDIA)
    print("\n🧮 Calcolo media finale...")
    
    # Calcola la media solo dove ci sono stati contributi (weights > 0)
    # Usa np.where per evitare errori di divisione per zero in modo controllato
    final_mosaic = np.where(mosaic_weights > 0, 
                            mosaic_sum / mosaic_weights, 
                            0.0).astype(np.float32)

    # 7. SALVATAGGIO
    print(f"💾 Salvataggio Super Mosaico in: {output_file.name}...")
    
    # Usa l'header di Hubble come base
    new_header = h_header.copy()
    new_header['HISTORY'] = f'Super-Mosaic: Created from {success_count} Obs files projected on Hubble Grid'
    new_header['ORIGIN'] = 'Observatory Data (Reprojected)'
    
    fits.PrimaryHDU(data=final_mosaic, header=new_header).writeto(output_file, overwrite=True)
    
    print("\n✅ COMPLETATO!")
    print(f"   File salvato: {output_file}")
    print("   Ora 'aligned_observatory.fits' può essere utilizzato nello Step 3 (Cutting).")

if __name__ == "__main__":
    
    # GESTIONE INPUT AUTOMATIZZATA (DA STEP PRECEDENTI)
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1]).resolve()
        if target_path.exists() and target_path.is_dir():
            print(f"\n🤖 Modalità Automatica: Target ricevuto {target_path.name}")
            create_super_mosaic(target_path)
            
        else:
            print(f"❌ Errore: Path fornito non valido: {target_path}")
            
    else:
        # Modalità manuale
        target = select_target_directory()
        if target:
            create_super_mosaic(target)