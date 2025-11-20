"""
MODELLO - STEP 3: VERIFICA DATASET (SMART VERSION)
Ignora i file extra e trova automaticamente la coppia LR/HR corretta.
"""

import sys
import json
import subprocess
from pathlib import Path
from astropy.io import fits

# CONFIGURAZIONE PATH
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.append(str(PROJECT_ROOT))
ROOT_DATA_DIR = PROJECT_ROOT / "data"

try:
    from src.dataset import AstronomicalDataset
    print("✅ Modulo 'src.dataset' caricato.")
except ImportError:
    try:
        from src.light.dataset import AstronomicalDataset
        print("⚠️ Usato 'src.light.dataset'.")
    except: sys.exit(1)

def select_target_directory():
    try:
        subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and (d/"6_patches_from_cropped"/"splits").exists()]
    except: return None
    if not subdirs: return None
    print("\nDataset pronti:")
    for i, d in enumerate(subdirs): print(f"   {i+1}: {d.name}")
    while True:
        try:
            c = input("👉 Scelta: ").strip()
            idx = int(c) - 1
            if 0 <= idx < len(subdirs): return subdirs[idx]
        except: pass

def check_dataset(target_dir):
    print(f"\n🔍 VERIFICA: {target_dir.name}")
    train_dir = target_dir / "6_patches_from_cropped" / "splits" / "train"
    
    pairs = list(train_dir.glob("pair_*"))
    if not pairs:
        print("❌ Nessuna cartella 'pair' trovata.")
        return False
    
    # COSTRUZIONE LISTA SMART
    train_list = []
    
    # Analizziamo 10 coppie
    print("\n🔬 Analisi Smart (ignoro file extra)...")
    for pair_dir in pairs[:10]:
        fits_files = list(pair_dir.glob("*.fits"))
        
        candidates = []
        for f in fits_files:
            try:
                with fits.open(f) as hdul:
                    if hdul[0].data is not None:
                        h, w = hdul[0].data.shape[-2:]
                        candidates.append({'path': f, 'area': h*w, 'dim': (h, w)})
            except: pass
            
        if len(candidates) < 2: continue
        
        # Ordina per area: [LR, ..., HR]
        candidates.sort(key=lambda x: x['area'])
        lr = candidates[0]
        hr = candidates[-1]
        
        print(f"   ✅ {pair_dir.name}: LR={lr['dim']} | HR={hr['dim']}")
        train_list.append({"ground_path": str(lr['path']), "hubble_path": str(hr['path'])})
        
    if not train_list:
        print("❌ Impossibile creare lista valida.")
        return False
        
    # TEST CARICAMENTO
    temp = train_dir.parent / "temp_check.json"
    with open(temp, 'w') as f: json.dump(train_list, f)
    
    try:
        ds = AstronomicalDataset(temp, base_path=PROJECT_ROOT, augment=False)
        print(f"\n✅ PyTorch Dataset caricato con {len(ds)} elementi.")
        s = ds[0]
        print(f"   📄 Sample Tensor: LR {s['lr'].shape} -> HR {s['hr'].shape}")
        temp.unlink()
        return True
    except Exception as e:
        print(f"❌ Errore Dataset: {e}")
        return False

if __name__ == "__main__":
    td = select_target_directory()
    if td and check_dataset(td):
        print("\n🚀 TUTTO OK! Avvio Training Light...")
        subprocess.run([sys.executable, str(HERE / "Modello_4_train_light.py")])