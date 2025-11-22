import os
import sys
import subprocess
from pathlib import Path

def run_cmd(cmd):
    print(f"🚀 Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def setup_project():
    # PATH ASSOLUTO PER RUNPOD
    PROJECT_ROOT = Path("/root/SuperResolution")
    if not PROJECT_ROOT.exists(): PROJECT_ROOT = Path(__file__).resolve().parent.parent

    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    
    REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt" 
    
    print("="*60)
    print("🛠️  RUNPOD ENVIRONMENT SETUP")
    print(f"📂  Project Root: {PROJECT_ROOT}")
    print("="*60)

    # 1. Installazione Librerie Mancanti (NO PyTorch, già presente)
    print("\n📦 Installazione dipendenze aggiuntive...")
    packages = [
        "tensorboard", 
        "astropy", 
        "scikit-image", 
        "scipy", 
        "tqdm", 
        "reproject", 
        "astroalign",
        "matplotlib"
    ]
    
    cmd = f"{sys.executable} -m pip install " + " ".join(packages)
    try:
        run_cmd(cmd)
    except Exception as e:
        print(f"⚠️ Errore pip: {e}")

    # 2. Controllo requirements.txt opzionale
    if REQUIREMENTS_FILE.exists():
        print(f"\n📦 Trovato requirements.txt, installo resto...")
        run_cmd(f"{sys.executable} -m pip install -r {str(REQUIREMENTS_FILE)}")

    print("\n✅ SETUP COMPLETE! (Pronto per l'uso su GPU Cloud)")

if __name__ == "__main__":
    setup_project()