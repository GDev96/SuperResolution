import os
import sys
import subprocess
from pathlib import Path

def run_cmd(cmd):
    print(f"🚀 Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def setup_project():
    # Ottiene il percorso di QUESTO script (dentro 'scripts')
    HERE = Path(__file__).resolve().parent
    # La root del progetto è il genitore di 'scripts'
    PROJECT_ROOT = HERE.parent 
    
    # Modelli e Requirements sono nella ROOT
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Il requirements.txt potrebbe essere in scripts o in root.
    REQUIREMENTS_FILE = HERE / "requirements.txt" 

    print("="*60)
    print("🛠️  ASTRO SUPER-RES SETUP")
    print(f"📂  Project Root: {PROJECT_ROOT}")
    print("="*60)

    # --- NUOVA AGGIUNTA: Installazione TensorBoard e PyTorch CUDA ---
    print("\n📦 Configurazione ambiente GPU (CUDA)...")
    try:
        # 1. Installiamo TensorBoard (libreria standard)
        run_cmd(f"{sys.executable} -m pip install tensorboard")

        # 2. Disinstalliamo versioni vecchie per evitare conflitti CPU/GPU
        print("   🧹 Pulizia vecchie versioni torch...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )

        # 3. Installiamo PyTorch + Torchvision con supporto CUDA 11.8 (Stabile per RTX 2060)
        print("   📥 Scaricamento e installazione PyTorch CUDA (questo richiederà un po' di tempo)...")
        # Nota: --index-url punta ai wheel specifici per NVIDIA
        cuda_install_cmd = (
            f"{sys.executable} -m pip install "
            "torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu118"
        )
        run_cmd(cuda_install_cmd)

    except subprocess.CalledProcessError as e:
        print(f"⚠️  Attenzione: Errore nell'installazione delle librerie GPU: {e}")
        print("   Potrebbe essere necessario installarle manualmente.")
    # ----------------------------------------------------

    # Installazione Requirements (altre dipendenze come numpy, cv2, ecc.)
    if REQUIREMENTS_FILE.exists():
        print(f"\n📦 Installazione altre dipendenze da {REQUIREMENTS_FILE.name}...")
        run_cmd(f"{sys.executable} -m pip install -r {str(REQUIREMENTS_FILE)}")
    else:
        print(f"\nℹ️  File {REQUIREMENTS_FILE.name} non trovato (ok se hai già le dipendenze base).")
    
    print("\n✅ SETUP COMPLETE!")

def ask_continue_to_next_step():
    """Chiede se proseguire con il prossimo script."""
    next_script_name = 'Modello_2_prepare_data.py'
    HERE = Path(__file__).resolve().parent
    next_script_path = HERE / next_script_name

    print("\n" + "="*70)
    print("🎯 STEP 1 COMPLETATO!")
    print("="*70)
    print(f"\n📋 PROSSIMO STEP: Preparazione Dati ({next_script_name})")
    print("   Questo script scansionerà i dati e creerà i file JSON per il training.")
    
    while True:
        print("\n" + "─"*70)
        choice = input(f"👉 Vuoi avviare '{next_script_name}' ora? [S/n]: ").strip().lower()
        if choice in ('', 's', 'si', 'y', 'yes'):
            if next_script_path.exists():
                print(f"\n🚀 Avvio {next_script_name}...")
                try:
                    subprocess.run([sys.executable, str(next_script_path)])
                except Exception as e:
                    print(f"❌ Errore durante l'avvio dello script: {e}")
            else:
                print(f"❌ Errore: File non trovato: {next_script_path}")
            return
        elif choice in ('n', 'no'):
            print(f"\n👋 Ok. Puoi eseguire lo script manualmente in seguito: python {next_script_name}")
            return
        else:
            print("❌ Scelta non valida.")

if __name__ == "__main__":
    setup_project()
    ask_continue_to_next_step()