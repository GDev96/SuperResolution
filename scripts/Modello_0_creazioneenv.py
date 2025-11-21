import os
import sys
import subprocess
import platform
from pathlib import Path

def manage_env():
    print("="*60)
    print("🔧 MODELLO 0: SETUP AMBIENTE")
    print("="*60)

    # Rilevamento Sistema Operativo
    sistema = platform.system()
    print(f"🖥️  Sistema rilevato: {sistema}")

    if sistema == "Windows":
        # --- LOGICA WINDOWS (VECCHIA) ---
        HERE = Path(__file__).resolve().parent.parent
        venv_path = HERE / "venv"
        activate_script = venv_path / "Scripts" / "activate.bat"

        if not activate_script.exists():
            print(f"🔨 Creazione venv Windows in {venv_path}...")
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
        
        print(f"🚀 Apro nuova console attivata...")
        subprocess.run(f'cmd /k "{activate_script}"', shell=True)

    else:
        # --- LOGICA LINUX / RUNPOD (NUOVA) ---
        print("🐧 Configurazione per Linux/RunPod...")
        
        # Su RunPod NON serve il venv (siamo già in un container isolato).
        # Invece, ci assicuriamo che pip e i tool base siano aggiornati nel sistema globale.
        
        try:
            print("📦 Aggiornamento pip e strumenti base...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
            
            # Controllo se unzip e git sono installati (spesso mancano nei container base)
            # Nota: Questo richiede permessi di root (che su RunPod hai).
            print("🛠️  Verifica pacchetti di sistema (unzip, git, libgl)...")
            try:
                # Aggiorna apt e installa librerie grafiche necessarie per OpenCV/Scikit-Image
                subprocess.run("apt-get update && apt-get install -y unzip git libgl1-mesa-glx libglib2.0-0", shell=True, check=True)
            except Exception as e:
                print(f"⚠️  Non sono riuscito a usare apt-get (forse non sei root?): {e}")
                print("   Proseguo lo stesso, potrebbe non servire.")

            print("\n✅ AMBIENTE LINUX PRONTO!")
            print("👉 Non serve attivare nulla. Esegui direttamente:")
            print("   python scripts/Modello_1_setup_environment.py")

        except Exception as e:
            print(f"❌ Errore durante il setup: {e}")

if __name__ == "__main__":
    manage_env()