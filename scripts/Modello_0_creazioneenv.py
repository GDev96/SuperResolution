import os
import sys
import subprocess
from pathlib import Path
import platform

def manage_venv():
    # La cartella corrente è "Modello"
    HERE = Path(__file__).resolve().parent
    
    # La venv va creata qui dentro
    venv_path = HERE / "venv"
    
    # Percorso dello script di attivazione (specifico per Windows)
    activate_script = venv_path / "Scripts" / "activate.bat"

    # Verifica OS
    if platform.system() != "Windows":
        print("❌ Questo script è ottimizzato solo per Windows.")
        return

    print(f"📂 Cartella di lavoro: {HERE}")
    print(f"🐍 Target Venv: {venv_path}")

    # 2. CONTROLLO E CREAZIONE AUTOMATICA
    if not activate_script.exists():
        print(f"⚠️  Cartella 'venv' non trovata.")
        print(f"🔨  Creazione automatica dell'ambiente virtuale in corso...")
        
        try:
            # Esegue il comando: python -m venv venv
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
            print("✅ Cartella venv creata con successo!")
            print("⚠️  NOTA: Ora esegui '1_setup_environment.py' per installare le librerie!")
        except subprocess.CalledProcessError:
            print("❌ Errore critico nella creazione della venv.")
            input("Premi Invio per uscire...")
            return

    # 3. ATTIVAZIONE
    print(f"🚀 Avvio console con venv attivata...")
    
    # Comando per aprire CMD, attivare venv e rimanere aperto (/k)
    command = f'cmd /k "{activate_script}"'
    
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    manage_venv()