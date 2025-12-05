import subprocess
import sys
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent  # Mantengo come oggetto Path per le operazioni sui file
DATA_DIR = PROJECT_ROOT / "data"

def select_target():
    print(f"\n🔍 Scansione target in: {DATA_DIR}")
    
    # Verifica esistenza cartella data
    if not DATA_DIR.exists():
        print(f"❌ Errore: Cartella 'data' non trovata in {PROJECT_ROOT}")
        print("   Assicurati di eseguire lo script dalla posizione corretta.")
        sys.exit(1)

    # Elenco cartelle valide (escludendo logs, splits, ecc.)
    excluded = ['splits', 'logs', 'models', '__pycache__']
    targets = [
        d.name for d in DATA_DIR.iterdir() 
        if d.is_dir() and d.name not in excluded and not d.name.startswith('.')
    ]
    targets.sort()

    if not targets:
        print("❌ Nessun target valido trovato nella cartella data.")
        sys.exit(1)

    print("\nTarget disponibili:")
    for i, t in enumerate(targets):
        print(f"  {i+1}: {t}")

    # Loop di selezione
    while True:
        choice = input(f"\nScegli il target (1-{len(targets)}) [Invio per '{targets[0]}']: ").strip()
        
        # Default (prima opzione) se si preme Invio
        if not choice:
            return targets[0]
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(targets):
                return targets[idx]
            else:
                print("❌ Selezione fuori range.")
        except ValueError:
            print("❌ Inserisci un numero valido.")

# --- SELEZIONE UTENTE ---
TARGET_NAME = select_target()

print("\n" + "="*50)
print(f"🚀 LANCIO TRAINING DUAL-GPU (0,1): {TARGET_NAME}")
print("="*50 + "\n")

# Configurazione Ambiente
env = os.environ.copy()
# Convertiamo PROJECT_ROOT in stringa per l'environment
env["PYTHONPATH"] = f"{str(PROJECT_ROOT)}:{env.get('PYTHONPATH', '')}"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print(f" ⚡ Abilitazione GPU 0 e 1...")
# --- Limitiamo la visibilità alle sole GPU 0 e 1 ---
env["CUDA_VISIBLE_DEVICES"] = "0,1" 

# Comando di lancio
cmd = [sys.executable, "Modello_supporto.py", "--target", TARGET_NAME]
p = subprocess.Popen(cmd, env=env)

try:
    p.wait()
except KeyboardInterrupt:
    print("\n🛑 Interruzione manuale rilevata. Terminazione processo...")
    p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
    print("Processo terminato.")