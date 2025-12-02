"""
LAUNCHER (Il Capo)
Questo file DEVE chiamarsi: scripts/LAUNCHER_7_GPUS.py
Avvia 7 processi di training indipendenti su 7 GPU diverse.
"""
import subprocess
import sys
import time
import os
from pathlib import Path

# CONFIGURA QUI IL TARGET
TARGET_NAME = "M33" 

print("="*60)
print(f"🚀 LANCIO DI 7 TRAINING PARALLELI SU {TARGET_NAME}")
print("="*60)

processes = []
script_path = Path("scripts/Modello_4_train_independent.py")

if not script_path.exists():
    print(f"❌ Errore: Script non trovato in {script_path.resolve()}")
    sys.exit(1)

for i in range(7):
    print(f"   ⚡ Avvio Worker {i} su GPU {i}...")
    
    # Assegna una specifica GPU al processo
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(i)
    
    # Chiama lo script Worker
    cmd = [
        sys.executable, 
        str(script_path),
        "--target", TARGET_NAME,
        "--rank", str(i)
    ]
    
    # Avvia in background
    p = subprocess.Popen(cmd, env=env)
    processes.append(p)
    time.sleep(2) # Piccola pausa per evitare race conditions in creazione cartelle

print("\n✅ Tutti i 7 worker sono partiti!")
print("   Per fermarli tutti, premi CTRL+C qui.")

try:
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    print("\n🛑 STOP GENERALE. Arresto di tutti i processi...")
    for p in processes:
        p.terminate()