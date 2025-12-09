import subprocess
import sys
import os
from pathlib import Path

# ================= CONFIGURAZIONE PATH =================
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def clear_screen():
    os.system('clear')

def get_available_targets():
    targets = []
    if ROOT_DATA_DIR.exists():
        for d in ROOT_DATA_DIR.iterdir():
            if d.is_dir() and d.name not in ['splits', 'logs', '.ipynb_checkpoints']:
                if (d / "7_dataset_ready_LOG").exists() or (d / "7_dataset_ready").exists():
                    targets.append(d.name)
    return sorted(targets)

def select_target():
    targets = get_available_targets()
    if not targets:
        print(f"{Colors.FAIL}❌ Nessun target pronto in {ROOT_DATA_DIR}{Colors.ENDC}")
        sys.exit(1)
    
    print(f"\n{Colors.HEADER}📂 SELEZIONE TARGET (Dataset Pronti):{Colors.ENDC}")
    for i, t in enumerate(targets):
        print(f"   {Colors.CYAN}[{i+1}]{Colors.ENDC} {t}")
    
    while True:
        try:
            choice = input(f"\n{Colors.BOLD}Scegli numero target: {Colors.ENDC}").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(targets):
                return targets[idx]
            print(f"{Colors.WARNING}Numero non valido.{Colors.ENDC}")
        except ValueError:
            print(f"{Colors.WARNING}Inserisci un numero valido.{Colors.ENDC}")

def select_gpus():
    """Menu selezione GPU per Linux (Multi-GPU)."""
    print(f"\n{Colors.HEADER}🖥️  CONFIGURAZIONE GPU (NVIDIA RTX 5000){Colors.ENDC}")
    print(f"{Colors.BLUE}   Esempi:{Colors.ENDC}")
    print(f"   '0,1' -> Usa entrambe le GPU (Massima Potenza)")
    print(f"   '0'   -> Usa solo la prima")
    print(f"   '1'   -> Usa solo la seconda")

    while True:
        selection = input(f"\n{Colors.BOLD}Quali GPU usare? [0,1] > {Colors.ENDC}").strip()
        
        if selection == "":
            selection = "0,1" # Default entrambe
            
        # Validazione semplice
        valid_chars = set("0123456789,")
        if not all(c in valid_chars for c in selection):
            print(f"{Colors.FAIL}❌ Input non valido.{Colors.ENDC}")
            continue
            
        return selection

def main():
    clear_screen()
    print(f"{Colors.HEADER}{Colors.BOLD}🚀 LINUX DUAL-GPU SANITY CHECK{Colors.ENDC}")
    print(f"{Colors.HEADER}=============================={Colors.ENDC}")

    target_name = select_target()
    gpu_str = select_gpus()
    
    num_gpus = len(gpu_str.split(','))
    
    print(f"\n{Colors.GREEN}✅ RIEPILOGO AVVIO:{Colors.ENDC}")
    print(f"   🎯 Target:     {Colors.BOLD}{target_name}{Colors.ENDC}")
    print(f"   🖥️  GPU IDs:    {Colors.BOLD}{gpu_str}{Colors.ENDC} (Totale: {num_gpus})")
    
    input(f"\nPremi {Colors.BOLD}[INVIO]{Colors.ENDC} per avviare il Worker...")

    # --- CONFIGURAZIONE AMBIENTE LINUX ---
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    
    # Configurazione CUDA
    env["CUDA_VISIBLE_DEVICES"] = gpu_str
    
    # Ottimizzazione PyTorch per RTX 5000
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    worker_script = HERE / "Modello_supporto.py"
    
    if not worker_script.exists():
        print(f"\n{Colors.FAIL}❌ File non trovato: {worker_script}{Colors.ENDC}")
        sys.exit(1)

    # Usa l'eseguibile Python corrente
    cmd = [sys.executable, str(worker_script), "--target", target_name]

    print(f"\n{Colors.CYAN}⚡ Avvio sottoprocesso Python...{Colors.ENDC}\n")
    print("-" * 50)
    
    try:
        p = subprocess.Popen(cmd, env=env)
        p.wait()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}🛑 Interrotto.{Colors.ENDC}")
        p.terminate()
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ Errore: {e}{Colors.ENDC}")

if __name__ == "__main__":
    main()