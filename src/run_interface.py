#!/usr/bin/env python3
"""
Script per avviare l'interfaccia web di Super-Risoluzione
"""

import sys
from pathlib import Path

# Aggiungi src al path
sys.path.append(str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    try:
        from ui.launch_ui import main
        main()
    except ImportError as e:
        print(f"❌ Errore nell'importazione: {e}")
        print("💡 Verifica che tutti i moduli siano presenti nella cartella src/")
        print("📁 Struttura attesa:")
        print("   src/")
        print("   ├── ui/")
        print("   │   ├── gradio_interface.py")
        print("   │   └── launch_ui.py")
        print("   └── utils/")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Errore nell'avvio: {e}")
        sys.exit(1)