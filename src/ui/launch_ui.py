"""
Script per avviare l'interfaccia utente del progetto Super Resolution
"""
import sys
import os

# Aggiungi il percorso src al PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ui.gradio_interface import SuperResolutionUI

def main():
    print("🌟 Avvio interfaccia Super Resolution...")
    print("📡 L'interfaccia sarà disponibile su: http://localhost:7860")
    print("🌐 Link pubblico temporaneo verrà mostrato dopo l'avvio")
    print("⏹️  Premi Ctrl+C per fermare il server")
    
    ui = SuperResolutionUI()
    ui.launch(
        share=True,
        server_port=7860,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()