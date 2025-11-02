import sys
from pathlib import Path

# Aggiungi il percorso src al path
sys.path.append(str(Path(__file__).parent.parent))

from ui.gradio_interface import create_interface

def main():
    """Lancia l'interfaccia web"""
    print("🚀 Avvio interfaccia Super-Risoluzione...")
    print("📍 L'interfaccia sarà disponibile su: http://localhost:7860")
    
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        show_api=False,
        quiet=False
    )

if __name__ == "__main__":
    main()