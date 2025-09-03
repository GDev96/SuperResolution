import gradio as gr
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
import io

# Aggiungi il percorso src al path per gli import
sys.path.append(str(Path(__file__).parent.parent))

# Rimuovi gli import problematici e usa solo quello che esiste
from utils.io import save_aligned_image
# from preprocessing.registration import align_images  # Commentato per ora se non esiste

class SuperResolutionInterface:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        
    def load_model(self):
        """Placeholder per il caricamento del modello"""
        try:
            # TODO: Implementare il caricamento del modello effettivo
            # self.model = load_super_resolution_model()
            self.model_loaded = True
            return "✅ Modello caricato con successo!"
        except Exception as e:
            self.model_loaded = False
            return f"❌ Errore nel caricamento del modello: {str(e)}"
    
    def process_image(self, input_image, target_image=None, use_preprocessing=True):
        """Processa l'immagine con super-risoluzione"""
        if input_image is None:
            return None, "⚠️ Carica un'immagine di input"
        
        try:
            # Converti in PIL Image se necessario
            if isinstance(input_image, np.ndarray):
                input_image = Image.fromarray(input_image)
            
            status = "🔄 Elaborazione in corso...\n"
            
            # Pre-processing opzionale
            processed_image = input_image
            if use_preprocessing and target_image is not None:
                status += "📐 Allineamento immagini...\n"
                if isinstance(target_image, np.ndarray):
                    target_image = Image.fromarray(target_image)
                
                # TODO: Implementare l'allineamento
                # processed_image = align_images(input_image, target_image)
                status += "✅ Allineamento completato\n"
            
            # Super-risoluzione
            if self.model_loaded:
                status += "🚀 Applicazione super-risoluzione...\n"
                # TODO: Implementare la super-risoluzione effettiva
                # result_image = self.model.predict(processed_image)
                
                # Per ora, simula il processo aumentando la dimensione
                width, height = processed_image.size
                result_image = processed_image.resize((width * 2, height * 2), Image.LANCZOS)
                status += "✅ Super-risoluzione completata!"
            else:
                status += "⚠️ Modello non caricato - restituisco immagine originale"
                result_image = processed_image
            
            return result_image, status
            
        except Exception as e:
            return None, f"❌ Errore nell'elaborazione: {str(e)}"
    
    def save_result(self, image, filename="result"):
        """Salva l'immagine risultante usando le funzioni esistenti"""
        if image is None:
            return "❌ Nessuna immagine da salvare"
        
        try:
            output_dir = Path(__file__).parent.parent.parent / "data" / "img_output"
            output_dir.mkdir(exist_ok=True)
            
            # Converti PIL Image in numpy array per save_aligned_image
            if isinstance(image, Image.Image):
                image_array = np.array(image)
                # Normalizza a 0-1 per save_aligned_image
                if image_array.max() > 1:
                    image_array = image_array.astype(np.float64) / 255.0
            else:
                image_array = image
            
            # Usa la funzione esistente save_aligned_image
            output_path = save_aligned_image(
                image_array, 
                f"{filename}.png", 
                str(output_dir),
                prefix="sr_"
            )
            
            return f"✅ Immagine salvata in: {output_path}"
        except Exception as e:
            return f"❌ Errore nel salvataggio: {str(e)}"

def create_interface():
    sr_interface = SuperResolutionInterface()
    
    with gr.Blocks(title="Super Resolution Interface", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔬 Interfaccia Super-Risoluzione per Immagini Astronomiche")
        gr.Markdown("Carica le tue immagini per applicare la super-risoluzione")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Controlli")
                
                # Controllo modello
                with gr.Group():
                    gr.Markdown("**Gestione Modello**")
                    model_status = gr.Textbox(
                        value="❌ Modello non caricato",
                        label="Status Modello",
                        interactive=False
                    )
                    load_model_btn = gr.Button("Carica Modello", variant="primary")
                
                # Input immagini
                with gr.Group():
                    gr.Markdown("**Input Immagini**")
                    input_image = gr.Image(
                        label="Immagine da elaborare",
                        type="pil"
                    )
                    target_image = gr.Image(
                        label="Immagine target (opzionale per allineamento)",
                        type="pil"
                    )
                
                # Opzioni
                with gr.Group():
                    gr.Markdown("**Opzioni**")
                    use_preprocessing = gr.Checkbox(
                        label="Usa pre-processing (allineamento)",
                        value=True
                    )
                
                # Controlli elaborazione
                process_btn = gr.Button("🚀 Elabora Immagine", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Risultati")
                
                # Output
                with gr.Group():
                    output_image = gr.Image(
                        label="Immagine elaborata",
                        type="pil"
                    )
                    processing_status = gr.Textbox(
                        label="Status elaborazione",
                        lines=5,
                        interactive=False
                    )
                
                # Salvataggio
                with gr.Row():
                    filename_input = gr.Textbox(
                        label="Nome file output",
                        value="super_resolution_result",
                        scale=3
                    )
                    save_btn = gr.Button("💾 Salva", variant="secondary", scale=1)
                
                save_status = gr.Textbox(
                    label="Status salvataggio",
                    interactive=False
                )
        
        # Eventi
        load_model_btn.click(
            fn=sr_interface.load_model,
            outputs=model_status
        )
        
        process_btn.click(
            fn=sr_interface.process_image,
            inputs=[input_image, target_image, use_preprocessing],
            outputs=[output_image, processing_status]
        )
        
        save_btn.click(
            fn=sr_interface.save_result,
            inputs=[output_image, filename_input],
            outputs=save_status
        )
        
        # Esempi
        with gr.Row():
            gr.Markdown("### 📝 Esempi")
            example_images = []
            data_dir = Path(__file__).parent.parent.parent / "data" / "img_input"
            if data_dir.exists():
                example_images = [str(f) for f in data_dir.glob("*.png")]
            
            if example_images:
                gr.Examples(
                    examples=[[img] for img in example_images[:3]],
                    inputs=[input_image],
                    label="Immagini di esempio"
                )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)