import gradio as gr
import numpy as np
from PIL import Image
import os
import sys

# Aggiungi il percorso src al PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.io import load_image, save_image
from preprocessing.registration import align_images

class SuperResolutionUI:
    def __init__(self):
        self.title = "🌟 Super Resolution - Astronomical Image Enhancement"
        self.description = """
        Carica un'immagine astronomica per migliorarne la risoluzione utilizzando tecniche di deep learning.
        Supporta formati: FITS, TIFF, PNG, JPEG
        """
        
    def process_single_image(self, image, scale_factor, model_type):
        """Processa una singola immagine per super resolution"""
        try:
            if image is None:
                return None, "❌ Nessuna immagine caricata"
            
            # Placeholder per il processing - da implementare con i modelli reali
            # Qui andrà la logica del modello di super resolution
            enhanced_image = self._placeholder_enhancement(image, scale_factor)
            
            status = f"✅ Immagine processata con successo! Fattore di scala: {scale_factor}x, Modello: {model_type}"
            return enhanced_image, status
            
        except Exception as e:
            return None, f"❌ Errore durante il processing: {str(e)}"
    
    def _placeholder_enhancement(self, image, scale_factor):
        """Placeholder per il miglioramento dell'immagine - da sostituire con il modello reale"""
        # Converte in numpy array se necessario
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        # Semplice resize come placeholder
        height, width = img_array.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        
        # Usa PIL per il resize
        pil_image = Image.fromarray(img_array)
        enhanced = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        return enhanced
    
    def process_batch(self, files, scale_factor, model_type):
        """Processa multiple immagini"""
        if not files:
            return "❌ Nessun file selezionato"
        
        results = []
        for i, file in enumerate(files):
            try:
                image = Image.open(file.name)
                enhanced, _ = self.process_single_image(image, scale_factor, model_type)
                
                # Salva l'immagine processata
                output_path = f"data/img_output/enhanced_{i+1}.png"
                enhanced.save(output_path)
                results.append(f"✅ File {i+1}: salvato in {output_path}")
                
            except Exception as e:
                results.append(f"❌ File {i+1}: errore - {str(e)}")
        
        return "\n".join(results)
    
    def create_interface(self):
        """Crea l'interfaccia Gradio"""
        
        with gr.Blocks(title=self.title, theme=gr.themes.Soft()) as interface:
            gr.Markdown(f"# {self.title}")
            gr.Markdown(self.description)
            
            with gr.Tabs():
                # Tab per singola immagine
                with gr.Tab("🖼️ Singola Immagine"):
                    with gr.Row():
                        with gr.Column():
                            input_image = gr.Image(
                                label="Carica Immagine",
                                type="pil",
                                height=400
                            )
                            
                            with gr.Row():
                                scale_factor = gr.Slider(
                                    minimum=2,
                                    maximum=8,
                                    value=4,
                                    step=1,
                                    label="Fattore di Scala"
                                )
                                
                                model_type = gr.Dropdown(
                                    choices=["SRCNN", "ESRGAN", "Real-ESRGAN"],
                                    value="SRCNN",
                                    label="Modello"
                                )
                            
                            process_btn = gr.Button("🚀 Migliora Immagine", variant="primary")
                        
                        with gr.Column():
                            output_image = gr.Image(
                                label="Immagine Migliorata",
                                height=400
                            )
                            status_text = gr.Textbox(
                                label="Status",
                                interactive=False
                            )
                    
                    process_btn.click(
                        fn=self.process_single_image,
                        inputs=[input_image, scale_factor, model_type],
                        outputs=[output_image, status_text]
                    )
                
                # Tab per batch processing
                with gr.Tab("📁 Batch Processing"):
                    with gr.Column():
                        batch_files = gr.File(
                            label="Carica Multiple Immagini",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        
                        with gr.Row():
                            batch_scale = gr.Slider(
                                minimum=2,
                                maximum=8,
                                value=4,
                                step=1,
                                label="Fattore di Scala"
                            )
                            
                            batch_model = gr.Dropdown(
                                choices=["SRCNN", "ESRGAN", "Real-ESRGAN"],
                                value="SRCNN",
                                label="Modello"
                            )
                        
                        batch_btn = gr.Button("🚀 Processa Batch", variant="primary")
                        batch_status = gr.Textbox(
                            label="Risultati Batch",
                            lines=10,
                            interactive=False
                        )
                    
                    batch_btn.click(
                        fn=self.process_batch,
                        inputs=[batch_files, batch_scale, batch_model],
                        outputs=batch_status
                    )
                
                # Tab per informazioni
                with gr.Tab("ℹ️ Info"):
                    gr.Markdown("""
                    ## 📊 Metriche di Qualità
                    - **PSNR**: Peak Signal-to-Noise Ratio (target > 30 dB)
                    - **SSIM**: Structural Similarity Index (target > 0.85)
                    
                    ## 🔧 Modelli Disponibili
                    - **SRCNN**: Super-Resolution Convolutional Neural Network (veloce)
                    - **ESRGAN**: Enhanced Super-Resolution GAN (qualità alta)
                    - **Real-ESRGAN**: Real-World Super-Resolution (immagini reali)
                    
                    ## 📁 Formati Supportati
                    - FITS (immagini astronomiche)
                    - TIFF, PNG, JPEG (immagini standard)
                    
                    ## 🎯 Uso Consigliato
                    1. Carica un'immagine astronomica
                    2. Seleziona il fattore di scala (2x-8x)
                    3. Scegli il modello più adatto
                    4. Clicca "Migliora Immagine"
                    """)
        
        return interface
    
    def launch(self, **kwargs):
        """Lancia l'interfaccia"""
        interface = self.create_interface()
        interface.launch(**kwargs)

def main():
    """Funzione principale per avviare l'interfaccia"""
    ui = SuperResolutionUI()
    ui.launch(
        share=True,  # Crea un link pubblico temporaneo
        server_name="0.0.0.0",  # Accessibile da altre macchine nella rete
        server_port=7860,  # Porta personalizzata
        show_error=True  # Mostra errori dettagliati
    )

if __name__ == "__main__":
    main()