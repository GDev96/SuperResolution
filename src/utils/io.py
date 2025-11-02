import os
import numpy as np
from PIL import Image
from astropy.io import fits
import matplotlib.pyplot as plt

def load_image(filename, file_content):
    """Carica immagini FITS o PNG/JPG dai file uploadati"""
    try:
        if filename.lower().endswith(('.fits', '.fit')):
            # Per file FITS
            with io.BytesIO(file_content) as f:
                try:
                    with fits.open(f) as hdul:
                        data = hdul[0].data
                        if data.ndim > 2:
                            data = data[0]  # Usa il primo canale se 3D
                        return data.astype(np.float64)
                except Exception as e:
                    print(f"Errore durante l'apertura del file FITS: {e}")
                    return None
        else:
            # Per immagini comuni (PNG, JPG, etc.)
            try:
                if not file_content:
                    print("Errore: il contenuto del file è vuoto.")
                    return None

                img = Image.open(io.BytesIO(file_content)).convert('L')
                return np.array(img).astype(np.float64)
            except UnidentifiedImageError as e:
                print(f"Errore: Impossibile identificare il formato dell'immagine: {e}")
                return None
            except Exception as e:
                print(f"Errore durante l'apertura dell'immagine con PIL: {e}")
                return None
    except Exception as e:
        print(f"Errore generale durante il caricamento dell'immagine: {e}")
        return None


def save_aligned_image(image, filename, output_dir, original_dtype=np.uint8, prefix=""):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    out_path = os.path.join(output_dir, f"{prefix}{base_name}_aligned.png")

    image_uint8 = (image * 255).astype(original_dtype)
    plt.imsave(out_path, image_uint8, cmap='gray')
    return out_path