import os
import numpy as np
from PIL import Image
from astropy.io import fits
import matplotlib.pyplot as plt

def load_image_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.fits', '.fit']:
        data = fits.getdata(path)
        if data.ndim > 2:
            data = data[0]
        return data.astype(np.float64)
    else:
        img = Image.open(path).convert('L')
        return np.array(img).astype(np.float64)

def save_aligned_image(image, filename, output_dir, original_dtype=np.uint8):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    out_path = os.path.join(output_dir, f"{base_name}_aligned.png")

    image_uint8 = (image * 255).astype(original_dtype)
    plt.imsave(out_path, image_uint8, cmap='gray')
    return out_path
