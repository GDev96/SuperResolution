import os
from src.preprocessing.registration import register_images
from src.utils.io import load_image_from_file, save_aligned_image

def align_images():
    input_dir = "data/img_input"
    target_dir = "data/img_hubble_Target"
    output_dir = "data/img_preprocessed"

    for input_fname in os.listdir(input_dir):
        if not input_fname.lower().endswith(('.fits', '.fit', '.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            continue

        # Extract the base name (excluding the first two characters)
        base_name = input_fname[2:]
        ref_fname = None

        # Find the matching target image
        for target_fname in os.listdir(target_dir):
            if target_fname[2:] == base_name and target_fname.lower().endswith(('.tif', '.tiff')):
                ref_fname = target_fname
                break

        if ref_fname:
            print(f"\n🔍 Trovata coppia: {input_fname} <-> {ref_fname}")

            input_path = os.path.join(input_dir, input_fname)
            ref_path = os.path.join(target_dir, ref_fname)

            try:
                target_img = load_image_from_file(input_path)
                ref_img = load_image_from_file(ref_path)

                aligned_img, method, stats = register_images(ref_img, target_img)
                out_path = save_aligned_image(aligned_img, input_fname, output_dir, prefix="O-")

                print(f"✅ {input_fname} ➜ {method}")
                print(f"   📊 Corr: {stats['correlation']:.5f} | MSE: {stats['mse']:.6e} | MAE: {stats['mae']:.6e}")
                print(f"   💾 Salvato in: {out_path}")

            except Exception as e:
                print(f"❌ Errore durante l'allineamento di {input_fname}: {e}")
        else:
            print(f"❌ Nessuna immagine di riferimento trovata per {input_fname}")

if __name__ == "__main__":
    align_images()