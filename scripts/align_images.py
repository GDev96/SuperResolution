import os
import argparse
from src.preprocessing.registration import register_images
from src.utils.io import load_image_from_file, save_aligned_image

def align_all_images(ref_path, target_dir, output_dir):
    ref_img = load_image_from_file(ref_path)

    for fname in os.listdir(target_dir):
        if fname.startswith(".") or not fname.lower().endswith(('.fits', '.fit', '.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            continue

        target_path = os.path.join(target_dir, fname)
        print(f"\n🛰️ Allineamento: {fname}")
        target_img = load_image_from_file(target_path)

        try:
            aligned_img, method, stats = register_images(ref_img, target_img)
            out_path = save_aligned_image(aligned_img, fname, output_dir)
            print(f"✅ {fname} ➜ {method}")
            print(f"   📊 Corr: {stats['correlation']:.5f} | MSE: {stats['mse']:.6e} | MAE: {stats['mae']:.6e}")
            print(f"   💾 Salvato in: {out_path}")
        except Exception as e:
            print(f"❌ Errore durante l'allineamento di {fname}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Allineamento immagini astronomiche")
    parser.add_argument("--ref", required=True, help="Path dell'immagine di riferimento")
    parser.add_argument("--target_dir", required=True, help="Directory con immagini da allineare")
    parser.add_argument("--output_dir", required=True, help="Directory di output per immagini allineate")
    args = parser.parse_args()

    align_all_images(args.ref, args.target_dir, args.output_dir)
