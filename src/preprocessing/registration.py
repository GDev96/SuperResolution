import numpy as np
from scipy import ndimage
from skimage import transform, registration
import cv2
import astroalign as aa

def normalize_image(img):
    img_float = img.astype(np.float64)
    p1, p99 = np.percentile(img_float, [1, 99])
    img_clipped = np.clip(img_float, p1, p99)
    img_norm = (img_clipped - img_clipped.min()) / (img_clipped.max() - img_clipped.min())
    return img_norm

def find_peaks_simple(image, threshold_factor=3.0, min_distance=10):
    smooth = ndimage.gaussian_filter(image, sigma=1.5)
    threshold = np.mean(smooth) + threshold_factor * np.std(smooth)

    peaks = []
    h, w = image.shape
    for i in range(min_distance, h - min_distance):
        for j in range(min_distance, w - min_distance):
            region = smooth[i - min_distance:i + min_distance + 1,
                            j - min_distance:j + min_distance + 1]
            if smooth[i, j] > threshold and smooth[i, j] == np.max(region):
                peaks.append([i, j])
    return np.array(peaks)

def template_match_alignment(ref_img, target_img, template_size=200):
    h, w = ref_img.shape
    center_h, center_w = h // 2, w // 2
    half_template = template_size // 2
    template = ref_img[center_h - half_template:center_h + half_template,
                       center_w - half_template:center_w + half_template]

    result = cv2.matchTemplate(target_img.astype(np.float32),
                               template.astype(np.float32),
                               cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    shift_x = max_loc[0] - (center_w - half_template)
    shift_y = max_loc[1] - (center_h - half_template)
    return shift_x, shift_y, max_val

def register_images(ref_img, target_img):
    ref_img_norm = normalize_image(ref_img)
    target_img_norm = normalize_image(target_img)

    try:
        aligned_img, _ = aa.register(target_img_norm, ref_img_norm)
        method = "AstroAlign"
    except Exception:
        try:
            best_confidence = 0
            best_result = None
            for size in [150, 200, 250, 300]:
                try:
                    shift_x, shift_y, confidence = template_match_alignment(ref_img_norm, target_img_norm, size)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = (shift_x, shift_y)
                except:
                    continue
            if best_result and best_confidence > 0.5:
                tform = transform.SimilarityTransform(translation=best_result)
                aligned_img = transform.warp(target_img_norm, tform.inverse, output_shape=ref_img_norm.shape)
                method = f"Template Matching (conf: {best_confidence:.3f})"
            else:
                raise ValueError("Template Matching failed")
        except Exception:
            shift, error, _ = registration.phase_cross_correlation(ref_img_norm, target_img_norm, upsample_factor=100)
            tform = transform.SimilarityTransform(translation=(-shift[1], -shift[0]))
            aligned_img = transform.warp(target_img_norm, tform.inverse, output_shape=ref_img_norm.shape)
            method = f"Phase Correlation (error: {error:.4f})"

    stats = {
        "correlation": np.corrcoef(ref_img_norm.flatten(), aligned_img.flatten())[0, 1],
        "mse": np.mean((ref_img_norm - aligned_img)**2),
        "mae": np.mean(np.abs(ref_img_norm - aligned_img))
    }

    return aligned_img, method, stats
