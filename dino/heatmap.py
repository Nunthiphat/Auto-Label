import cv2
import numpy as np

def resize_heatmap(heatmap, img_w, img_h):
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-6)
    return cv2.resize(
        heatmap,
        (img_w, img_h),
        interpolation=cv2.INTER_CUBIC
    )

def heatmap_to_colormap(heatmap):
    heatmap_uint8 = np.uint8(255 * heatmap)
    return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

def overlay_heatmap(image_bgr, heatmap_color, alpha=0.45):
    return cv2.addWeighted(
        image_bgr, 1 - alpha,
        heatmap_color, alpha,
        0
    )

def visualize_heatmap(image_path, heatmap):
    img = cv2.imdecode(
        np.fromfile(image_path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    h, w, _ = img.shape
    heatmap_resized = resize_heatmap(heatmap, w, h)
    heatmap_color = heatmap_to_colormap(heatmap_resized)
    return overlay_heatmap(img, heatmap_color)