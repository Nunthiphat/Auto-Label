import cv2
import numpy as np
import torch
from .dino_model import DinoFeatureExtractor

def load_image_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

class DinoAutoLabeler:
    def __init__(self):
        self.extractor = DinoFeatureExtractor()
        self.overlay = None
        self.show_overlay = True

    def auto_label(self, image_path, example_paths, label):
        heatmap = self.extractor.compute_heatmap_sliding(
            image_path,
            example_paths,
            window_size=512,
            stride=256
        )

        img = cv2.imdecode(
            np.fromfile(image_path, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        h, w, _ = img.shape

        boxes = self.extractor.heatmap_to_boxes(
            heatmap,
            w,
            h,
            threshold=None   # ⭐ adaptive
        )

        results = []
        for box in boxes:
            results.append({
                "label": label,
                "bbox": box,
                "auto": True
            })

        return results, heatmap