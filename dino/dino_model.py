import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.ops import nms
from PIL import Image
import torch.nn.functional as F

class DinoFeatureExtractor:
    def __init__(self, device="cpu"):
        self.device = device

        self.model = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vits16"
        )
        self.model.eval().to(device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _extract_tensor(self, img_pil):
        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(tensor)

        feat = feat.squeeze()
        feat = F.normalize(feat, dim=0)  # ⭐ สำคัญมาก

        return feat

    def extract(self, image_path):
        img = Image.open(image_path).convert("RGB")
        return self._extract_tensor(img)

    def extract_from_array(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        return self._extract_tensor(img_pil)
    
    def extract_feature_map(self, img_pil):
        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.model.get_intermediate_layers(
                tensor, n=1
            )[0]   # shape: [1, num_patches+1, C]

        feats = feats[:, 1:, :]   # ตัด CLS token
        feats = feats.squeeze(0) # [N, C]

        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats
    
    def compute_heatmap_sliding(
        self,
        image_path,
        example_paths,
        window_size=512,
        stride=256
    ):
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        H, W, _ = img_np.shape

        prototype = self.extract_prototype(example_paths)

        global_heatmap = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)

        for y in range(0, H - window_size + 1, stride):
            for x in range(0, W - window_size + 1, stride):

                crop = img_np[y:y+window_size, x:x+window_size]
                crop_pil = Image.fromarray(crop)

                feat_map = self.extract_feature_map(crop_pil)
                sim = torch.matmul(feat_map, prototype)

                # ⭐ clamp negative similarity ออก
                sim = torch.clamp(sim, min=0)

                size = int(sim.shape[0] ** 0.5)
                heatmap = sim.reshape(size, size).cpu().numpy()

                heatmap = cv2.resize(
                    heatmap,
                    (window_size, window_size)
                )

                global_heatmap[y:y+window_size, x:x+window_size] += heatmap
                count_map[y:y+window_size, x:x+window_size] += 1

        global_heatmap /= (count_map + 1e-6)

        return global_heatmap
    
    def heatmap_to_boxes(
        self,
        heatmap,
        img_w,
        img_h,
        threshold=None,
        min_area_ratio=0.002
    ):

        mean = heatmap.mean()
        std = heatmap.std()

        heatmap = (heatmap - mean) / (std + 1e-6)
        heatmap = np.clip(heatmap, -3, 3)

        heatmap = (heatmap + 3) / 6

        heatmap = cv2.GaussianBlur(heatmap, (5,5), 0)

        if threshold is None:
            threshold = np.percentile(heatmap, 95)

        mask = (heatmap > threshold).astype(np.uint8)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels = cv2.connectedComponents(mask)

        boxes = []
        img_area = img_w * img_h

        for i in range(1, num_labels):
            ys, xs = np.where(labels == i)
            if len(xs) == 0:
                continue

            x1 = xs.min()
            y1 = ys.min()
            x2 = xs.max()
            y2 = ys.max()

            w = x2 - x1
            h = y2 - y1
            area = w * h

            if area > 0.4 * img_area:
                continue

            if area < min_area_ratio * img_area:
                continue

            score = heatmap[ys, xs].mean()

            boxes.append({
                "bbox": [int(x1), int(y1), int(w), int(h)],
                "score": float(score)
            })

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(
                [[b["bbox"][0],
                b["bbox"][1],
                b["bbox"][0] + b["bbox"][2],
                b["bbox"][1] + b["bbox"][3]] for b in boxes],
                dtype=torch.float32
            )

            scores_tensor = torch.tensor(
                [b["score"] for b in boxes],
                dtype=torch.float32
            )

            keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.3)

            boxes = [boxes[i] for i in keep]

        return boxes

    # def auto_label(self, image_path, example_path, label):
    #     img = load_image_unicode(image_path)
    #     h, w, _ = img.shape

    #     heatmap = self.compute_heatmap(image_path, example_path)
    #     boxes = heatmap_to_boxes(heatmap, w, h, threshold=0.6)

    #     return [
    #         {"label": label, "bbox": box}
    #         for box in boxes
    #     ]
    
    def extract_prototype(self, example_paths):
        feats = []

        for path in example_paths:
            img = Image.open(path).convert("RGB")
            feat = self._extract_tensor(img)
            feats.append(feat)

        feats = torch.stack(feats, dim=0)   # [N, C]
        prototype = feats.mean(dim=0)       # [C]
        prototype = torch.nn.functional.normalize(prototype, dim=0)

        return prototype