# io_utils.py
import os
import json


# ---------- Annotation Model ----------
class ImageAnnotation:
    def __init__(self, image_path, width, height):
        self.image_path = image_path
        self.width = width
        self.height = height
        self.boxes = []


# ---------- Load ----------
def load_annotation(image_path, width, height):
    return ImageAnnotation(image_path, width, height)


# ---------- Save Dispatcher ----------
def save_annotation(annotation, final=False):
    os.makedirs("labels", exist_ok=True)
    os.makedirs("annotations", exist_ok=True)

    save_yolo(annotation)
    save_coco(annotation)

    print("✅ Exported YOLO + COCO")

# ---------- YOLO ----------
def save_yolo(annotation):
    label_map = get_label_map(annotation)

    base = os.path.splitext(os.path.basename(annotation.image_path))[0]
    yolo_path = os.path.join("labels", base + ".txt")

    with open(yolo_path, "w", encoding="utf-8") as f:
        for box in annotation.boxes:
            bbox = box.get("bbox")

            # 🔒 GUARD
            if (
                not isinstance(bbox, (list, tuple))
                or len(bbox) != 4
                or not all(isinstance(v, (int, float)) for v in bbox)
            ):
                print("⚠️ Skip invalid bbox:", bbox)
                continue

            # ✅ ทุกอย่างผ่านแล้ว ค่อยสร้างตัวแปร
            class_id = label_map[box["label"]]

            x, y, w, h = bbox

            x_center = (x + w / 2) / annotation.width
            y_center = (y + h / 2) / annotation.height
            w_norm = w / annotation.width
            h_norm = h / annotation.height

            f.write(
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{w_norm:.6f} {h_norm:.6f}\n"
            )

# ---------- COCO ----------
def save_coco(annotation):
    label_map = get_label_map(annotation)

    base = os.path.splitext(os.path.basename(annotation.image_path))[0]
    coco_path = os.path.join("annotations", base + ".json")

    categories = [
        {"id": cid, "name": name}
        for name, cid in label_map.items()
    ]

    annotations = []
    ann_id = 1

    for box in annotation.boxes:
        bbox = box.get("bbox")

        # 🔒 GUARD
        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not all(isinstance(v, (int, float)) for v in bbox)
        ):
            print("⚠️ Skip invalid bbox (COCO):", bbox)
            continue

        annotations.append({
            "id": ann_id,
            "image_id": 1,
            "category_id": label_map[box["label"]],
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        })
        ann_id += 1

    coco = {
        "images": [{
            "id": 1,
            "file_name": os.path.basename(annotation.image_path),
            "width": annotation.width,
            "height": annotation.height
        }],
        "annotations": annotations,
        "categories": categories
    }

    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)


# ---------- Label Map ----------
def get_label_map(annotation):
    labels = sorted(set(box["label"] for box in annotation.boxes))
    return {label: idx for idx, label in enumerate(labels)}