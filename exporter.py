from models import ImageAnnotation

def export_yolo(annotation: ImageAnnotation, class_map: dict, out_path):
    with open(out_path, "w") as f:
        for b in annotation.boxes:
            cid = class_map[b.label]
            cx = ((b.x1 + b.x2) / 2) / annotation.width
            cy = ((b.y1 + b.y2) / 2) / annotation.height
            bw = (b.x2 - b.x1) / annotation.width
            bh = (b.y2 - b.y1) / annotation.height
            f.write(f"{cid} {cx} {cy} {bw} {bh}\n")