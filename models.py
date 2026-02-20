from dataclasses import dataclass
from typing import List


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str


@dataclass
class ImageAnnotation:
    image_path: str
    width: int
    height: int
    boxes: List[BoundingBox]