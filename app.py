# app.py
import sys
import os
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QToolBar,
    QScrollArea,
    QInputDialog
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt

from io_utils import load_annotation, save_annotation
from canvas import Canvas

from dino.auto_label import DinoAutoLabeler
from dino.heatmap import visualize_heatmap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Offline Auto Label Editor")
        self.resize(640, 480)

        self.canvas = None
        self.scroll_area = None

        self._create_toolbar()

    def _create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        open_action = QAction("Open Image", self)
        open_action.triggered.connect(self.open_image)
        toolbar.addAction(open_action)

        save_action = QAction("Save Final", self)
        save_action.triggered.connect(self.save_final)
        toolbar.addAction(save_action)

        auto_action = QAction("Auto Label (DINO)", self)
        auto_action.triggered.connect(self.auto_label)
        toolbar.addAction(auto_action)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "images",
            "Images (*.png *.jpg *.jpeg)"
        )
        if not path:
            return

        # ✅ รองรับ path ภาษาไทย (Windows)
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if img is None:
            print("❌ Cannot load image:", path)
            return

        h, w, _ = img.shape

        annotation = load_annotation(path, w, h)

        # ----- Canvas -----
        self.canvas = Canvas(annotation)

        # ----- Scroll Area (สำคัญมาก) -----
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(True)

        self.setCentralWidget(self.scroll_area)

    def save_final(self):
        if self.canvas is not None:
            save_annotation(self.canvas.annotation, final=True)

    def auto_label(self):
        if self.canvas is None:
            return

        examples, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Example Images",
            "examples",
            "Images (*.png *.jpg *.jpeg)"
        )
        if not examples:
            return

        label, ok = QInputDialog.getText(
            self,
            "Label",
            "Enter label name:",
            text="object"
        )
        if not ok:
            return

        dino = DinoAutoLabeler()
        boxes, heatmap = dino.auto_label(
            self.canvas.annotation.image_path,
            examples,
            label
        )

        for b in boxes:
            raw_bbox = b["bbox"]["bbox"]   # ⭐ ดึง list ออกมา
            score = b["bbox"]["score"]

            x, y, w, h = raw_bbox

            self.canvas.annotation.boxes.append({
                "label": b["label"],
                "bbox": [int(x), int(y), int(w), int(h)],
                "score": float(score),
                "auto": True
            })

        overlay = visualize_heatmap(
            self.canvas.annotation.image_path,
            heatmap
        )
        self.canvas.set_overlay_heatmap(heatmap)
        self.canvas.update()
        print("AUTO BOXES:", boxes)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())