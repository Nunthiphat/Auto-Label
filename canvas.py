from PySide6.QtWidgets import QWidget, QInputDialog
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QImage
from PySide6.QtCore import Qt, QRect
import cv2
import numpy as np

class Canvas(QWidget):
    def __init__(self, annotation):
        super().__init__()
        self.annotation = annotation

        self.image = QPixmap(self.annotation.image_path)
        if self.image.isNull():
            raise RuntimeError("Failed to load image")

        self.setMinimumSize(
            self.image.width(),
            self.image.height()
        )
 
        # 🔥 overlay heatmap
        self.overlay_pixmap = None
        self.heatmap = None   # ⭐ เพิ่ม
        self.show_overlay = True

        # mouse state
        self.drawing = False
        self.start_point = None
        self.current_point = None

        # selection state
        self.selected_index = None
        self.setFocusPolicy(Qt.ClickFocus)

    # ---------------- PAINT ----------------
    def paintEvent(self, event):
        painter = QPainter(self)

        try:
            # base image
            painter.drawPixmap(0, 0, self.image)

            # overlay heatmap
            if self.overlay_pixmap is not None and self.show_overlay:
                painter.setOpacity(0.45)
                painter.drawPixmap(0, 0, self.overlay_pixmap)
                painter.setOpacity(1.0)

            # draw boxes
            for i, box in enumerate(self.annotation.boxes):
                bbox = box.get("bbox", None)

                # 🔥 กันข้อมูลพัง
                if not bbox or len(bbox) != 4:
                    continue

                x, y, w, h = bbox

                if box.get("auto", False):
                    pen = QPen(QColor(255, 140, 0), 2)   # ส้ม = auto
                else:
                    pen = QPen(QColor(255, 0, 0), 2)     # แดง = manual

                painter.setPen(pen)
                painter.drawRect(x, y, w, h)

                label = box.get("label", "")
                if label:
                    painter.drawText(x + 4, y + 14, label)

            # preview box
            if self.drawing and self.start_point and self.current_point:
                pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
                painter.setPen(pen)
                rect = QRect(self.start_point, self.current_point)
                painter.drawRect(rect.normalized())

        finally:
            painter.end()

    # ---------------- MOUSE ----------------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setFocus()
            pos = event.position().toPoint()

            # check select existing box (top-most first)
            for i in reversed(range(len(self.annotation.boxes))):
                box = self.annotation.boxes[i]
                bbox = box.get("bbox", None)

                # 🔥 กันข้อมูลพัง
                if not bbox or len(bbox) != 4:
                    continue

                x, y, w, h = bbox
                if QRect(x, y, w, h).contains(pos):
                    self.selected_index = i
                    self.update()
                    return

            # click empty → start drawing
            self.selected_index = None
            self.drawing = True
            self.start_point = pos
            self.current_point = pos
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.current_point = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            end_point = event.position().toPoint()

            rect = QRect(self.start_point, end_point).normalized()

            if rect.width() > 5 and rect.height() > 5:
                label, ok = QInputDialog.getText(
                    self,
                    "Label",
                    "Enter label name:",
                    text="object"
                )

                if ok:
                    label = label.strip() if label.strip() else "object"
                    self.annotation.boxes.append({
                        "label": label,
                        "bbox": [rect.x(), rect.y(), rect.width(), rect.height()]
                    })

            self.start_point = None
            self.current_point = None
            self.update()

    # ---------------- KEY ----------------
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.selected_index is not None:
                del self.annotation.boxes[self.selected_index]
                self.selected_index = None
                self.update()
        elif event.key() == Qt.Key_H:
            self.show_overlay = not self.show_overlay
            self.update()

    # ---------------- OVERLAY ----------------
    def set_overlay_image(self, img_bgr):
        """
        img_bgr: numpy array (H, W, 3) จาก OpenCV
        """
        h, w, ch = img_bgr.shape
        bytes_per_line = ch * w

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        qimg = QImage(
            img_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )

        self.overlay_pixmap = QPixmap.fromImage(qimg)
        self.update()

    def set_overlay_heatmap(self, heatmap):
        self.heatmap = heatmap   # ⭐ เก็บไว้ใช้ต่อ

        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-6)

        heatmap = cv2.resize(
            heatmap,
            (self.image.width(), self.image.height())
        )

        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        qimg = QImage(
            heatmap_color.data,
            heatmap_color.shape[1],
            heatmap_color.shape[0],
            heatmap_color.strides[0],
            QImage.Format_BGR888
        )

        self.overlay_pixmap = QPixmap.fromImage(qimg)
        self.update()

    def mouseDoubleClickEvent(self, event):
        if self.heatmap is None:
            return

        pos = event.position().toPoint()
        x, y = pos.x(), pos.y()

        # map pixel → heatmap coordinate
        h_map, w_map = self.heatmap.shape
        hx = int(x / self.width() * w_map)
        hy = int(y / self.height() * h_map)

        score = self.heatmap[hy, hx]

        if score < 0.5:
            return

        # TODO: region grow / flood fill
        print("Heatmap click:", hx, hy, "score:", score)