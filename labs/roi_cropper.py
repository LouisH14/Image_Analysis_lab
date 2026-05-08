from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SavedCrop:
    name: str
    bbox: tuple[int, int, int, int]
    path: Path


def discover_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    images = [
        path
        for path in folder.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()
    ]
    return sorted(images)


def sanitize_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())
    return cleaned.strip("_")


def ensure_unique_path(
    output_dir: Path, base_name: str, extension: str = ".png"
) -> Path:
    candidate = output_dir / f"{base_name}{extension}"
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = output_dir / f"{base_name}_{index:03d}{extension}"
        if not candidate.exists():
            return candidate
        index += 1


class ImageAnnotator(QWidget):
    statusMessage = Signal(str)
    imageChanged = Signal()
    saveCropRequested = Signal()

    def __init__(self, output_dir: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.image_path: Optional[Path] = None
        self.pixmap: Optional[QPixmap] = None
        self.current_points: list[QPointF] = []
        self.saved_crops: list[SavedCrop] = []
        self.drag_index: Optional[int] = None
        self.active_point_radius = 7
        self.hit_radius = 12
        self.setMinimumSize(900, 700)
        self.setFocusPolicy(Qt.StrongFocus)

    def load_image(self, image_path: Path) -> None:
        self.image_path = image_path
        self.pixmap = QPixmap(str(image_path))
        self.current_points = []
        self.saved_crops = []
        self.drag_index = None

        if self.pixmap.isNull():
            self.statusMessage.emit(f"Impossible de charger l'image: {image_path.name}")
        else:
            self.statusMessage.emit(
                f"Image chargée: {image_path.name}. Clic droit pour ajouter 4 points, puis 'Découper'."
            )
        self.imageChanged.emit()
        self.update()

    def has_image(self) -> bool:
        return (
            self.pixmap is not None
            and not self.pixmap.isNull()
            and self.image_path is not None
        )

    def has_active_points(self) -> bool:
        return len(self.current_points) > 0

    def has_full_selection(self) -> bool:
        return len(self.current_points) == 4

    def image_rect(self) -> QRectF:
        if not self.has_image():
            return QRectF()

        pixmap = self.pixmap
        assert pixmap is not None
        widget_rect = QRectF(self.rect())
        scaled = pixmap.size().scaled(widget_rect.size().toSize(), Qt.KeepAspectRatio)
        x = (widget_rect.width() - scaled.width()) / 2
        y = (widget_rect.height() - scaled.height()) / 2
        return QRectF(x, y, scaled.width(), scaled.height())

    def widget_to_image(self, pos: QPointF) -> Optional[QPointF]:
        if not self.has_image():
            return None

        rect = self.image_rect()
        if rect.isNull() or not rect.contains(pos):
            return None

        assert self.pixmap is not None
        x_ratio = (pos.x() - rect.left()) / rect.width()
        y_ratio = (pos.y() - rect.top()) / rect.height()
        return QPointF(x_ratio * self.pixmap.width(), y_ratio * self.pixmap.height())

    def image_to_widget(self, pos: QPointF) -> QPointF:
        rect = self.image_rect()
        if rect.isNull() or not self.has_image():
            return QPointF()

        assert self.pixmap is not None
        x = rect.left() + (pos.x() / self.pixmap.width()) * rect.width()
        y = rect.top() + (pos.y() / self.pixmap.height()) * rect.height()
        return QPointF(x, y)

    def clamp_point(self, point: QPointF) -> QPointF:
        assert self.pixmap is not None
        x = min(max(point.x(), 0.0), float(self.pixmap.width() - 1))
        y = min(max(point.y(), 0.0), float(self.pixmap.height() - 1))
        return QPointF(x, y)

    def current_bbox(self) -> Optional[tuple[int, int, int, int]]:
        if len(self.current_points) != 4:
            return None

        xs = [point.x() for point in self.current_points]
        ys = [point.y() for point in self.current_points]
        left = max(0, int(min(xs)))
        top = max(0, int(min(ys)))
        right = int(max(xs)) + 1
        bottom = int(max(ys)) + 1

        assert self.pixmap is not None
        right = min(right, self.pixmap.width())
        bottom = min(bottom, self.pixmap.height())

        if right <= left:
            right = min(left + 1, self.pixmap.width())
        if bottom <= top:
            bottom = min(top + 1, self.pixmap.height())

        return left, top, right, bottom

    def add_point(self, point: QPointF) -> None:
        if len(self.current_points) >= 4:
            self.statusMessage.emit(
                "Les 4 points sont déjà posés. Déplace un point ou clique sur Découper."
            )
            return

        self.current_points.append(self.clamp_point(point))
        self.statusMessage.emit(f"Point {len(self.current_points)}/4 ajouté.")
        self.update()

    def move_point(self, index: int, point: QPointF) -> None:
        if index < 0 or index >= len(self.current_points):
            return

        self.current_points[index] = self.clamp_point(point)
        self.update()

    def point_hit_test(self, pos: QPointF) -> Optional[int]:
        if not self.current_points:
            return None

        for index, point in enumerate(self.current_points):
            widget_point = self.image_to_widget(point)
            if (widget_point - pos).manhattanLength() <= self.hit_radius:
                return index
        return None

    def undo_last_point(self) -> None:
        if not self.current_points:
            self.statusMessage.emit("Aucun point à annuler.")
            return

        self.current_points.pop()
        self.statusMessage.emit(
            f"Point retiré. Il reste {len(self.current_points)} point(s)."
        )
        self.update()

    def reset_current_selection(self) -> None:
        self.current_points = []
        self.drag_index = None
        self.statusMessage.emit("Sélection courante réinitialisée.")
        self.update()

    def save_current_crop(self, crop_name: str) -> Optional[SavedCrop]:
        if not self.has_image() or self.image_path is None:
            self.statusMessage.emit("Aucune image chargée.")
            return None

        if len(self.current_points) != 4:
            self.statusMessage.emit("Il faut exactement 4 points avant de découper.")
            return None

        bbox = self.current_bbox()
        if bbox is None:
            self.statusMessage.emit("Sélection invalide.")
            return None

        base_name = sanitize_name(crop_name)
        if not base_name:
            base_name = f"{self.image_path.stem}_roi_{len(self.saved_crops) + 1:03d}"
        else:
            base_name = (
                f"{self.image_path.stem}_{base_name}_{len(self.saved_crops) + 1:03d}"
            )

        output_path = ensure_unique_path(self.output_dir, base_name, ".png")

        with Image.open(self.image_path) as source:
            crop = source.crop(bbox)
            crop.save(output_path)

        saved = SavedCrop(name=base_name, bbox=bbox, path=output_path)
        self.saved_crops.append(saved)
        self.current_points = []
        self.drag_index = None
        self.statusMessage.emit(f"Crop enregistré: {output_path.name}")
        self.update()
        return saved

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(28, 30, 34))
        painter.setRenderHint(QPainter.Antialiasing)

        if not self.has_image():
            painter.setPen(QColor(220, 220, 220))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(
                self.rect(), Qt.AlignCenter, "Charge une image pour commencer"
            )
            return

        assert self.pixmap is not None
        rect = self.image_rect()
        painter.drawPixmap(rect.toRect(), self.pixmap)

        overlay_pen_saved = QPen(QColor(46, 204, 113), 3)
        overlay_pen_saved.setJoinStyle(Qt.RoundJoin)
        overlay_pen_active = QPen(QColor(243, 156, 18), 3)
        overlay_pen_active.setJoinStyle(Qt.RoundJoin)

        for crop in self.saved_crops:
            left, top, right, bottom = crop.bbox
            top_left = self.image_to_widget(QPointF(left, top))
            bottom_right = self.image_to_widget(QPointF(right, bottom))
            crop_rect = QRectF(top_left, bottom_right).normalized()
            painter.setPen(overlay_pen_saved)
            painter.setBrush(QColor(46, 204, 113, 45))
            painter.drawRect(crop_rect)

        if len(self.current_points) == 4:
            bbox = self.current_bbox()
            if bbox is not None:
                left, top, right, bottom = bbox
                top_left = self.image_to_widget(QPointF(left, top))
                bottom_right = self.image_to_widget(QPointF(right, bottom))
                active_rect = QRectF(top_left, bottom_right).normalized()
                painter.setPen(overlay_pen_active)
                painter.setBrush(QColor(243, 156, 18, 40))
                painter.drawRect(active_rect)

        painter.setPen(QPen(QColor(243, 156, 18), 2))
        painter.setBrush(QColor(243, 156, 18))
        for index, point in enumerate(self.current_points):
            widget_point = self.image_to_widget(point)
            painter.drawEllipse(
                widget_point, self.active_point_radius, self.active_point_radius
            )
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.drawText(widget_point + QPointF(10, -10), str(index + 1))

        if self.current_points:
            painter.setFont(QFont("Segoe UI", 10))
            painter.setPen(QColor(255, 214, 138))
            painter.drawText(
                16,
                26,
                "Clic droit pour poser les points. Glisse un point pour le corriger. Backspace annule le dernier.",
            )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if not self.has_image():
            return

        pos = event.position()
        if event.button() == Qt.LeftButton:
            hit = self.point_hit_test(pos)
            if hit is not None:
                self.drag_index = hit
                self.statusMessage.emit(f"Déplacement du point {hit + 1}.")
        elif event.button() == Qt.RightButton:
            image_pos = self.widget_to_image(pos)
            if image_pos is not None:
                self.add_point(image_pos)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.drag_index is None or not self.has_image():
            return

        image_pos = self.widget_to_image(event.position())
        if image_pos is not None:
            self.move_point(self.drag_index, image_pos)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.drag_index = None

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            self.undo_last_point()
            return
        if event.key() == Qt.Key_Escape:
            self.reset_current_selection()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            if event.key() == Qt.Key_Space:
                self.saveCropRequested.emit()
            return
        super().keyPressEvent(event)


class CropperWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ROI Cropper")

        self.workspace_root = Path(__file__).resolve().parents[1]
        self.source_dir = self.workspace_root / "data" / "reference_images"
        self.output_dir = self.workspace_root / "data" / "images_crop"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.images = discover_images(self.source_dir)
        self.current_index = 0

        self.annotator = ImageAnnotator(self.output_dir)
        self.annotator.statusMessage.connect(self.update_status)
        self.annotator.imageChanged.connect(self.refresh_controls)
        self.annotator.saveCropRequested.connect(self.crop_current_roi)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Nom du crop (optionnel)")

        self.crop_button = QPushButton("Découper")
        self.undo_button = QPushButton("Annuler point")
        self.reset_button = QPushButton("Réinitialiser")
        self.next_button = QPushButton("Image suivante")

        self.crop_button.clicked.connect(self.crop_current_roi)
        self.undo_button.clicked.connect(self.annotator.undo_last_point)
        self.reset_button.clicked.connect(self.annotator.reset_current_selection)
        self.next_button.clicked.connect(self.next_image)

        button_bar = QHBoxLayout()
        button_bar.addWidget(self.crop_button)
        button_bar.addWidget(self.undo_button)
        button_bar.addWidget(self.reset_button)
        button_bar.addWidget(self.next_button)
        button_bar.addWidget(self.name_edit, 1)

        self.hint_label = QPushButton(
            "Clic droit: poser un point | Glisser un point: corriger | Backspace: annuler | Esc: reset"
        )
        self.hint_label.setEnabled(False)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.annotator, 1)
        layout.addLayout(button_bar)
        layout.addWidget(self.hint_label)
        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar())
        self.update_status(
            f"{len(self.images)} image(s) détectée(s) dans {self.source_dir}"
        )

        if self.images:
            self.load_image(0)
        else:
            self.annotator.statusMessage.emit(
                f"Aucune image trouvée dans {self.source_dir}"
            )
            self.refresh_controls()

    def update_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def refresh_controls(self) -> None:
        has_image = self.annotator.has_image()
        has_full_selection = self.annotator.has_full_selection()
        self.crop_button.setEnabled(has_image and has_full_selection)
        self.undo_button.setEnabled(has_image and self.annotator.has_active_points())
        self.reset_button.setEnabled(has_image and self.annotator.has_active_points())
        self.next_button.setEnabled(has_image)

    def load_image(self, index: int) -> None:
        if not self.images:
            return

        self.current_index = index % len(self.images)
        image_path = self.images[self.current_index]
        self.annotator.load_image(image_path)
        self.name_edit.setText("")
        self.setWindowTitle(
            f"ROI Cropper - {image_path.name} ({self.current_index + 1}/{len(self.images)})"
        )
        self.refresh_controls()

    def crop_current_roi(self) -> None:
        saved = self.annotator.save_current_crop(self.name_edit.text())
        if saved is None:
            return

        self.name_edit.clear()
        self.refresh_controls()

    def next_image(self) -> None:
        if self.annotator.has_active_points():
            answer = QMessageBox.question(
                self,
                "Sélection non enregistrée",
                "Une sélection est en cours. Voulez-vous la supprimer et passer à l'image suivante ?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        if not self.images:
            return

        next_index = self.current_index + 1
        if next_index >= len(self.images):
            QMessageBox.information(
                self, "Fin", "Toutes les images ont été parcourues."
            )
            return

        self.load_image(next_index)


def main() -> int:
    app = QApplication(sys.argv)
    window = CropperWindow()
    window.resize(1400, 1000)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
