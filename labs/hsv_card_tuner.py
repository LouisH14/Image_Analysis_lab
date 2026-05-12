"""Standalone HSV tuning tool for cropped UNO card images.

The script lets you browse the images in ``data/images_crop`` with Previous / Next
buttons, tune six HSV threshold sliders, and inspect the original image, the
binary thresholded result, and the HSV histograms with threshold markers.

The chosen defaults are stored in a small JSON file next to this script so the
next launch reuses the last validated tuning values.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent
from matplotlib.widgets import Button, Slider

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "images_crop"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "hsv_card_tuner_defaults.json"


@dataclass
class HsvDefaults:
    lower_h: int = 0
    lower_s: int = 0
    lower_v: int = 0
    upper_h: int = 179
    upper_s: int = 255
    upper_v: int = 255
    kernel_size: int = 1
    morph_op: str = "none"  # 'none', 'close', 'open'

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "HsvDefaults":
        values = cls()
        for field_name in values.__dataclass_fields__:
            if field_name in data:
                val = data[field_name]
                if field_name == "morph_op":
                    setattr(values, field_name, str(val))
                else:
                    setattr(values, field_name, int(val))
        values._validate()
        return values

    def _validate(self) -> None:
        self.lower_h = int(np.clip(self.lower_h, 0, 179))
        self.upper_h = int(np.clip(self.upper_h, 0, 179))
        self.lower_s = int(np.clip(self.lower_s, 0, 255))
        self.upper_s = int(np.clip(self.upper_s, 0, 255))
        self.lower_v = int(np.clip(self.lower_v, 0, 255))
        self.upper_v = int(np.clip(self.upper_v, 0, 255))
        self.kernel_size = int(np.clip(self.kernel_size, 1, 51))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.morph_op = str(self.morph_op).lower() if self.morph_op else "none"
        if self.morph_op not in ("none", "close", "open"):
            self.morph_op = "none"
        if self.lower_h > self.upper_h:
            self.lower_h, self.upper_h = self.upper_h, self.lower_h
        if self.lower_s > self.upper_s:
            self.lower_s, self.upper_s = self.upper_s, self.lower_s
        if self.lower_v > self.upper_v:
            self.lower_v, self.upper_v = self.upper_v, self.lower_v

    def as_lower(self) -> tuple[int, int, int]:
        return self.lower_h, self.lower_s, self.lower_v

    def as_upper(self) -> tuple[int, int, int]:
        return self.upper_h, self.upper_s, self.upper_v


def list_images(images_dir: Path) -> list[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_paths: list[Path] = []
    for pattern in patterns:
        image_paths.extend(images_dir.glob(pattern))
    return sorted(image_paths)


def load_image(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image_bgr


def load_defaults(config_path: Path) -> HsvDefaults:
    if not config_path.exists():
        return HsvDefaults()
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return HsvDefaults()
    if not isinstance(data, dict):
        return HsvDefaults()
    return HsvDefaults.from_dict(data)


def save_defaults(config_path: Path, defaults: HsvDefaults) -> None:
    config_path.write_text(json.dumps(asdict(defaults), indent=2), encoding="utf-8")


def build_mask(
    image_bgr: np.ndarray, defaults: HsvDefaults
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(defaults.as_lower(), dtype=np.uint8)
    upper = np.array(defaults.as_upper(), dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    if defaults.kernel_size > 1 and defaults.morph_op != "none":
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (defaults.kernel_size, defaults.kernel_size)
        )
        if defaults.morph_op == "close":
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif defaults.morph_op == "open":
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    return hsv, mask, masked


def visualize_morph_effect(
    mask: np.ndarray, kernel_size: int, morph_op: str
) -> np.ndarray:
    """Visualize effect of morphology on a real mask patch: left=before, right=after."""
    if kernel_size <= 1 or morph_op == "none" or mask.size == 0:
        return np.zeros((80, 80), dtype=np.uint8)

    h, w = mask.shape
    if h < 40 or w < 40:
        patch = mask.copy()
    else:
        patch_size = 40
        y_start = h // 2 - patch_size // 2
        x_start = w // 2 - patch_size // 2
        patch = mask[
            y_start : y_start + patch_size, x_start : x_start + patch_size
        ].copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if morph_op == "close":
        result = cv2.morphologyEx(patch, cv2.MORPH_CLOSE, kernel)
    elif morph_op == "open":
        result = cv2.morphologyEx(patch, cv2.MORPH_OPEN, kernel)
    else:
        result = patch

    viz = np.hstack([patch, result])
    if viz.shape[0] < 80:
        pad = 80 - viz.shape[0]
        viz = np.vstack([viz, np.zeros((pad, viz.shape[1]), dtype=np.uint8)])
    if viz.shape[1] < 80:
        pad = 80 - viz.shape[1]
        viz = np.hstack([viz, np.zeros((viz.shape[0], pad), dtype=np.uint8)])

    return viz[:80, :80]


def plot_histograms(
    axes: list[Axes],
    hsv: np.ndarray,
    defaults: HsvDefaults,
    image_label: str,
) -> None:
    channel_specs = [
        (0, "H", 180, "#ff7850"),
        (1, "S", 256, "#50b4ff"),
        (2, "V", 256, "#78dc78"),
    ]
    thresholds = [
        (defaults.lower_h, defaults.upper_h),
        (defaults.lower_s, defaults.upper_s),
        (defaults.lower_v, defaults.upper_v),
    ]

    for axis, (channel_index, label, bins, color), (low, high) in zip(
        axes, channel_specs, thresholds
    ):
        axis.clear()
        values = hsv[:, :, channel_index].ravel()
        axis.hist(values, bins=bins, range=(0, bins - 1), color=color, alpha=0.85)
        axis.axvline(low, color="crimson", linestyle="--", linewidth=2, label="lower")
        axis.axvline(
            high, color="deepskyblue", linestyle="--", linewidth=2, label="upper"
        )
        axis.set_xlim(0, bins - 1)
        axis.set_title(f"{label} histogram")
        axis.legend(loc="upper right", fontsize=8)
        axis.grid(alpha=0.2)

    axes[0].text(
        0.01,
        1.12,
        image_label,
        transform=axes[0].transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


class HsvCardTuner:
    def __init__(
        self, image_paths: list[Path], defaults: HsvDefaults, config_path: Path
    ):
        if not image_paths:
            raise FileNotFoundError("No images found in data/images_crop")
        self.image_paths = image_paths
        self.config_path = config_path
        self.defaults = defaults
        self.morph_op_state = defaults.morph_op
        self.index = 0
        self.current_image = load_image(self.image_paths[self.index])

        self.fig = plt.figure(figsize=(18, 10))
        grid = self.fig.add_gridspec(3, 7, height_ratios=[2.2, 1.4, 1.0])
        self.ax_original = self.fig.add_subplot(grid[0, :2])
        self.ax_mask = self.fig.add_subplot(grid[0, 2:4])
        self.ax_kernel = self.fig.add_subplot(grid[0, 4:6])
        self.ax_result = self.fig.add_subplot(grid[0, 6])
        self.ax_hist_h = self.fig.add_subplot(grid[1, :2])
        self.ax_hist_s = self.fig.add_subplot(grid[1, 2:4])
        self.ax_hist_v = self.fig.add_subplot(grid[1, 4:])

        self.fig.suptitle(
            "HSV + Morphology tuner for data/images_crop",
            fontsize=16,
            fontweight="bold",
        )
        self.fig.subplots_adjust(
            left=0.06, right=0.98, top=0.90, bottom=0.30, hspace=0.60, wspace=0.35
        )

        self.original_artist = self.ax_original.imshow(
            cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        )
        self.mask_artist = self.ax_mask.imshow(
            np.zeros(self.current_image.shape[:2], dtype=np.uint8),
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        self.kernel_artist = self.ax_kernel.imshow(
            np.zeros((21, 21), dtype=np.uint8), cmap="gray", vmin=0, vmax=255
        )
        self.result_artist = self.ax_result.imshow(
            cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        )
        self.ax_original.set_title("Original")
        self.ax_mask.set_title("HSV mask")
        self.ax_kernel.set_title("Kernel shape")
        self.ax_result.set_title("Final result")
        self.ax_original.axis("off")
        self.ax_mask.axis("off")
        self.ax_kernel.axis("off")
        self.ax_result.axis("off")

        self.slider_axes = self._create_slider_axes()
        self.sliders = self._create_sliders()
        self._create_buttons()

        self.status_text = self.fig.text(
            0.06, 0.96, "", fontsize=10, ha="left", va="top"
        )
        self.help_text = self.fig.text(
            0.70,
            0.96,
            "Keys: left/right arrows navigate, s saves defaults",
            fontsize=9,
            ha="left",
            va="top",
            color="#444444",
        )

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self._refresh_view()

    def _create_slider_axes(self) -> dict[str, Axes]:
        return {
            "lower_h": self.fig.add_axes([0.08, 0.20, 0.28, 0.025]),
            "upper_h": self.fig.add_axes([0.08, 0.16, 0.28, 0.025]),
            "lower_s": self.fig.add_axes([0.08, 0.12, 0.28, 0.025]),
            "upper_s": self.fig.add_axes([0.08, 0.08, 0.28, 0.025]),
            "lower_v": self.fig.add_axes([0.08, 0.04, 0.28, 0.025]),
            "upper_v": self.fig.add_axes([0.38, 0.20, 0.28, 0.025]),
            "kernel": self.fig.add_axes([0.38, 0.08, 0.28, 0.12], projection=None),
        }

    def _create_sliders(self) -> dict[str, Slider]:
        sliders = {
            "lower_h": Slider(
                self.slider_axes["lower_h"],
                "H low",
                0,
                179,
                valinit=self.defaults.lower_h,
                valstep=1,
            ),
            "upper_h": Slider(
                self.slider_axes["upper_h"],
                "H high",
                0,
                179,
                valinit=self.defaults.upper_h,
                valstep=1,
            ),
            "lower_s": Slider(
                self.slider_axes["lower_s"],
                "S low",
                0,
                255,
                valinit=self.defaults.lower_s,
                valstep=1,
            ),
            "upper_s": Slider(
                self.slider_axes["upper_s"],
                "S high",
                0,
                255,
                valinit=self.defaults.upper_s,
                valstep=1,
            ),
            "lower_v": Slider(
                self.slider_axes["lower_v"],
                "V low",
                0,
                255,
                valinit=self.defaults.lower_v,
                valstep=1,
            ),
            "upper_v": Slider(
                self.slider_axes["upper_v"],
                "V high",
                0,
                255,
                valinit=self.defaults.upper_v,
                valstep=1,
            ),
            "kernel": Slider(
                self.slider_axes["kernel"],
                "Kernel size",
                1,
                25,
                valinit=self.defaults.kernel_size,
                valstep=2,
                orientation="vertical",
            ),
        }
        for key, slider in sliders.items():
            slider.on_changed(self._on_slider_change)
        return sliders

    def _create_buttons(self) -> None:
        prev_ax = self.fig.add_axes([0.68, 0.12, 0.08, 0.05])
        next_ax = self.fig.add_axes([0.77, 0.12, 0.08, 0.05])
        reset_ax = self.fig.add_axes([0.86, 0.12, 0.08, 0.05])
        save_ax = self.fig.add_axes([0.68, 0.04, 0.26, 0.06])

        morph_ax = self.fig.add_axes([0.38, 0.02, 0.28, 0.04])

        self.prev_button = Button(prev_ax, "Previous")
        self.next_button = Button(next_ax, "Next")
        self.reset_button = Button(reset_ax, "Reset")
        self.save_button = Button(save_ax, "Save defaults")
        self.morph_button = Button(morph_ax, "Morph: None | Close | Open")

        self.prev_button.on_clicked(self.previous_image)
        self.next_button.on_clicked(self.next_image)
        self.reset_button.on_clicked(self.reset_defaults)
        self.save_button.on_clicked(self.save_current_defaults)
        self.morph_button.on_clicked(self.toggle_morph_op)

    def current_defaults(self) -> HsvDefaults:
        defaults = HsvDefaults(
            lower_h=int(self.sliders["lower_h"].val),
            lower_s=int(self.sliders["lower_s"].val),
            lower_v=int(self.sliders["lower_v"].val),
            upper_h=int(self.sliders["upper_h"].val),
            upper_s=int(self.sliders["upper_s"].val),
            upper_v=int(self.sliders["upper_v"].val),
            kernel_size=int(self.sliders["kernel"].val),
            morph_op=self.morph_op_state,
        )
        defaults._validate()
        return defaults

    def _on_slider_change(self, _value: float) -> None:
        self._refresh_view()

    def _refresh_view(self) -> None:
        defaults = self.current_defaults()
        hsv, mask, masked = build_mask(self.current_image, defaults)
        original_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        masked_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
        morph_viz = visualize_morph_effect(
            mask, defaults.kernel_size, defaults.morph_op
        )

        self.original_artist.set_data(original_rgb)
        self.mask_artist.set_data(mask)
        self.kernel_artist.set_data(morph_viz)
        self.result_artist.set_data(masked_rgb)

        plot_histograms(
            [self.ax_hist_h, self.ax_hist_s, self.ax_hist_v],
            hsv,
            defaults,
            f"{self.index + 1}/{len(self.image_paths)} - {self.image_paths[self.index].name}",
        )

        mask_ratio = (
            100.0 * float(np.count_nonzero(mask)) / float(mask.size)
            if mask.size
            else 0.0
        )
        self.ax_original.set_title(f"Original: {self.image_paths[self.index].name}")
        self.ax_mask.set_title(f"HSV mask | {mask_ratio:.1f}% active")
        morph_label = f"Morph {defaults.morph_op.upper()} (k={defaults.kernel_size})"
        self.ax_kernel.set_title(f"{morph_label}\n[left=before, right=after]")
        self.ax_result.set_title(f"Final result")
        self.status_text.set_text(
            f"Image {self.index + 1}/{len(self.image_paths)} | H=[{defaults.lower_h},{defaults.upper_h}] S=[{defaults.lower_s},{defaults.upper_s}] V=[{defaults.lower_v},{defaults.upper_v}] {morph_label}"
        )

        morph_idx = ["none", "close", "open"].index(self.morph_op_state)
        self.morph_button.label.set_text(
            [
                "Morph: NONE | close | open",
                "Morph: none | CLOSE | open",
                "Morph: none | close | OPEN",
            ][morph_idx]
        )

        self.fig.canvas.draw_idle()

    def _set_slider_values(self, defaults: HsvDefaults) -> None:
        for name, value in [
            ("lower_h", defaults.lower_h),
            ("upper_h", defaults.upper_h),
            ("lower_s", defaults.lower_s),
            ("upper_s", defaults.upper_s),
            ("lower_v", defaults.lower_v),
            ("upper_v", defaults.upper_v),
            ("kernel", defaults.kernel_size),
        ]:
            self.sliders[name].eventson = False
            self.sliders[name].set_val(value)
            self.sliders[name].eventson = True

    def previous_image(self, _event: object | None = None) -> None:
        self.index = (self.index - 1) % len(self.image_paths)
        self.current_image = load_image(self.image_paths[self.index])
        self._refresh_view()

    def next_image(self, _event: object | None = None) -> None:
        self.index = (self.index + 1) % len(self.image_paths)
        self.current_image = load_image(self.image_paths[self.index])
        self._refresh_view()

    def reset_defaults(self, _event: object | None = None) -> None:
        self._set_slider_values(HsvDefaults())
        self._refresh_view()

    def save_current_defaults(self, _event: object | None = None) -> None:
        defaults = self.current_defaults()
        save_defaults(self.config_path, defaults)
        self.status_text.set_text(
            f"Saved defaults to {self.config_path.name}: lower={defaults.as_lower()} upper={defaults.as_upper()}"
        )
        self.fig.canvas.draw_idle()

    def toggle_morph_op(self, _event: object | None = None) -> None:
        ops = ["none", "close", "open"]
        current_idx = ops.index(self.morph_op_state)
        self.morph_op_state = ops[(current_idx + 1) % len(ops)]
        self._refresh_view()

    def on_key_press(self, event: KeyEvent) -> None:
        if event.key in {"left", "a"}:
            self.previous_image()
        elif event.key in {"right", "d"}:
            self.next_image()
        elif event.key == "r":
            self.reset_defaults()
        elif event.key == "s":
            self.save_current_defaults()
        elif event.key == "m":
            self.toggle_morph_op()

    def show(self) -> None:
        plt.show()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone HSV tuner for images in data/images_crop"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(DEFAULT_IMAGES_DIR),
        help="Directory containing the cropped card images",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the JSON file storing the last saved defaults",
    )
    parser.add_argument(
        "--start-index", type=int, default=0, help="Index of the first image to open"
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    images_dir = Path(args.images_dir)
    config_path = Path(args.config)
    image_paths = list_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    defaults = load_defaults(config_path)
    tuner = HsvCardTuner(
        image_paths=image_paths, defaults=defaults, config_path=config_path
    )
    tuner.index = int(np.clip(args.start_index, 0, len(image_paths) - 1))
    tuner.current_image = load_image(image_paths[tuner.index])
    tuner._set_slider_values(defaults)
    tuner._refresh_view()
    tuner.show()


if __name__ == "__main__":
    main()
