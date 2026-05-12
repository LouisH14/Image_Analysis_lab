"""Interactive HSV and morphology tuner for UNO card preprocessing.

The script visualizes how HSV thresholding, morphological closing, and small
component removal affect a card image. It is meant for manual parameter tuning
before wiring the chosen preprocessing into the detector pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.widgets import Button, Slider

DEFAULT_WINDOW_SIZE = 500


def list_images(images_dir: Path) -> list[Path]:
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images: list[Path] = []
    for pattern in extensions:
        images.extend(sorted(images_dir.glob(pattern)))
    return images


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


def resize_to_square(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def apply_hsv_threshold(
    image_bgr: np.ndarray,
    lower_hsv: Tuple[int, int, int],
    upper_hsv: Tuple[int, int, int],
) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def apply_closing(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def build_preview(
    image_bgr: np.ndarray,
    lower_hsv: Tuple[int, int, int],
    upper_hsv: Tuple[int, int, int],
    kernel_size: int,
    min_area: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.array(lower_hsv, dtype=np.uint8), np.array(upper_hsv, dtype=np.uint8)
    )
    closed = apply_closing(mask, kernel_size)
    cleaned = remove_small_components(closed, min_area)
    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=cleaned)
    return hsv, mask, cleaned, masked


def find_default_image(images_dir: Path) -> Path:
    images = list_images(images_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return images[0]


def make_tuner(
    image_bgr: np.ndarray,
    title: str,
    initial_lower: Tuple[int, int, int],
    initial_upper: Tuple[int, int, int],
    initial_kernel: int,
    initial_min_area: int,
    initial_x: int = 0,
    initial_y: int = 0,
    window_size: int = 500,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for ax in axes:
        ax.axis("off")

    h, w = image_bgr.shape[:2]
    max_x = max(0, w - window_size)
    max_y = max(0, h - window_size)
    initial_x = int(np.clip(initial_x, 0, max_x))
    initial_y = int(np.clip(initial_y, 0, max_y))

    def get_crop(image: np.ndarray, x: int, y: int) -> tuple[np.ndarray, int, int]:
        crop_h = min(window_size, image.shape[0] - y)
        crop_w = min(window_size, image.shape[1] - x)
        crop_h = max(1, crop_h)
        crop_w = max(1, crop_w)
        return image[y : y + crop_h, x : x + crop_w], crop_w, crop_h

    crop_bgr, crop_w, crop_h = get_crop(image_bgr, initial_x, initial_y)
    _, mask, cleaned, masked = build_preview(
        crop_bgr,
        initial_lower,
        initial_upper,
        initial_kernel,
        initial_min_area,
    )

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Panel 0: Original with 500x500 window indicator
    axes[0].imshow(img_rgb)
    # Draw a rectangle to show the selected 500x500 window region
    rect = mpatches.Rectangle(
        (initial_x, initial_y),
        crop_w,
        crop_h,
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
    )
    axes[0].add_patch(rect)
    origin_text = axes[0].text(
        10,
        20,
        f"window origin = ({initial_x}, {initial_y})",
        color="lime",
        fontsize=10,
        fontweight="bold",
        bbox=dict(facecolor="black", alpha=0.45, pad=2, edgecolor="none"),
    )
    axes[0].set_title(
        f"Original ({h}×{w})\nGreen box = movable {window_size}×{window_size} window"
    )

    # Panel 1: HSV mask
    mask_pct = 100 * np.count_nonzero(mask) / mask.size if mask.size > 0 else 0
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"HSV mask\n({mask_pct:.1f}% active)")

    # Panel 2: Cleaned mask
    cleaned_pct = (
        100 * np.count_nonzero(cleaned) / cleaned.size if cleaned.size > 0 else 0
    )
    axes[2].imshow(cleaned, cmap="gray")
    axes[2].set_title(f"After closing + area filter\n({cleaned_pct:.1f}% active)")

    # Panel 3: Masked image
    axes[3].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Masked image (result)")

    # Panel 4: Cropped 500x500 original
    img_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    axes[4].imshow(img_crop)
    axes[4].set_title(
        f"Cropped region at ({initial_x}, {initial_y})\n(actual size: {crop_h}×{crop_w})"
    )

    # Panel 5: Cropped 500x500 mask
    mask_crop = mask[:crop_h, :crop_w]
    crop_pct = (
        100 * np.count_nonzero(mask_crop) / mask_crop.size if mask_crop.size > 0 else 0
    )
    axes[5].imshow(mask_crop, cmap="gray")
    axes[5].set_title(f"Mask in crop\n({crop_pct:.1f}% active)")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.subplots_adjust(
        left=0.08, right=0.98, bottom=0.38, top=0.9, hspace=0.3, wspace=0.15
    )

    slider_axes = {
        "x": fig.add_axes([0.10, 0.30, 0.75, 0.03]),
        "y": fig.add_axes([0.10, 0.26, 0.75, 0.03]),
        "h_low": fig.add_axes([0.10, 0.22, 0.75, 0.03]),
        "s_low": fig.add_axes([0.10, 0.18, 0.75, 0.03]),
        "v_low": fig.add_axes([0.10, 0.14, 0.75, 0.03]),
        "h_high": fig.add_axes([0.10, 0.10, 0.75, 0.03]),
        "s_high": fig.add_axes([0.10, 0.06, 0.75, 0.03]),
        "v_high": fig.add_axes([0.10, 0.02, 0.75, 0.03]),
    }

    sliders = {
        "x": Slider(slider_axes["x"], "X", 0, max_x, valinit=initial_x, valstep=1),
        "y": Slider(slider_axes["y"], "Y", 0, max_y, valinit=initial_y, valstep=1),
        "h_low": Slider(
            slider_axes["h_low"], "H low", 0, 179, valinit=initial_lower[0], valstep=1
        ),
        "s_low": Slider(
            slider_axes["s_low"], "S low", 0, 255, valinit=initial_lower[1], valstep=1
        ),
        "v_low": Slider(
            slider_axes["v_low"], "V low", 0, 255, valinit=initial_lower[2], valstep=1
        ),
        "h_high": Slider(
            slider_axes["h_high"], "H high", 0, 179, valinit=initial_upper[0], valstep=1
        ),
        "s_high": Slider(
            slider_axes["s_high"], "S high", 0, 255, valinit=initial_upper[1], valstep=1
        ),
        "v_high": Slider(
            slider_axes["v_high"], "V high", 0, 255, valinit=initial_upper[2], valstep=1
        ),
    }

    kernel_ax = fig.add_axes([0.88, 0.22, 0.06, 0.18])
    area_ax = fig.add_axes([0.88, 0.02, 0.06, 0.18])
    kernel_slider = Slider(
        kernel_ax,
        "Kernel",
        1,
        31,
        valinit=initial_kernel,
        valstep=2,
        orientation="vertical",
    )
    area_slider = Slider(
        area_ax,
        "Min area",
        0,
        20000,
        valinit=initial_min_area,
        valstep=50,
        orientation="vertical",
    )

    reset_ax = fig.add_axes([0.88, 0.88, 0.08, 0.05])
    reset_button = Button(reset_ax, "Reset")

    def redraw(_value: object) -> None:
        x = int(sliders["x"].val)
        y = int(sliders["y"].val)
        lower = (
            int(sliders["h_low"].val),
            int(sliders["s_low"].val),
            int(sliders["v_low"].val),
        )
        upper = (
            int(sliders["h_high"].val),
            int(sliders["s_high"].val),
            int(sliders["v_high"].val),
        )
        kernel_size = int(kernel_slider.val)
        if kernel_size % 2 == 0:
            kernel_size += 1
        min_area = int(area_slider.val)

        crop_bgr, crop_w, crop_h = get_crop(image_bgr, x, y)
        _, mask, cleaned, masked = build_preview(
            crop_bgr, lower, upper, kernel_size, min_area
        )

        # Update all 6 panels
        origin_text.set_text(f"window origin = ({x}, {y})")

        # Update main image rectangle in place
        rect.set_xy((x, y))
        rect.set_width(crop_w)
        rect.set_height(crop_h)

        # Update crop preview panel
        axes[4].images[0].set_data(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        axes[4].set_title(
            f"Cropped region at ({x}, {y})\n(actual size: {crop_h}×{crop_w})"
        )

        # Panel 1: HSV mask
        mask_pct = 100 * np.count_nonzero(mask) / mask.size if mask.size > 0 else 0
        axes[1].images[0].set_data(mask)
        axes[1].set_title(f"HSV mask\n({mask_pct:.1f}% active)")

        # Panel 2: Cleaned mask
        cleaned_pct = (
            100 * np.count_nonzero(cleaned) / cleaned.size if cleaned.size > 0 else 0
        )
        axes[2].images[0].set_data(cleaned)
        axes[2].set_title(f"After closing + area filter\n({cleaned_pct:.1f}% active)")

        # Panel 3: Masked image
        axes[3].images[0].set_data(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))

        # Panel 5: Cropped mask
        mask_crop = mask[:crop_h, :crop_w]
        crop_pct = (
            100 * np.count_nonzero(mask_crop) / mask_crop.size
            if mask_crop.size > 0
            else 0
        )
        axes[5].images[0].set_data(mask_crop)
        axes[5].set_title(f"Mask in crop\n({crop_pct:.1f}% active)")

        fig.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(redraw)
    kernel_slider.on_changed(redraw)
    area_slider.on_changed(redraw)

    def reset(_event: object) -> None:
        for slider in sliders.values():
            slider.reset()
        kernel_slider.reset()
        area_slider.reset()

    reset_button.on_clicked(reset)

    plt.show()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive HSV and morphology tuner for UNO cards"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to an image to inspect"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/test_images",
        help="Directory of test images",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Image index to use when --image is not provided",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=0,
        help="Resize image to a square of this size before tuning",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Expected card window size for reference",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    images_dir = Path(args.images_dir)

    if args.image is not None:
        image_path = Path(args.image)
    else:
        images = list_images(images_dir)
        if not images:
            raise FileNotFoundError(f"No images found in {images_dir}")
        image_path = images[min(max(args.index, 0), len(images) - 1)]

    image_bgr = load_image(image_path)

    # Note: Do NOT resize here; pass original size and let make_tuner handle it
    make_tuner(
        image_bgr=image_bgr,
        title=f"HSV + Morphology Tuner: {image_path.name} | window size {args.window_size}×{args.window_size}",
        initial_lower=(0, 0, 0),
        initial_upper=(179, 255, 255),
        initial_kernel=5,
        initial_min_area=250,
        initial_x=0,
        initial_y=0,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main()
