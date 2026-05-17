#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

def crop_and_transfer(src_path: str, dest_path: str, box: tuple):
    """
    Crops a part of an image from a source path and saves it to a destination path.
    
    Args:
        src_path: String path to the source image file.
        dest_path: String path where the cropped image will be saved.
        box: A tuple (left, top, right, bottom) defining the crop region.
    """
    src = Path(src_path)
    dst = Path(dest_path)
    
    if not src.exists():
        print(f"Error: Source file {src_path} does not exist.")
        return

    # Handle filename collision to avoid overwriting
    if dst.exists():
        base = dst.stem
        ext = dst.suffix
        parent = dst.parent
        counter = 1
        while (parent / f"{base}_{counter}{ext}").exists():
            counter += 1
        dst = parent / f"{base}_{counter}{ext}"

    try:
        with Image.open(src) as img:
            # PIL crop box is (left, top, right, bottom)
            cropped = img.crop(box)
            
            # Ensure destination directory exists
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the cropped image. Format is inferred from extension.
            cropped.save(dst)
            print(f"Successfully cropped {src.name} and saved to {dst}")
    except Exception as e:
        print(f"An error occurred during cropping: {e}")

def interactive_crop_and_transfer(src_path: str, dest_path: str):
    """
    Allows the user to select points on an image to define a contour for cropping.
    The result is saved with a transparent background (PNG).
    
    Args:
        src_path: String path to the source image file.
        dest_path: String path where the cropped image will be saved (use .png).
    """
    src = Path(src_path)
    dst = Path(dest_path)

    if not src.exists():
        print(f"Error: Source file {src_path} does not exist.")
        return

    # Handle filename collision to avoid overwriting existing crops
    if dst.exists():
        base = dst.stem
        ext = dst.suffix
        parent = dst.parent
        counter = 1
        while (parent / f"{base}_{counter}{ext}").exists():
            counter += 1
        dst = parent / f"{base}_{counter}{ext}"

    # Open image and ensure it's in RGBA for transparency
    img = Image.open(src).convert("RGBA")
    
    # Interactive selection using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title("Left-click to place points. Right-click to undo. Press Enter to finish.")
    print(f"Interacting with {src.name}...")
    print("- Click points on the image to define a polygon.")
    print("- Press 'Enter' when the contour is finished.")
    
    points = plt.ginput(n=-1, timeout=0)
    plt.close()

    if len(points) < 3:
        print("Operation cancelled: At least 3 points are required to define a contour.")
        return

    # Create a mask for the polygon
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(points, fill=255)

    # Apply mask to the image (transparent background)
    result = Image.new("RGBA", img.size, (0, 0, 0, 0))
    result.paste(img, (0, 0), mask=mask)

    # Calculate bounding box of the points to crop the resulting image tightly
    points_np = np.array(points)
    min_x, min_y = points_np.min(axis=0)
    max_x, max_y = points_np.max(axis=0)
    
    crop_box = (
        max(0, int(min_x)),
        max(0, int(min_y)),
        min(img.width, int(max_x)),
        min(img.height, int(max_y))
    )

    cropped_result = result.crop(crop_box)

    # Save the result
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        cropped_result.save(dst)
        print(f"Successfully saved interactive crop to {dst}")
    except Exception as e:
        print(f"An error occurred while saving: {e}")

def main():
    # Calculate the project root directory (one level up from this script's directory)
    project_root = Path(__file__).resolve().parent.parent
    
    # Define paths relative to the project root
    src_path = project_root / "data" / "train_images" / "L1000776.jpg"
    dest_path = project_root / "data" / "jeton" / "manual_crop.png"

    # Now the script will find the file regardless of where you execute it from
    interactive_crop_and_transfer(src_path, dest_path)

if __name__ == "__main__":
    main()