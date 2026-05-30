#!/usr/bin/env python3
"""
Compose a UV texture for a box from 6 face images.

Layout (top-left origin, x = left→right, y = top→down), for output size N:
- front: (N/4,   N/4)
- up:    (N/4,   0)
- back:  (N/4,   N/2)
- down:  (N/4,   3N/4)
- left:  (0,     N/4)
- right: (N/2,   N/4)

Any empty area stays black.

CLI supports per-face rotation flags (degrees counter-clockwise):
--rot-front, --rot-up, --rot-back, --rot-down, --rot-left, --rot-right
Each accepts one of: 0, 90, 180, 270 (default 0).
"""

from pathlib import Path
from PIL import Image
import argparse
import sys

# Faces in the required layout order with their top-left coordinates as lambdas of N
FACE_POSITIONS = {
    "front": lambda N: (N // 4, N // 4),
    "up":    lambda N: (N // 4, 0),
    "down":  lambda N: (N // 4, N // 2),
    "back":  lambda N: (N // 4, (3 * N) // 4),
    "left":  lambda N: (0,      N // 4),
    "right": lambda N: (N // 2, N // 4),
}

# Try common image extensions; filenames are case-insensitive
EXTS = [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"]

def find_image(folder: Path, stem: str) -> Path:
    """Find an image file with the given stem in folder, trying common extensions."""
    # First try exact match (e.g., right.jpg) to honor strict naming if provided
    exact = folder / f"{stem}.jpg"
    if exact.exists():
        return exact
    # Otherwise search case-insensitively across known extensions
    stem_lower = stem.lower()
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() in EXTS and p.stem.lower() == stem_lower:
            return p
    raise FileNotFoundError(f"Missing required face image '{stem}' in {folder}")

def _rotate_image_ccw(img: Image.Image, degrees: int) -> Image.Image:
    """Rotate image by 0/90/180/270 degrees counter-clockwise using transpose (no resample)."""
    if not degrees:
        return img
    # Pillow 9+ exposes constants under Image.Transpose; older versions use Image.* directly
    try:
        Transpose = Image.Transpose
    except AttributeError:
        Transpose = Image
    mapping = {
        90: Transpose.ROTATE_90,
        180: Transpose.ROTATE_180,
        270: Transpose.ROTATE_270,
    }
    op = mapping.get(degrees)
    if op is None:
        raise ValueError(f"Unsupported rotation: {degrees}. Expected one of 0, 90, 180, 270.")
    return img.transpose(op)


def compose_uv(folder: Path, size: int, output: Path, rotations: dict[str, int] | None = None):
    if size <= 0:
        raise ValueError("Output size must be a positive integer.")
    # Prepare black canvas (RGB)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    tile = size // 4
    if size % 4 != 0:
        print(f"[warning] size {size} is not divisible by 4; "
              f"tiles will be {tile}×{tile} and may leave thin borders.", file=sys.stderr)

    # Load, convert, resize, and paste each face
    for face, pos_fn in FACE_POSITIONS.items():
        src_path = find_image(folder, face)
        img = Image.open(src_path)
        # Flatten alpha against black to avoid fringes; then ensure RGB
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
            img = Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")
        else:
            img = img.convert("RGB")
        # Apply per-face rotation before resizing
        deg = 0
        if rotations:
            deg = int(rotations.get(face, 0) or 0)
        if deg:
            img = _rotate_image_ccw(img, deg)
        img = img.resize((tile, tile), Image.LANCZOS)
        x, y = pos_fn(size)
        canvas.paste(img, (x, y))

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output

def main():
    parser = argparse.ArgumentParser(description="Stitch six cube-face images into a UV texture.")
    parser.add_argument("folder", type=Path,
                        help="Path to folder containing face images named front, up, back, down, left, right (e.g., .jpg)")
    parser.add_argument("size", type=int,
                        help="Output texture size (e.g., 1024). Each face will be size/4 square.")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output file path (default: <folder>/uv_texture_<size>.png)")
    # Per-face rotations (degrees CCW)
    rot_help = "Rotation in degrees CCW for this face (choices: 0, 90, 180, 270)."
    parser.add_argument("--rot-front", type=int, choices=[0, 90, 180, 270], default=0, help=rot_help)
    parser.add_argument("--rot-up", type=int, choices=[0, 90, 180, 270], default=0, help=rot_help)
    parser.add_argument("--rot-back", type=int, choices=[0, 90, 180, 270], default=0, help=rot_help)
    parser.add_argument("--rot-down", type=int, choices=[0, 90, 180, 270], default=0, help=rot_help)
    parser.add_argument("--rot-left", type=int, choices=[0, 90, 180, 270], default=0, help=rot_help)
    parser.add_argument("--rot-right", type=int, choices=[0, 90, 180, 270], default=0, help=rot_help)
    args = parser.parse_args()

    out = args.output or (args.folder / f"uv_texture_{args.size}.png")
    try:
        rotations = {
            "front": args.rot_front,
            "up": args.rot_up,
            "back": args.rot_back,
            "down": args.rot_down,
            "left": args.rot_left,
            "right": args.rot_right,
        }
        result = compose_uv(args.folder, args.size, out, rotations)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {result}")

if __name__ == "__main__":
    main()
