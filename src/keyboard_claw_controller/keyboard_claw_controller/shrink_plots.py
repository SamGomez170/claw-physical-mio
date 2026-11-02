"""
Shrink all PNG/JPG images in a folder so Overleaf compiles faster.

Usage:
    pip install pillow
    python shrink_images.py path/to/images

Smaller copies (same file names) are written to path/to/images/small/,
so your originals are never touched. Upload the files from "small" to Overleaf.
"""

import sys
from pathlib import Path

from PIL import Image

MAX_WIDTH = 1800   # pixels; plenty for a figure up to full page width
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def shrink(src: Path, dst: Path) -> None:
    with Image.open(src) as img:
        if img.width > MAX_WIDTH:
            new_height = round(img.height * MAX_WIDTH / img.width)
            img = img.resize((MAX_WIDTH, new_height), Image.LANCZOS)

        if src.suffix.lower() == ".png":
            # Drop transparency (plots rarely need it) -> much smaller files
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, "white")
                rgba = img.convert("RGBA")
                background.paste(rgba, mask=rgba.split()[-1])
                img = background
            img.save(dst, optimize=True)
        else:
            img.convert("RGB").save(dst, quality=85, optimize=True)


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = folder / "small"
    out.mkdir(exist_ok=True)

    for src in sorted(folder.iterdir()):
        if src.suffix.lower() not in EXTENSIONS:
            continue
        dst = out / src.name
        before = src.stat().st_size / 1e6
        shrink(src, dst)
        after = dst.stat().st_size / 1e6
        print(f"{src.name}: {before:.2f} MB -> {after:.2f} MB")


if __name__ == "__main__":
    main()