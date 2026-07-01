"""
OCR a PDF document into a text file using Tesseract + PyMuPDF.

Usage:
  python ocr_pdf.py input.pdf                   # writes input.txt
  python ocr_pdf.py input.pdf -o result.txt
  python ocr_pdf.py input.pdf --dpi 400        # higher DPI for better accuracy
  python ocr_pdf.py input.pdf --preprocess light  # light = no binarization (often better for faded scans)
"""

import argparse
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_light(img: Image.Image) -> Image.Image:
    """Grayscale, sharpen, boost contrast. Tesseract does its own binarization."""
    img = img.convert("L")
    img = ImageOps.autocontrast(img, cutoff=2)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    return img


def preprocess_strong(img: Image.Image, threshold: int = 180) -> Image.Image:
    """Grayscale, contrast, sharpen, then binarize. Good for clean printed text."""
    img = img.convert("L")
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.point(lambda x: 0 if x < threshold else 255, "1")
    return img


def ocr_pdf(
    pdf_path: str,
    output_path: str,
    dpi: int = 300,
    lang: str = "eng",
    preprocess: str = "light",
    threshold: int = 180,
) -> None:
    pdf = Path(pdf_path)
    if not pdf.exists():
        print(f"Error: file not found: {pdf}")
        sys.exit(1)

    doc = fitz.open(str(pdf))
    total = len(doc)
    print(f"Opened PDF with {total} page(s). OCR (DPI={dpi}, preprocess={preprocess}) ...\n")

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    tesseract_config = "--oem 1 --psm 6"

    all_text: list[str] = []
    start = time.perf_counter()

    for i, page in enumerate(doc, start=1):
        print(f"  Page {i}/{total} ...", end=" ", flush=True)
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        if preprocess == "light":
            img = preprocess_light(img)
        else:
            img = preprocess_strong(img, threshold=threshold)
        text = pytesseract.image_to_string(img, lang=lang, config=tesseract_config)
        all_text.append(text)
        print("done")

    doc.close()
    elapsed = time.perf_counter() - start

    out = Path(output_path)
    out.write_text("\n".join(all_text), encoding="utf-8")

    print(f"\nOCR complete in {elapsed:.1f}s")
    print(f"Output saved to: {out.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR a PDF into a text file (Tesseract).")
    parser.add_argument("pdf", help="Path to the input PDF file.")
    parser.add_argument(
        "-o", "--output",
        help="Output text file path (default: same name as PDF with .txt).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for rendering pages (default: 300). Use 400 for better accuracy.",
    )
    parser.add_argument(
        "--lang", default="eng",
        help="Tesseract language code (default: eng).",
    )
    parser.add_argument(
        "--preprocess", choices=("light", "strong"), default="light",
        help="light = grayscale+contrast+sharpen only (recommended for faded scans). strong = binarize.",
    )
    parser.add_argument(
        "--threshold", type=int, default=180,
        help="Binarization threshold when preprocess=strong (0-255, default: 180).",
    )
    args = parser.parse_args()

    output = args.output or str(Path(args.pdf).with_suffix(".txt"))
    ocr_pdf(
        args.pdf,
        output,
        dpi=args.dpi,
        lang=args.lang,
        preprocess=args.preprocess,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
