"""Set the poster's page box to exactly DIN A1 landscape.

Chrome's print-to-PDF rounds the page height up by about 0.1 mm regardless of
how the CSS @page size is written. That is far inside any print tolerance, but
the examination guidelines state the format explicitly, so the box is set to
the exact value rather than left to rounding.

    python poster/finalize_pdf.py poster/poster.pdf
"""
import sys
from decimal import Decimal
from pathlib import Path

from pypdf import PdfReader, PdfWriter

MM = Decimal("72") / Decimal("25.4")
W = Decimal("841") * MM   # 2383.937... pt
H = Decimal("594") * MM   # 1683.779... pt


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "poster/poster.pdf")
    reader = PdfReader(str(path))
    writer = PdfWriter()
    for page in reader.pages:
        page.mediabox.lower_left = (0, 0)
        page.mediabox.upper_right = (W, H)
        for box in ("cropbox", "trimbox", "bleedbox", "artbox"):
            try:
                b = getattr(page, box)
                b.lower_left, b.upper_right = (0, 0), (W, H)
            except Exception:
                pass
        writer.add_page(page)
    with path.open("wb") as fh:
        writer.write(fh)
    b = PdfReader(str(path)).pages[0].mediabox
    print(f"{path}: {float(b.width) / 72 * 25.4:.4f} x {float(b.height) / 72 * 25.4:.4f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
