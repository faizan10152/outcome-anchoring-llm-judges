"""Assemble the appendix PDF and the final submission ZIP.

The appendix must be one PDF containing the references, the declaration on the
use of generative AI tools, and the signed Declaration of Academic Integrity.
The first two are generated from appendix.html; the signed declaration can only
come from you, so this script merges it in.

    python appendix/build_submission.py --signed appendix/declaration_signed.pdf \
                                        --student-id 255678

It refuses to build if the signed declaration is missing or still looks like the
blank form, because submitting an unsigned declaration fails the examination.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent.parent
APPENDIX = ROOT / "appendix"


def looks_unsigned(path: Path) -> bool:
    """Crude check that the form still has empty name and ID fields."""
    try:
        text = PdfReader(str(path)).pages[0].extract_text() or ""
    except Exception:
        return False
    # The blank form has these labels with nothing after them on the same line.
    for label in ("Last name, first name:", "Student ID number:"):
        idx = text.find(label)
        if idx == -1:
            continue
        tail = text[idx + len(label): idx + len(label) + 40].strip()
        if not tail or tail.startswith(("Student ID", "hereby")):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signed", required=True,
                    help="your filled-in and signed Declaration of Academic Integrity (PDF)")
    ap.add_argument("--student-id", required=True, help="used to name the ZIP")
    ap.add_argument("--allow-unsigned", action="store_true",
                    help="build anyway if the signature check misfires (check it yourself)")
    args = ap.parse_args()

    body = APPENDIX / "appendix_body.pdf"
    signed = Path(args.signed)
    poster = ROOT / "poster" / "poster.pdf"
    for f in (body, signed, poster):
        if not f.exists():
            print(f"ERROR: missing {f}", file=sys.stderr)
            return 2

    if looks_unsigned(signed) and not args.allow_unsigned:
        print(f"ERROR: {signed.name} still looks like the blank form: the name and "
              f"student ID fields appear empty.\nFill it in, sign it, and re-export. "
              f"Use --allow-unsigned only if you have checked it yourself.",
              file=sys.stderr)
        return 3

    # 1. appendix = body + signed declaration
    out_appendix = APPENDIX / "appendix.pdf"
    writer = PdfWriter()
    for page in PdfReader(str(body)).pages:
        writer.add_page(page)
    for page in PdfReader(str(signed)).pages:
        writer.add_page(page)
    with out_appendix.open("wb") as fh:
        writer.write(fh)
    print(f"appendix.pdf: {len(PdfReader(str(out_appendix)).pages)} pages")

    # 2. the ZIP: exactly two PDFs, named by student id
    sid = args.student_id.strip()
    out_zip = ROOT / "submission" / f"{sid}.zip"
    out_zip.parent.mkdir(exist_ok=True)
    staged = out_zip.parent / f"{sid}_poster.pdf"
    shutil.copy(poster, staged)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(poster, "poster.pdf")
        z.write(out_appendix, "appendix.pdf")
    staged.unlink(missing_ok=True)

    with zipfile.ZipFile(out_zip) as z:
        names = z.namelist()
    print(f"\n{out_zip.relative_to(ROOT)}  ({out_zip.stat().st_size / 1e6:.2f} MB)")
    print(f"  contents: {names}")
    if len(names) != 2 or not all(n.endswith('.pdf') for n in names):
        print("ERROR: the archive must contain exactly two PDF files", file=sys.stderr)
        return 4
    print("\nReady to upload to the posters folder on StudIP.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
