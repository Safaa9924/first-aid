"""
================================================================================
 STAGE 01 · DOCUMENT LOADING
 First Aid Reference Guide (St. John Ambulance Canada) — RAG Pipeline
================================================================================
Loads the source PDF with Docling, preserving headings / lists / structure,
and writes the raw extracted text to disk so Stage 02 (preprocessing) can
pick it up. Run this file once whenever the source PDF changes.

Usage:
    python 01_documents.py
================================================================================
"""

import os

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# ==================================================================
# Configuration
# ==================================================================

PDF_PATH = os.environ.get(
    "FIRST_AID_PDF_PATH",
    "First aid reference guide_V4.1_Public.pdf",
)

PUBLICATION_YEAR = 2019  # Fourth Edition, January 2019
SOURCE_TITLE = "First Aid Reference Guide, 4th Edition — St. John Ambulance Canada"

DATA_DIR = "data"
RAW_TEXT_PATH = os.path.join(DATA_DIR, "raw_docling_text.txt")

os.makedirs(DATA_DIR, exist_ok=True)

# ==================================================================
# Initialize Docling Converter
# ==================================================================

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.generate_picture_images = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)


def load_pdf_document(pdf_path):
    """
    Load a PDF with Docling while preserving document structure
    (headings, lists, paragraphs).

    Returns a dict with:
        source_file, raw_text, document, char_count, word_count
    """

    result = converter.convert(pdf_path)
    doc = result.document

    text_parts = []

    for item, _ in doc.iterate_items():

        # Skip non-text items
        if not hasattr(item, "text"):
            continue

        text = item.text.strip()

        if not text:
            continue

        item_type = item.__class__.__name__

        # --------------------------------------------------
        # Section headings
        # --------------------------------------------------
        if item_type == "SectionHeaderItem":
            text_parts.append(f"\n## {text}\n")

        # --------------------------------------------------
        # List items
        # --------------------------------------------------
        elif item_type == "ListItem":
            text_parts.append(f"- {text}")

        # --------------------------------------------------
        # Everything else (paragraphs, table cells, captions...)
        # --------------------------------------------------
        else:
            text_parts.append(text)

    raw_text = "\n".join(text_parts)

    return {
        "source_file": os.path.basename(pdf_path),
        "raw_text": raw_text,
        "document": doc,
        "char_count": len(raw_text),
        "word_count": len(raw_text.split()),
    }


if __name__ == "__main__":

    print("=" * 60)
    print("STAGE 01 · DOCUMENT LOADING")
    print("=" * 60)
    print(f"Source PDF : {PDF_PATH}")

    pdf_document = load_pdf_document(PDF_PATH)

    print(f"Characters : {pdf_document['char_count']:,}")
    print(f"Words      : {pdf_document['word_count']:,}")

    with open(RAW_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write(pdf_document["raw_text"])

    print(f"\nSaved raw text -> {RAW_TEXT_PATH}")
    print("Done. Next: run 02_preprocessing.py")
