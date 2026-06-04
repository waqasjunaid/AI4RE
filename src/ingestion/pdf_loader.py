"""
pdf_loader.py
Loads PDF files using PyMuPDF (fitz).

Strategy:
1. Try heading-based section detection (works for reports, manuals, SRS docs)
2. If the document looks like a slide deck OR fewer than 3 real headings found,
   fall back to page-per-section
3. If zero text is extractable (scanned/image PDF), return empty list with a
   clear warning printed so the caller knows why
"""

import fitz
from typing import List, Dict, Any
from src.ingestion.base_loader import BaseLoader


class PDFLoader(BaseLoader):

    HEADING_RATIO   = 1.20   # heading font must be 20% larger than median body
    MIN_HEADING_LEN = 15     # headings must be at least this many characters
    MAX_HEADING_LEN = 150    # headings must be shorter than this

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        document = fitz.open(file_path)

        if document.is_encrypted:
            document.close()
            raise ValueError("PDF is encrypted: {}".format(file_path))

        sections = self._extract_sections(document)
        document.close()
        return sections

    def _extract_sections(self, document) -> List[Dict[str, Any]]:
        # Collect all text spans with font info
        all_spans = []
        for page_num, page in enumerate(document, start=1):
            try:
                blocks = page.get_text("dict")["blocks"]
            except Exception:
                continue
            for block in blocks:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            all_spans.append({
                                "page":  page_num,
                                "size":  round(span["size"], 1),
                                "text":  text,
                                "flags": span["flags"],
                            })

        if not all_spans:
            print("    WARNING: No text could be extracted from this PDF.")
            print("    The file may be scanned/image-based or use embedded fonts.")
            print("    Try downloading the .odt/.docx version instead.")
            return []

        # Detect median body font size
        from statistics import median
        body_size = median(s["size"] for s in all_spans)

        # Detect if this is a slide deck:
        # In slide decks almost every line is "large", so most text looks like headings.
        large_count = sum(1 for s in all_spans if s["size"] >= body_size * self.HEADING_RATIO)
        is_slide_deck = large_count > len(all_spans) * 0.4   # >40% large = slide deck

        if is_slide_deck:
            # For slide decks, one section per page is better
            return self._page_fallback(document)

        # Heading-based section detection for structured documents
        def is_real_heading(span):
            text    = span["text"]
            big     = span["size"] >= body_size * self.HEADING_RATIO
            bold    = bool(span["flags"] & 16)
            long_enough  = len(text) >= self.MIN_HEADING_LEN
            short_enough = len(text) <= self.MAX_HEADING_LEN
            not_bullet   = not text.startswith(("*", "-", "+", "#", ">", "chr(0x2022)", chr(183)))
            return (big or bold) and long_enough and short_enough and not_bullet

        sections = []
        current_heading = "Preamble"
        current_page    = 1
        current_lines   = []

        def flush(heading, page, lines):
            text = self._clean(" \n".join(lines))
            if text:
                sections.append({
                    "section_id":  len(sections) + 1,
                    "section_ref": "{} (p.{})".format(heading[:80], page),
                    "text":        text,
                })

        for span in all_spans:
            if is_real_heading(span):
                flush(current_heading, current_page, current_lines)
                current_heading = span["text"]
                current_page    = span["page"]
                current_lines   = []
            else:
                current_lines.append(span["text"])

        flush(current_heading, current_page, current_lines)

        # If detection produced fewer than 3 real sections, fall back to pages
        if len(sections) < 3:
            return self._page_fallback(document)

        return sections

    def _page_fallback(self, document) -> List[Dict[str, Any]]:
        """One section per page."""
        sections = []
        for page_num, page in enumerate(document, start=1):
            try:
                text = self._clean(page.get_text())
            except Exception:
                text = ""
            if text:
                sections.append({
                    "section_id":  page_num,
                    "section_ref": "page {}".format(page_num),
                    "text":        text,
                })
        return sections