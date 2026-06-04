"""
docx_loader.py
--------------
Loads .docx files using python-docx.

Strategy:
- Detects heading paragraphs by style name ("Heading 1", "Heading 2", etc.)
- Each heading starts a new section; body paragraphs accumulate until next heading.
- Preserves heading level in section_ref for downstream traceability.

Returns unified schema: [{section_id, section_ref, text}, ...]
"""

from docx import Document
from docx.oxml.ns import qn
from typing import List, Dict, Any
from src.ingestion.base_loader import BaseLoader


class DOCXLoader(BaseLoader):

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        doc = Document(file_path)
        return self._extract_sections(doc)

    def _extract_sections(self, doc) -> List[Dict[str, Any]]:
        sections: List[Dict[str, Any]] = []
        current_heading = "Preamble"
        current_level = 0
        current_lines: List[str] = []

        def _is_heading(para) -> bool:
            return para.style.name.startswith("Heading")

        def _heading_level(para) -> int:
            name = para.style.name  # e.g. "Heading 1"
            parts = name.split()
            try:
                return int(parts[-1])
            except (ValueError, IndexError):
                return 1

        def _flush(heading, lines):
            text = self._clean("\n".join(lines))
            if text:
                sections.append({
                    "section_id": len(sections) + 1,
                    "section_ref": heading,
                    "text": text,
                })

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            if _is_heading(para):
                _flush(current_heading, current_lines)
                level = _heading_level(para)
                current_heading = f"H{level}: {text}"
                current_lines = []
            else:
                current_lines.append(text)

        _flush(current_heading, current_lines)

        # --- Fallback: plain paragraphs with no headings ---
        if not sections:
            for idx, para in enumerate(doc.paragraphs, start=1):
                text = para.text.strip()
                if text:
                    sections.append({
                        "section_id": idx,
                        "section_ref": f"paragraph {idx}",
                        "text": text,
                    })

        return sections