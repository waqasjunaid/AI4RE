"""
text_loader.py
Loads plain text, JSON, YAML, and PlantUML (.puml) files.

Plain text strategy:
- Detects section headings (numbered lines, ALL CAPS, underlined with ===)
- Groups paragraphs under their nearest heading
- Merges tiny paragraphs (< 30 tokens) into their neighbours
- Caps output at MAX_SECTIONS sections to keep processing manageable
- Each section targets MIN_TOKENS to MAX_TOKENS tokens

PlantUML (.puml) files:
- Splits on diagram block boundaries (package, class, enum, etc.)
"""

import re
import os
from typing import List, Dict, Any
from src.ingestion.base_loader import BaseLoader

MAX_SECTIONS = 150    # never produce more than this many sections
MIN_TOKENS   = 50     # merge sections smaller than this into neighbours
TARGET_CHARS = 1500   # target characters per section (~300 tokens)


def _approx_tokens(text: str) -> int:
    return len(text.split())


class TextLoader(BaseLoader):

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        ext = os.path.splitext(file_path)[1].lower()
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        if ext == ".puml":
            return self._extract_puml_sections(raw)
        else:
            return self._extract_text_sections(raw)

    # ------------------------------------------------------------------
    # Plain text
    # ------------------------------------------------------------------
    def _extract_text_sections(self, raw: str) -> List[Dict[str, Any]]:
        lines = raw.splitlines()

        # Detect heading lines
        def is_heading(line: str, next_line: str = "") -> bool:
            s = line.strip()
            if not s or len(s) > 100:
                return False
            # Numbered: "1 Introduction", "2.3 Options", "Chapter 4"
            if re.match(r'^(\d+[\.\d]*|Chapter\s+\d+|Appendix\s+[A-Z])\s+\S', s):
                return True
            # Underlined with = or - on next line
            if next_line and re.match(r'^[=\-\*]{4,}$', next_line.strip()):
                return True
            # ALL CAPS short line
            if s.isupper() and 4 <= len(s) <= 60:
                return True
            # Ends with no period and starts with capital, short
            if (len(s) < 60 and s[0].isupper()
                    and not s.endswith('.')
                    and not s.endswith(',')
                    and re.match(r'^[A-Z][^\n]{3,59}$', s)
                    and s.count(' ') < 8):
                return True
            return False

        # Group lines into raw blocks under headings
        blocks = []          # list of (heading, text_lines)
        current_heading = "Introduction"
        current_lines   = []

        for i, line in enumerate(lines):
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if is_heading(line, next_line):
                if current_lines:
                    blocks.append((current_heading, current_lines))
                current_heading = line.strip()
                current_lines   = []
            else:
                # Skip underline rows
                if not re.match(r'^[=\-\*]{4,}$', line.strip()):
                    current_lines.append(line)

        if current_lines:
            blocks.append((current_heading, current_lines))

        # Convert blocks to sections, merging small ones
        sections = []
        pending_heading = ""
        pending_text    = ""

        for heading, blines in blocks:
            text = self._clean("\n".join(blines))
            if not text:
                continue

            # If pending + current still small, merge
            combined = (pending_text + "\n\n" + text).strip() if pending_text else text
            if pending_text and _approx_tokens(combined) < MIN_TOKENS * 3:
                pending_text = combined
                continue

            # Flush pending
            if pending_text:
                sections.append({
                    "section_id":  len(sections) + 1,
                    "section_ref": pending_heading[:80],
                    "text":        pending_text,
                })

            pending_heading = heading
            pending_text    = text

        # Flush last
        if pending_text:
            sections.append({
                "section_id":  len(sections) + 1,
                "section_ref": pending_heading[:80],
                "text":        pending_text,
            })

        # If still too many sections, merge consecutive small ones
        sections = self._merge_small_sections(sections, MAX_SECTIONS)

        return sections

    def _merge_small_sections(
        self,
        sections: List[Dict[str, Any]],
        max_count: int
    ) -> List[Dict[str, Any]]:
        """Merge consecutive sections until count <= max_count."""
        while len(sections) > max_count:
            # Find the two smallest adjacent sections and merge them
            best_i  = 0
            best_sz = float("inf")
            for i in range(len(sections) - 1):
                sz = _approx_tokens(sections[i]["text"]) + _approx_tokens(sections[i+1]["text"])
                if sz < best_sz:
                    best_sz = sz
                    best_i  = i

            merged = {
                "section_id":  sections[best_i]["section_id"],
                "section_ref": sections[best_i]["section_ref"],
                "text":        sections[best_i]["text"] + "\n\n" + sections[best_i+1]["text"],
            }
            sections = sections[:best_i] + [merged] + sections[best_i+2:]

        # Re-number
        for i, s in enumerate(sections, start=1):
            s["section_id"] = i

        return sections

    # ------------------------------------------------------------------
    # PlantUML
    # ------------------------------------------------------------------
    def _extract_puml_sections(self, raw: str) -> List[Dict[str, Any]]:
        lines = raw.splitlines()
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("'"):
                content = stripped[1:].strip(" -=~_")
                if len(content) > 4:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)

        cleaned = "\n".join(cleaned_lines)

        BLOCK_STARTERS = [
            r'^package\s+',
            r'^class\s+',
            r'^abstract\s+class\s+',
            r'^interface\s+',
            r'^enum\s+',
            r'^component\s+',
            r'^database\s+',
            r'^actor\s+',
            r'^note\s+as\s+',
            r'^sequence\s+',
            r'^@startuml',
            r'^title\s+',
        ]
        block_pattern = re.compile(
            r'(?=^(?:' + '|'.join(BLOCK_STARTERS) + r'))',
            re.MULTILINE | re.IGNORECASE
        )

        blocks = block_pattern.split(cleaned)
        sections = []

        for block in blocks:
            text = self._clean(block)
            if not text or len(text) < 10:
                continue
            first_line = text.splitlines()[0][:80]
            sections.append({
                "section_id":  len(sections) + 1,
                "section_ref": "puml:{}".format(first_line),
                "text":        text,
            })

        if not sections:
            text = self._clean(raw)
            if text:
                sections.append({
                    "section_id":  1,
                    "section_ref": "puml:full",
                    "text":        text,
                })

        return sections