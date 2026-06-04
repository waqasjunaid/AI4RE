"""
xml_loader.py
-------------
Loads .xml, .uml, and XMI files (e.g. ArgoUML, PlantUML XMI exports).

Strategy:
- Groups elements by their top-level parent tag to form logical sections.
- Only collects text from LEAF nodes (no children with text) to avoid duplication.
- Handles both plain XML and UML/XMI namespace formats.

Returns unified schema: [{section_id, section_ref, text}, ...]
"""

from lxml import etree
from typing import List, Dict, Any
from src.ingestion.base_loader import BaseLoader
import re


class XMLLoader(BaseLoader):

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            tree = etree.parse(file_path)
        except etree.XMLSyntaxError as e:
            # Try recovery parser for malformed XML
            parser = etree.XMLParser(recover=True)
            tree = etree.parse(file_path, parser)

        root = tree.getroot()
        return self._extract_sections(root)

    def _strip_ns(self, tag: str) -> str:
        """Remove namespace prefix: {http://...}TagName -> TagName"""
        return re.sub(r'\{[^}]+\}', '', tag)

    def _is_leaf_with_text(self, elem) -> bool:
        """True if element has text and no child elements with text."""
        if not (elem.text and elem.text.strip()):
            return False
        for child in elem:
            if child.text and child.text.strip():
                return False
        return True

    def _extract_sections(self, root) -> List[Dict[str, Any]]:
        sections: List[Dict[str, Any]] = []

        # Group by top-level children of root
        for top_elem in root:
            tag = self._strip_ns(top_elem.tag)
            lines: List[str] = []

            # Collect all meaningful text from descendants
            for elem in top_elem.iter():
                if self._is_leaf_with_text(elem):
                    elem_tag = self._strip_ns(elem.tag)
                    text = elem.text.strip()
                    # Include attribute hints for UML elements
                    name_attr = elem.get("name") or elem.get("xmi:id") or ""
                    if name_attr:
                        lines.append(f"[{elem_tag}:{name_attr}] {text}")
                    else:
                        lines.append(f"[{elem_tag}] {text}")

                # Also capture meaningful attributes (e.g. UML element names)
                name = elem.get("name", "").strip()
                kind = elem.get("kind", "").strip() or elem.get("type", "").strip()
                elem_tag = self._strip_ns(elem.tag)
                if name and elem_tag not in ("XMI", "Model"):
                    entry = f"[{elem_tag}] name={name}"
                    if kind:
                        entry += f", kind={kind}"
                    if entry not in lines:
                        lines.append(entry)

            text = self._clean("\n".join(lines))
            if text:
                sections.append({
                    "section_id": len(sections) + 1,
                    "section_ref": f"xml:{tag}",
                    "text": text,
                })

        # Fallback: flat extraction if no top-level children produced output
        if not sections:
            lines = []
            for elem in root.iter():
                if self._is_leaf_with_text(elem):
                    tag = self._strip_ns(elem.tag)
                    lines.append(f"[{tag}] {elem.text.strip()}")
            text = self._clean("\n".join(lines))
            if text:
                sections.append({
                    "section_id": 1,
                    "section_ref": "xml:root",
                    "text": text,
                })

        return sections