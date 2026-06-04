"""
base_loader.py
Unified output schema for all loaders.
Every loader returns List[Dict] where each dict contains:
    section_id  : int   (1-based sequential index)
    section_ref : str   (human-readable position, e.g. "page 3" or "H1: Intro")
    text        : str   (cleaned section text)
DocumentManager injects source_id and artifact_type after load.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import re


class BaseLoader(ABC):

    @abstractmethod
    def load(self, file_path: str) -> List[Dict[str, Any]]:
        pass

    @staticmethod
    def _clean(text: str) -> str:
        """Strip excess whitespace, collapse 3+ newlines to 2."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = '\n'.join(line.rstrip() for line in text.splitlines())
        return text.strip()