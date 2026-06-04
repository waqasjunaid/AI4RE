"""
semantic_chunker.py
Stage 1 chunker. Splits loaded sections into token-bounded chunks
with overlap, respecting section boundaries.

Chunk config per artifact_type:
    srs          : max 512 tokens, 51 overlap, sentence split
    user_manual  : max 512 tokens, 51 overlap, sentence split
    runtime      : max 256 tokens, 26 overlap, line split
    design       : max 384 tokens, 38 overlap, element split

chunk_id format: {source_id}__s{section_id:04d}__c{chunk_idx:04d}
"""

import re
import json
import os
import sys
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
from schemas.entity_schema import DocumentChunk, ArtifactType


# Token counting: use tiktoken if available, else word count
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text):
        return len(_enc.encode(text))
except ImportError:
    def _count_tokens(text):
        return len(text.split())


CHUNK_CONFIG = {
    "srs":         {"max_tokens": 512, "overlap_tokens": 51, "split_on": "sentence"},
    "user_manual": {"max_tokens": 512, "overlap_tokens": 51, "split_on": "sentence"},
    "runtime":     {"max_tokens": 256, "overlap_tokens": 26, "split_on": "line"},
    "design":      {"max_tokens": 384, "overlap_tokens": 38, "split_on": "element"},
}
DEFAULT_CONFIG = {"max_tokens": 512, "overlap_tokens": 51, "split_on": "sentence"}


class SemanticChunker:

    def chunk_sections(self, sections: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """Chunk a list of loaded sections into DocumentChunk objects."""
        if not sections:
            return []
        artifact_type = sections[0]["artifact_type"]
        config = CHUNK_CONFIG.get(artifact_type, DEFAULT_CONFIG)
        all_chunks = []
        for section in sections:
            all_chunks.extend(self._chunk_section(section, config))
        return all_chunks

    def _chunk_section(self, section: Dict, config: Dict) -> List[DocumentChunk]:
        text          = section["text"]
        source_id     = section["source_id"]
        artifact_type = section["artifact_type"]
        section_id    = section["section_id"]
        section_ref   = section["section_ref"]
        max_tok       = config["max_tokens"]
        overlap_tok   = config["overlap_tokens"]
        split_mode    = config["split_on"]

        # Fits in one chunk?
        if _count_tokens(text) <= max_tok:
            return [DocumentChunk(
                chunk_id      = "{}__s{:04d}__c0001".format(source_id, section_id),
                source_id     = source_id,
                artifact_type = ArtifactType(artifact_type),
                content       = text,
                page_ref      = section_ref,
                metadata      = {
                    "section_id":  section_id,
                    "chunk_index": 1,
                    "token_count": _count_tokens(text),
                    "overlap":     False,
                }
            )]

        units      = self._split_into_units(text, split_mode)
        chunk_list = self._group_with_overlap(units, max_tok, overlap_tok)

        result = []
        for cidx, (chunk_text, is_overlap) in enumerate(chunk_list, start=1):
            result.append(DocumentChunk(
                chunk_id      = "{}__s{:04d}__c{:04d}".format(source_id, section_id, cidx),
                source_id     = source_id,
                artifact_type = ArtifactType(artifact_type),
                content       = chunk_text,
                page_ref      = section_ref,
                metadata      = {
                    "section_id":  section_id,
                    "chunk_index": cidx,
                    "token_count": _count_tokens(chunk_text),
                    "overlap":     is_overlap,
                }
            ))
        return result

    def _split_into_units(self, text: str, mode: str) -> List[str]:
        if mode == "sentence":
            units = re.split(r'(?<=[.!?])\s+', text)
        elif mode == "line":
            units = [l for l in text.splitlines() if l.strip()]
        elif mode == "element":
            units = re.split(r'(?=\[)', text)
        else:
            units = re.split(r'(?<=[.!?])\s+', text)
        return [u.strip() for u in units if u.strip()]

    def _group_with_overlap(
        self, units: List[str], max_tok: int, overlap_tok: int
    ) -> List[Tuple[str, bool]]:
        chunks = []
        i = 0
        prev_tail = ""

        while i < len(units):
            current = []
            cur_tok = 0

            if prev_tail:
                current.append(prev_tail)
                cur_tok = _count_tokens(prev_tail)

            while i < len(units):
                ut = _count_tokens(units[i])
                if cur_tok + ut > max_tok and current:
                    break
                current.append(units[i])
                cur_tok += ut
                i += 1

            chunks.append((" ".join(current).strip(), bool(prev_tail)))

            # Build tail for next chunk overlap
            tail = []
            tail_tok = 0
            for u in reversed(current):
                ut = _count_tokens(u)
                if tail_tok + ut <= overlap_tok:
                    tail.insert(0, u)
                    tail_tok += ut
                else:
                    break
            prev_tail = " ".join(tail).strip() if tail else ""

        return chunks

    def save_chunks(self, chunks: List[DocumentChunk], output_dir: str) -> str:
        """Save chunks to data/processed/chunks/{source_id}.json"""
        if not chunks:
            return ""
        source_id = chunks[0].source_id
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "{}.json".format(source_id))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in chunks], f, indent=2, ensure_ascii=False)
        return out_path