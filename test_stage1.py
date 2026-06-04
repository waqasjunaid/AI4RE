"""
test_stage1.py
Tests Stage 1 (Document Parsing + Chunking) with real documents.
Run from the AI4RE-Test/ root directory:
    python test_stage1.py
"""

import sys
import os
import time

sys.path.insert(0, os.getcwd())

from src.ingestion.document_manager import DocumentManager
from src.chunking.semantic_chunker import SemanticChunker

DOCUMENTS = [
    {
        "file_path":     "data/raw/srs/nasa_srs.pdf",
        "artifact_type": "srs",
        "source_id":     "nasa_srs_v1",
    },
    {
        "file_path":     "data/raw/user_manual/libreoffice_writer.txt",
        "artifact_type": "user_manual",
        "source_id":     "bash_user_manual",
    },
    {
        "file_path":     "data/raw/runtime/swagger_petstore.json",
        "artifact_type": "runtime",
        "source_id":     "swagger_petstore",
    },
    {
        "file_path":     "data/raw/design_docs/auth_system_design.puml",
        "artifact_type": "design",
        "source_id":     "auth_system_design",
    },
]

CHUNKS_OUTPUT_DIR = "data/processed/chunks"


def divider(title):
    print("\n" + "=" * 60)
    print("  " + title)
    print("=" * 60)


def test_document(cfg, manager, chunker):
    path          = cfg["file_path"]
    artifact_type = cfg["artifact_type"]
    source_id     = cfg["source_id"]

    if not os.path.exists(path):
        print("  SKIP -- file not found: {}".format(path))
        return False

    divider("{} [{}]".format(source_id, artifact_type))

    # ---- Step 1: Load -----------------------------------------------
    print("\n[1] Loading {} ...".format(path))
    t0 = time.time()
    sections = manager.load_document(path, artifact_type, source_id)
    print("    Sections loaded : {}  ({:.1f}s)".format(
        len(sections), time.time() - t0
    ))

    if not sections:
        print("    RESULT: FAILED -- no sections extracted.")
        return False

    required = {"section_id", "section_ref", "source_id", "artifact_type", "text"}
    for s in sections:
        missing = required - set(s.keys())
        if missing:
            print("    SCHEMA ERROR -- missing fields: {}".format(missing))
            return False
    print("    Schema check    : OK")

    print("\n    First 3 sections:")
    for s in sections[:3]:
        preview = s["text"][:90].replace("\n", " ")
        print("      [{}] {}".format(s["section_id"], s["section_ref"]))
        print("           \"{}\"".format(preview))

    # ---- Step 2: Chunk ----------------------------------------------
    print("\n[2] Chunking ...")
    t0 = time.time()
    chunks = chunker.chunk_sections(sections)
    elapsed = time.time() - t0
    print("    Chunks produced : {}  ({:.1f}s)".format(len(chunks), elapsed))

    if not chunks:
        print("    RESULT: FAILED -- no chunks produced.")
        return False

    token_counts  = [c.metadata["token_count"] for c in chunks]
    overlap_count = sum(1 for c in chunks if c.metadata["overlap"])
    tiny          = sum(1 for t in token_counts if t < 20)

    print("    Token range     : {} - {}".format(
        min(token_counts), max(token_counts)
    ))
    print("    Mean tokens     : {:.0f}".format(
        sum(token_counts) / len(token_counts)
    ))
    print("    Overlap chunks  : {}".format(overlap_count))

    if tiny > len(chunks) * 0.4:
        print("    WARNING: {}/{} chunks under 20 tokens".format(
            tiny, len(chunks)
        ))

    print("\n    First 2 chunks:")
    for c in chunks[:2]:
        print("      chunk_id     : {}".format(c.chunk_id))
        print("      page_ref     : {}".format(c.page_ref))
        print("      token_count  : {}".format(c.metadata["token_count"]))
        print("      content[:80] : \"{}\"".format(
            c.content[:80].replace("\n", " ")
        ))
        print()

    # ---- Step 3: Save -----------------------------------------------
    out_path = chunker.save_chunks(chunks, CHUNKS_OUTPUT_DIR)
    size_kb  = os.path.getsize(out_path) / 1024
    print("[3] Saved to  : {}  ({:.1f} KB)".format(out_path, size_kb))
    return True


def main():
    print("\nAI4RE-Test  |  Stage 1: Document Parsing + Chunking")
    print("-" * 60)

    manager = DocumentManager()
    chunker = SemanticChunker()
    os.makedirs(CHUNKS_OUTPUT_DIR, exist_ok=True)

    results = {"passed": 0, "skipped": 0, "failed": 0}

    for cfg in DOCUMENTS:
        try:
            if not os.path.exists(cfg["file_path"]):
                results["skipped"] += 1
                divider("{} [{}]".format(
                    cfg["source_id"], cfg["artifact_type"]
                ))
                print("  SKIP -- {}".format(cfg["file_path"]))
            else:
                ok = test_document(cfg, manager, chunker)
                if ok:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
        except Exception as e:
            results["failed"] += 1
            print("\n  ERROR in {}: {}".format(cfg["source_id"], e))
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("  Stage 1 complete.")
    print("  Passed: {}   Skipped: {}   Failed: {}".format(
        results["passed"], results["skipped"], results["failed"]
    ))
    print("  Chunk files in: {}".format(CHUNKS_OUTPUT_DIR))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()