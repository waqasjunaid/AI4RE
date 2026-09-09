"""
test_stage1_extraction.py
Run: python test_stage1_extraction.py
Add --debug to print raw model responses for troubleshooting.
"""
import sys, os, json, time
from collections import Counter

sys.path.insert(0, os.getcwd())

from src.extraction.entity_extractor import EntityExtractor, MODEL_NAME, OLLAMA_HOST
from schemas.entity_schema import DocumentChunk, ArtifactType

# -- Config -------------------------------------------------------------------
CHUNKS_DIR     = "data/processed/chunks"
ENTITIES_DIR   = "data/processed/entities"
CHUNKS_PER_DOC = None   # change to None for full extraction

DEBUG = "--debug" in sys.argv   # python test_stage1_extraction.py --debug
FORCE = "--force" in sys.argv   # re-extract even if entity file already exists

DOCUMENTS = [
    {"chunk_file": "nasa_srs_v1.json",        "source_id": "nasa_srs_v1"},
    {"chunk_file": "bash_user_manual.json",   "source_id": "bash_user_manual"},
    {"chunk_file": "swagger_petstore.json",   "source_id": "swagger_petstore"},
    {"chunk_file": "auth_system_design.json", "source_id": "auth_system_design"},
    # -- Same-system bundle added in response to reviewer comment R1 --
    {"chunk_file": "auth_system_srs.json",      "source_id": "auth_system_srs"},
    {"chunk_file": "auth_system_manual.json",   "source_id": "auth_system_manual"},
    {"chunk_file": "auth_system_runtime.json",  "source_id": "auth_system_runtime"},
]

# -- Helpers ------------------------------------------------------------------

def divider(title):
    print("\n" + "=" * 60)
    print("  " + title)
    print("=" * 60)


def load_chunks(chunk_file):
    path = os.path.join(CHUNKS_DIR, chunk_file)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = []
    for item in data:
        if isinstance(item.get("artifact_type"), str):
            item["artifact_type"] = ArtifactType(item["artifact_type"])
        chunks.append(DocumentChunk(**item))
    return chunks


def print_summary(entities):
    if not entities:
        print("  No entities extracted.")
        return
    counts = Counter(e.entity_type.value for e in entities)
    print("\n  Entity type breakdown:")
    for etype, count in sorted(counts.items(), key=lambda x: -x[1]):
        print("    {:25s}: {}".format(etype, count))
    print("\n  Sample entities (first 2 per type):")
    shown = {}
    for e in entities:
        t = e.entity_type.value
        shown.setdefault(t, 0)
        if shown[t] < 2:
            print("    [{:20s}] ({:.2f})  {}".format(
                t, e.confidence_score, e.content[:65]
            ))
            shown[t] += 1


# -- Debug extractor wrapper --------------------------------------------------

class DebugExtractor:
    """Wraps EntityExtractor to print raw responses when --debug is set."""

    def __init__(self, extractor):
        self._e = extractor

    def extract_from_chunks(self, chunks, max_chunks=None, verbose=True):
        if max_chunks is not None:
            chunks = chunks[:max_chunks]

        all_entities = []
        for i, chunk in enumerate(chunks, start=1):
            if verbose:
                print("  [{:3d}/{}] {} ... ".format(
                    i, len(chunks), chunk.chunk_id
                ), end="", flush=True)
            try:
                t0 = time.time()

                # Build and send prompt manually so we can print the raw response
                user_msg = self._e._build_user_message(chunk)
                raw      = self._e._call_ollama_chat(user_msg)
                entities = []

                if DEBUG:
                    print()
                    print("    --- RAW RESPONSE ---")
                    print(raw[:600])
                    print("    --- END RESPONSE ---")

                items = self._e._parse_json(raw)
                for item in items:
                    try:
                        from schemas.entity_schema import ExtractedEntity, EntityType
                        etype   = str(item.get("entity_type","")).strip()
                        content = str(item.get("content","")).strip()
                        score   = float(item.get("confidence_score", 0.8))
                        if etype and content:
                            entities.append(ExtractedEntity(
                                source_id        = chunk.source_id,
                                artifact_type    = chunk.artifact_type,
                                entity_type      = EntityType(etype),
                                content          = content,
                                page_ref         = chunk.page_ref,
                                confidence_score = max(0.0, min(1.0, score)),
                            ))
                    except (KeyError, ValueError):
                        continue

                elapsed = time.time() - t0
                all_entities.extend(entities)
                if verbose:
                    if not DEBUG:
                        print("{} entities  ({:.1f}s)".format(len(entities), elapsed))
                    else:
                        print("  => {} entities  ({:.1f}s)".format(len(entities), elapsed))
            except Exception as ex:
                if verbose:
                    print("ERROR: {}".format(ex))
                    if DEBUG:
                        import traceback; traceback.print_exc()

        return all_entities

    def save_entities(self, *args, **kwargs):
        return self._e.save_entities(*args, **kwargs)


# -- Main ---------------------------------------------------------------------

def main():
    print("\nAI4RE-Test  |  Stage 1 Extraction  |  {}".format(MODEL_NAME))
    print("-" * 60)
    print("Host   : {}".format(OLLAMA_HOST))
    print("Model  : {}".format(MODEL_NAME))
    print("Chunks : {}".format(CHUNKS_PER_DOC or "ALL"))
    print("Debug  : {}".format(DEBUG))

    try:
        base_extractor = EntityExtractor()
    except ConnectionError as e:
        print("\n{}".format(e))
        sys.exit(1)

    extractor = DebugExtractor(base_extractor)
    os.makedirs(ENTITIES_DIR, exist_ok=True)

    total_entities = 0
    results = {"passed": 0, "skipped": 0}

    for cfg in DOCUMENTS:
        divider("{} -- {} chunks".format(
            cfg["source_id"], CHUNKS_PER_DOC or "ALL"
        ))

        suffix_check = "_sample" if CHUNKS_PER_DOC else ""
        existing_path = os.path.join(
            ENTITIES_DIR, "{}{}.json".format(cfg["source_id"], suffix_check)
        )
        if os.path.exists(existing_path) and not FORCE:
            print("  SKIP -- entities already exist: {}".format(existing_path))
            print("  (pass --force to re-extract)")
            results["skipped"] += 1
            continue

        chunks = load_chunks(cfg["chunk_file"])
        if not chunks:
            print("  SKIP -- run test_stage1.py first")
            results["skipped"] += 1
            continue

        n = CHUNKS_PER_DOC if CHUNKS_PER_DOC else len(chunks)
        print("\n  {} chunks total, processing {}\n".format(len(chunks), n))

        t0       = time.time()
        entities = extractor.extract_from_chunks(
            chunks, max_chunks=CHUNKS_PER_DOC, verbose=True
        )
        elapsed  = time.time() - t0

        print("\n  Extracted: {} entities  ({:.1f}s)".format(
            len(entities), elapsed
        ))
        print_summary(entities)

        suffix   = "_sample" if CHUNKS_PER_DOC else ""
        out_path = extractor.save_entities(
            entities, ENTITIES_DIR,
            source_id=cfg["source_id"] + suffix
        )
        if out_path:
            print("\n  Saved: {}  ({:.1f} KB)".format(
                out_path, os.path.getsize(out_path)/1024
            ))

        total_entities += len(entities)
        results["passed"] += 1

    print("\n" + "=" * 60)
    print("  Done.  Passed:{}  Skipped:{}  Total entities:{}".format(
        results["passed"], results["skipped"], total_entities
    ))
    print("=" * 60)

    if CHUNKS_PER_DOC and total_entities > 0:
        print("\nLooks good! Set CHUNKS_PER_DOC = None for full extraction.")
    elif total_entities == 0:
        print("\nStill 0 entities. Run with --debug to see raw model output:")
        print("  python test_stage1_extraction.py --debug")
    print()


if __name__ == "__main__":
    main()