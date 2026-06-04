"""
test_stage2_demand.py
Stage 2: Demand Understanding
Run: python test_stage2_demand.py
"""
import sys, os, json, time
sys.path.insert(0, os.getcwd())

from schemas.entity_schema import ExtractedEntity, ArtifactType, EntityType
from schemas.demand_schema  import DemandModel
from src.reasoning.demand_modeller import DemandModeller

ENTITIES_DIR      = "data/processed/entities"
DEMAND_DIR        = "data/processed/demand_model"
USE_FULL_ENTITIES = True   # full extraction is done -- use complete files

DOCUMENTS = [
    {"source_id": "nasa_srs_v1",        "artifact_type": "srs"},
    {"source_id": "swagger_petstore",   "artifact_type": "runtime"},
    {"source_id": "auth_system_design", "artifact_type": "design"},
    {"source_id": "bash_user_manual",   "artifact_type": "user_manual"},
]


def load_entities(source_id: str) -> list:
    suffix = "" if USE_FULL_ENTITIES else "_sample"
    path   = os.path.join(ENTITIES_DIR, "{}{}.json".format(source_id, suffix))
    if not os.path.exists(path):
        # Try sample as fallback
        path = os.path.join(ENTITIES_DIR, "{}_sample.json".format(source_id))
        if not os.path.exists(path):
            return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entities = []
    for item in data:
        try:
            if isinstance(item.get("artifact_type"), str):
                item["artifact_type"] = ArtifactType(item["artifact_type"])
            if isinstance(item.get("entity_type"), str):
                item["entity_type"] = EntityType(item["entity_type"])
            entities.append(ExtractedEntity(**item))
        except Exception:
            continue
    return entities


def divider(title):
    print("\n" + "=" * 60)
    print("  " + title)
    print("=" * 60)


def print_model_summary(model: DemandModel):
    print("\n  System objectives    : {}".format(len(model.system_objectives)))
    for o in model.system_objectives[:4]:
        print("    - {}".format(o[:72]))

    print("\n  User roles           : {}".format(len(model.user_roles)))
    for r in model.user_roles[:6]:
        print("    - {}".format(r))

    print("\n  Use cases            : {}".format(len(model.use_cases)))
    for uc in model.use_cases[:3]:
        print("    [{}] {} -> {}".format(
            uc.use_case_id, uc.actor[:18], uc.goal[:52]
        ))

    print("\n  Functional reqs      : {}".format(len(model.functional_reqs)))
    for fr in model.functional_reqs[:4]:
        print("    [{}] ({}) {}".format(
            fr.req_id, fr.priority, fr.description[:62]
        ))

    print("\n  NFR constraints      : {}".format(len(model.nfr_constraints)))
    for c in model.nfr_constraints[:4]:
        print("    [{}] ({}) {}".format(
            c.constraint_id, c.category, c.description[:55]
        ))

    print("\n  Runtime constraints  : {}".format(len(model.runtime_constraints)))
    for c in model.runtime_constraints[:2]:
        print("    [{}] {}".format(c.constraint_id, c.description[:60]))

    print("\n  Glossary terms       : {}".format(len(model.glossary)))
    for term, defn in list(model.glossary.items())[:3]:
        print("    {}: {}".format(term[:20], defn[:50]))

    print("\n  Exception handlers   : {}".format(len(model.exception_handlers)))
    for ex in model.exception_handlers[:3]:
        print("    - {}".format(ex[:70]))


def main():
    print("\nAI4RE-Test  |  Stage 2: Demand Understanding")
    print("-" * 60)
    print("Using: {} entity files".format(
        "FULL" if USE_FULL_ENTITIES else "SAMPLE"
    ))

    modeller = DemandModeller()
    os.makedirs(DEMAND_DIR, exist_ok=True)

    results = {"passed": 0, "skipped": 0}

    for cfg in DOCUMENTS:
        source_id = cfg["source_id"]
        divider("{} [{}]".format(source_id, cfg["artifact_type"]))

        entities = load_entities(source_id)
        if not entities:
            print("  SKIP -- no entity file found for: {}".format(source_id))
            results["skipped"] += 1
            continue

        print("\n  Loaded {} entities from {}{}".format(
            len(entities), source_id,
            ".json" if USE_FULL_ENTITIES else "_sample.json"
        ))

        t0    = time.time()
        model = modeller.build_demand_model(
            entities  = entities,
            source_id = source_id,
            verbose   = True,
        )
        elapsed = time.time() - t0

        print("\n  Build time: {:.1f}s".format(elapsed))
        print_model_summary(model)

        out_path = modeller.save_demand_model(model, DEMAND_DIR)
        size_kb  = os.path.getsize(out_path) / 1024
        print("\n  Saved: {}  ({:.1f} KB)".format(out_path, size_kb))
        results["passed"] += 1

    print("\n" + "=" * 60)
    print("  Stage 2 complete.")
    print("  Passed: {}  Skipped: {}".format(
        results["passed"], results["skipped"]
    ))
    print("  Demand models in: {}".format(DEMAND_DIR))
    print("=" * 60)
    print()
    if results["passed"] == len(DOCUMENTS):
        print("Stage 2 done. Next: python test_stage3_consistency.py")
    print()


if __name__ == "__main__":
    main()