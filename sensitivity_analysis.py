"""
sensitivity_analysis.py
--------------------------
Reviewer comment R3a: quantifies how the DEMAND_ITEM_CAP affects the
demand model and downstream results on bash_user_manual -- the document
most affected by capping, since it has by far the most entities (2,390)
of any evaluation document.

Runs Stage 2 (demand modelling) on bash_user_manual at several cap
values, using the AI4RE_DEMAND_ITEM_CAP environment variable added to
demand_modeller.py, and reports how demand model size and composition
change as the cap increases.

Usage:
    python sensitivity_analysis.py --caps 10,20,30,50 --source-id bash_user_manual

Requires: entities/bash_user_manual.json to already exist (Stage 1 must
have already run).
"""
import argparse
import json
import os
import subprocess
import sys

ENTITIES_DIR = "data/processed/entities"
OUT_DIR = "sensitivity_results"


def run_stage2_at_cap(source_id, cap, artifact_type):
    """Runs Stage 2 demand modelling for one document at one cap value,
    via a small inline Python invocation (avoids touching test_stage2_demand.py,
    which has its own skip-if-exists logic we don't want here).

    IMPORTANT: max_tokens is scaled proportionally with cap. A fixed
    output token budget (the module default, 1500) is sized for the
    default cap of 10; asking the model to return more items (larger
    cap) in the SAME fixed budget risks silent JSON truncation --
    exactly the use_cases=0 anomaly seen at cap=20/30 in initial runs
    of this script, while cap=10 and cap=50 both happened to produce
    valid output. Scaling max_tokens with cap removes this confound so
    the sensitivity curve reflects genuine content availability, not an
    unrelated token-budget artifact."""
    env = os.environ.copy()
    env["AI4RE_DEMAND_ITEM_CAP"] = str(cap)
    # ~150 tokens/item is a rough but generous estimate for a fully
    # populated use-case JSON object (6 fields, several of them arrays);
    # floor at the original 1500 default so small caps are unaffected.
    env["AI4RE_MAX_TOKENS"] = str(max(1500, cap * 150))

    script = """
import sys, os, json
sys.path.insert(0, os.getcwd())
from schemas.entity_schema import ExtractedEntity, ArtifactType, EntityType
from src.reasoning.demand_modeller import DemandModeller

with open("{entities_path}", encoding="utf-8") as f:
    data = json.load(f)
entities = []
for item in data:
    if isinstance(item.get("artifact_type"), str):
        item["artifact_type"] = ArtifactType(item["artifact_type"])
    if isinstance(item.get("entity_type"), str):
        item["entity_type"] = EntityType(item["entity_type"])
    entities.append(ExtractedEntity(**item))

modeller = DemandModeller()
model = modeller.build_demand_model(entities=entities, source_id="{source_id}", verbose=False)
print(json.dumps(model.model_dump()))
""".format(
        entities_path=os.path.join(ENTITIES_DIR, source_id + ".json"),
        source_id=source_id,
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env, capture_output=True, text=True, timeout=3600
    )
    if result.returncode != 0:
        print("  ERROR at cap={}: {}".format(cap, result.stderr[-2000:]))
        return None
    # The demand model JSON is the last line of stdout (in case of any
    # stray prints from dependencies).
    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    print("  ERROR at cap={}: could not find JSON output".format(cap))
    return None


def summarize(model):
    if model is None:
        return None
    return {
        "user_roles": len(model.get("user_roles", [])),
        "use_cases": len(model.get("use_cases", [])),
        "functional_reqs": len(model.get("functional_reqs", [])),
        "nfr_constraints": len(model.get("nfr_constraints", [])),
        "runtime_constraints": len(model.get("runtime_constraints", [])),
        "exception_handlers": len(model.get("exception_handlers", [])),
        "glossary_terms": len(model.get("glossary", {})),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="10,20,30,50",
                     help="comma-separated cap values to test")
    ap.add_argument("--source-id", default="bash_user_manual")
    ap.add_argument("--artifact-type", default="user_manual")
    args = ap.parse_args()

    caps = [int(c) for c in args.caps.split(",")]
    os.makedirs(OUT_DIR, exist_ok=True)

    entities_path = os.path.join(ENTITIES_DIR, args.source_id + ".json")
    if not os.path.exists(entities_path):
        raise SystemExit("Entities file not found: {}. Run Stage 1 extraction first.".format(entities_path))

    all_results = {}
    for cap in caps:
        print("Running Stage 2 for {} at DEMAND_ITEM_CAP={} ...".format(args.source_id, cap))
        model = run_stage2_at_cap(args.source_id, cap, args.artifact_type)
        if model is not None:
            out_path = os.path.join(OUT_DIR, "{}_cap{}.json".format(args.source_id, cap))
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(model, f, indent=2, ensure_ascii=False)
            all_results[cap] = summarize(model)
            print("  -> saved {}".format(out_path))
        else:
            all_results[cap] = None

    print("\n" + "=" * 78)
    print("  Sensitivity analysis: demand model size vs. DEMAND_ITEM_CAP ({})".format(args.source_id))
    print("=" * 78)
    header = "{:>6s}".format("cap")
    fields = ["user_roles", "use_cases", "functional_reqs", "nfr_constraints",
              "runtime_constraints", "exception_handlers", "glossary_terms"]
    for f in fields:
        header += " {:>12s}".format(f[:12])
    print(header)
    for cap in caps:
        row = "{:>6d}".format(cap)
        s = all_results[cap]
        if s is None:
            row += "  (failed)"
        else:
            for f in fields:
                row += " {:>12d}".format(s[f])
        print(row)

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("\nWrote {}/summary.json".format(OUT_DIR))
    print("\nNext step: run Stage 4/5 on each {}_cap*.json to see how TR count".format(args.source_id))
    print("and coverage change with cap size (see run_stage4_5_at_caps.py).")


if __name__ == "__main__":
    main()