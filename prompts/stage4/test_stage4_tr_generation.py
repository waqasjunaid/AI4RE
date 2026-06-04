"""
test_stage4_tr_generation.py
Stage 4: Test Requirement Generation
Run: python test_stage4_tr_generation.py
"""
import sys, os, json, time
sys.path.insert(0, os.getcwd())

from schemas.demand_schema      import DemandModel
from schemas.consistency_schema import ConsistencyReport
from schemas.tr_schema          import TRDoc, TRCategory
from src.generation.tr_generator import TRGenerator

DEMAND_DIR      = "data/processed/demand_model"
CONSISTENCY_DIR = "data/processed/consistency"
TR_DIR          = "data/processed/tr_doc"

SOURCES = [
    "nasa_srs_v1",
    "swagger_petstore",
    "auth_system_design",
    "bash_user_manual",
]


def load_demand(source_id):
    path = os.path.join(DEMAND_DIR, "{}.json".format(source_id))
    with open(path, "r", encoding="utf-8") as f:
        return DemandModel(**json.load(f))


def load_consistency():
    path = os.path.join(CONSISTENCY_DIR, "consistency_report.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [ConsistencyReport(**r) for r in data]


def divider(title):
    print("\n" + "=" * 60)
    print("  " + title)
    print("=" * 60)


def print_tr_summary(doc: TRDoc):
    from collections import Counter
    cats = Counter(tr.category.value for tr in doc.trs)
    print("\n  TR breakdown by category:")
    for cat, count in sorted(cats.items()):
        print("    {:30s}: {}".format(cat, count))

    print("\n  Sample TRs (first in each category):")
    shown = set()
    for tr in doc.trs:
        cat = tr.category.value
        if cat not in shown:
            shown.add(cat)
            print("\n  [{}] {}".format(tr.tr_id, cat))
            print("  TR-H: {}".format(tr.tr_h[:90]))
            if tr.tr_a.preconditions:
                print("  Pre : {}".format(tr.tr_a.preconditions[0][:60]))
            if tr.tr_a.stimuli:
                print("  Stim: {}".format(tr.tr_a.stimuli[0][:60]))
            if tr.tr_a.expected_outputs:
                print("  Exp : {}".format(tr.tr_a.expected_outputs[0][:60]))


def main():
    print("\nAI4RE-Test  |  Stage 4: Test Requirement Generation")
    print("-" * 60)

    consistency_reports = load_consistency()
    total_issues = sum(len(r.issues) for r in consistency_reports)
    print("Loaded {} consistency reports ({} issues)".format(
        len(consistency_reports), total_issues
    ))

    generator = TRGenerator()
    os.makedirs(TR_DIR, exist_ok=True)

    results = {"passed": 0, "skipped": 0}
    grand_total = 0

    for source_id in SOURCES:
        path = os.path.join(DEMAND_DIR, "{}.json".format(source_id))
        if not os.path.exists(path):
            divider("{} -- SKIP".format(source_id))
            print("  Demand model not found. Run test_stage2_demand.py first.")
            results["skipped"] += 1
            continue

        divider(source_id)
        demand = load_demand(source_id)
        print("\n  Demand model: {} func_reqs, {} NFR, {} exceptions".format(
            len(demand.functional_reqs),
            len(demand.nfr_constraints),
            len(demand.exception_handlers),
        ))

        t0  = time.time()
        doc = generator.generate(
            demand_model        = demand,
            consistency_reports = consistency_reports,
            verbose             = True,
        )
        elapsed = time.time() - t0

        print("\n  Total TRs generated : {}  ({:.1f}s)".format(
            doc.total_count, elapsed
        ))
        print_tr_summary(doc)

        out_path = generator.save_tr_doc(doc, TR_DIR)
        size_kb  = os.path.getsize(out_path) / 1024
        print("\n  Saved: {}  ({:.1f} KB)".format(out_path, size_kb))

        grand_total += doc.total_count
        results["passed"] += 1

    print("\n" + "=" * 60)
    print("  Stage 4 complete.")
    print("  Passed: {}  Skipped: {}".format(
        results["passed"], results["skipped"]
    ))
    print("  Total TRs generated: {}".format(grand_total))
    print("  TR documents in: {}".format(TR_DIR))
    print("=" * 60)
    print()
    print("Next: python test_stage5_review.py")
    print()


if __name__ == "__main__":
    main()
