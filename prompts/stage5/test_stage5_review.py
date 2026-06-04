"""
test_stage5_review.py  --  Stage 5: Test Tracking and Review
Run: python test_stage5_review.py
"""
import sys, os, json, time
sys.path.insert(0, os.getcwd())

from schemas.demand_schema import DemandModel
from schemas.tr_schema     import TRDoc, TestRequirement, TR_A, TRCategory
from src.evaluation.tr_reviewer import TRReviewer

DEMAND_DIR = "data/processed/demand_model"
TR_DIR     = "data/processed/tr_doc"
REVIEW_DIR = "data/processed/evaluation/reviews"

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


def load_tr_doc(source_id):
    path = os.path.join(TR_DIR, "{}_tr_doc.json".format(source_id))
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Rebuild TRDoc from JSON
    trs = []
    for item in raw.get("trs", []):
        try:
            tr_a_data = item.get("tr_a", {})
            tr_a = TR_A(
                tr_id              = tr_a_data.get("tr_id", item["tr_id"]),
                preconditions      = tr_a_data.get("preconditions", []),
                stimuli            = tr_a_data.get("stimuli", []),
                expected_outputs   = tr_a_data.get("expected_outputs", []),
                coverage_criterion = tr_a_data.get("coverage_criterion", ""),
                priority           = tr_a_data.get("priority", "medium"),
                source_req_ids     = tr_a_data.get("source_req_ids", []),
            )
            trs.append(TestRequirement(
                tr_id          = item["tr_id"],
                category       = TRCategory(item["category"]),
                source_id      = item["source_id"],
                tr_h           = item["tr_h"],
                tr_a           = tr_a,
                source_req_ids = item.get("source_req_ids", []),
            ))
        except Exception:
            continue

    return TRDoc(
        source_id   = raw["source_id"],
        trs         = trs,
        tr_h_list   = [t.tr_h for t in trs],
        tr_a_list   = [t.tr_a for t in trs],
        total_count = len(trs),
    )


def divider(title):
    print("\n" + "=" * 60)
    print("  " + title)
    print("=" * 60)


def print_review(review: dict):
    m = review["metrics"]
    print("\n  Metrics:")
    print("    Q_H  (human acceptance)  : {:.2f}  (target >= {:.2f})  {}".format(
        m["q_h"], 0.78, "PASS" if m["q_h"] >= 0.78 else "FAIL"
    ))
    print("    Q_A  (schema validity)   : {:.2f}  (target >= {:.2f})  {}".format(
        m["q_a"], 0.85, "PASS" if m["q_a"] >= 0.85 else "FAIL"
    ))
    print("    Traceability             : {:.2f}  (target >= {:.2f})  {}".format(
        m["traceability"], 0.90, "PASS" if m["traceability"] >= 0.90 else "FAIL"
    ))
    print("    Coverage                 : {:.2f}  (target >= {:.2f})  {}".format(
        m["coverage"], 0.95, "PASS" if m["coverage"] >= 0.95 else "FAIL"
    ))

    dims = review.get("dimensions", {})

    dup = dims.get("quality", {}).get("duplicate_count", 0)
    unt = dims.get("quality", {}).get("untestable_count", 0)
    print("\n  Quality defects  : {} duplicates, {} untestable".format(dup, unt))

    uncov = dims.get("coverage", {}).get("uncovered", [])
    if uncov:
        print("\n  Uncovered demand items:")
        for item in uncov[:4]:
            print("    - {}".format(item[:70]))

    flagged = review.get("flagged_tr_ids", [])
    if flagged:
        print("\n  Flagged TR-IDs   : {}".format(", ".join(flagged[:6])))

    notes = review.get("correction_notes", [])
    if notes:
        print("\n  Correction notes:")
        for note in notes:
            print("    * {}".format(note[:80]))

    status = "PASS" if review["passed"] else "NEEDS REVIEW"
    print("\n  Overall status   : {}".format(status))


def main():
    print("\nAI4RE-Test  |  Stage 5: Test Tracking and Review")
    print("-" * 60)

    reviewer = TRReviewer()
    os.makedirs(REVIEW_DIR, exist_ok=True)

    results     = {"passed": 0, "needs_review": 0, "skipped": 0}
    grand_total = 0

    for source_id in SOURCES:
        divider(source_id)

        tr_doc = load_tr_doc(source_id)
        if tr_doc is None:
            print("  SKIP -- TR doc not found. Run test_stage4_tr_generation.py.")
            results["skipped"] += 1
            continue

        demand = load_demand(source_id)
        print("\n  TR doc loaded   : {} TRs".format(tr_doc.total_count))

        t0     = time.time()
        review = reviewer.review(tr_doc, demand, verbose=True)
        elapsed= time.time() - t0

        print("\n  Review time: {:.1f}s".format(elapsed))
        print_review(review)

        out_path = reviewer.save_review(review, REVIEW_DIR)
        size_kb  = os.path.getsize(out_path) / 1024
        print("\n  Saved: {}  ({:.1f} KB)".format(out_path, size_kb))

        grand_total += tr_doc.total_count
        if review["passed"]:
            results["passed"] += 1
        else:
            results["needs_review"] += 1

    print("\n" + "=" * 60)
    print("  Stage 5 complete.")
    print("  Pass: {}  Needs review: {}  Skipped: {}".format(
        results["passed"], results["needs_review"], results["skipped"]
    ))
    print("  Total TRs reviewed : {}".format(grand_total))
    print("  Review files in    : {}".format(REVIEW_DIR))
    print("=" * 60)
    print()

    # Final pipeline summary
    print("PIPELINE COMPLETE -- All 5 stages done.")
    print()
    print("Output artifacts:")
    print("  data/processed/chunks/          Stage 1a: document chunks")
    print("  data/processed/entities/        Stage 1b: extracted entities")
    print("  data/processed/demand_model/    Stage 2:  demand models")
    print("  data/processed/consistency/     Stage 3:  consistency issues")
    print("  data/processed/tr_doc/          Stage 4:  test requirement docs")
    print("  data/processed/evaluation/      Stage 5:  quality reviews")
    print()


if __name__ == "__main__":
    main()