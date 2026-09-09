"""
run_b2_no_consistency.py
----------------------------
Baseline B2 (paper text): "generating TRs from the demand model only
without any consistency analysis" -- Stage 3 is skipped ENTIRELY (not
just cross-document checks; intra-source too), so Stage 4 generates
with an empty consistency_reports list and can produce zero Cat.5
(consistency-driven) TRs.

Run on auth_system_srs, matching B1's document choice, so B1/B2/Full
are all directly comparable on the same document.

Usage:
    python run_b2_no_consistency.py --source-id auth_system_srs
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from schemas.demand_schema import DemandModel
from src.generation.tr_generator import TRGenerator
from src.evaluation.tr_reviewer import TRReviewer
from src.evaluation.feedback_loop import FeedbackLoop

DIRS = {
    "tr": "data/processed/tr_doc",
    "reviews": "data/processed/evaluation/reviews",
    "final": "ablation_results",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id", default="auth_system_srs")
    ap.add_argument("--no-feedback", action="store_true")
    args = ap.parse_args()

    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

    demand_path = "data/processed/demand_model/{}.json".format(args.source_id)
    with open(demand_path, encoding="utf-8") as f:
        demand = DemandModel(**json.load(f))

    generator = TRGenerator()
    reviewer = TRReviewer()
    feedback = FeedbackLoop()

    print("[B2] Generating TRs for {} with Stage 3 SKIPPED (consistency_reports=[]) ...".format(
        args.source_id
    ))
    t0 = time.time()
    doc = generator.generate(demand, consistency_reports=[], verbose=False)
    print("  {} TRs generated  ({:.0f}s)".format(doc.total_count, time.time() - t0))

    cat5_count = sum(1 for t in doc.trs if t.category.value == "consistency_driven")
    print("  Cat.5 (consistency-driven) TRs: {} (expect 0, by construction)".format(cat5_count))

    print("[B2] Reviewing ...")
    review = reviewer.review(doc, demand, verbose=False)
    m = review["metrics"]
    print("  Q_H={:.2f} Q_A={:.2f} Trace={:.2f} Coverage={:.2f}".format(
        m["q_h"], m["q_a"], m["traceability"], m["coverage"]
    ))

    if not args.no_feedback and not review["passed"]:
        print("[B2] Running feedback loop ...")
        doc = feedback.run(doc, review, demand, max_iterations=2, verbose=True)
        review = reviewer.review(doc, demand, verbose=False)
        m = review["metrics"]
        print("  After feedback: Q_H={:.2f} Q_A={:.2f} Trace={:.2f} Coverage={:.2f}".format(
            m["q_h"], m["q_a"], m["traceability"], m["coverage"]
        ))

    out_path = os.path.join(DIRS["final"], "b2_{}_tr_doc.json".format(args.source_id))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc.model_dump(), f, indent=2, ensure_ascii=False)

    result = {
        "source_id": args.source_id,
        "total_trs": doc.total_count,
        "cat5_trs": cat5_count,
        "metrics": review["metrics"],
        "passed": review["passed"],
    }
    with open(os.path.join(DIRS["final"], "b2_{}_result.json".format(args.source_id)), "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print("  B2 result for {}".format(args.source_id))
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print("\nWrote {}".format(out_path))


if __name__ == "__main__":
    main()
