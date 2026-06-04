"""
run_pipeline.py
---------------
Full pipeline runner with feedback loop.
Runs all 5 stages end-to-end, then applies Stage 5 -> Stage 4
feedback for documents that need review.

Run: python run_pipeline.py

Options:
  --sources  nasa_srs_v1,swagger_petstore   (comma-separated, default=all)
  --no-feedback                              (skip feedback loop)
  --nohup                                    (print status only, no progress)
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.getcwd())

from schemas.demand_schema       import DemandModel
from schemas.consistency_schema  import ConsistencyReport
from schemas.tr_schema           import TRDoc, TestRequirement, TR_A, TRCategory
from src.reasoning.demand_modeller    import DemandModeller
from src.reasoning.consistency_checker import ConsistencyChecker
from src.generation.tr_generator      import TRGenerator
from src.evaluation.tr_reviewer       import TRReviewer
from src.evaluation.feedback_loop     import FeedbackLoop

# -- Directories --------------------------------------------------------------
DIRS = {
    "demand":      "data/processed/demand_model",
    "consistency": "data/processed/consistency",
    "tr":          "data/processed/tr_doc",
    "reviews":     "data/processed/evaluation/reviews",
    "final":       "data/processed/evaluation/final",
}

ALL_SOURCES = [
    "nasa_srs_v1",
    "swagger_petstore",
    "auth_system_design",
    "bash_user_manual",
]


# -- Loaders ------------------------------------------------------------------

def load_demand(sid):
    p = os.path.join(DIRS["demand"], "{}.json".format(sid))
    with open(p) as f: return DemandModel(**json.load(f))

def load_consistency():
    p = os.path.join(DIRS["consistency"], "consistency_report.json")
    if not os.path.exists(p): return []
    with open(p) as f: data = json.load(f)
    return [ConsistencyReport(**r) for r in data]

def load_tr_doc(sid):
    p = os.path.join(DIRS["tr"], "{}_tr_doc.json".format(sid))
    if not os.path.exists(p): return None
    with open(p) as f: raw = json.load(f)
    trs = []
    for item in raw.get("trs", []):
        try:
            a = item.get("tr_a", {})
            tr_a = TR_A(
                tr_id=a.get("tr_id", item["tr_id"]),
                preconditions=a.get("preconditions",[]),
                stimuli=a.get("stimuli",[]),
                expected_outputs=a.get("expected_outputs",[]),
                coverage_criterion=a.get("coverage_criterion",""),
                priority=a.get("priority","medium"),
                source_req_ids=a.get("source_req_ids",[]),
            )
            trs.append(TestRequirement(
                tr_id=item["tr_id"], category=TRCategory(item["category"]),
                source_id=item["source_id"], tr_h=item["tr_h"], tr_a=tr_a,
                source_req_ids=item.get("source_req_ids",[]),
            ))
        except Exception: continue
    return TRDoc(source_id=raw["source_id"], trs=trs,
                 tr_h_list=[t.tr_h for t in trs],
                 tr_a_list=[t.tr_a for t in trs], total_count=len(trs))

def load_review(sid):
    p = os.path.join(DIRS["reviews"], "{}_review.json".format(sid))
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

def save_tr_doc(doc, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    p = os.path.join(output_dir, "{}_final_tr_doc.json".format(doc.source_id))
    with open(p, "w") as f:
        json.dump(doc.model_dump(), f, indent=2, ensure_ascii=False)
    return p


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default=",".join(ALL_SOURCES))
    parser.add_argument("--no-feedback", action="store_true")
    args = parser.parse_args()

    sources     = [s.strip() for s in args.sources.split(",")]
    run_feedback= not args.no_feedback

    print("\nAI4RE-Test  |  Full Pipeline Runner with Feedback Loop")
    print("=" * 60)
    print("Sources      : {}".format(", ".join(sources)))
    print("Feedback loop: {}".format("YES" if run_feedback else "NO"))
    print("=" * 60)

    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

    modeller  = DemandModeller()
    checker   = ConsistencyChecker()
    generator = TRGenerator()
    reviewer  = TRReviewer()
    feedback  = FeedbackLoop()

    # - Stage 2: rebuild demand models if needed -
    print("\n[Stage 2] Loading demand models ...")
    models = {}
    for sid in sources:
        p = os.path.join(DIRS["demand"], "{}.json".format(sid))
        if os.path.exists(p):
            models[sid] = load_demand(sid)
            print("  Loaded: {}".format(sid))
        else:
            print("  SKIP (not found): {}  -- run test_stage2_demand.py".format(sid))

    if not models:
        print("No demand models found. Run test_stage2_demand.py first.")
        return

    # - Stage 3: consistency check -
    print("\n[Stage 3] Consistency check ...")
    t0 = time.time()
    consistency_reports = checker.check_all(models, verbose=True)
    checker.save_reports(consistency_reports, DIRS["consistency"])
    total_issues = sum(len(r.issues) for r in consistency_reports)
    print("  {} issues found  ({:.0f}s)".format(total_issues, time.time()-t0))

    # - Stage 4 + 5 + feedback per source -
    summary = {}
    for sid in sources:
        if sid not in models:
            continue

        print("\n[Stage 4+5] {}".format(sid))
        demand = models[sid]

        # Stage 4
        print("  Generating TRs ...")
        t0  = time.time()
        doc = generator.generate(demand, consistency_reports, verbose=False)
        generator.save_tr_doc(doc, DIRS["tr"])
        print("  {} TRs generated  ({:.0f}s)".format(
            doc.total_count, time.time()-t0
        ))

        # Stage 5
        print("  Reviewing ...")
        t0     = time.time()
        review = reviewer.review(doc, demand, verbose=False)
        reviewer.save_review(review, DIRS["reviews"])
        m = review["metrics"]
        print("  Q_H={:.2f} Q_A={:.2f} Trace={:.2f} Coverage={:.2f}  ({:.0f}s)".format(
            m["q_h"], m["q_a"], m["traceability"], m["coverage"], time.time()-t0
        ))

        # Feedback loop
        if run_feedback and not review["passed"]:
            print("  Running feedback loop ...")
            t0  = time.time()
            doc = feedback.run(doc, review, demand, max_iterations=2, verbose=True)
            # Re-review
            review = reviewer.review(doc, demand, verbose=False)
            reviewer.save_review(review, DIRS["reviews"])
            m = review["metrics"]
            print("  After feedback: Q_H={:.2f} Q_A={:.2f} Trace={:.2f} "
                  "Coverage={:.2f}  ({:.0f}s)".format(
                m["q_h"], m["q_a"], m["traceability"],
                m["coverage"], time.time()-t0
            ))

        # Save final TR doc
        out = save_tr_doc(doc, DIRS["final"])
        size_kb = os.path.getsize(out) / 1024
        print("  Final TR-Doc saved: {}  ({:.1f} KB, {} TRs)".format(
            out, size_kb, doc.total_count
        ))

        summary[sid] = {
            "total_trs": doc.total_count,
            "passed":    review["passed"],
            "metrics":   review["metrics"],
        }

    # - Final summary -
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    for sid, info in summary.items():
        status = "PASS" if info["passed"] else "NEEDS REVIEW"
        m = info["metrics"]
        print("  {:25s}  {:12s}  {} TRs  Q_H={:.2f} Q_A={:.2f}".format(
            sid, status, info["total_trs"], m["q_h"], m["q_a"]
        ))
    print()
    print("Output artifacts:")
    for label, d in DIRS.items():
        print("  {:15s} : {}".format(label, d))
    print()


if __name__ == "__main__":
    main()
