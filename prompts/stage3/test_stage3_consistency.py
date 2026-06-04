"""
test_stage3_consistency.py
Stage 3: Cross-document Consistency Check
Run: python test_stage3_consistency.py
"""
import sys, os, json, time
sys.path.insert(0, os.getcwd())

from schemas.demand_schema       import DemandModel
from schemas.consistency_schema  import ConsistencyReport, IssueType
from src.reasoning.consistency_checker import ConsistencyChecker

DEMAND_DIR      = "data/processed/demand_model"
CONSISTENCY_DIR = "data/processed/consistency"

SOURCES = [
    "nasa_srs_v1",
    "bash_user_manual",
    "swagger_petstore",
    "auth_system_design",
]


def load_demand_model(source_id: str) -> DemandModel:
    path = os.path.join(DEMAND_DIR, "{}.json".format(source_id))
    with open(path, "r", encoding="utf-8") as f:
        return DemandModel(**json.load(f))


def print_report(report: ConsistencyReport):
    print("\n  Comparison : {} vs {}".format(report.source_a, report.source_b))
    print("  Issues     : {}  (conflicts:{} omissions:{} ambiguities:{} "
          "to_confirm:{})".format(
        len(report.issues),
        report.total_conflicts,
        report.total_omissions,
        report.total_ambiguities,
        report.total_to_confirm,
    ))
    for issue in report.issues[:5]:
        print("    [{}] ({}/{}) {}".format(
            issue.issue_id,
            issue.issue_type.value,
            issue.severity.value,
            issue.description[:65],
        ))
        if issue.suggested_fix:
            print("         Fix: {}".format(issue.suggested_fix[:60]))


def main():
    print("\nAI4RE-Test  |  Stage 3: Consistency Check")
    print("-" * 60)

    # Load all demand models
    models = {}
    for src in SOURCES:
        path = os.path.join(DEMAND_DIR, "{}.json".format(src))
        if not os.path.exists(path):
            print("  SKIP -- demand model not found: {}".format(src))
            print("  Run test_stage2_demand.py first.")
            continue
        models[src] = load_demand_model(src)
        print("  Loaded: {}  ({} func reqs, {} roles)".format(
            src,
            len(models[src].functional_reqs),
            len(models[src].user_roles),
        ))

    if len(models) < 2:
        print("\nNeed at least 2 demand models to check consistency.")
        return

    print("\nRunning {} consistency checks ...\n".format(4))

    checker = ConsistencyChecker()
    os.makedirs(CONSISTENCY_DIR, exist_ok=True)

    t0      = time.time()
    reports = checker.check_all(models, verbose=True)
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("  Results  ({:.1f}s total)".format(elapsed))
    print("=" * 60)

    total_issues = 0
    for report in reports:
        print_report(report)
        total_issues += len(report.issues)

    out_path = checker.save_reports(reports, CONSISTENCY_DIR)
    size_kb  = os.path.getsize(out_path) / 1024
    print("\n" + "=" * 60)
    print("  Total issues found : {}".format(total_issues))
    print("  Saved to : {}  ({:.1f} KB)".format(out_path, size_kb))
    print("=" * 60)
    print()
    if total_issues > 0:
        print("Consistency issues will feed into Stage 4 TR generation.")
        print("Each issue generates a consistency-driven test requirement.")
    print("Next: python test_stage4_tr_generation.py")
    print()


if __name__ == "__main__":
    main()