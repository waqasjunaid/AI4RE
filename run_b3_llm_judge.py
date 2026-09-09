"""
run_b3_llm_judge.py
-----------------------
Baseline B3 (paper text): applies an LLM-as-judge Q_H rubric to the
SAME TRs the full pipeline already generated, replacing only the
deterministic EARS checker's score. Per the paper: "with all other
metrics identical to Full by construction" -- so this script does NOT
regenerate TRs, Q_A, Trace, or Coverage; it only recomputes Q_H via an
LLM judge and reports it alongside Full's unchanged other metrics.

Uses the SAME six criteria the deterministic checker uses (Section 8.2),
but asks the LLM to judge them holistically per TR, rather than via
regex/keyword pattern matching. This matches "6-criterion scoring
prompt" as described in the paper text.

Usage:
    python run_b3_llm_judge.py --tr-doc data/processed/tr_doc/auth_system_srs_tr_doc.json \
        --demand-model data/processed/demand_model/auth_system_srs.json
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.getcwd())

from schemas.demand_schema import DemandModel
from schemas.tr_schema import TRDoc, TestRequirement, TR_A, TRCategory
from src.evaluation.tr_reviewer import TRReviewer

OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME = "llama3.1:70b"
TIMEOUT = int(os.environ.get("AI4RE_TIMEOUT", "600"))
MAX_RETRIES = 1
RETRY_DELAY_S = 10

JUDGE_PROMPT = """Evaluate this test requirement against 6 EARS quality criteria. For each, answer PASS or FAIL:

(a) shall presence: contains the word "shall"
(b) trigger keyword: contains When/If/While/Upon, or a clear "the X shall" subject
(c) expected output length: the expected outcome is described in at least 4 words
(d) preconditions: has at least one precondition specified
(e) stimuli: has at least one triggering stimulus specified
(f) semantic specificity: contains a measurable element (a number+unit, an HTTP status code, a REST verb, or an API endpoint path)

TR-H: {tr_h}
Preconditions: {preconditions}
Stimuli: {stimuli}
Expected outputs: {expected_outputs}

Respond with exactly 6 words, one per criterion in order (a) through (f), each either PASS or FAIL, space-separated. Nothing else.
Example: PASS PASS FAIL PASS PASS FAIL"""


def call_judge(tr):
    prompt = JUDGE_PROMPT.format(
        tr_h=tr.tr_h,
        preconditions=", ".join(tr.tr_a.preconditions) or "(none)",
        stimuli=", ".join(tr.tr_a.stimuli) or "(none)",
        expected_outputs=", ".join(tr.tr_a.expected_outputs) or "(none)",
    )
    payload = {
        "model": MODEL_NAME, "stream": False,
        "options": {"temperature": 0.0, "num_predict": 50},
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        OLLAMA_HOST + "/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read())
            words = data["message"]["content"].strip().upper().split()
            passes = sum(1 for w in words[:6] if w.startswith("PASS"))
            return passes, words[:6]
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < MAX_RETRIES:
                print("    (retrying in {}s: {})".format(RETRY_DELAY_S, e))
                time.sleep(RETRY_DELAY_S)
                continue
            return None, ["ERROR:" + str(e)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tr-doc", required=True,
                     help="Full's already-generated TR-doc for this document "
                          "(use the SAME TRs, do not regenerate)")
    ap.add_argument("--demand-model", required=True)
    args = ap.parse_args()

    with open(args.tr_doc, encoding="utf-8") as f:
        raw = json.load(f)
    trs = []
    for item in raw.get("trs", raw if isinstance(raw, list) else []):
        a = item.get("tr_a", {})
        tr_a = TR_A(
            tr_id=a.get("tr_id", item.get("tr_id", "")),
            preconditions=a.get("preconditions", []), stimuli=a.get("stimuli", []),
            expected_outputs=a.get("expected_outputs", []),
            coverage_criterion=a.get("coverage_criterion", ""),
            priority=a.get("priority", "medium"),
            source_req_ids=a.get("source_req_ids", []),
        )
        trs.append(TestRequirement(
            tr_id=item["tr_id"], category=TRCategory(item["category"]),
            source_id=item.get("source_id", ""), tr_h=item["tr_h"], tr_a=tr_a,
            source_req_ids=item.get("source_req_ids", []),
        ))

    with open(args.demand_model, encoding="utf-8") as f:
        demand = DemandModel(**json.load(f))

    print("[B3] LLM-judging Q_H for {} TRs (same TRs as Full, only Q_H recomputed) ...".format(len(trs)))
    scores = []
    for i, tr in enumerate(trs, 1):
        passes, raw_words = call_judge(tr)
        scores.append(passes)
        print("  [{}/{}] {} -> {}/6 ({})".format(i, len(trs), tr.tr_id, passes, " ".join(raw_words)))

    valid_scores = [s for s in scores if s is not None]
    q_h_llm = sum(1 for s in valid_scores if s >= 5) / len(valid_scores) if valid_scores else 0.0

    # Unchanged from Full "by construction": Q_A, Trace, Coverage use the
    # exact same deterministic logic, on the exact same TRs.
    reviewer = TRReviewer()
    schema = reviewer._dim5_schema(trs)
    trace = reviewer._dim3_traceability(trs)
    coverage = reviewer._dim1_coverage(trs, demand) if hasattr(reviewer, "_dim1_coverage") else None

    result = {
        "source_id": demand.source_id,
        "total_trs": len(trs),
        "q_h_llm_judge": q_h_llm,
        "q_h_score_distribution": {str(i): valid_scores.count(i) for i in range(7)},
        "q_a": schema["score"],
        "traceability": trace["score"],
        "coverage": coverage["score"] if coverage else "see Full's reported value (unchanged by construction)",
        "note": "TRs, Q_A, Trace, Coverage identical to Full by construction; "
                "only Q_H recomputed via LLM-as-judge instead of the deterministic checker.",
    }

    out_path = "b3_{}_result.json".format(demand.source_id)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print("  B3 result for {}".format(demand.source_id))
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print("\nWrote {}".format(out_path))


if __name__ == "__main__":
    main()
