"""
faithfulness_check.py
------------------------
Cross-model automated faithfulness check (reviewer comment 2c/2a).

Uses a SECOND, DIFFERENT LLM (not llama3.1:70b, the model that generated
the TRs) to independently judge whether each TR-H statement is actually
grounded in its own recorded source_req_ids text.

Sample window (fixed, reproducible): --n selects the FIRST N TRs in the
tr-doc file, always the same N regardless of how many runs/retries you
do. Resuming only fills in TRs within that fixed window that previously
errored or were unparseable -- it never drifts past the window to "top
up" the count with TRs you were never asked to check. (An earlier
version of this script had that bug: skipped already-done TRs didn't
count against --n, so retries silently expanded the sample each time
you resumed. Fixed here.)

Robustness:
  - each judge call is wrapped in error handling; a single failed call
    is recorded as an ERROR outcome and does NOT crash the run
  - one automatic retry on transient server errors (HTTP 500/502/503)
  - results are written to disk after EVERY TR, never just at the end

NOTE on model choice: qwen2.5:14b has now shown corrupted/garbled
output (repeated fragments, leaked chat-template tokens like
<|IM_START|>, non-English character intrusions) on real runs against
this exact task, at a rate around 30-35% of calls. Combined with the
same model's documented corruption during entity extraction earlier in
this project, this is a real, repeated reliability problem with this
model in this environment, not a fluke. Recommend switching to
mistral:7b or gemma2:9b -- see below.

Usage:
    python faithfulness_check.py \
        --tr-doc data/processed/evaluation/final/auth_system_design_final_tr_doc.json \
        --judge-model mistral:7b \
        --n 30
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

OLLAMA_HOST = "http://127.0.0.1:11434"
TIMEOUT = 600
MAX_RETRIES = 1
RETRY_DELAY_S = 5

JUDGE_PROMPT = """You are an independent fact-checker. You are given a SOURCE requirement description and a STATEMENT that claims to test that requirement.

Your only job: does the STATEMENT stay faithful to the SOURCE, or does it introduce specific claims (numbers, thresholds, behaviors, conditions) that are NOT present in or reasonably implied by the SOURCE?

SOURCE: {source}

STATEMENT: {statement}

Respond with exactly one word: SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED.
- SUPPORTED: the statement's claims are all present in or directly implied by the source.
- PARTIALLY_SUPPORTED: the general topic matches, but the statement adds a specific detail (a number, threshold, condition) not present in the source.
- UNSUPPORTED: the statement's core claim is not present in the source at all.

Answer with only the single word, nothing else."""


def call_judge(model, source, statement):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
            source=source, statement=statement
        )}],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    req = urllib.request.Request(
        OLLAMA_HOST + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read())
            raw = data["message"]["content"].strip().upper()
            for label in ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"]:
                if label in raw:
                    return label
            return "UNPARSEABLE:" + raw[:60].replace("\n", " ")
        except urllib.error.HTTPError as e:
            last_err = "HTTP_{}: {}".format(e.code, e.reason)
            if e.code in (500, 502, 503) and attempt < MAX_RETRIES:
                print("    (server error, retrying in {}s ...)".format(RETRY_DELAY_S))
                time.sleep(RETRY_DELAY_S)
                continue
            break
        except Exception as e:
            last_err = "{}: {}".format(type(e).__name__, e)
            break
    return "ERROR:" + last_err


def load_trs(tr_doc_path):
    with open(tr_doc_path, encoding="utf-8") as f:
        data = json.load(f)
    trs = data.get("trs") or data.get("TRs") or data
    if isinstance(trs, dict):
        trs = trs.get("items", [])
    return trs


def get_source_text(tr):
    ids = tr.get("source_req_ids") or tr.get("tr_a", {}).get("source_req_ids", [])
    ids = [s for s in ids if s and s.strip()]
    return " / ".join(ids) if ids else None


def classify(verdict):
    if verdict in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"):
        return verdict
    if verdict.startswith("UNPARSEABLE"):
        return "UNPARSEABLE"
    return "ERROR"


def save(out_path, detail):
    summary = {"SUPPORTED": 0, "PARTIALLY_SUPPORTED": 0, "UNSUPPORTED": 0,
               "UNPARSEABLE": 0, "ERROR": 0}
    for d in detail:
        summary[classify(d["verdict"])] += 1
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detail": detail}, f, indent=2, ensure_ascii=False)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tr-doc", required=True)
    ap.add_argument("--judge-model", default="mistral:7b")
    ap.add_argument("--n", type=int, default=30,
                     help="fixed sample size: always the first N TRs in the file, "
                          "same window every run/resume")
    ap.add_argument("--out", default="faithfulness_results.json")
    args = ap.parse_args()

    all_trs = load_trs(args.tr_doc)
    window = all_trs[:args.n]  # FIXED slice -- identical across every run/resume
    window_ids = {t.get("tr_id") for t in window}

    # detail_by_id holds one entry per TR in `window`; prior successful
    # results are loaded, prior errors/unparseable are dropped (will retry).
    detail_by_id = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            prior = json.load(f)
        for d in prior.get("detail", []):
            if d["tr_id"] in window_ids and classify(d["verdict"]) not in ("ERROR", "UNPARSEABLE"):
                detail_by_id[d["tr_id"]] = d
        if detail_by_id:
            print("Resuming: {}/{} TRs in the sample window already completed successfully, "
                  "skipping them.".format(len(detail_by_id), len(window)))

    n_to_process = len(window) - len(detail_by_id)
    processed = 0
    for t in window:
        tr_id = t.get("tr_id")
        if tr_id in detail_by_id:
            continue  # already have a good result; does not need re-checking

        tr_h = t.get("tr_h", "")
        source_text = get_source_text(t)
        if not source_text:
            detail_by_id[tr_id] = {"tr_id": tr_id, "tr_h": tr_h, "source": None,
                                    "verdict": "SKIPPED_NO_SOURCE"}
            continue

        verdict = call_judge(args.judge_model, source_text, tr_h)
        detail_by_id[tr_id] = {"tr_id": tr_id, "tr_h": tr_h, "source": source_text, "verdict": verdict}
        processed += 1
        print("[{}/{}] {} -> {}".format(processed, n_to_process, tr_id, verdict))

        # Preserve original file order when saving.
        ordered_detail = [detail_by_id[t2.get("tr_id")] for t2 in window if t2.get("tr_id") in detail_by_id]
        save(args.out, ordered_detail)

    ordered_detail = [detail_by_id[t.get("tr_id")] for t in window if t.get("tr_id") in detail_by_id]
    summary = save(args.out, ordered_detail)

    print("\n" + "=" * 60)
    print("  Cross-model faithfulness check ({} judging)".format(args.judge_model))
    print("  Sample window: first {} TRs in {}".format(len(window), os.path.basename(args.tr_doc)))
    print("=" * 60)
    total = len(ordered_detail)
    for k, v in summary.items():
        pct = (v / total * 100) if total else 0
        print("  {:22s}: {:3d}  ({:.1f}%)".format(k, v, pct))
    print("\nWrote {} ({} TRs in fixed window)".format(args.out, total))
    if summary["ERROR"] or summary["UNPARSEABLE"]:
        print("\nNOTE: {} TRs had ERROR/UNPARSEABLE outcomes. Re-run the exact same".format(
            summary["ERROR"] + summary["UNPARSEABLE"]
        ))
        print("command (same --tr-doc, same --n, same --out) to retry ONLY those --")
        print("the sample window stays fixed, it will not drift to new TRs.")


if __name__ == "__main__":
    main()