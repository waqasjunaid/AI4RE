"""
single_prompt_baseline.py
-----------------------------
Reviewer comment R3b: "The baseline cannot justify the architecture...
add comparisons to show whether results come from the five-stage design
or simply from using a capable model, e.g. the same model given the
same artifacts in more prompts."

This implements exactly that: a single-prompt (or minimal-prompt)
baseline "B4" that gives llama3.1:70b the SAME raw document text used
by AI4RE-Test, in ONE prompt, and asks it to directly produce TR-H/TR-A
test requirements -- with none of AI4RE-Test's five-stage decomposition
(no separate entity extraction, demand modelling, consistency checking,
or quality-gated feedback loop).

The output is scored using the EXACT SAME deterministic Stage 5 checker
(Q_H, Q_A, Trace, Cov) that scores the full pipeline's output, for a
fair, apples-to-apples comparison -- not a different metric that could
be argued to favor one approach.

IMPORTANT SCOPE NOTE: Cov (coverage) is computed against a DemandModel.
B4 does not produce a demand model (that is one of the things the
5-stage pipeline does that this baseline does not), so Cov cannot be
computed for B4 on the same basis. This script reports Q_H, Q_A, and
Trace for both, and reports TR count only (not Cov) for B4, with an
explanation rather than a fabricated/estimated number.

Usage:
    python single_prompt_baseline.py --source-id nasa_srs_v1 \
        --raw-doc data/raw/srs/nasa_srs.pdf
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.getcwd())

from schemas.tr_schema import TRDoc, TestRequirement, TR_A, TRCategory
from src.evaluation.tr_reviewer import TRReviewer
from src.ingestion.document_manager import DocumentManager

OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME = "llama3.1:70b"
TIMEOUT = int(os.environ.get("AI4RE_TIMEOUT", "2400"))  # 40 min default; a single
                                                          # open-ended "produce all TRs
                                                          # for this document" call can
                                                          # legitimately run much longer
                                                          # than a typical per-stage call
MAX_RETRIES = 1
RETRY_DELAY_S = 10

import urllib.request
import urllib.error
import time


def chat(prompt, max_tokens=4000):
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": max_tokens},
        "messages": [
            {"role": "system", "content": "You output ONLY valid JSON. No markdown, no explanation."},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        OLLAMA_HOST + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read())
            return data["message"]["content"]
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                print("  (request failed: {} -- retrying in {}s ...)".format(e, RETRY_DELAY_S))
                time.sleep(RETRY_DELAY_S)
                continue
            break
    raise SystemExit(
        "Ollama request failed after {} attempt(s): {}\n"
        "This model/prompt combination may just need more time than TIMEOUT={}s "
        "allows on this hardware -- try increasing it:\n"
        "  AI4RE_TIMEOUT=3600 python single_prompt_baseline.py ...".format(
            MAX_RETRIES + 1, last_err, TIMEOUT
        )
    )


SINGLE_PROMPT_TEMPLATE = """You are a test requirements engineer. Read the following software document and directly produce a set of test requirements (TRs).

DOCUMENT TEXT:
{doc_text}

For each test requirement, produce BOTH:
1. TR-H: a natural language statement in EARS format ("When <trigger>, the <system> shall <response>")
2. TR-A: a structured JSON object with fields: tr_id, category (one of: functional, nfr_constraint, runtime_environment, interface_exception), preconditions (array), stimuli (array), expected_outputs (array), coverage_criterion (string), priority (high/medium/low)

Produce as many test requirements as you judge necessary to reasonably cover the document's testable behavior -- there is no fixed target count.

NOTE: do not use a "consistency" category -- that category applies only to
defects found by comparing this document against OTHER related documents,
which you do not have access to here.

Return a JSON array. Each item has exactly these keys: tr_id, category, tr_h, tr_a (an object with the TR-A fields above).

JSON array:"""


def build_tr_doc_from_json(items, source_id):
    trs = []
    for item in items:
        try:
            a = item.get("tr_a", {})
            tr_a = TR_A(
                tr_id=item.get("tr_id", ""),
                preconditions=a.get("preconditions", []),
                stimuli=a.get("stimuli", []),
                expected_outputs=a.get("expected_outputs", []),
                coverage_criterion=a.get("coverage_criterion", ""),
                priority=a.get("priority", "medium"),
                source_req_ids=a.get("source_req_ids", []),
            )
            trs.append(TestRequirement(
                tr_id=item["tr_id"], category=TRCategory(item.get("category", "functional")),
                source_id=source_id, tr_h=item.get("tr_h", ""), tr_a=tr_a,
                source_req_ids=[],
            ))
        except Exception as e:
            print("  (skipped one malformed TR: {})".format(e))
            continue
    return TRDoc(source_id=source_id, trs=trs,
                 tr_h_list=[t.tr_h for t in trs],
                 tr_a_list=[t.tr_a for t in trs], total_count=len(trs))


def salvage_json_array(text):
    """Parse a JSON array from LLM output robustly. Tries a strict parse
    first (fast path); if the response has a malformed or truncated tail
    (a real risk in a single long, open-ended generation -- e.g. an
    unescaped quote inside one TR's text, or hitting max_tokens mid-object),
    falls back to salvaging as many complete, valid top-level objects as
    possible in order, stopping at the first one that fails to parse
    rather than discarding everything already generated successfully.
    """
    start = text.find("[")
    if start == -1:
        return [], "no '[' found in response"

    end = text.rfind("]")
    if end != -1:
        try:
            return json.loads(text[start:end + 1]), None
        except json.JSONDecodeError as e:
            salvage_note = "strict parse failed ({}), falling back to salvage mode".format(e)
    else:
        salvage_note = "no closing ']' found, falling back to salvage mode"

    decoder = json.JSONDecoder()
    idx = start + 1
    items = []
    n = len(text)
    while idx < n:
        while idx < n and text[idx] in " \t\n\r,":
            idx += 1
        if idx >= n or text[idx] == "]":
            break
        try:
            obj, next_idx = decoder.raw_decode(text, idx)
            items.append(obj)
            idx = next_idx
        except json.JSONDecodeError:
            break  # stop at first unparseable item; keep everything salvaged so far
    return items, "{} -- salvaged {} complete item(s) before the parse error".format(
        salvage_note, len(items)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--raw-doc", required=True, help="path to the raw document file")
    ap.add_argument("--artifact-type", default="srs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or "baseline_b4_{}.json".format(args.source_id)

    print("Loading raw document via existing DocumentManager (same loader AI4RE-Test uses) ...")
    dm = DocumentManager()
    sections = dm.load_document(args.raw_doc, args.artifact_type, source_id=args.source_id)
    doc_text = "\n\n".join(s["text"] for s in sections)
    print("  {} sections, {} chars total".format(len(sections), len(doc_text)))

    # llama3.1:70b has a 128k context window per Table 17; most evaluation
    # documents fit. Truncate defensively for very large documents (e.g.
    # bash_user_manual) rather than silently failing.
    MAX_CHARS = 100000
    if len(doc_text) > MAX_CHARS:
        print("  WARNING: document text truncated from {} to {} chars to fit context.".format(
            len(doc_text), MAX_CHARS
        ))
        doc_text = doc_text[:MAX_CHARS]

    prompt = SINGLE_PROMPT_TEMPLATE.format(doc_text=doc_text)
    print("Calling {} in a single prompt (no pipeline stages) ...".format(MODEL_NAME))
    raw = chat(prompt)

    # Always persist the raw response, success or failure -- never lose
    # information a debugging session might need.
    raw_out_path = "baseline_b4_{}_raw_response.txt".format(args.source_id)
    with open(raw_out_path, "w", encoding="utf-8") as f:
        f.write(raw)
    print("  Raw model response saved to {} ({} chars)".format(raw_out_path, len(raw)))

    s = raw.strip()
    if "```" in s:
        s = "\n".join(l for l in s.splitlines() if not l.strip().startswith("```"))

    items, salvage_note = salvage_json_array(s)
    if salvage_note:
        print("  NOTE: {}".format(salvage_note))
    if not items:
        raise SystemExit(
            "Could not parse any TR items from the model's response.\n"
            "Full raw response saved to {} for inspection.".format(raw_out_path)
        )
    print("  Parsed {} TR items from single-prompt output".format(len(items)))

    tr_doc = build_tr_doc_from_json(items, args.source_id)

    # Score with the EXACT SAME checker logic used for the full pipeline.
    # Q_H, Q_A, and Trace are each computed by TRReviewer methods that only
    # need the TR list (no demand model); Coverage requires a DemandModel,
    # which B4 does not produce -- see docstring for why that's expected,
    # not a bug.
    reviewer = TRReviewer()
    trs = tr_doc.trs

    ears = reviewer._ears_compliance(trs)
    schema = reviewer._dim5_schema(trs)
    trace = reviewer._dim3_traceability(trs)
    quality = reviewer._dim4_quality(trs)

    result = {
        "source_id": args.source_id,
        "total_trs": tr_doc.total_count,
        "metrics": {
            "q_h": ears["score"],
            "q_a": schema["score"],
            "traceability": trace["score"],
            "coverage": None,
        },
        "coverage_note": "Not computed: B4 produces no DemandModel to measure "
                          "coverage against (see script docstring) -- this is "
                          "an architectural difference being measured, not a "
                          "missing number.",
        "duplicate_count": quality["duplicate_count"],
        "untestable_count": quality["untestable_count"],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tr_doc": tr_doc.model_dump(), "result": result}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("  Single-prompt baseline (B4) result for {}".format(args.source_id))
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print("\nWrote {}".format(out_path))
    print("\nCompare against the full pipeline's Table 22 row for {} :".format(args.source_id))
    print("  same model, same document, five stages + feedback loop vs. one prompt.")


if __name__ == "__main__":
    main()