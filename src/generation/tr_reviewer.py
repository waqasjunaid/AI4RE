"""
tr_reviewer.py  --  Stage 5: Test Tracking and Review

Five-dimension quality review:
  1. Coverage completeness   : all demand model items covered
  2. Constraint coverage     : all NFR thresholds explicitly tested
  3. Traceability            : every TR has source_req_ids link
  4. Quality defects         : duplicates and untestable TRs
  5. Schema suitability      : TR-A fields present and non-empty

Q_H is now computed DETERMINISTICALLY using EARS compliance rules:
  A TR-H passes if it meets >= 4 of these 5 criteria:
    (a) contains "shall" (requirement verb)
    (b) contains a trigger word (When/If/While/Upon) OR a clear subject
    (c) expected output is non-empty and specific (>= 4 words)
    (d) preconditions are non-empty
    (e) stimuli are non-empty
  This is reproducible, fast (no LLM call), and calibrated to EARS standard.

The LLM rubric is kept as an optional method for manual spot-checking.
"""

import json, os, sys, re, urllib.request
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
from schemas.demand_schema import DemandModel
from schemas.tr_schema     import TRDoc, TestRequirement, TRCategory

OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME  = "llama3.1:70b"
TEMPERATURE = 0.0
MAX_TOKENS  = 1200
TIMEOUT     = 600

TARGET_Q_H          = 0.78
TARGET_Q_A          = 0.85
TARGET_TRACEABILITY = 0.90
TARGET_COVERAGE     = 0.95

SYSTEM_PROMPT = (
    "You are a test requirements quality reviewer. "
    "You output ONLY valid JSON arrays. "
    "Never output markdown or text outside the JSON."
)


class TRReviewer:

    def __init__(self, host=OLLAMA_HOST, model=MODEL_NAME,
                 temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                 timeout=TIMEOUT):
        self.host        = host.rstrip("/")
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout

    # -- Public ---------------------------------------------------------------

    def review(self, tr_doc, demand_model, verbose=True):
        report = {
            "source_id":        tr_doc.source_id,
            "total_trs":        tr_doc.total_count,
            "dimensions":       {},
            "flagged_tr_ids":   [],
            "correction_notes": [],
            "metrics": {"q_h":0.0,"q_a":0.0,"traceability":0.0,"coverage":0.0},
            "passed": False,
        }
        trs = tr_doc.trs
        if not trs:
            report["correction_notes"].append("No TRs to review.")
            return report

        if verbose: print("  Dim 1: Coverage completeness ...")
        d1 = self._dim1_coverage(trs, demand_model)
        report["dimensions"]["coverage"] = d1
        report["metrics"]["coverage"] = d1["score"]
        if verbose: print("    -> {:.2f}  ({} uncovered)".format(
            d1["score"], len(d1["uncovered"])))

        if verbose: print("  Dim 2: Constraint coverage ...")
        d2 = self._dim2_constraints(trs, demand_model)
        report["dimensions"]["constraints"] = d2
        if verbose: print("    -> {}/{} constraints covered".format(
            d2["covered"], d2["total"]))

        if verbose: print("  Dim 3: Traceability ...")
        d3 = self._dim3_traceability(trs)
        report["dimensions"]["traceability"] = d3
        report["metrics"]["traceability"] = d3["score"]
        if verbose: print("    -> {:.2f}  ({} untraceable)".format(
            d3["score"], len(d3["untraceable"])))

        if verbose: print("  Dim 4: Quality defects ...")
        d4 = self._dim4_quality(trs)
        report["dimensions"]["quality"] = d4
        if verbose: print("    -> {} duplicates, {} untestable".format(
            d4["duplicate_count"], d4["untestable_count"]))

        if verbose: print("  Dim 5: Schema suitability ...")
        d5 = self._dim5_schema(trs)
        report["dimensions"]["schema"] = d5
        report["metrics"]["q_a"] = d5["score"]
        if verbose: print("    -> Q_A={:.2f}  ({} schema issues)".format(
            d5["score"], len(d5["invalid_ids"])))

        # Q_H: deterministic EARS compliance check
        if verbose: print("  EARS compliance check (Q_H) ...")
        q_h_result = self._ears_compliance(trs)
        report["metrics"]["q_h"] = q_h_result["score"]
        report["dimensions"]["ears"] = q_h_result
        if verbose: print("    -> Q_H={:.2f}  ({}/{} TRs EARS-compliant)".format(
            q_h_result["score"],
            q_h_result["compliant"],
            q_h_result["total"]))

        # Collect flagged IDs
        all_flagged = set(
            d3["untraceable"] +
            d4["flagged_ids"] +
            d5["invalid_ids"] +
            q_h_result["non_compliant_ids"][:5]
        )
        report["flagged_tr_ids"] = sorted(all_flagged)

        # Correction notes
        notes = []
        if d1["score"] < TARGET_COVERAGE:
            notes.append("Coverage {:.0f}% < {:.0f}%. Missing: {}".format(
                d1["score"]*100, TARGET_COVERAGE*100,
                "; ".join(d1["uncovered"][:3])))
        if d3["score"] < TARGET_TRACEABILITY:
            notes.append("Traceability {:.0f}% < {:.0f}%. Fix: {}".format(
                d3["score"]*100, TARGET_TRACEABILITY*100,
                ", ".join(d3["untraceable"][:3])))
        if d5["score"] < TARGET_Q_A:
            notes.append("Q_A {:.2f} < {:.2f}. Fix schema for: {}".format(
                d5["score"], TARGET_Q_A, ", ".join(d5["invalid_ids"][:3])))
        if q_h_result["score"] < TARGET_Q_H:
            notes.append(
                "Q_H {:.2f} < {:.2f}. "
                "{} TRs fail EARS compliance. Top issues: {}".format(
                    q_h_result["score"], TARGET_Q_H,
                    len(q_h_result["non_compliant_ids"]),
                    "; ".join(q_h_result["top_issues"][:2])
                )
            )
        report["correction_notes"] = notes

        report["passed"] = (
            report["metrics"]["q_h"]         >= TARGET_Q_H and
            report["metrics"]["q_a"]          >= TARGET_Q_A and
            report["metrics"]["traceability"] >= TARGET_TRACEABILITY and
            report["metrics"]["coverage"]     >= TARGET_COVERAGE
        )
        return report

    def save_review(self, review, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir,
                           "{}_review.json".format(review["source_id"]))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(review, f, indent=2, ensure_ascii=False)
        return out

    # -- Dim 1: Coverage ------------------------------------------------------

    def _dim1_coverage(self, trs, model):
        items = (
            [r.description[:60] for r in model.functional_reqs] +
            [uc.goal[:60] for uc in model.use_cases] +
            [c.description[:60] for c in model.nfr_constraints]
        )
        if not items:
            return {"score":1.0,"uncovered":[],"total":0,"covered":0}
        tr_texts = " ".join(tr.tr_h.lower() for tr in trs)
        covered, uncovered = [], []
        for item in items:
            kws = [w for w in item.lower().split() if len(w) > 4]
            if kws and any(k in tr_texts for k in kws[:3]):
                covered.append(item)
            else:
                uncovered.append(item)
        score = len(covered)/len(items) if items else 1.0
        return {"score":score,"covered":len(covered),
                "uncovered":uncovered[:10],"total":len(items)}

    # -- Dim 2: Constraints ---------------------------------------------------

    def _dim2_constraints(self, trs, model):
        constraints = (
            [c.description for c in model.nfr_constraints] +
            [c.description for c in model.runtime_constraints]
        )
        if not constraints:
            return {"covered":0,"total":0,"uncovered":[]}
        nfr_trs  = [tr for tr in trs if tr.category == TRCategory.NFR_CONSTRAINT]
        tr_texts = " ".join(tr.tr_h.lower() for tr in nfr_trs)
        covered, uncovered = [], []
        for c in constraints:
            kws = [w for w in c.lower().split() if len(w) > 4]
            if kws and any(k in tr_texts for k in kws[:3]):
                covered.append(c)
            else:
                uncovered.append(c)
        return {"covered":len(covered),"total":len(constraints),
                "uncovered":uncovered[:5]}

    # -- Dim 3: Traceability --------------------------------------------------

    def _dim3_traceability(self, trs):
        untraceable = [
            tr.tr_id for tr in trs
            if not tr.tr_a.source_req_ids or
               all(not s.strip() for s in tr.tr_a.source_req_ids)
        ]
        total  = len(trs)
        traced = total - len(untraceable)
        score  = traced/total if total > 0 else 1.0
        return {"score":score,"traced":traced,
                "untraceable":untraceable[:10],"total":total}

    # -- Dim 4: Quality -------------------------------------------------------

    def _dim4_quality(self, trs):
        def jaccard(a, b):
            sa, sb = set(a.lower().split()), set(b.lower().split())
            if not sa or not sb: return 0.0
            return len(sa & sb)/len(sa | sb)

        dup_pairs, flagged = [], []
        for i in range(len(trs)):
            for j in range(i+1, min(i+20, len(trs))):
                if jaccard(trs[i].tr_h, trs[j].tr_h) > 0.75:
                    dup_pairs.append((trs[i].tr_id, trs[j].tr_id))
                    flagged.append(trs[j].tr_id)

        bad_kws = ["shall be good","shall be nice","shall be appropriate",
                   "shall work properly","shall function correctly"]
        untestable = [
            tr.tr_id for tr in trs
            if any(kw in tr.tr_h.lower() for kw in bad_kws)
        ]
        flagged += untestable
        return {
            "duplicate_count":  len(dup_pairs),
            "duplicate_pairs":  dup_pairs[:5],
            "untestable_count": len(untestable),
            "untestable_ids":   untestable[:5],
            "flagged_ids":      list(set(flagged))[:10],
        }

    # -- Dim 5: Schema --------------------------------------------------------

    def _dim5_schema(self, trs):
        invalid = [
            tr.tr_id for tr in trs
            if not tr.tr_a.preconditions or
               not tr.tr_a.stimuli or
               not tr.tr_a.expected_outputs or
               not tr.tr_a.coverage_criterion
        ]
        total = len(trs)
        valid = total - len(invalid)
        score = valid/total if total > 0 else 1.0
        return {"score":score,"valid":valid,
                "invalid_ids":invalid[:10],"total":total}

    # -- EARS compliance (replaces LLM rubric) --------------------------------

    def _ears_compliance(self, trs: List[TestRequirement]) -> Dict:
        """
        Deterministic EARS compliance check.
        A TR-H passes if it meets >= 4 of these 5 criteria:
          (a) contains 'shall'
          (b) has a trigger (When/If/While/Upon/On) or clear system subject
          (c) expected output is specific (>= 4 words)
          (d) preconditions list is non-empty
          (e) stimuli list is non-empty
        """
        compliant_ids     = []
        non_compliant_ids = []
        issue_counts      = {}

        trigger_pattern = re.compile(
            r'\b(when|if|while|upon|on)\b', re.IGNORECASE
        )
        subject_pattern = re.compile(
            r'\bthe\s+\w+\s+shall\b', re.IGNORECASE
        )

        for tr in trs:
            tr_h  = tr.tr_h.lower()
            tr_a  = tr.tr_a
            score = 0
            issues = []

            # (a) shall
            if "shall" in tr_h:
                score += 1
            else:
                issues.append("missing 'shall'")

            # (b) trigger or clear subject
            if trigger_pattern.search(tr_h) or subject_pattern.search(tr.tr_h):
                score += 1
            else:
                issues.append("no trigger (When/If) or clear subject")

            # (c) specific expected output
            exp = " ".join(tr_a.expected_outputs)
            if len(exp.split()) >= 4:
                score += 1
            else:
                issues.append("expected output too vague")

            # (d) preconditions non-empty
            if tr_a.preconditions and any(p.strip() for p in tr_a.preconditions):
                score += 1
            else:
                issues.append("no preconditions")

            # (e) stimuli non-empty
            if tr_a.stimuli and any(s.strip() for s in tr_a.stimuli):
                score += 1
            else:
                issues.append("no stimuli")

            if score >= 4:
                compliant_ids.append(tr.tr_id)
            else:
                non_compliant_ids.append(tr.tr_id)
                for iss in issues:
                    issue_counts[iss] = issue_counts.get(iss, 0) + 1

        total     = len(trs)
        compliant = len(compliant_ids)
        score_val = compliant/total if total > 0 else 1.0

        # Top issues by frequency
        top_issues = sorted(issue_counts, key=lambda k: -issue_counts[k])

        return {
            "score":              score_val,
            "compliant":          compliant,
            "non_compliant":      len(non_compliant_ids),
            "non_compliant_ids":  non_compliant_ids[:10],
            "top_issues":         top_issues[:3],
            "total":              total,
        }

    # -- Optional LLM rubric (for manual spot-check only) --------------------

    def llm_spot_check(self, trs: List[TestRequirement], n: int = 5) -> Dict:
        """
        Call this manually for spot-checking a sample of TRs.
        Not used in the automated pipeline.
        """
        sample = trs[:n]
        sample_lines = []
        for tr in sample:
            sample_lines.append(
                "TR-ID: {}\nTR-H: {}\n"
                "Preconditions: {}\nStimuli: {}\nExpected: {}".format(
                    tr.tr_id, tr.tr_h,
                    "; ".join(tr.tr_a.preconditions[:2]),
                    "; ".join(tr.tr_a.stimuli[:2]),
                    "; ".join(tr.tr_a.expected_outputs[:2]),
                )
            )
        prompt = (
            "Evaluate these test requirements. Score each 0-5 on "
            "completeness, testability, clarity. "
            "accept=true if average>=3.\n\n"
            "{}\n\nJSON array with tr_id, completeness, testability, "
            "clarity, accept:"
        ).format("\n\n".join(sample_lines))

        try:
            raw  = self._chat(prompt)
            text = raw.strip()
            if "```" in text:
                text = "\n".join(
                    l for l in text.splitlines()
                    if not l.strip().startswith("```")
                ).strip()
            s = text.find("["); e = text.rfind("]")
            if s == -1 or e == -1: return {"raw": raw, "parsed": []}
            data = json.loads(text[s:e+1])
            return {"raw": raw[:300], "parsed": data}
        except Exception as ex:
            return {"error": str(ex)}

    def _chat(self, user_message):
        url = "{}/api/chat".format(self.host)
        payload = json.dumps({
            "model":  self.model, "stream": True,
            "options": {"temperature":self.temperature,
                        "num_predict":self.max_tokens,"stop":[]},
            "messages": [
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user",  "content":user_message},
            ],
        }).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type":"application/json"}, method="POST")
        tokens = []
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line: continue
                try:
                    d = json.loads(line)
                    tokens.append(d.get("message",{}).get("content",""))
                    if d.get("done"): break
                except json.JSONDecodeError: continue
        return "".join(tokens).strip()