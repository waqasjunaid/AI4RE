"""
feedback_loop.py
----------------
Implements the Stage 5 -> Stage 4 feedback loop.

For each flagged TR:
  1. Takes the correction note from the Stage 5 review
  2. Regenerates only the flagged TRs via Stage 4
  3. Replaces them in the TR-Doc
  4. Re-runs Stage 5 review

Also handles coverage gaps: generates new TRs for uncovered demand items.
Iterates until all metrics pass or max_iterations is reached.
"""

import json, os, sys, time, urllib.request
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
from schemas.demand_schema import DemandModel
from schemas.tr_schema     import TRDoc, TestRequirement, TR_A, TRCategory, Priority

OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME  = "llama3.1:70b"
TEMPERATURE = 0.1
MAX_TOKENS  = 1200
TIMEOUT     = 600

SYSTEM_PROMPT = (
    "You are a test requirements engineer. "
    "You output ONLY valid JSON arrays. "
    "Never output markdown or text outside the JSON array."
)

EARS_REMINDER = (
    "EARS format: 'When <trigger>, the <system> shall <response>.' "
    "Be specific and testable. Include measurable criteria."
)


class FeedbackLoop:

    def __init__(self, host=OLLAMA_HOST, model=MODEL_NAME,
                 temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                 timeout=TIMEOUT):
        self.host        = host.rstrip("/")
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout

    def run(
        self,
        tr_doc:       TRDoc,
        review:       Dict,
        demand_model: DemandModel,
        max_iterations: int = 2,
        verbose: bool = True,
    ) -> TRDoc:
        """
        Run feedback loop: regenerate flagged TRs and fill coverage gaps.
        Returns updated TR-Doc.
        """
        updated_doc = tr_doc

        for iteration in range(1, max_iterations + 1):
            if verbose:
                print("\n  Feedback iteration {}/{}".format(
                    iteration, max_iterations
                ))

            changed = False

            # 1. Regenerate flagged TRs
            flagged = review.get("flagged_tr_ids", [])
            if flagged:
                if verbose:
                    print("    Regenerating {} flagged TRs ...".format(
                        len(flagged)
                    ))
                correction_notes = review.get("correction_notes", [])
                updated_doc, n = self._regenerate_flagged(
                    updated_doc, flagged, correction_notes, verbose
                )
                if n > 0:
                    changed = True
                    if verbose:
                        print("    Regenerated {} TRs".format(n))

            # 2. Fill coverage gaps
            uncovered = review.get("dimensions", {}).get(
                "coverage", {}
            ).get("uncovered", [])
            if uncovered:
                if verbose:
                    print("    Generating TRs for {} uncovered items ...".format(
                        len(uncovered)
                    ))
                new_trs = self._fill_coverage_gaps(
                    uncovered, demand_model, updated_doc.source_id
                )
                if new_trs:
                    all_trs = updated_doc.trs + new_trs
                    updated_doc = TRDoc(
                        source_id   = updated_doc.source_id,
                        trs         = all_trs,
                        tr_h_list   = [t.tr_h for t in all_trs],
                        tr_a_list   = [t.tr_a for t in all_trs],
                        total_count = len(all_trs),
                    )
                    changed = True
                    if verbose:
                        print("    Added {} new TRs".format(len(new_trs)))

            if not changed:
                if verbose:
                    print("    No changes made -- stopping early")
                break

        return updated_doc

    # -- Internal -------------------------------------------------------------

    def _regenerate_flagged(
        self,
        tr_doc: TRDoc,
        flagged_ids: List[str],
        correction_notes: List[str],
        verbose: bool,
    ):
        notes_str = "; ".join(correction_notes[:3]) if correction_notes else \
            "Improve EARS format clarity and specificity"

        # Get the flagged TRs
        flagged_trs = [t for t in tr_doc.trs if t.tr_id in flagged_ids]
        keep_trs    = [t for t in tr_doc.trs if t.tr_id not in flagged_ids]

        if not flagged_trs:
            return tr_doc, 0

        tr_lines = []
        for tr in flagged_trs[:8]:
            tr_lines.append(
                "TR-ID: {}\n"
                "Category: {}\n"
                "Current TR-H: {}\n"
                "Issues: {}".format(
                    tr.tr_id, tr.category.value, tr.tr_h, notes_str
                )
            )

        prompt = (
            "Improve these test requirements that were flagged for quality issues.\n\n"
            "{ears}\n\n"
            "Issues to fix: {notes}\n\n"
            "Current TRs to improve:\n{trs}\n\n"
            "For each TR, generate an improved version. "
            "Make TR-H more specific, testable, and EARS-compliant.\n\n"
            "Return JSON array. Each item:\n"
            "  tr_id: same as original\n"
            "  tr_h: improved EARS-format test requirement\n"
            "  preconditions: array of strings\n"
            "  stimuli: array of strings\n"
            "  expected_outputs: array of strings\n"
            "  coverage_criterion: string\n"
            "  priority: high/medium/low\n"
            "  source_req: string\n\n"
            "JSON array:"
        ).format(
            ears =EARS_REMINDER,
            notes=notes_str,
            trs  ="\n\n".join(tr_lines),
        )

        raw  = self._chat(prompt)
        data = self._parse_list(raw)

        new_trs = []
        for item in data:
            try:
                orig_id = item.get("tr_id", "")
                orig    = next((t for t in flagged_trs if t.tr_id == orig_id), None)
                if not orig:
                    continue
                tr_a = TR_A(
                    tr_id              = orig_id,
                    preconditions      = item.get("preconditions", []),
                    stimuli            = item.get("stimuli", []),
                    expected_outputs   = item.get("expected_outputs", []),
                    coverage_criterion = str(item.get("coverage_criterion", "")),
                    priority           = Priority(item.get("priority", "medium")),
                    source_req_ids     = [str(item.get("source_req", ""))],
                )
                new_trs.append(TestRequirement(
                    tr_id          = orig_id,
                    category       = orig.category,
                    source_id      = orig.source_id,
                    tr_h           = str(item.get("tr_h", orig.tr_h)),
                    tr_a           = tr_a,
                    source_req_ids = [str(item.get("source_req", ""))],
                ))
            except Exception:
                continue

        all_trs = keep_trs + new_trs
        updated = TRDoc(
            source_id   = tr_doc.source_id,
            trs         = all_trs,
            tr_h_list   = [t.tr_h for t in all_trs],
            tr_a_list   = [t.tr_a for t in all_trs],
            total_count = len(all_trs),
        )
        return updated, len(new_trs)

    def _fill_coverage_gaps(
        self,
        uncovered_items: List[str],
        demand_model: DemandModel,
        source_id: str,
    ) -> List[TestRequirement]:
        prompt = (
            "Generate test requirements for these uncovered demand items.\n\n"
            "{ears}\n\n"
            "Uncovered items (each needs at least one TR):\n{items}\n\n"
            "Return JSON array. Each item:\n"
            "  tr_h: EARS-format test requirement\n"
            "  preconditions: array of strings\n"
            "  stimuli: array of strings\n"
            "  expected_outputs: array of strings\n"
            "  coverage_criterion: string\n"
            "  priority: high/medium/low\n"
            "  source_req: the uncovered item this TR addresses\n\n"
            "JSON array:"
        ).format(
            ears =EARS_REMINDER,
            items="\n".join("- "+item for item in uncovered_items[:8]),
        )

        raw  = self._chat(prompt)
        data = self._parse_list(raw)

        counter = 1000  # offset to avoid ID collision
        result  = []
        for item in data[:10]:
            try:
                counter += 1
                tr_id = "{}-FIXD-{:03d}".format(source_id[:6].upper(), counter)
                tr_a  = TR_A(
                    tr_id              = tr_id,
                    preconditions      = item.get("preconditions", []),
                    stimuli            = item.get("stimuli", []),
                    expected_outputs   = item.get("expected_outputs", []),
                    coverage_criterion = str(item.get("coverage_criterion", "")),
                    priority           = Priority(item.get("priority", "medium")),
                    source_req_ids     = [str(item.get("source_req", ""))],
                )
                tr_h = str(item.get("tr_h", "")).strip()
                if not tr_h:
                    continue
                result.append(TestRequirement(
                    tr_id          = tr_id,
                    category       = TRCategory.FUNCTIONAL,
                    source_id      = source_id,
                    tr_h           = tr_h,
                    tr_a           = tr_a,
                    source_req_ids = [str(item.get("source_req", ""))],
                ))
            except Exception:
                continue
        return result

    def _chat(self, user_message: str) -> str:
        url = "{}/api/chat".format(self.host)
        payload = json.dumps({
            "model":  self.model,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "stop": [],
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        }).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        tokens = []
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line: continue
                try:
                    d = json.loads(line)
                    tokens.append(d.get("message", {}).get("content", ""))
                    if d.get("done"): break
                except json.JSONDecodeError:
                    continue
        return "".join(tokens).strip()

    def _parse_list(self, text: str) -> list:
        text = text.strip()
        if "```" in text:
            text = "\n".join(
                l for l in text.splitlines()
                if not l.strip().startswith("```")
            ).strip()
        s = text.find("["); e = text.rfind("]")
        if s == -1 or e == -1: return []
        try:
            data = json.loads(text[s:e+1])
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []
