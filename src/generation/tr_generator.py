"""
tr_generator.py  --  Stage 4: Test Requirement Generation

Generates structured test requirements from:
  - demand model (Stage 2 output)
  - consistency issue list (Stage 3 output)

Five TR categories:
  1. functional           : from functional_reqs in demand model
  2. nfr_constraint       : from nfr_constraints + runtime_constraints
  3. runtime_environment  : from runtime_constraints + exception_handlers
  4. interface_exception  : from interfaces in entity data + exceptions
  5. consistency_driven   : one TR per consistency issue

Output:
  TR-H (EARS NL): "When <trigger>, the <system> shall <response>."
  TR-A (JSON)   : {tr_id, preconditions[], stimuli[], expected_outputs[],
                   coverage_criterion, priority, source_req_ids[]}
"""

import json, os, sys, time, urllib.request
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
from schemas.demand_schema       import DemandModel
from schemas.consistency_schema  import ConsistencyReport
from schemas.tr_schema           import (
    TestRequirement, TRDoc, TR_A, TRCategory, Priority
)

OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME  = "llama3.1:70b"
TEMPERATURE = 0.1
MAX_TOKENS  = 1500
TIMEOUT     = 600

SYSTEM_PROMPT = (
    "You are a test requirements engineer. "
    "You output ONLY valid JSON arrays. "
    "Never output markdown, explanation, or text outside the JSON array."
)

# EARS template reminder for the LLM
EARS_REMINDER = (
    "EARS format for TR-H: "
    "'When <trigger>, the <system> shall <response>.'"
    " OR "
    "'The <system> shall <response> within <constraint>.'"
)


class TRGenerator:

    def __init__(self, host=OLLAMA_HOST, model=MODEL_NAME,
                 temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                 timeout=TIMEOUT):
        self.host        = host.rstrip("/")
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout
        self._tr_counter = 0

    # -- Public ---------------------------------------------------------------

    def generate(
        self,
        demand_model:        DemandModel,
        consistency_reports: List[ConsistencyReport],
        verbose: bool = True,
    ) -> TRDoc:
        self._tr_counter = 0
        source_id = demand_model.source_id
        all_trs: List[TestRequirement] = []

        # Category 1: Functional TRs
        if verbose: print("  Cat 1: Functional TRs ...")
        t0 = time.time()
        trs = self._gen_functional(demand_model)
        all_trs.extend(trs)
        if verbose: print("    -> {}  ({:.0f}s)".format(len(trs), time.time()-t0))

        # Category 2: NFR Constraint TRs
        if verbose: print("  Cat 2: NFR/Constraint TRs ...")
        t0 = time.time()
        trs = self._gen_nfr(demand_model)
        all_trs.extend(trs)
        if verbose: print("    -> {}  ({:.0f}s)".format(len(trs), time.time()-t0))

        # Category 3: Runtime/Environment TRs
        if verbose: print("  Cat 3: Runtime/Environment TRs ...")
        t0 = time.time()
        trs = self._gen_runtime(demand_model)
        all_trs.extend(trs)
        if verbose: print("    -> {}  ({:.0f}s)".format(len(trs), time.time()-t0))

        # Category 4: Interface/Exception TRs
        if verbose: print("  Cat 4: Interface/Exception TRs ...")
        t0 = time.time()
        trs = self._gen_interface_exception(demand_model)
        all_trs.extend(trs)
        if verbose: print("    -> {}  ({:.0f}s)".format(len(trs), time.time()-t0))

        # Category 5: Consistency-driven TRs
        all_issues = [
            issue
            for report in consistency_reports
            for issue in report.issues
            if source_id in (report.source_a, report.source_b)
               or report.source_a == "all_sources"
               or "intra_{}".format(source_id) in report.comparison_id
        ]
        if all_issues:
            if verbose:
                print("  Cat 5: Consistency-driven TRs ({} issues) ...".format(
                    len(all_issues)
                ))
            t0 = time.time()
            trs = self._gen_consistency_driven(demand_model, all_issues)
            all_trs.extend(trs)
            if verbose: print("    -> {}  ({:.0f}s)".format(len(trs), time.time()-t0))

        doc = TRDoc(
            source_id   = source_id,
            trs         = all_trs,
            tr_h_list   = [tr.tr_h for tr in all_trs],
            tr_a_list   = [tr.tr_a for tr in all_trs],
            total_count = len(all_trs),
        )
        return doc

    def save_tr_doc(self, doc: TRDoc, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "{}_tr_doc.json".format(doc.source_id))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc.model_dump(), f, indent=2, ensure_ascii=False)
        return out_path

    # -- Category generators --------------------------------------------------

    def _gen_functional(self, model: DemandModel) -> List[TestRequirement]:
        reqs = [r.description for r in model.functional_reqs[:12]]
        if not reqs:
            return []

        # Scale the output cap to genuinely accommodate the stated
        # normal/boundary/negative triple pattern (Section 7.1.1), rather
        # than a flat cap that silently discards TRs whenever there are
        # more than 5 input requirements (5*3=15, the previous flat cap).
        # See reviewer comment R4a.
        max_items = len(reqs) * 3

        prompt = (
            "Generate functional test requirements from these software "
            "requirements.\n\n"
            "{ears}\n\n"
            "Requirements:\n{reqs}\n\n"
            "For each requirement generate:\n"
            "  1. A normal-path TR\n"
            "  2. A boundary-condition TR\n"
            "  3. A negative-path TR (invalid input or error state)\n\n"
            "Return JSON array (max {max_items} items). Each item:\n"
            "  tr_h: EARS-format test requirement string\n"
            "  preconditions: array of strings\n"
            "  stimuli: array of strings (inputs/triggers)\n"
            "  expected_outputs: array of strings\n"
            "  coverage_criterion: string\n"
            "  priority: high/medium/low\n"
            "  source_req: the original requirement this TR covers\n\n"
            "JSON array:"
        ).format(
            ears=EARS_REMINDER,
            reqs="\n".join("- "+r for r in reqs),
            max_items=max_items,
        )

        # Scale the output token budget with max_items -- the previous fix
        # (scaling max_items to len(reqs)*3) asked the model for up to
        # twice as many fully-detailed JSON objects without also scaling
        # the fixed 1500-token output budget, causing truncation-induced
        # complete generation failures on exactly the documents with more
        # functional requirements (confirmed: 4/7 documents regressed to
        # 0 Cat.1 TRs after that change). ~120 tokens/item is a generous
        # estimate for one TR-H/TR-A object (7 fields, 3 array fields);
        # floor at the original 1500 default so small requests are
        # unaffected.
        call_max_tokens = max(1500, max_items * 120)

        for attempt in range(2):
            trs = self._parse_and_build(
                self._chat(prompt, max_tokens=call_max_tokens),
                model.source_id, TRCategory.FUNCTIONAL
            )
            if trs:
                return trs
        return trs  # both attempts failed; return the (empty) result rather than raise

    def _gen_nfr(self, model: DemandModel) -> List[TestRequirement]:
        nfr = [c.description for c in model.nfr_constraints[:10]]
        rt  = [c.description for c in model.runtime_constraints[:6]]
        if not nfr and not rt:
            return []
        all_c = nfr + rt
        prompt = (
            "Generate NFR and constraint test requirements.\n\n"
            "{ears}\n\n"
            "Constraints to test:\n{constraints}\n\n"
            "For each constraint generate EXACTLY ONE test requirement that "
            "verifies the constraint is met, including the measurable "
            "threshold. Do not generate more than one TR per constraint.\n\n"
            "Return JSON array (max 12 items). Each item:\n"
            "  tr_h: EARS-format test requirement with measurable threshold\n"
            "  preconditions: array of strings\n"
            "  stimuli: array of strings\n"
            "  expected_outputs: array of strings\n"
            "  coverage_criterion: string\n"
            "  priority: high/medium/low\n"
            "  source_req: the constraint this TR covers\n\n"
            "JSON array:"
        ).format(
            ears=EARS_REMINDER,
            constraints="\n".join("- "+c for c in all_c),
        )
        return self._parse_and_build(
            self._chat(prompt), model.source_id, TRCategory.NFR_CONSTRAINT
        )

    def _gen_runtime(self, model: DemandModel) -> List[TestRequirement]:
        exc = model.exception_handlers[:10]
        rt  = [c.description for c in model.runtime_constraints[:8]]
        if not exc and not rt:
            return []
        prompt = (
            "Generate runtime and environment test requirements.\n\n"
            "{ears}\n\n"
            "Runtime constraints:\n{rt}\n"
            "Exception conditions:\n{exc}\n\n"
            "For each item generate EXACTLY ONE test requirement that "
            "verifies the system handles this runtime condition correctly. "
            "Do not generate more than one TR per item.\n\n"
            "Return JSON array (max 10 items). Each item:\n"
            "  tr_h: EARS-format test requirement\n"
            "  preconditions: array of strings\n"
            "  stimuli: array of strings\n"
            "  expected_outputs: array of strings\n"
            "  coverage_criterion: string\n"
            "  priority: high/medium/low\n"
            "  source_req: the condition this TR covers\n\n"
            "JSON array:"
        ).format(
            ears=EARS_REMINDER,
            rt ="\n".join("- "+r for r in rt) if rt else "(none)",
            exc="\n".join("- "+e for e in exc) if exc else "(none)",
        )
        return self._parse_and_build(
            self._chat(prompt), model.source_id, TRCategory.RUNTIME_ENVIRONMENT
        )

    def _gen_interface_exception(
        self, model: DemandModel
    ) -> List[TestRequirement]:
        use_cases = [
            "{} -> {}".format(uc.actor, uc.goal)
            for uc in model.use_cases[:8]
        ]
        exc = model.exception_handlers[:8]
        if not use_cases:
            return []
        prompt = (
            "Generate interface and exception handling test requirements.\n\n"
            "{ears}\n\n"
            "Use case flows:\n{ucs}\n"
            "Exception conditions:\n{exc}\n\n"
            "Generate test requirements covering:\n"
            "1. Interface interaction paths from the use cases\n"
            "2. Exception handling for each exception condition\n"
            "3. Process flow verification\n\n"
            "Return JSON array (max 12 items). Each item:\n"
            "  tr_h: EARS-format test requirement\n"
            "  preconditions: array of strings\n"
            "  stimuli: array of strings\n"
            "  expected_outputs: array of strings\n"
            "  coverage_criterion: string\n"
            "  priority: high/medium/low\n"
            "  source_req: the use case or exception this TR covers\n\n"
            "JSON array:"
        ).format(
            ears=EARS_REMINDER,
            ucs ="\n".join("- "+u for u in use_cases),
            exc ="\n".join("- "+e for e in exc) if exc else "(none)",
        )
        return self._parse_and_build(
            self._chat(prompt), model.source_id, TRCategory.INTERFACE_EXCEPTION
        )

    def _gen_consistency_driven(
        self, model: DemandModel, issues
    ) -> List[TestRequirement]:
        # Previously capped at issues[:10], silently violating the stated
        # "one TR per consistency issue" rule whenever a document had more
        # than 10 relevant issues (confirmed to affect auth_system_srs:
        # 14 relevant issues, only 9 Cat.5 TRs produced -- see reviewer
        # comment R4b). Raised to a generous ceiling (50) that comfortably
        # covers every document observed in this evaluation (max 14) while
        # still guarding against a pathological future document with an
        # unreasonably large issue count blowing the prompt size.
        issue_lines = [
            "[{}] ({}) {}".format(
                iss.issue_id, iss.issue_type.value, iss.description
            )
            for iss in issues[:50]
        ]
        prompt = (
            "Generate consistency-driven test requirements. Each TR must "
            "specifically target one consistency issue to expose it as a "
            "verifiable test failure.\n\n"
            "{ears}\n\n"
            "Consistency issues:\n{issues}\n\n"
            "For each issue generate one test requirement that would fail "
            "if the issue is not resolved.\n\n"
            "Return JSON array. Each item:\n"
            "  tr_h: EARS-format test requirement targeting the specific issue\n"
            "  preconditions: array of strings\n"
            "  stimuli: array of strings\n"
            "  expected_outputs: array of strings\n"
            "  coverage_criterion: string\n"
            "  priority: high/medium/low\n"
            "  source_req: the issue_id this TR targets\n\n"
            "JSON array:"
        ).format(
            ears  =EARS_REMINDER,
            issues="\n".join("- "+l for l in issue_lines),
        )
        return self._parse_and_build(
            self._chat(prompt), model.source_id, TRCategory.CONSISTENCY_DRIVEN
        )

    # -- LLM + parsing --------------------------------------------------------

    def _chat(self, user_message: str, max_tokens: int = None) -> str:
        url = "{}/api/chat".format(self.host)
        payload = json.dumps({
            "model":  self.model,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
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

    def _parse_and_build(
        self, text: str, source_id: str, category: TRCategory
    ) -> List[TestRequirement]:
        text = text.strip()
        if "```" in text:
            text = "\n".join(
                l for l in text.splitlines()
                if not l.strip().startswith("```")
            ).strip()
        s = text.find("["); e = text.rfind("]")
        if s == -1 or e == -1:
            return []
        try:
            data = json.loads(text[s:e+1])
        except json.JSONDecodeError:
            return []

        result = []
        # Previously hardcoded at data[:15] -- a SHARED cap applied after
        # parsing, downstream of and uncoordinated with each category's own
        # prompt-level cap (Functional's len(reqs)*3, NFR's 12, Runtime's
        # 10, etc.). Confirmed as the actual binding constraint for several
        # Cat1 documents that landed at exactly 15 despite their own prompt
        # requesting far more (e.g. nasa_srs_v1: 10 FRs, prompt cap 30, but
        # this line silently truncated the result back to 15 regardless).
        # Raised to a generous ceiling well above every category's own cap,
        # so each category's own prompt-level limit is the actual binding
        # constraint, with this line acting only as a safety net against a
        # pathologically oversized LLM response.
        for item in data[:50]:
            try:
                self._tr_counter += 1
                tr_id = "{}-{}-{:03d}".format(
                    source_id[:6].upper(),
                    category.value[:4].upper(),
                    self._tr_counter
                )
                tr_a = TR_A(
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
                    category       = category,
                    source_id      = source_id,
                    tr_h           = tr_h,
                    tr_a           = tr_a,
                    source_req_ids = [str(item.get("source_req", ""))],
                ))
            except (KeyError, ValueError):
                continue
        return result