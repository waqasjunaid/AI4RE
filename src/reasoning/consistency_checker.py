"""
consistency_checker.py  --  Stage 3: Consistency Check

Five checks, scoped to documents that share a bundle_id (i.e. describe
the SAME system):
  1. SRS vs User Manual       : function coverage gaps
  2. Runtime vs SRS           : execution condition compatibility
  3. Design vs SRS            : constraint implementation gaps
  4. Intra-source checks      : each document checked against itself
     (functional reqs vs NFR, functional reqs vs exceptions)
  5. Terminology              : term usage consistency within a bundle

Cross-document checks (1-3, 5) are only run between documents that share
a bundle_id, so that consistency findings are always about the same
underlying system. Intra-source checks (4) run for every document
regardless of bundling, since they only ever inspect one document
against itself.

NOTE ON A PRIOR VERSION OF THIS MODULE: an earlier version of check_all()
ran cross-document checks over the full `models` dict without any
notion of "same system," which meant Checks 1-3 could compare artifacts
describing unrelated systems (e.g. a NASA SRS against a GNU Bash
manual) when both were present in the same pipeline run. That behavior
has been removed; see check_all()'s docstring for the corrected,
bundle-aware default.
"""

import json, os, sys, time, urllib.request
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
from schemas.demand_schema      import DemandModel
from schemas.consistency_schema import (
    ConsistencyIssue, ConsistencyReport, IssueType, Severity
)

OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME  = "llama3.1:70b"
TEMPERATURE = 0.0
MAX_TOKENS  = 1500
TIMEOUT     = 600

SYSTEM_PROMPT = (
    "You are a software requirements consistency analyst. "
    "You output ONLY valid JSON arrays. "
    "Never output markdown, explanation, or text outside the JSON array."
)


class ConsistencyChecker:

    def __init__(self, host=OLLAMA_HOST, model=MODEL_NAME,
                 temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                 timeout=TIMEOUT):
        self.host        = host.rstrip("/")
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout

    # -- Public ---------------------------------------------------------------

    def check_all(
        self,
        models: Dict[str, DemandModel],
        bundles: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ) -> List[ConsistencyReport]:
        """
        Run Stage 3 consistency checks.

        IMPORTANT (see Section 9.1.1 / reviewer comment R1):
        Cross-document checks (1-3) and the terminology check (5) are only
        meaningful between artifacts that describe THE SAME SYSTEM. This
        method therefore groups `models` into bundles via `bundles`
        (a source_id -> bundle_id mapping) and only cross-checks documents
        that share a bundle_id.

        If `bundles` is not supplied, every document is treated as its own
        singleton bundle by default -- i.e. NO cross-document checks fire.
        This is the safe default: it prevents silently re-introducing the
        cross-domain comparison bug (e.g. comparing a NASA SRS against a
        GNU Bash manual) that produced non-meaningful "cross-document"
        issues in the original evaluation. To obtain cross-document
        findings, the caller must explicitly declare which documents
        belong to the same system via `bundles`.

        The intra-source check (4) is unaffected by bundling: it only ever
        inspects one document against itself, so it always runs for every
        document regardless of bundle membership.
        """
        reports: List[ConsistencyReport] = []

        if bundles is None:
            bundles = {sid: sid for sid in models}  # each doc = its own bundle
            if verbose:
                print("  No bundle mapping supplied -- each document treated as "
                      "its own singleton bundle (cross-document checks disabled "
                      "by default; see check_all() docstring).")

        bundle_groups: Dict[str, List[str]] = {}
        for sid in models:
            bid = bundles.get(sid, sid)  # unmapped docs default to their own bundle
            bundle_groups.setdefault(bid, []).append(sid)

        for bundle_id, sids in bundle_groups.items():
            bundle_models = {sid: models[sid] for sid in sids}

            if verbose:
                print("  Bundle '{}': {} document(s) -- {}".format(
                    bundle_id, len(bundle_models), ", ".join(sids)
                ))

            if len(bundle_models) >= 2:
                srs     = self._find(bundle_models, ["srs", "nasa"])
                manual  = self._find(bundle_models, ["manual", "bash", "user"])
                runtime = self._find(bundle_models, ["swagger", "petstore", "runtime"])
                design  = self._find(bundle_models, ["design"])  # "auth" removed: was
                                                                    # ambiguous once multiple
                                                                    # auth_system_* docs coexist

                if srs and manual:
                    if verbose: print("    Check 1: SRS vs User Manual ...")
                    r = self._check_srs_vs_manual(srs, manual)
                    reports.append(r)
                    if verbose: print("      -> {} issues".format(len(r.issues)))

                if runtime and srs:
                    if verbose: print("    Check 2: Runtime vs SRS ...")
                    r = self._check_runtime_vs_srs(runtime, srs)
                    reports.append(r)
                    if verbose: print("      -> {} issues".format(len(r.issues)))

                if design and srs:
                    if verbose: print("    Check 3: Design vs SRS ...")
                    r = self._check_design_vs_srs(design, srs)
                    reports.append(r)
                    if verbose: print("      -> {} issues".format(len(r.issues)))

                if verbose: print("    Check 5: Terminology (within bundle) ...")
                r = self._check_terminology(bundle_models)
                if r.issues:
                    reports.append(r)
                if verbose: print("      -> {} issues".format(len(r.issues)))
            else:
                if verbose:
                    print("    Only {} document(s) in this bundle -- skipping "
                          "cross-document checks 1-3 and terminology check 5 "
                          "(need >=2 same-system artifacts).".format(len(bundle_models)))

            # Intra-source check: always runs per document, independent of bundling
            for sid, model in bundle_models.items():
                r = self._check_intra_source(model)
                if r.issues:
                    reports.append(r)
                    if verbose:
                        print("    Check 4 (intra-source) -> {} ({} issues)".format(
                            sid, len(r.issues)
                        ))

        return reports

    def save_reports(
        self, reports: List[ConsistencyReport], output_dir: str
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "consistency_report.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                [r.model_dump() for r in reports],
                f, indent=2, ensure_ascii=False
            )
        return out_path

    # -- Cross-document checks ------------------------------------------------

    def _check_srs_vs_manual(
        self, srs: DemandModel, manual: DemandModel
    ) -> ConsistencyReport:
        srs_reqs    = [r.description for r in srs.functional_reqs[:12]]
        manual_objs = manual.system_objectives[:12]
        manual_ucs  = [uc.goal for uc in manual.use_cases[:8]]

        prompt = (
            "Compare two software document summaries for consistency gaps.\n\n"
            "SOURCE A (Requirements Specification):\n"
            "{srs_reqs}\n\n"
            "SOURCE B (User Manual/Scenarios):\n"
            "Objectives: {manual_objs}\n"
            "Use case goals: {manual_ucs}\n\n"
            "Find:\n"
            "1. User-visible features in B with no matching requirement in A "
            "(omission)\n"
            "2. Requirements in A for user-visible features absent from B "
            "(omission)\n"
            "3. Contradictory statements between A and B (conflict)\n"
            "4. Vague items needing clarification (ambiguity)\n\n"
            "Return JSON array. Each item: issue_id, issue_type "
            "(conflict/omission/ambiguity/to_confirm), description, "
            "severity (blocker/major/minor), suggested_fix.\n"
            "Return [] if no issues.\nJSON array:"
        ).format(
            srs_reqs    ="\n".join("- "+r for r in srs_reqs),
            manual_objs ="\n".join("- "+o for o in manual_objs),
            manual_ucs  ="\n".join("- "+u for u in manual_ucs),
        )
        raw    = self._chat(prompt)
        issues = self._parse_issues(raw, srs.source_id, manual.source_id)
        return self._make_report("srs_vs_manual", srs.source_id, manual.source_id, issues)

    def _check_runtime_vs_srs(
        self, runtime: DemandModel, srs: DemandModel
    ) -> ConsistencyReport:
        rt_items = (
            [c.description for c in runtime.nfr_constraints[:10]] +
            [c.description for c in runtime.runtime_constraints[:10]] +
            runtime.exception_handlers[:8]
        )
        srs_reqs = [r.description for r in srs.functional_reqs[:12]]
        srs_nfr  = [c.description for c in srs.nfr_constraints[:6]]
        srs_rt   = [c.description for c in srs.runtime_constraints[:8]]

        prompt = (
            "Check if runtime environment constraints are compatible with "
            "software requirements.\n\n"
            "RUNTIME constraints/exceptions:\n"
            "{rt}\n\n"
            "SRS requirements:\n"
            "Functional: {func}\n"
            "NFR: {nfr}\n"
            "Runtime/operational constraints: {srs_rt}\n\n"
            "Find:\n"
            "1. Runtime conditions preventing requirements from being met "
            "(conflict) -- INCLUDING numeric mismatches (e.g. a rate limit, "
            "duration, or threshold that differs between RUNTIME and SRS)\n"
            "2. Requirements assuming undocumented runtime conditions "
            "(omission)\n"
            "3. Implicit execution conditions not addressed by runtime specs "
            "(to_confirm)\n\n"
            "Return JSON array. Each item: issue_id, issue_type, description, "
            "severity, suggested_fix.\n"
            "Return [] if no issues.\nJSON array:"
        ).format(
            rt    ="\n".join("- "+r for r in rt_items[:20]),
            func  ="\n".join("- "+r for r in srs_reqs),
            nfr   ="\n".join("- "+r for r in srs_nfr) if srs_nfr else "(none)",
            srs_rt="\n".join("- "+r for r in srs_rt) if srs_rt else "(none)",
        )
        raw    = self._chat(prompt)
        issues = self._parse_issues(raw, runtime.source_id, srs.source_id)
        return self._make_report("runtime_vs_srs", runtime.source_id, srs.source_id, issues)

    def _check_design_vs_srs(
        self, design: DemandModel, srs: DemandModel
    ) -> ConsistencyReport:
        d_reqs  = [r.description for r in design.functional_reqs[:12]]
        d_nfr   = [c.description for c in design.nfr_constraints[:10]]
        d_rt    = [c.description for c in design.runtime_constraints[:8]]
        d_roles = design.user_roles[:10]
        srs_reqs= [r.description for r in srs.functional_reqs[:12]]
        srs_nfr = [c.description for c in srs.nfr_constraints[:6]]
        srs_rt  = [c.description for c in srs.runtime_constraints[:8]]

        prompt = (
            "Check whether the system design correctly addresses the "
            "software requirements.\n\n"
            "DESIGN components/functions/constraints:\n"
            "Components: {roles}\n"
            "Functions: {d_func}\n"
            "Constraints (NFR): {d_nfr}\n"
            "Constraints (runtime/operational): {d_rt}\n\n"
            "SRS requirements:\n"
            "Functional: {srs_func}\n"
            "NFR: {srs_nfr}\n"
            "Runtime/operational constraints: {srs_rt}\n\n"
            "Find:\n"
            "1. SRS requirements not addressed in the design (omission)\n"
            "2. Design decisions contradicting SRS constraints (conflict) -- "
            "INCLUDING numeric mismatches (e.g. a duration, threshold, or "
            "limit that differs between the DESIGN constraints and the SRS "
            "constraints above, even if both describe the same concept using "
            "different wording)\n"
            "3. Design components not traceable to any requirement (to_confirm)\n\n"
            "Return JSON array. Each item: issue_id, issue_type, description, "
            "severity, suggested_fix.\n"
            "Return [] if no issues.\nJSON array:"
        ).format(
            roles   =", ".join(d_roles),
            d_func  ="\n".join("- "+r for r in d_reqs),
            d_nfr   ="\n".join("- "+c for c in d_nfr) if d_nfr else "(none)",
            d_rt    ="\n".join("- "+c for c in d_rt) if d_rt else "(none)",
            srs_func="\n".join("- "+r for r in srs_reqs),
            srs_nfr ="\n".join("- "+c for c in srs_nfr) if srs_nfr else "(none)",
            srs_rt  ="\n".join("- "+c for c in srs_rt) if srs_rt else "(none)",
        )
        raw    = self._chat(prompt)
        issues = self._parse_issues(raw, design.source_id, srs.source_id)
        return self._make_report("design_vs_srs", design.source_id, srs.source_id, issues)

    # -- Intra-source check ---------------------------------------------------

    def _check_intra_source(self, model: DemandModel) -> ConsistencyReport:
        """
        Check a single document for internal consistency:
        - Functional reqs vs NFR constraints (does any req violate a constraint?)
        - Functional reqs vs exception handlers (are all exceptions covered?)
        - Use case goals vs functional reqs (are all goals addressed by reqs?)
        """
        func_reqs  = [r.description for r in model.functional_reqs[:12]]
        nfr        = [c.description for c in model.nfr_constraints[:8]]
        exceptions = model.exception_handlers[:10]
        uc_goals   = [uc.goal for uc in model.use_cases[:8]]

        if not func_reqs:
            return self._make_report(
                "intra_{}".format(model.source_id),
                model.source_id, model.source_id, []
            )

        prompt = (
            "Check this software document for internal consistency issues.\n\n"
            "Document: {source_id}\n\n"
            "Functional requirements:\n{func}\n\n"
            "NFR constraints:\n{nfr}\n\n"
            "Exception handlers:\n{exc}\n\n"
            "Use case goals:\n{goals}\n\n"
            "Find:\n"
            "1. Functional reqs that may violate an NFR constraint (conflict)\n"
            "2. Use case goals with no matching functional requirement (omission)\n"
            "3. Exception conditions not covered by any requirement (omission)\n"
            "4. Ambiguous or untestable requirement statements (ambiguity)\n"
            "5. Requirements missing measurable acceptance criteria (to_confirm)\n\n"
            "Return JSON array (max 8 items). Each item: issue_id, issue_type "
            "(conflict/omission/ambiguity/to_confirm), description, "
            "severity (blocker/major/minor), suggested_fix.\n"
            "Return [] if no issues.\nJSON array:"
        ).format(
            source_id=model.source_id,
            func ="\n".join("- "+r for r in func_reqs),
            nfr  ="\n".join("- "+c for c in nfr) if nfr else "(none)",
            exc  ="\n".join("- "+e for e in exceptions) if exceptions else "(none)",
            goals="\n".join("- "+g for g in uc_goals) if uc_goals else "(none)",
        )
        raw    = self._chat(prompt)
        issues = self._parse_issues(raw, model.source_id, model.source_id)
        return self._make_report(
            "intra_{}".format(model.source_id),
            model.source_id, model.source_id, issues
        )

    # -- Terminology check ----------------------------------------------------

    def _check_terminology(
        self, models: Dict[str, DemandModel]
    ) -> ConsistencyReport:
        all_terms: Dict[str, List[str]] = {}
        for src_id, model in models.items():
            for term in list(model.glossary.keys())[:20]:
                all_terms.setdefault(term.lower(), []).append(src_id)
            for role in model.user_roles[:8]:
                all_terms.setdefault(role.lower(), []).append(src_id)

        if not all_terms:
            return self._make_report("terminology", "all", "all", [])

        term_lines = [
            "{} (in: {})".format(term, ", ".join(set(srcs)))
            for term, srcs in list(all_terms.items())[:40]
        ]

        prompt = (
            "Review these terms used across software documents and identify "
            "terminology inconsistencies.\n\n"
            "Terms and sources:\n{terms}\n\n"
            "Find:\n"
            "1. Same concept with different names across documents (ambiguity)\n"
            "2. Same term used with different meanings (conflict)\n"
            "3. Key terms used but not defined in any glossary (to_confirm)\n\n"
            "Return JSON array (max 6 items). Each item: issue_id, issue_type, "
            "description, severity, suggested_fix.\n"
            "Return [] if no issues.\nJSON array:"
        ).format(terms="\n".join("- "+t for t in term_lines))

        raw    = self._chat(prompt)
        issues = self._parse_issues(raw, "all_sources", "all_sources")
        return self._make_report("terminology", "all_sources", "all_sources", issues)

    # -- LLM ------------------------------------------------------------------

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

    # -- Helpers --------------------------------------------------------------

    def _parse_issues(
        self, text: str, src_a: str, src_b: str
    ) -> List[ConsistencyIssue]:
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
        issues = []
        for i, item in enumerate(data[:20], start=1):
            try:
                issues.append(ConsistencyIssue(
                    issue_id      = item.get("issue_id", "ISS-{:03d}".format(i)),
                    issue_type    = IssueType(item.get("issue_type", "to_confirm")),
                    source_a_id   = src_a,
                    source_b_id   = src_b,
                    description   = str(item.get("description", "")),
                    severity      = Severity(item.get("severity", "minor")),
                    suggested_fix = item.get("suggested_fix"),
                ))
            except (KeyError, ValueError):
                continue
        return issues

    def _make_report(
        self, cmp_id: str, src_a: str, src_b: str,
        issues: List[ConsistencyIssue]
    ) -> ConsistencyReport:
        return ConsistencyReport(
            comparison_id      = cmp_id,
            source_a           = src_a,
            source_b           = src_b,
            issues             = issues,
            total_conflicts    = sum(1 for i in issues if i.issue_type == IssueType.CONFLICT),
            total_omissions    = sum(1 for i in issues if i.issue_type == IssueType.OMISSION),
            total_ambiguities  = sum(1 for i in issues if i.issue_type == IssueType.AMBIGUITY),
            total_to_confirm   = sum(1 for i in issues if i.issue_type == IssueType.TO_CONFIRM),
        )

    def _find(
        self, models: Dict[str, DemandModel], keywords: List[str]
    ) -> Optional[DemandModel]:
        for k in keywords:
            for src_id, model in models.items():
                if k.lower() in src_id.lower():
                    return model
        return None