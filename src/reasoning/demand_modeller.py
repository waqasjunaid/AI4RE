"""
demand_modeller.py
Stage 2: Demand Understanding

Builds a structured DemandModel from ExtractedEntity objects.
Uses a 4-step Chain-of-Thought (CoT) prompt sequence.

Changes from v1:
- Handles large entity sets (2000+) by sampling top entities per type
- Better prompts for role extraction (handles system components as roles)
- Runtime constraint step now also reads interface entities for runtime docs
- Glossary and exception steps improved
"""

import json, os, sys, time, urllib.request
from typing import List, Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
from schemas.entity_schema import ExtractedEntity, EntityType
from schemas.demand_schema  import (
    DemandModel, UseCase, FunctionalRequirement,
    NFRConstraint, RuntimeConstraint
)

OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME  = "llama3.1:70b"
TEMPERATURE = 0.0
MAX_TOKENS  = 1500
TIMEOUT     = 600

# Max entities to include in each prompt to stay within context window
MAX_PER_TYPE = 30

SYSTEM_PROMPT = (
    "You are a requirements engineer. "
    "You output ONLY valid JSON. "
    "Never output markdown, explanation, or any text outside the JSON."
)


class DemandModeller:

    def __init__(self, host=OLLAMA_HOST, model=MODEL_NAME,
                 temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                 timeout=TIMEOUT):
        self.host        = host.rstrip("/")
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout

    # -- Public ---------------------------------------------------------------

    def build_demand_model(
        self,
        entities:  List[ExtractedEntity],
        source_id: str,
        verbose:   bool = True,
    ) -> DemandModel:
        if verbose:
            print("  Building demand model for: {}  ({} entities)".format(
                source_id, len(entities)
            ))

        by_type = self._group_by_type(entities)

        # Step 1: user roles
        if verbose: print("  Step 1: user roles ...")
        t0 = time.time()
        roles = self._step1_roles(by_type)
        if verbose: print("    -> {}  ({:.0f}s)".format(roles, time.time()-t0))

        # Step 2: use cases
        if verbose: print("  Step 2: use cases ...")
        t0 = time.time()
        use_cases = self._step2_use_cases(by_type, roles)
        if verbose: print("    -> {} use cases  ({:.0f}s)".format(
            len(use_cases), time.time()-t0))

        # Step 3: functional requirements
        if verbose: print("  Step 3: functional requirements ...")
        t0 = time.time()
        func_reqs = self._step3_functional_reqs(by_type, source_id)
        if verbose: print("    -> {} reqs  ({:.0f}s)".format(
            len(func_reqs), time.time()-t0))

        # Step 4: constraints
        if verbose: print("  Step 4: constraints ...")
        t0 = time.time()
        nfr, runtime = self._step4_constraints(by_type, source_id)
        if verbose: print("    -> {} NFR + {} runtime  ({:.0f}s)".format(
            len(nfr), len(runtime), time.time()-t0))

        # Glossary: parse glossary_term entities (term: definition format)
        glossary = {}
        for e in by_type.get("glossary_term", [])[:50]:
            parts = e.content.split(":", 1)
            if len(parts) == 2:
                glossary[parts[0].strip()] = parts[1].strip()
            else:
                glossary[e.content[:40]] = e.content

        # Exceptions
        exceptions = [
            e.content for e in by_type.get("exception_condition", [])[:30]
        ]

        # System objectives: highest-confidence function_points
        fps = sorted(
            by_type.get("function_point", []),
            key=lambda e: e.confidence_score, reverse=True
        )
        objectives = list(dict.fromkeys(e.content for e in fps[:8]))

        return DemandModel(
            source_id           = source_id,
            system_objectives   = objectives,
            user_roles          = roles,
            use_cases           = use_cases,
            functional_reqs     = func_reqs,
            nfr_constraints     = nfr,
            runtime_constraints = runtime,
            glossary            = glossary,
            exception_handlers  = exceptions,
        )

    def save_demand_model(self, model: DemandModel, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "{}.json".format(model.source_id))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(model.model_dump(), f, indent=2, ensure_ascii=False)
        return out_path

    # -- CoT Steps ------------------------------------------------------------

    def _step1_roles(self, by_type: Dict) -> List[str]:
        actors = [e.content for e in by_type.get("actor_role", [])[:MAX_PER_TYPE]]
        # Also include interface_name for runtime/design docs
        # (components act as actors in those contexts)
        ifaces = [e.content for e in by_type.get("interface_name", [])[:10]]
        if not actors and not ifaces:
            return ["User", "System"]

        all_mentions = actors + ifaces
        prompt = (
            "Given these actor/role/component mentions from a software document:\n"
            "{mentions}\n\n"
            "Return a JSON array of unique, clean role names.\n"
            "Rules:\n"
            "- Merge duplicates and near-duplicates\n"
            "- Include both human roles (User, Admin) and system components "
            "(AuthService, Database) if they appear as actors\n"
            "- Remove noise words and non-role items\n"
            "- Maximum 10 items\n\n"
            "Example: [\"Administrator\", \"End User\", "
            "\"Authentication Service\", \"Database\"]\n\n"
            "JSON array:"
        ).format(mentions="\n".join("- " + a for a in all_mentions[:40]))

        raw = self._chat(prompt)
        return self._parse_list(raw)

    def _step2_use_cases(
        self, by_type: Dict, roles: List[str]
    ) -> List[UseCase]:
        fps = [e.content for e in by_type.get("function_point", [])[:MAX_PER_TYPE]]
        procs = [e.content for e in by_type.get("process_step", [])[:15]]
        if not fps:
            return []

        role_str = ", ".join(roles[:8]) if roles else "User, System"
        prompt = (
            "Given these system functions:\n{functions}\n\n"
            "And these process steps:\n{steps}\n\n"
            "User roles: {roles}\n\n"
            "Create use cases. Return a JSON array (max 10 items). "
            "Each item must have exactly these keys:\n"
            "  use_case_id    (e.g. UC-001)\n"
            "  actor          (one of the roles)\n"
            "  goal           (what the actor achieves)\n"
            "  preconditions  (array of strings)\n"
            "  steps          (array of strings, 2-4 steps)\n"
            "  postconditions (array of strings)\n"
            "  exceptions     (array of strings)\n\n"
            "JSON array:"
        ).format(
            functions="\n".join("- " + f for f in fps[:20]),
            steps="\n".join("- " + p for p in procs[:10]) if procs else "(none)",
            roles=role_str,
        )

        raw  = self._chat(prompt)
        data = self._parse_json_list(raw)

        use_cases = []
        for i, item in enumerate(data[:12], start=1):
            try:
                use_cases.append(UseCase(
                    use_case_id    = item.get("use_case_id", "UC-{:03d}".format(i)),
                    actor          = str(item.get("actor", "User")),
                    goal           = str(item.get("goal", "")),
                    preconditions  = item.get("preconditions", []),
                    steps          = item.get("steps", []),
                    postconditions = item.get("postconditions", []),
                    exceptions     = item.get("exceptions", []),
                ))
            except Exception:
                continue
        return use_cases

    def _step3_functional_reqs(
        self, by_type: Dict, source_id: str
    ) -> List[FunctionalRequirement]:
        fps = [e.content for e in by_type.get("function_point", [])[:MAX_PER_TYPE]]
        if not fps:
            return []

        prompt = (
            "Convert these system function descriptions into formal "
            "functional requirements using 'shall' statements.\n\n"
            "Functions:\n{functions}\n\n"
            "Return a JSON array (max 15 items). Each item:\n"
            "  req_id      (e.g. FR-001)\n"
            "  description (clear 'The system shall...' statement)\n"
            "  priority    (high / medium / low)\n\n"
            "JSON array:"
        ).format(functions="\n".join("- " + f for f in fps[:25]))

        raw  = self._chat(prompt)
        data = self._parse_json_list(raw)

        reqs = []
        for i, item in enumerate(data[:15], start=1):
            try:
                reqs.append(FunctionalRequirement(
                    req_id      = item.get("req_id", "FR-{:03d}".format(i)),
                    description = str(item.get("description", "")),
                    source_ids  = [source_id],
                    priority    = item.get("priority", "medium"),
                ))
            except Exception:
                continue
        return reqs

    def _step4_constraints(
        self, by_type: Dict, source_id: str
    ):
        constraints = [
            e.content for e in by_type.get("constraint", [])[:MAX_PER_TYPE]
        ]
        if not constraints:
            return [], []

        prompt = (
            "Classify these software constraints into NFR constraints "
            "and runtime constraints.\n\n"
            "Constraints:\n{constraints}\n\n"
            "Return a JSON object with exactly two keys:\n\n"
            "nfr_constraints: array of items with keys:\n"
            "  constraint_id  (e.g. NFR-001)\n"
            "  category       (performance / security / usability / "
            "reliability / compatibility)\n"
            "  description    (the constraint)\n"
            "  threshold      (measurable value if present, else null)\n\n"
            "runtime_constraints: array of items with keys:\n"
            "  constraint_id  (e.g. RT-001)\n"
            "  description    (the constraint)\n"
            "  affects_reqs   (array of strings, can be empty)\n\n"
            "JSON object:"
        ).format(
            constraints="\n".join("- " + c for c in constraints[:20])
        )

        raw  = self._chat(prompt)
        data = self._parse_json_obj(raw)

        nfr_list = []
        for i, item in enumerate(data.get("nfr_constraints", [])[:15], 1):
            try:
                nfr_list.append(NFRConstraint(
                    constraint_id = item.get("constraint_id", "NFR-{:03d}".format(i)),
                    category      = item.get("category", "performance"),
                    description   = str(item.get("description", "")),
                    threshold     = item.get("threshold"),
                    source_ids    = [source_id],
                ))
            except Exception:
                continue

        rt_list = []
        for i, item in enumerate(data.get("runtime_constraints", [])[:10], 1):
            try:
                rt_list.append(RuntimeConstraint(
                    constraint_id = item.get("constraint_id", "RT-{:03d}".format(i)),
                    description   = str(item.get("description", "")),
                    affects_reqs  = item.get("affects_reqs", []),
                    source_ids    = [source_id],
                ))
            except Exception:
                continue

        return nfr_list, rt_list

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

    # -- Parsers --------------------------------------------------------------

    def _strip_fences(self, text: str) -> str:
        if "```" in text:
            text = "\n".join(
                l for l in text.splitlines()
                if not l.strip().startswith("```")
            ).strip()
        return text

    def _parse_list(self, text: str) -> List[str]:
        text = self._strip_fences(text)
        s = text.find("["); e = text.rfind("]")
        if s == -1 or e == -1: return []
        try:
            data = json.loads(text[s:e+1])
            return [str(x) for x in data if x] if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    def _parse_json_list(self, text: str) -> list:
        text = self._strip_fences(text)
        s = text.find("["); e = text.rfind("]")
        if s == -1 or e == -1: return []
        try:
            data = json.loads(text[s:e+1])
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    def _parse_json_obj(self, text: str) -> dict:
        text = self._strip_fences(text)
        s = text.find("{"); e = text.rfind("}")
        if s == -1 or e == -1: return {}
        try:
            data = json.loads(text[s:e+1])
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _group_by_type(
        self, entities: List[ExtractedEntity]
    ) -> Dict[str, List[ExtractedEntity]]:
        groups: Dict[str, List[ExtractedEntity]] = {}
        for e in entities:
            key = e.entity_type.value
            groups.setdefault(key, []).append(e)
        return groups