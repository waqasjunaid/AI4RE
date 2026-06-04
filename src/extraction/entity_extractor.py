"""
entity_extractor.py  --  Stage 1 NER extraction via Ollama /api/chat

Works with llama3.2:3b (fast), llama3.1:70b (best quality).
qwen2.5:14b is known corrupted on this machine -- do not use it.
"""

import json, os, sys, time, urllib.request, urllib.error
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
from schemas.entity_schema import (
    ExtractedEntity, EntityType, ArtifactType, DocumentChunk
)

# -- Config -------------------------------------------------------------------
OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME  = "llama3.1:70b"  # confirmed working on this machine
TEMPERATURE = 0.0              # 0 = deterministic, most consistent JSON output
MAX_TOKENS  = 800              # enough for ~15 entities per chunk
TIMEOUT     = 600              # llama3.1:70b needs up to 10 min on first load

# -- Entity types per artifact ------------------------------------------------
ARTIFACT_ENTITY_TYPES = {
    "srs":         ["function_point","constraint","actor_role",
                    "interface_name","exception_condition"],
    "user_manual": ["function_point","actor_role","process_step",
                    "glossary_term","exception_condition"],
    "runtime":     ["interface_name","constraint",
                    "exception_condition","function_point"],
    "design":      ["interface_name","actor_role","function_point",
                    "constraint","glossary_term"],
}

SYSTEM_PROMPT = (
    "You are a JSON API. "
    "You output ONLY valid JSON arrays. "
    "Never output markdown, code fences, explanation, or any text "
    "outside the JSON array. "
    "If you find no entities output exactly: []"
)


class EntityExtractor:

    def __init__(self, host=OLLAMA_HOST, model=MODEL_NAME,
                 temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                 timeout=TIMEOUT):
        self.host        = host.rstrip("/")
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout
        self._check_connection()

    # -- Public ---------------------------------------------------------------

    def extract_from_chunks(self, chunks, max_chunks=None, verbose=True):
        if max_chunks is not None:
            chunks = chunks[:max_chunks]
        all_entities = []
        total = len(chunks)
        for i, chunk in enumerate(chunks, start=1):
            if verbose:
                print("  [{:3d}/{}] {} ... ".format(
                    i, total, chunk.chunk_id), end="", flush=True)
            try:
                t0 = time.time()
                entities = self._extract_chunk(chunk)
                elapsed  = time.time() - t0
                all_entities.extend(entities)
                if verbose:
                    print("{} entities  ({:.1f}s)".format(len(entities), elapsed))
            except Exception as e:
                if verbose:
                    print("ERROR: {}".format(e))
        return all_entities

    def save_entities(self, entities, output_dir, source_id=None):
        if not entities:
            print("  No entities to save.")
            return ""
        sid = source_id or entities[0].source_id
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "{}.json".format(sid))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([e.model_dump() for e in entities],
                      f, indent=2, ensure_ascii=False)
        return out_path

    # -- Internal -------------------------------------------------------------

    def _check_connection(self):
        try:
            with urllib.request.urlopen(
                "{}/api/tags".format(self.host), timeout=5
            ) as resp:
                data  = json.loads(resp.read().decode())
                names = [m["name"] for m in data.get("models", [])]
            base = self.model.split(":")[0]
            if not any(base in n for n in names):
                print("WARNING: '{}' not found. Available: {}".format(
                    self.model, names))
                print("Run: ollama pull {}".format(self.model))
            else:
                print("Ollama OK  |  model: {}  |  host: {}".format(
                    self.model, self.host))
        except Exception as e:
            raise ConnectionError(
                "Cannot reach Ollama at {}.\nRun: ollama serve\n"
                "Error: {}".format(self.host, e))

    def _build_user_message(self, chunk):
        atype      = chunk.artifact_type.value
        type_names = ARTIFACT_ENTITY_TYPES.get(atype, [
            "function_point","constraint","actor_role"])
        text = chunk.content[:1200]

        return (
            "Extract software entities from this {atype} document fragment.\n\n"
            "Return a JSON array. Each item must have exactly these three keys:\n"
            "  entity_type  (one of: {types})\n"
            "  content      (the extracted text, max 20 words)\n"
            "  confidence_score  (float 0.0-1.0)\n\n"
            "Entity type definitions:\n"
            "  function_point      = a system capability or behaviour\n"
            "  constraint          = a measurable rule or limit\n"
            "  actor_role          = a person, system, or component\n"
            "  interface_name      = an API endpoint, database, or named component\n"
            "  process_step        = a sequential action in a workflow\n"
            "  exception_condition = an error state or failure mode\n"
            "  glossary_term       = a defined term\n\n"
            "Document fragment:\n"
            "{text}\n\n"
            "JSON array:"
        ).format(atype=atype, types=", ".join(type_names), text=text)

    def _call_ollama_chat(self, user_message):
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
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    tokens.append(d.get("message", {}).get("content", ""))
                    if d.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
        return "".join(tokens).strip()

    def _extract_chunk(self, chunk):
        user_msg = self._build_user_message(chunk)
        raw      = self._call_ollama_chat(user_msg)
        items    = self._parse_json(raw)
        result   = []
        for item in items:
            try:
                etype   = str(item.get("entity_type","")).strip()
                content = str(item.get("content","")).strip()
                score   = float(item.get("confidence_score", 0.8))
                if not etype or not content:
                    continue
                result.append(ExtractedEntity(
                    source_id        = chunk.source_id,
                    artifact_type    = chunk.artifact_type,
                    entity_type      = EntityType(etype),
                    content          = content,
                    page_ref         = chunk.page_ref,
                    confidence_score = max(0.0, min(1.0, score)),
                ))
            except (KeyError, ValueError):
                continue
        return result

    def _parse_json(self, text):
        text = text.strip()
        if "```" in text:
            text = "\n".join(
                l for l in text.splitlines()
                if not l.strip().startswith("```")
            ).strip()
        start = text.find("[")
        end   = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]
        try:
            data = json.loads(text)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []