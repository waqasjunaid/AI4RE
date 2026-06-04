# AI4RE-Test: LLM-Driven Multi-Source Test Requirement Generation

Five-stage pipeline for automated test requirement generation from
heterogeneous software engineering documents using local LLMs.

**Author:** Dr. Waqas Junaid, School of Software,
Northwestern Polytechnical University, Xian, P.R. China

**Model:** llama3.1:70b via Ollama (local, no API key required)

---

## Quick Start

```bash
# Prerequisites
ollama serve                     # Terminal 1 -- keep running
ollama pull llama3.1:70b

# Install dependencies
pip install pydantic pymupdf python-docx lxml tiktoken

# Run all 5 stages
cd AI4RE-Test
python test_stage1.py            # Stage 1a: parse and chunk documents
python test_stage1_extraction.py # Stage 1b: extract entities (takes hours)
python test_stage2_demand.py     # Stage 2:  build demand models
python test_stage3_consistency.py# Stage 3:  check consistency
python test_stage4_tr_generation.py  # Stage 4: generate TRs
python test_stage5_review.py     # Stage 5:  quality review

# Or run stages 3-5 together with feedback loop
python run_pipeline.py

# Generate summary report
python generate_report.py
```

---

## Project Structure

```
AI4RE-Test/
|
+-- schemas/
|   +-- entity_schema.py       Stage 1 entity types and DocumentChunk
|   +-- demand_schema.py       Stage 2 demand model (roles, use cases, reqs)
|   +-- consistency_schema.py  Stage 3 consistency issues and reports
|   +-- tr_schema.py           Stage 4 TR-H and TR-A schemas
|
+-- src/
|   +-- ingestion/
|   |   +-- base_loader.py     Abstract base: unified section schema
|   |   +-- pdf_loader.py      PDF: heading detection, slide deck fallback
|   |   +-- docx_loader.py     DOCX: heading hierarchy extraction
|   |   +-- text_loader.py     TXT/PUML: paragraph and PlantUML splitting
|   |   +-- xml_loader.py      XML/XMI/UML: leaf-node element extraction
|   |   +-- json_loader.py     JSON: OpenAPI/Swagger endpoint splitting
|   |   +-- document_manager.py Dispatcher: injects source_id, artifact_type
|   |
|   +-- chunking/
|   |   +-- semantic_chunker.py Token-bounded chunks with overlap per artifact
|   |
|   +-- extraction/
|   |   +-- entity_extractor.py Few-shot NER via Ollama /api/chat
|   |
|   +-- reasoning/
|   |   +-- demand_modeller.py  4-step CoT demand understanding
|   |   +-- consistency_checker.py Cross-doc and intra-source checks
|   |
|   +-- generation/
|   |   +-- tr_generator.py    5-category TR generation (TR-H + TR-A)
|   |
|   +-- evaluation/
|       +-- tr_reviewer.py     5-dimension quality review + EARS checker
|       +-- feedback_loop.py   Stage 5 -> Stage 4 regeneration loop
|
+-- data/
|   +-- raw/
|   |   +-- srs/               NASA HDTN SRS (PDF)
|   |   +-- user_manual/       GNU Bash Reference Manual (TXT)
|   |   +-- runtime/           Swagger Petstore API (JSON)
|   |   +-- design_docs/       Auth System Design (PUML)
|   |
|   +-- processed/
|       +-- chunks/            Stage 1a: DocumentChunk JSON files
|       +-- entities/          Stage 1b: ExtractedEntity JSON files
|       +-- demand_model/      Stage 2:  DemandModel JSON files
|       +-- consistency/       Stage 3:  ConsistencyReport JSON
|       +-- tr_doc/            Stage 4:  TRDoc JSON (TR-H + TR-A)
|       +-- evaluation/
|           +-- reviews/       Stage 5:  quality review JSON files
|           +-- final/         Stage 5:  final TR-Doc after feedback
|           +-- pipeline_report.md  Full summary report
|
+-- test_stage1.py             Test Stage 1a: parsing and chunking
+-- test_stage1_extraction.py  Test Stage 1b: entity extraction
+-- test_stage2_demand.py      Test Stage 2: demand understanding
+-- test_stage3_consistency.py Test Stage 3: consistency check
+-- test_stage4_tr_generation.py Test Stage 4: TR generation
+-- test_stage5_review.py      Test Stage 5: quality review
+-- run_pipeline.py            Full pipeline runner with feedback loop
+-- generate_report.py         Generate pipeline_report.md
+-- debug_ollama.py            Diagnose Ollama connection and model issues
+-- patch_reviewer.py          Patch: replace LLM rubric with EARS checker
+-- patch_coverage.py          Patch: fix keyword length threshold
+-- patch_q_h_semantic.py      Patch: add semantic specificity criterion (f)
+-- run_full_extraction.sh     nohup wrapper for overnight extraction
```

---

## Five-Stage Pipeline

### Stage 1 -- Document Parsing and Entity Extraction

**Input:** Raw documents (PDF, DOCX, TXT, JSON, PUML)

**Processing:**
- Loaders extract text into unified section dicts
- SemanticChunker splits sections into token-bounded chunks
  - SRS/Manual: 512 tokens, 10% overlap, sentence boundary
  - Runtime: 256 tokens (dense API specs)
  - Design: 384 tokens, element boundary
- EntityExtractor calls llama3.1:70b with few-shot NER prompt
  - 7 entity types: function_point, constraint, actor_role,
    interface_name, process_step, exception_condition, glossary_term

**Output:** data/processed/entities/{source_id}.json

**Results:** 2,624 entities across 4 documents

### Stage 2 -- Demand Understanding

**Input:** ExtractedEntity lists from Stage 1

**Processing:** 4-step Chain-of-Thought prompting:
1. Identify user roles from actor_role entities
2. Map goals and use cases from function_point entities
3. Extract pre/postconditions and exception handlers
4. Classify constraints into NFR and runtime categories

**Output:** data/processed/demand_model/{source_id}.json

### Stage 3 -- Consistency Check

**Input:** DemandModel objects from Stage 2

**Processing:** 5 check types:
1. SRS vs User Manual: function coverage gaps
2. Runtime vs SRS: execution condition compatibility
3. Design vs SRS: constraint implementation gaps
4. Intra-source: each document vs itself
5. Terminology: term consistency across documents

**Output:** data/processed/consistency/consistency_report.json

**Results:** 34 issues (11 cross-doc, 23 intra-source)

### Stage 4 -- Test Requirement Generation

**Input:** DemandModel + ConsistencyReport

**Processing:** 5 category sub-prompts:
1. Functional: normal, boundary, negative paths per requirement
2. NFR constraint: threshold verification TRs
3. Runtime/environment: error condition handling
4. Interface/exception: use case flows and exception handling
5. Consistency-driven: one TR per consistency issue

**Output:** TR-H (EARS NL) + TR-A (JSON schema) per TR

**Results:** 191 TRs total

### Stage 5 -- Test Tracking and Review

**Input:** TRDoc + DemandModel

**Processing:** 5 dimensions:
1. Coverage: keyword overlap with all demand model items
2. Constraint coverage: NFR thresholds tested
3. Traceability: source_req_ids presence
4. Quality defects: Jaccard deduplication, untestable detection
5. Schema (Q_A): all TR-A fields present and non-empty

**Q_H metric:** 6-criterion EARS compliance checker
- (a) contains "shall"
- (b) trigger word or clear subject
- (c) expected outputs >= 4 words
- (d) preconditions non-empty
- (e) stimuli non-empty
- (f) measurable threshold: number+unit, HTTP code, REST verb, endpoint
- Pass threshold: >= 5 of 6 criteria

**Output:** Quality review JSON + corrected final TR-Doc

---

## Final Quality Results

| Document | Q_H | Q_A | Traceability | Coverage | Status |
|---|---|---|---|---|---|
| nasa_srs_v1 | 0.79 | 0.86 | 1.00 | 1.00 | PASS |
| swagger_petstore | 0.93 | 1.00 | 1.00 | 1.00 | PASS |
| auth_system_design | 0.98 | 1.00 | 1.00 | 1.00 | PASS |
| bash_user_manual | 0.83 | 1.00 | 1.00 | 0.95 | PASS |

Q_H variation reflects domain specificity:
- Auth design (0.98): TRs contain exact values (TTL=15min, HTTP 401)
- Swagger (0.93): REST TRs naturally contain GET/POST and /endpoints
- Bash manual (0.83): Shell TRs describe behaviour without thresholds
- NASA SRS (0.79): Abstract protocol TRs lack measurable specifics

---

## Quality Targets

| Metric | Target | Rationale |
|---|---|---|
| Q_H >= 0.78 | EARS compliance rate | Industry standard for TR acceptance |
| Q_A >= 0.85 | Schema validity rate | Ensures machine-executable TR-A |
| Traceability >= 0.90 | Source linkage | Enables impact analysis |
| Coverage >= 0.95 | Demand coverage | Minimises untested requirements |
