# AI4RE-Test: LLM-Based Test Requirement Generation from Heterogeneous Artifacts

Five-stage pipeline for automated test requirement (TR) generation from heterogeneous software engineering artifacts using a local large language model.

**Author:** Dr. Waqas Junaid, School of Software,  
Northwestern Polytechnical University, Xi'an, P.R. China

**Model:** `llama3.1:70b` via Ollama (local inference; no cloud API dependency)

---

## Overview

AI4RE-Test processes four supported artifact types:

- Software Requirements Specifications (SRS)
- User manuals
- Runtime/API specifications
- Design documents

The pipeline consists of five stages:

1. **Document Parsing and Entity Extraction**
2. **Demand Understanding**
3. **Consistency Checking**
4. **Test Requirement Generation**
5. **Quality Review and Feedback**

Stage 3 performs consistency analysis before TR generation, while Stage 5 uses a deterministic six-criterion EARS compliance checker and a feedback loop to support targeted regeneration.

The current evaluation contains **seven documents**: four domain-diverse single-document cases plus a dedicated four-document `auth_system` same-system bundle for cross-document reasoning. The design document `auth_system_design` is shared between these two evaluation settings.

---

## Quick Start

```bash
# Prerequisites
ollama serve
ollama pull llama3.1:70b

# Install dependencies
pip install pydantic pymupdf python-docx lxml tiktoken

# Run Stage 1
cd AI4RE-Test
python test_stage1.py
python test_stage1_extraction.py

# Run Stage 2
python test_stage2_demand.py

# Run Stage 3
python test_stage3_consistency.py

# Run Stage 4
python test_stage4_tr_generation.py

# Run Stage 5
python test_stage5_review.py

# Or run the integrated Stage 3-5 pipeline with feedback
python run_pipeline.py

# Generate the summary report
python generate_report.py
```

Stage 1 entity extraction can require substantial local compute time, especially for the 151-page Bash manual.

---

## Project Structure

```text
AI4RE-Test/
|
+-- schemas/
|   +-- entity_schema.py          Stage 1 entity types and DocumentChunk
|   +-- demand_schema.py          Stage 2 demand model
|   +-- consistency_schema.py     Stage 3 consistency issues and reports
|   +-- tr_schema.py              Stage 4 TR-H and TR-A schemas
|
+-- src/
|   +-- ingestion/
|   |   +-- base_loader.py        Unified section schema
|   |   +-- pdf_loader.py         PDF parsing
|   |   +-- docx_loader.py        DOCX heading hierarchy extraction
|   |   +-- text_loader.py        TXT/PUML parsing
|   |   +-- xml_loader.py         XML/XMI/UML extraction
|   |   +-- json_loader.py        JSON/OpenAPI/Swagger parsing
|   |   +-- document_manager.py   Source/artifact metadata dispatcher
|   |
|   +-- chunking/
|   |   +-- semantic_chunker.py   Artifact-calibrated semantic chunking
|   |
|   +-- extraction/
|   |   +-- entity_extractor.py   Zero-shot NER via Ollama
|   |
|   +-- reasoning/
|   |   +-- demand_modeller.py    Four-step CoT demand understanding
|   |   +-- consistency_checker.py Cross-document and intra-source checks
|   |
|   +-- generation/
|   |   +-- tr_generator.py       Five-category TR generation
|   |
|   +-- evaluation/
|       +-- tr_reviewer.py        Structural quality review + EARS checker
|       +-- feedback_loop.py      Stage 5 -> Stage 4 feedback loop
|
+-- data/
|   +-- raw/
|   |   +-- srs/                  Evaluation SRS inputs
|   |   +-- user_manual/          User-manual inputs
|   |   +-- runtime/              Runtime/API inputs
|   |   +-- design_docs/          Design-document inputs
|   |
|   +-- processed/
|       +-- chunks/               Stage 1 semantic chunks
|       +-- entities/             Stage 1 extracted entities
|       +-- demand_model/         Stage 2 DemandModel JSON
|       +-- consistency/          Stage 3 consistency reports
|       +-- tr_doc/               Stage 4 TRDoc JSON
|       +-- evaluation/
|           +-- reviews/          Stage 5 review reports
|           +-- final/            Final TR-Doc after feedback
|           +-- pipeline_report.md
|
+-- test_stage1.py
+-- test_stage1_extraction.py
+-- test_stage2_demand.py
+-- test_stage3_consistency.py
+-- test_stage4_tr_generation.py
+-- test_stage5_review.py
+-- run_pipeline.py
+-- generate_report.py
+-- debug_ollama.py
+-- patch_reviewer.py
+-- patch_coverage.py
+-- patch_q_h_semantic.py
+-- run_full_extraction.sh
```

The patch scripts are retained for reproducibility of development history. They are not required for a normal end-to-end run of the current pipeline.

---

## Five-Stage Pipeline

### Stage 1 -- Document Parsing and Entity Extraction

**Input:** Raw documents (`PDF`, `DOCX`, `TXT`, `JSON`, `PUML`)

**Processing:**

- Artifact-specific loaders convert inputs to a unified document structure.
- Semantic chunking uses artifact-calibrated token sizes:
  - SRS/manual: 512 tokens
  - runtime/API: 256 tokens
  - design: 384 tokens
  - 10% overlap
- A definition-driven **zero-shot NER** extractor identifies seven entity types: `function_point`, `constraint`, `actor_role`, `interface_name`, `process_step`, `exception_condition`, and `glossary_term`.
- Extracted entities retain source provenance.

**Current evaluation result:**

- **1,785 extracted entities**
- **290 chunks**
- **6.2 entities/chunk mean yield**
- `bash_user_manual`: **1,477 entities**, approximately 83% of the total

**Output:**

```text
data/processed/entities/{source_id}.json
```

---

### Stage 2 -- Demand Understanding

**Input:** Stage 1 fragment library for each document

Stage 2 builds an eight-field `DemandModel` using a four-step Chain-of-Thought process for:

1. user roles
2. use cases
3. functional requirements
4. NFR and runtime constraints

`system_objectives`, `glossary`, and `exception_handlers` are populated by significance-ranked selection from the fragment library without an additional LLM call.

The default `DEMAND_ITEM_CAP` is 10 for the main demand fields. Exception handlers and glossary terms use a `3 x DEMAND_ITEM_CAP` selection limit.

Functional requirements additionally use a Jaccard deduplication step before the final cap is applied.

**Current evaluation result:**

```text
Roles:               36
Use cases:            59
Functional reqs:      57
NFR constraints:      10
Runtime constraints:  17
Exception handlers:   42
Glossary terms:       43
```

Coverage is computed against this capped demand model and therefore does not independently validate the cap value.

---

### Stage 3 -- Consistency Check

**Input:** DemandModel objects

Five checks are used:

1. **SRS <-> Manual** -- cross-document omissions/conflicts
2. **Runtime <-> SRS** -- runtime-condition compatibility
3. **Design <-> SRS** -- design/requirement alignment
4. **Intra-source** -- internal omissions, conflicts, ambiguities, and confirmation items
5. **Terminology** -- cross-source terminology consistency within a same-system bundle

Cross-document checks are applied only when artifacts are scoped to the same underlying system. Singleton bundles receive the intra-source check only.

**Current evaluation result:**

- **37 typed consistency issues**
- **6 cross-document**
- **31 intra-source**
- 15 omissions
- 1 conflict
- 8 ambiguities
- 13 `to_confirm`

The dedicated `auth_system` bundle contains four documents describing the same authentication system.

**Output:**

```text
data/processed/consistency/consistency_report.json
```

---

### Stage 4 -- Test Requirement Generation

**Input:** DemandModel + Stage 3 ConsistencyReport

TR generation uses five category-specific sub-prompts:

1. **Functional**
2. **NFR constraint**
3. **Runtime/environment**
4. **Interface/exception**
5. **Consistency-driven**

Each generated TR is represented in two complementary forms:

- **TR-H:** EARS-structured natural-language requirement
- **TR-A:** machine-readable structured JSON representation

TR-A contains:

```text
tr_id
category
priority
preconditions[]
stimuli[]
expected_outputs[]
coverage_criterion
source_req_ids[]
```

`source_req_ids[]` records the associated source requirement, source item, or consistency issue. Category 5 uses structured consistency-issue identifiers; Categories 1-4 currently use text-based source references.

### Evaluation convention

The paper reports two different counts for the representative official Run 2:

- **332 TRs:** Stage 4 output before Stage 5 feedback
- **308 TRs:** final post-feedback output

The three official full-pipeline runs are used separately for stability analysis and are summarized by QH ranges, means, and pass rates.

**Representative official Run 2:**

```text
Pre-feedback Stage 4 output:  332 TRs
Post-feedback output:         308 TRs
```

**Output:**

```text
data/processed/tr_doc/{source_id}.json
```

---

### Stage 5 -- Test Tracking and Review

Stage 5 evaluates four scored metrics:

1. **QH** -- deterministic EARS structural/lexical conformance
2. **QA** -- TR-A schema completeness
3. **Trace** -- source_req_ids presence
4. **Cov** -- lexical coverage of the Stage 2 demand model

It also reports two diagnostic dimensions:

- Constraint Coverage
- Quality Defects

These two diagnostic dimensions are not independently thresholded.

#### QH: deterministic six-criterion EARS checker

The checker makes no LLM calls for a fixed TR set.

The six criteria are:

- **(a)** `shall` is present in TR-H
- **(b)** an EARS trigger keyword or clear system subject is present
- **(c)** `expected_outputs` contains at least four words
- **(d)** `preconditions[]` is non-empty
- **(e)** `stimuli[]` is non-empty
- **(f)** lexical measurability is detected

Criterion (f) is a deterministic **lexical measurability heuristic**, not a semantic correctness judgment. It checks for observable patterns such as:

- number + unit
- HTTP status code
- REST verb
- API endpoint path
- bare number with at least two digits

A TR passes QH when it satisfies at least **5 of 6** criteria.

#### Feedback loop

When one or more scored targets are not met, Stage 5 can send flagged TRs and coverage gaps back to Stage 4 for targeted regeneration.

The current feedback configuration permits at most:

```text
K = 2 regeneration iterations
```

The feedback loop is intended to improve the generated TR set; it does not execute the resulting TRs against a system under test.

---

## Official Evaluation Results

### Three official full-pipeline runs

The three official runs are used to assess run-to-run variation under the fixed model, prompts, thresholds, and pipeline configuration.

| Document | Q_H range | Mean Q_H | Q_A | Trace | Cov. | Pass rate |
|---|---:|---:|---:|---:|---:|---:|
| `nasa_srs_v1` | 0.62-0.91 | 0.77 | 1.00 | 1.00 | 0.95-1.00 | 2/3 |
| `swagger_petstore` | 0.97-0.97 | 0.97 | 1.00 | 1.00 | 1.00 | 3/3 |
| `auth_system_design` | 0.66-0.79 | 0.75 | 1.00 | 1.00 | 1.00 | 2/3 |
| `bash_user_manual` | 0.38-0.57 | 0.50 | 1.00 | 1.00 | 1.00 | 0/3 |
| `auth_system_srs` | 0.84-0.85 | 0.84 | 1.00 | 1.00 | 0.96-1.00 | 3/3 |
| `auth_system_manual` | 0.82-1.00 | 0.94 | 1.00 | 1.00 | 1.00 | 3/3 |
| `auth_system_runtime` | 0.68-0.91 | 0.83 | 1.00 | 1.00 | 1.00 | 2/3 |

**Overall:** mean QH = **0.80** and **15/21 (71.4%)** document-run observations pass all four quality targets.

Three documents pass all three runs, three pass two of three runs, and `bash_user_manual` fails all three runs because QH remains below the 0.78 target.

### Representative official Run 2

The detailed feedback analysis uses official Run 2:

```text
Stage 4 before feedback: 332 TRs
Stage 5 -> Stage 4 feedback: final 308 TRs
Compliant TRs: 240 / 308 (77.9%)
Non-compliant TRs: 68 / 308
```

The 308-TR post-feedback set has complete population of the four checked TR-A fields (`preconditions[]`, `stimuli[]`, `expected_outputs[]`, `coverage_criterion`) and Traceability = 1.00 for all seven documents.

The detailed 308-TR diagnostic breakdown is specific to representative Run 2 and should not be interpreted as a run-independent failure distribution.

---

## Quality Targets

| Metric | Target | Interpretation |
|---|---:|---|
| Q_H | >= 0.78 | At least 78% of TRs satisfy at least 5/6 checker criteria |
| Q_A | >= 0.85 | TR-A schema completeness |
| Trace | >= 0.90 | Non-empty source_req_ids for generated TRs |
| Coverage | >= 0.95 | Lexical coverage of capped Stage 2 demand-model items |

The thresholds were selected during pipeline development and fixed before the reported evaluation. They are operational criteria for this study rather than independently validated universal standards.

---

## Reproducibility Notes

The reported results depend on the specified local model and inference configuration. The experiments use:

```text
Model: llama3.1:70b
Inference: Ollama
Stages 1-3 temperature: 0.0
Stage 4 temperature: 0.1
Feedback iterations: K = 2
```

The Stage 5 checker itself is deterministic for a fixed TR set. Variation in the three official runs comes from the generative stages, particularly Stage 4, rather than from stochasticity inside the checker.

The current evaluation is descriptive rather than inferential: the 21 document-run observations reuse the same seven documents and therefore should not be treated as 21 independent samples.

---

## Current Scope and Limitations

The current implementation and evaluation do **not** establish:

- semantic correctness of generated TRs
- practical usefulness to test engineers
- superiority over human-authored requirements
- universal validity of the QH threshold
- generalization to safety-critical domains
- model-independent pipeline performance
- execution of generated TRs against a system under test

The cross-document consistency evaluation uses one dedicated `auth_system` same-system bundle. Two planted numeric conflicts in that bundle were not detected by the current LLM-based consistency checks, showing that semantic conflict detection remains a limitation.

The current traceability guarantee is strongest for Category 5. Categories 1-4 retain source references as free text rather than structured source IDs.

---

## Development and Replication Scripts

The repository contains several helper and patch scripts used during development, including:

```text
patch_reviewer.py
patch_coverage.py
patch_q_h_semantic.py
debug_ollama.py
run_full_extraction.sh
```

`patch_q_h_semantic.py` is a historical development artifact. In the current pipeline and manuscript, criterion (f) is described as **lexical measurability**, not semantic specificity.

Earlier exploratory baselines are not treated as main evaluation runs. In particular, unverified LLM-as-judge and single-prompt variants remain outside the official three-run evaluation.

---

## Data Availability

The replication package contains the implementation, prompts, processed artifacts, generated TR documents, review reports, and figure-generation materials used in the study.

The evaluation configuration is designed for local execution with Ollama and `llama3.1:70b`. Reproduction of the exact reported numbers requires matching the corresponding model and inference configuration.
