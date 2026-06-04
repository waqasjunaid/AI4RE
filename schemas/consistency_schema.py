"""
consistency_schema.py
Pydantic schemas for Stage 3 Consistency Check output.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class IssueType(str, Enum):
    CONFLICT   = "conflict"
    OMISSION   = "omission"
    AMBIGUITY  = "ambiguity"
    TO_CONFIRM = "to_confirm"


class Severity(str, Enum):
    BLOCKER = "blocker"
    MAJOR   = "major"
    MINOR   = "minor"


class ConsistencyIssue(BaseModel):
    issue_id:       str
    issue_type:     IssueType
    source_a_id:    str
    source_b_id:    str
    description:    str
    severity:       Severity = Severity.MINOR
    affected_req_ids: List[str] = []
    suggested_fix:  Optional[str] = None


class ConsistencyReport(BaseModel):
    comparison_id:  str
    source_a:       str
    source_b:       str
    issues:         List[ConsistencyIssue] = []
    total_conflicts:  int = 0
    total_omissions:  int = 0
    total_ambiguities: int = 0
    total_to_confirm:  int = 0
