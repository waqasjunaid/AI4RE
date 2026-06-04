"""
tr_schema.py
Pydantic schemas for Stage 4 Test Requirement output.
TR-H = human-readable EARS-format natural language
TR-A = machine-executable JSON schema
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class TRCategory(str, Enum):
    FUNCTIONAL           = "functional"
    NFR_CONSTRAINT       = "nfr_constraint"
    RUNTIME_ENVIRONMENT  = "runtime_environment"
    INTERFACE_EXCEPTION  = "interface_exception"
    CONSISTENCY_DRIVEN   = "consistency_driven"


class Priority(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


class TR_A(BaseModel):
    tr_id:              str
    preconditions:      List[str] = []
    stimuli:            List[str] = []
    expected_outputs:   List[str] = []
    coverage_criterion: str = ""
    priority:           Priority = Priority.MEDIUM
    source_req_ids:     List[str] = []


class TestRequirement(BaseModel):
    tr_id:          str
    category:       TRCategory
    source_id:      str
    tr_h:           str    # EARS natural language
    tr_a:           TR_A   # machine-executable
    source_req_ids: List[str] = []


class TRDoc(BaseModel):
    source_id:   str
    tr_h_list:   List[str]              = []
    tr_a_list:   List[TR_A]             = []
    trs:         List[TestRequirement]  = []
    total_count: int                    = 0
