"""
demand_schema.py
Pydantic schemas for the Stage 2 Demand Understanding Model.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum


class RequirementType(str, Enum):
    FUNCTIONAL    = "functional"
    NON_FUNCTIONAL = "non_functional"
    RUNTIME       = "runtime"
    INTERFACE     = "interface"


class UseCase(BaseModel):
    use_case_id:   str
    actor:         str
    goal:          str
    preconditions: List[str] = []
    steps:         List[str] = []
    postconditions:List[str] = []
    exceptions:    List[str] = []


class FunctionalRequirement(BaseModel):
    req_id:      str
    description: str
    source_ids:  List[str] = []
    priority:    str = "medium"


class NFRConstraint(BaseModel):
    constraint_id: str
    category:      str   # performance, security, usability, reliability
    description:   str
    threshold:     Optional[str] = None
    source_ids:    List[str] = []


class RuntimeConstraint(BaseModel):
    constraint_id: str
    description:   str
    affects_reqs:  List[str] = []
    source_ids:    List[str] = []


class DemandModel(BaseModel):
    source_id:            str
    bundle_id:            Optional[str] = None  # groups artifacts describing the SAME system;
                                                  # cross-document consistency checks (1-3, 5) only
                                                  # compare documents sharing a bundle_id. See
                                                  # ConsistencyChecker.check_all() and Section 9.1.1.
    system_objectives:    List[str] = []
    user_roles:           List[str] = []
    use_cases:            List[UseCase] = []
    functional_reqs:      List[FunctionalRequirement] = []
    nfr_constraints:      List[NFRConstraint] = []
    runtime_constraints:  List[RuntimeConstraint] = []
    glossary:             Dict[str, str] = {}
    exception_handlers:   List[str] = []
