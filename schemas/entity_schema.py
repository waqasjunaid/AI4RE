from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ArtifactType(str, Enum):
    SRS = "srs"
    USER_MANUAL = "user_manual"
    RUNTIME = "runtime"
    DESIGN = "design"


class EntityType(str, Enum):
    FUNCTION_POINT = "function_point"
    CONSTRAINT = "constraint"
    ACTOR_ROLE = "actor_role"
    GLOSSARY_TERM = "glossary_term"
    INTERFACE_NAME = "interface_name"
    PROCESS_STEP = "process_step"
    EXCEPTION_CONDITION = "exception_condition"


class ExtractedEntity(BaseModel):
    source_id: str = Field(..., description="Unique source document ID")

    artifact_type: ArtifactType

    entity_type: EntityType

    content: str = Field(
        ...,
        description="Extracted entity text"
    )

    page_ref: Optional[str] = Field(
        default=None,
        description="Page or section reference"
    )

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM confidence score"
    )


class DocumentChunk(BaseModel):
    chunk_id: str

    source_id: str

    artifact_type: ArtifactType

    content: str

    page_ref: Optional[str]

    metadata: dict = {}