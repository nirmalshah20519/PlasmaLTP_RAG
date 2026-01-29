"""Pydantic schemas for API requests and responses."""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any


class QueryRequest(BaseModel):
    """Request schema for query endpoint."""
    question: str = Field(..., description="User question", min_length=1)
    top_k: Optional[int] = Field(None, description="Number of evidence items to retrieve", ge=1, le=50)


class EvidenceItem(BaseModel):
    """Schema for evidence item."""
    score: float
    chunk_id: Optional[str] = None
    sentence_type: Optional[str] = None
    sent_index: Optional[int] = None
    sentence: str
    paper_title: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None
    topic_id: Optional[str] = None
    topic_name: Optional[str] = None

    @field_validator("topic_id", mode="before")
    @classmethod
    def coerce_topic_id_to_str(cls, v: object) -> Optional[str]:
        """Neo4j may return topic_id as int; coerce to str for API consistency."""
        if v is None:
            return None
        if isinstance(v, int):
            return str(v)
        return v if isinstance(v, str) else str(v)


class QueryResponse(BaseModel):
    """Response schema for query endpoint."""
    answer: str
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Retrieved evidence")

