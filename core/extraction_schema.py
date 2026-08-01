"""Pydantic schemas for LLM entity/relation extraction."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    """A single entity extracted from a conversation."""

    name: str = Field(..., description="Canonical entity name")
    type: str = Field(
        ...,
        description="person|project|tool|concept|file|task|skill|artifact",
    )
    label: str = Field(default="", description="Human-readable short label. Use name if unknown.")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    aliases: list[str] = Field(default_factory=list)
    context: str = Field(default="", description="Why this entity was extracted")


class ExtractedRelation(BaseModel):
    """A relationship between two entities."""

    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    type: str = Field(
        default="mentions",
        description="mentions|produces|uses|decides|references|blocks|resolves|assigns",
    )
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    context: str = Field(default="", description="Evidence/snippet from conversation")


class ExtractedKnowledge(BaseModel):
    """Complete extraction result for a single session."""

    session_id: str = ""
    summary: str = Field(default="", description="One-paragraph session summary")
    topics: list[str] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    outcome: str = Field(
        default="in_progress",
        description="completed|in_progress|failed|unknown",
    )
