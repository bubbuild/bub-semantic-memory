"""
Pydantic models for the semantic_memory plugin.

Entity
    Represents a named concept extracted from conversation context (e.g. a person,
    place, technology, or abstract idea). Each entity has a stable UUID, a type
    label, a human-readable name, and an open-ended metadata bag.

Relation
    Represents a directed edge between two entities (from_id -> to_id) with a
    type label (e.g. "uses", "belongs_to", "created_by") and optional metadata.

SemanticSnapshot
    An immutable point-in-time capture of a set of entities and relations,
    anchored to a specific tape position (tape_id + anchor_id). Snapshots are
    the unit of persistence and retrieval for the plugin.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A named concept node in the semantic graph.

    Attributes:
        id:       Stable UUID that uniquely identifies this entity.
        type:     A short label describing the category (e.g. "person", "tool").
        name:     Human-readable name for the entity.
        metadata: Arbitrary key-value pairs for extra context.
    """

    model_config = {
        "frozen": True,
        "json_encoders": {datetime: lambda v: v.isoformat()},
    }

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.id == other.id


class Relation(BaseModel):
    """A directed edge between two Entity nodes.

    Attributes:
        from_id:  UUID of the source entity.
        to_id:    UUID of the target entity.
        type:     Label describing the relationship (e.g. "uses", "created_by").
        metadata: Arbitrary key-value pairs for extra context.
    """

    model_config = {
        "frozen": True,
        "json_encoders": {datetime: lambda v: v.isoformat()},
    }

    from_id: str
    to_id: str
    type: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.from_id, self.to_id, self.type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relation):
            return NotImplemented
        return (
            self.from_id == other.from_id
            and self.to_id == other.to_id
            and self.type == other.type
        )


class SemanticSnapshot(BaseModel):
    """An immutable snapshot of entities and relations at a tape position.

    Attributes:
        entities:   Ordered list of Entity nodes captured at this snapshot.
        relations:  Ordered list of Relation edges captured at this snapshot.
        tape_id:    Identifier of the tape (conversation thread) this belongs to.
        anchor_id:  Identifier of the specific tape entry this snapshot is anchored to.
        created_at: UTC timestamp when this snapshot was created.
    """

    model_config = {
        "frozen": True,
        "json_encoders": {datetime: lambda v: v.isoformat()},
    }

    entities: tuple[Entity, ...] = Field(default_factory=tuple)
    relations: tuple[Relation, ...] = Field(default_factory=tuple)
    tape_id: str
    anchor_id: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC)
    )
