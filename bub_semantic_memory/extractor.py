"""
Semantic extraction for the semantic_memory plugin.

Provides extract_semantics(), which takes a list of TapeEntry objects and an LLM
instance, calls the LLM to identify entities and relations in the conversation, and
returns the result as a SemanticSnapshot.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from republic import LLM
from republic.tape.entries import TapeEntry

from bub_semantic_memory.models import Entity, Relation, SemanticSnapshot

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a semantic extraction assistant. Given a snippet of conversation history, \
identify the key entities and the relations between them.

Respond ONLY with a single valid JSON object — no markdown fences, no prose. \
The JSON must conform exactly to this schema:

{
  "entities": [
    {"id": "<short_stable_slug>", "type": "person|task|event|concept", "name": "<human name>", "metadata": {}}
  ],
  "relations": [
    {"from": "<entity_id>", "to": "<entity_id>", "type": "<relation_label>", "metadata": {}}
  ]
}

Guidelines:
- Use concise, lowercase slugs for entity ids (e.g. "alice", "deploy_task").
- Entity types must be one of: person, task, event, concept.
- Relation types are free-form verbs/labels (e.g. "creates", "depends_on", "mentions").
- Only include entities and relations that are clearly present or implied in the text.
- If nothing meaningful can be extracted, return {"entities": [], "relations": []}.
"""


def _entries_to_text(entries: list[TapeEntry]) -> str:
    """Convert a list of TapeEntry objects into a compact human-readable string."""
    lines: list[str] = []
    for entry in entries:
        kind = entry.kind
        payload = entry.payload

        if kind == "message":
            role = payload.get("role", "unknown")
            content = payload.get("content", "")
            if isinstance(content, list):
                # Multimodal content — extract text parts only
                content = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
                )
            lines.append(f"[{role}] {content}")

        elif kind == "system":
            content = payload.get("content", "")
            lines.append(f"[system] {content}")

        elif kind == "tool_call":
            for call in payload.get("calls", []):
                name = call.get("function", {}).get("name") or call.get("name", "unknown_tool")
                lines.append(f"[tool_call] {name}")

        elif kind == "tool_result":
            for result in payload.get("results", []):
                if isinstance(result, dict):
                    text = result.get("content", result.get("output", str(result)))
                else:
                    text = str(result)
                lines.append(f"[tool_result] {text}")

        elif kind == "event":
            event_name = payload.get("name", "")
            lines.append(f"[event] {event_name}")

        elif kind == "anchor":
            anchor_name = payload.get("name", "")
            lines.append(f"[anchor] {anchor_name}")

        # Skip other internal entry types silently

    return "\n".join(lines)


def _parse_llm_response(raw: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse raw LLM output into (entities_list, relations_list).

    Returns empty lists on any parse error.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("semantic_memory: failed to parse LLM JSON response: %s", exc)
        return [], []

    if not isinstance(data, dict):
        logger.warning("semantic_memory: LLM response is not a JSON object")
        return [], []

    entities = data.get("entities", [])
    relations = data.get("relations", [])

    if not isinstance(entities, list) or not isinstance(relations, list):
        logger.warning("semantic_memory: 'entities' or 'relations' is not a list")
        return [], []

    return entities, relations


def _build_snapshot(
    entities_raw: list[dict[str, Any]],
    relations_raw: list[dict[str, Any]],
    tape_id: str,
    anchor_id: str,
) -> SemanticSnapshot:
    """Convert raw dicts from the LLM into a SemanticSnapshot."""
    # Build a mapping from LLM slug -> UUID so relations can reference entities
    slug_to_uuid: dict[str, str] = {}
    entities: list[Entity] = []

    for item in entities_raw:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("id", ""))
        entity_type = str(item.get("type", "concept"))
        name = str(item.get("name", slug))
        metadata = item.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        entity = Entity(type=entity_type, name=name, metadata=metadata)
        entities.append(entity)
        if slug:
            slug_to_uuid[slug] = entity.id

    relations: list[Relation] = []
    for item in relations_raw:
        if not isinstance(item, dict):
            continue
        from_slug = str(item.get("from", ""))
        to_slug = str(item.get("to", ""))
        relation_type = str(item.get("type", "related_to"))
        metadata = item.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        from_id = slug_to_uuid.get(from_slug)
        to_id = slug_to_uuid.get(to_slug)
        if not from_id or not to_id:
            logger.debug(
                "semantic_memory: skipping relation '%s' -> '%s' (unknown slug)",
                from_slug,
                to_slug,
            )
            continue

        relations.append(Relation(from_id=from_id, to_id=to_id, type=relation_type, metadata=metadata))

    return SemanticSnapshot(
        entities=tuple(entities),
        relations=tuple(relations),
        tape_id=tape_id,
        anchor_id=anchor_id,
    )


async def extract_semantics(
    entries: list[TapeEntry],
    llm: LLM,
    *,
    tape_id: str = "",
    anchor_id: str = "",
    max_tokens: int = 1000,
) -> SemanticSnapshot:
    """Extract semantic entities and relations from a list of TapeEntry objects.

    Calls the LLM to analyse the conversation represented by *entries* and returns
    a SemanticSnapshot.  On any failure (LLM error, invalid JSON, …) an empty
    snapshot is returned rather than raising.

    Args:
        entries:   The tape entries to analyse.
        llm:       A republic.LLM instance used to call the model.
        tape_id:   Identifier for the tape these entries belong to.
        anchor_id: Identifier of the tape anchor this snapshot is tied to.
        max_tokens: Maximum tokens for the LLM response.

    Returns:
        A SemanticSnapshot (possibly empty on failure).
    """
    empty_snapshot = SemanticSnapshot(
        entities=(),
        relations=(),
        tape_id=tape_id,
        anchor_id=anchor_id,
    )

    if not entries:
        return empty_snapshot

    conversation_text = _entries_to_text(entries)
    if not conversation_text.strip():
        return empty_snapshot

    try:
        raw_response: str = await llm.chat_async(
            prompt=conversation_text,
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.warning("semantic_memory: LLM call failed: %s", exc)
        return empty_snapshot

    entities_raw, relations_raw = _parse_llm_response(raw_response)

    if not entities_raw and not relations_raw:
        return empty_snapshot

    try:
        return _build_snapshot(entities_raw, relations_raw, tape_id=tape_id, anchor_id=anchor_id)
    except Exception as exc:
        logger.warning("semantic_memory: failed to build snapshot: %s", exc)
        return empty_snapshot
