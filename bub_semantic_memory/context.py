"""Semantic memory context builder for the semantic_memory plugin."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from republic import LLM, TapeContext, TapeEntry

import bub.builtin.context as _builtin_context
from bub_semantic_memory import extractor
from bub_semantic_memory.models import SemanticSnapshot
from bub_semantic_memory.store import SemanticStore


async def build_semantic_context(
    entries: Iterable[TapeEntry],
    context: TapeContext,
    llm: LLM,
    store: SemanticStore,
) -> list[dict[str, Any]]:
    """Build a message list enriched with semantic memory.

    Steps:
    1. Collect base messages via the default tape context selector.
    2. Extract a SemanticSnapshot from the current entries using the LLM.
    3. Append the snapshot to the persistent store.
    4. Load all historical snapshots for this tape.
    5. Format snapshots into a system prompt block and append it to messages.
    6. Return the complete message list.

    If no snapshots exist after loading, the function returns only the base
    messages without adding an empty semantic block.
    """
    entries_list = list(entries)

    # Step 1: Build base messages using the default selector
    messages: list[dict[str, Any]] = _builtin_context._select_messages(entries_list, context)

    # Determine tape_id from context state (session_id is stored there at runtime)
    tape_id: str = str(context.state.get("session_id", "")) or "default"

    # Determine anchor_id from the last entry id (or empty string)
    anchor_id: str = str(entries_list[-1].id) if entries_list else ""

    # Step 2: Extract semantics from current entries
    snapshot: SemanticSnapshot = await extractor.extract_semantics(
        entries_list,
        llm,
        tape_id=tape_id,
        anchor_id=anchor_id,
    )

    # Step 3: Append snapshot to store (always persist, even if empty)
    await store.append(tape_id, snapshot)

    # Step 4: Load all historical snapshots for this tape
    snapshots: list[SemanticSnapshot] = await store.load(tape_id)

    if not snapshots:
        return messages

    # Step 5: Format snapshots into a Markdown system prompt block
    system_content = _format_snapshots(snapshots)

    # Step 6: Append semantic memory block to messages
    messages.append({"role": "system", "content": system_content})

    return messages


def _format_snapshots(snapshots: list[SemanticSnapshot]) -> str:
    """Render a list of SemanticSnapshot objects as a Markdown system prompt."""
    # Deduplicate entities and relations across all snapshots
    seen_entity_ids: set[str] = set()
    seen_relation_keys: set[tuple[str, str, str]] = set()

    all_entities = []
    all_relations = []

    # Build an id->name lookup for relation rendering
    id_to_name: dict[str, str] = {}

    for snap in snapshots:
        for entity in snap.entities:
            id_to_name[entity.id] = entity.name
            if entity.id not in seen_entity_ids:
                seen_entity_ids.add(entity.id)
                all_entities.append(entity)

        for relation in snap.relations:
            key = (relation.from_id, relation.to_id, relation.type)
            if key not in seen_relation_keys:
                seen_relation_keys.add(key)
                all_relations.append(relation)

    entity_count = len(all_entities)
    relation_count = len(all_relations)

    lines: list[str] = ["## Semantic Memory", ""]

    lines.append(f"### Entities ({entity_count}):")
    for entity in all_entities:
        lines.append(f"- {entity.type}:{entity.name} (id={entity.id})")

    lines.append("")
    lines.append(f"### Relations ({relation_count}):")
    for relation in all_relations:
        from_name = id_to_name.get(relation.from_id, relation.from_id)
        to_name = id_to_name.get(relation.to_id, relation.to_id)
        lines.append(f"- {from_name} --{relation.type}--> {to_name}")

    return "\n".join(lines)


def _format_snapshots_filtered(
    snapshots: list[SemanticSnapshot],
    cues: set[str],
) -> str:
    """Render snapshots, keeping only entities/relations relevant to *cues*.

    An entity is kept when any cue is a substring of its name or type.  A
    relation is kept when at least one endpoint is kept, or when any cue is a
    substring of the relation type.  When a relation is kept via the one-endpoint
    rule, the OTHER endpoint entity is also kept (1-hop relation traversal) —
    this bridges the recall gap when the question word ("research") differs from
    the answer word ("adoption agencies") but they are connected via a relation.
    """
    if not cues:
        # No usable cues means we cannot safely filter; fall back to the full view.
        return _format_snapshots(snapshots)

    seen_entity_ids: set[str] = set()
    seen_relation_keys: set[tuple[str, str, str]] = set()

    all_entities: list[Entity] = []
    all_relations: list[Relation] = []
    id_to_name: dict[str, str] = {}
    id_to_entity: dict[str, Entity] = {}

    def _matches_cues(text: str) -> bool:
        lowered = text.lower()
        return any(cue in lowered for cue in cues)

    # First pass: collect entities that match a cue.
    for snap in snapshots:
        for entity in snap.entities:
            id_to_name[entity.id] = entity.name
            id_to_entity[entity.id] = entity
            if entity.id in seen_entity_ids:
                continue
            if _matches_cues(entity.name) or _matches_cues(entity.type):
                seen_entity_ids.add(entity.id)
                all_entities.append(entity)

    # Second pass: keep relations with at least one endpoint kept, or type matches.
    # When keeping, expand the kept entity set to include the other endpoint (1-hop).
    new_entity_ids: set[str] = set()
    for snap in snapshots:
        for relation in snap.relations:
            key = (relation.from_id, relation.to_id, relation.type)
            if key in seen_relation_keys:
                continue
            from_kept = relation.from_id in seen_entity_ids
            to_kept = relation.to_id in seen_entity_ids
            one_endpoint_kept = from_kept or to_kept
            if one_endpoint_kept or _matches_cues(relation.type):
                seen_relation_keys.add(key)
                all_relations.append(relation)
                # 1-hop expansion: add the other endpoint entity.
                if from_kept and not to_kept and relation.to_id not in new_entity_ids:
                    new_entity_ids.add(relation.to_id)
                if to_kept and not from_kept and relation.from_id not in new_entity_ids:
                    new_entity_ids.add(relation.from_id)

    # Third pass: add entities discovered via 1-hop relation traversal.
    for eid in new_entity_ids:
        if eid not in seen_entity_ids and eid in id_to_entity:
            seen_entity_ids.add(eid)
            all_entities.append(id_to_entity[eid])

    entity_count = len(all_entities)
    relation_count = len(all_relations)

    lines: list[str] = ["## Semantic Memory", ""]

    lines.append(f"### Entities ({entity_count}):")
    for entity in all_entities:
        lines.append(f"- {entity.type}:{entity.name} (id={entity.id})")

    lines.append("")
    lines.append(f"### Relations ({relation_count}):")
    for relation in all_relations:
        from_name = id_to_name.get(relation.from_id, relation.from_id)
        to_name = id_to_name.get(relation.to_id, relation.to_id)
        lines.append(f"- {from_name} --{relation.type}--> {to_name}")

    return "\n".join(lines)
