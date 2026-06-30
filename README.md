# Semantic Memory Plugin for Bub

> **Latest:** v0.1.2 — Query-driven filtering, multilang support, LoCoMo eval suite

A plugin that extracts and retains semantic entities and relations from conversation histories, enriching agent context with semantic memory.

## Overview

This plugin intercepts the tape context building process to:
1. **Extract semantics** from conversation entries using an LLM
2. **Store snapshots** of entities (people, tasks, concepts) and relations between them
3. **Inject memory** into subsequent agent prompts, enabling long-context awareness

The plugin follows Bub's philosophy: it's completely optional, zero-config after installation, and hooks into the existing `build_tape_context` architecture without modifying core.

## Release Notes

### v0.1.2 (2026-06-29)

**Query-Driven Filtering** — Cuts memory block size by 78-81% with minimal accuracy loss.

- `extract_cues()`: Deterministic cue extraction from user queries (language-aware,
  14 languages supported). Cues are used to filter entities/relations before injection.
- `_format_snapshots_filtered()`: Renders only entities and relations relevant to the
  current question, with 1-hop relation traversal to preserve answer-reachable nodes.

**LoCoMo Evaluation Suite** — Two benchmark scripts against the ACL 2024 LoCoMo dataset.

- `scripts/eval_locomo.py`: Recall/precision benchmark with token savings per category.
- `scripts/eval_locomo_judge.py`: Mem0-protocol LLM-judge accuracy evaluation.
  Session timeline injection fixes temporal recall (0% → 50-100%).
  Baseline accuracy reaches 75-100% on DeepSeek (above Mem0's published 67%).

**Multilang Support** — 14 languages via i18n entity patterns.

- Language-aware stopwords, candidate patterns, and multi-word extraction for:
  `en`, `zh-CN`, `zh-TW`, `ja`, `ko`, `ru`, `de`, `fr`, `es`, `it`, `pt-br`,
  `be`, `hi`, `id`.
- `BUB_SEMANTIC_LANGS` env var controls active languages (default: `en`).
- Regex-based fallback for languages without i18n data.

**Performance** — LLM extraction uses `EmptyTapeStore` to avoid unnecessary persistence;
  snapshot loading is cached within a turn to avoid repeated I/O.

**Breaking changes:** None (additive release).

---

## Installation

The plugin is already registered in `pyproject.toml`:

```toml
[project.entry-points."bub"]
semantic_memory = "bub.plugins.semantic_memory.hook_impl:SemanticMemoryPlugin"
```

Bub's framework automatically loads and instantiates it on startup. No additional setup required.

## How It Works

### Per-Turn Flow

1. **Input**: Agent receives a new message, tape entries are loaded
2. **Extract**: LLM analyzes entries and identifies:
   - **Entities**: people, tasks, events, concepts
   - **Relations**: created, depends_on, mentions, etc.
3. **Store**: SemanticSnapshot is appended to `~/.bub/tapes/semantic/{tape_id}.jsonl`
4. **Load**: All historical snapshots for this tape are loaded
5. **Inject**: Semantic memory is formatted as a system prompt block and prepended to the context
6. **Output**: Agent receives enriched context with semantic awareness

### Example

Given this conversation:
```
User: "Alice created a task to deploy v1.0"
Agent: [responds]
User: "What did Alice do?"
```

On the second turn, the agent sees:

```
## Semantic Memory

### Entities (2):
- person:alice
- task:deploy_v1 (v1.0 deployment)

### Relations (1):
- alice --created--> deploy_v1

---

[rest of context]
```

## Architecture

### Core Modules

- **`models.py`**: Pydantic dataclasses for Entity, Relation, SemanticSnapshot
- **`extractor.py`**: LLM-based extraction from tape entries
- **`store.py`**: JSONL file storage at `~/.bub/tapes/semantic/`
- **`context.py`**: Formatting snapshots into system prompts
- **`hook_impl.py`**: Bub hookimpl that wires everything together

### Storage Format

Snapshots are stored as JSONL (one JSON object per line):

```json
{
  "entities": [
    {"id": "ent_abc123", "type": "person", "name": "Alice", "metadata": {}},
    {"id": "ent_def456", "type": "task", "name": "deploy_v1", "metadata": {"version": "1.0"}}
  ],
  "relations": [
    {"from": "ent_abc123", "to": "ent_def456", "type": "created", "metadata": {}}
  ],
  "tape_id": "527c9ae0c6f31e05__0b871d5e50e7c192",
  "anchor_id": "anchor_001",
  "created_at": "2026-06-06T09:35:00Z"
}
```

## Configuration

The plugin **reuses your main LLM settings** (`BUB_MODEL`, `BUB_API_KEY`, etc.):

```bash
# Your existing setup (e.g., DeepSeek)
export BUB_MODEL=deepseek:deepseek-chat
export BUB_API_KEY=sk-...
```

**Optional configuration:**

```bash
# Multilang support (default: en)
export BUB_SEMANTIC_LANGS=en,zh-CN,ja

# Query-driven filtering (off by default; set to enable)
export BUB_SEMANTIC_QUERY_DRIVEN=1
```

No separate credentials needed.

## Testing

Run the test suite:

```bash
uv run pytest tests/plugins/semantic_memory/test_semantic_memory.py -v
```

**Coverage**: 43 tests across unit and integration scenarios:
- Entity/Relation serialization
- JSONL storage I/O
- LLM extraction with mocks
- Context building
- Multi-turn memory retention

## Usage Examples

### Example 1: CLI Multi-Turn

```bash
$ uv run bub chat
bub > Alice is a data scientist.
Agent > Got it.

bub > What is Alice's profession?
Agent > Alice is a data scientist. (retrieved from semantic memory)

bub > ,tape.info
[Shows: 2 entries, 1 anchor, ... semantic snapshots: 2]
```

### Example 2: Telegram

```
You: "I need to fix a critical bug in the payment module"
Bot: [Uses semantic memory to track bug, module]

You: "What was I working on?"
Bot: [Recalls semantic memory: bug:critical_payment, module:payment]
```

### Example 3: Inspect Semantic Store

```bash
$ cat ~/.bub/tapes/semantic/527c9ae0c6f31e05__0b871d5e50e7c192.jsonl | python -m json.tool
[Shows stored entities and relations]
```

## Performance & Cost

### Token Usage
- Each extraction call: ~300-500 tokens (depends on entry volume)
- Estimated overhead: **+10-20%** per turn (configurable via extraction prompt)

### Storage
- JSONL format: ~1-2 KB per snapshot (grows with entities/relations)
- Typical session: ~50-100 KB

### Latency
- Extraction is async, non-blocking
- First turn (with extraction): ~500ms extra
- Subsequent turns: ~50ms extra (just loading snapshots)

## Graceful Degradation

If semantic extraction fails for any reason:
- LLM error: Returns empty snapshot, continues
- Invalid JSON: Logged as warning, continues
- Storage error: Logged, continues with base context

The agent **always** works, semantic memory is optional enhancement.

## Future Enhancements

### Phase 2: Smart Retrieval ✅ (partial — v0.1.2)
- ~~Vector embeddings for semantic similarity search~~
- ✅ Query-driven context filtering (reduce prompt bloat by 78-81%)
- ✅ Language-aware cue extraction (14 languages)

### Phase 3: Advanced Graphs
- Entity dependency analysis (who depends on what)
- Centrality metrics (who/what is most important)
- Causal reasoning (what led to what)

### Phase 4: Multi-Session Memory
- Cross-session entity resolution
- Long-term memory across multiple conversations
- Persistent entity graph (not just per-tape)

## Troubleshooting

**Q: Plugin not loading?**
A: Check that entry-point is registered:
```bash
python -c "import importlib.metadata; print(list(importlib.metadata.entry_points(group='bub')))"
```

**Q: Semantic snapshots not appearing?**
A: Check `~/.bub/tapes/semantic/` directory exists. Check logs with `BUB_VERBOSE=1`.

**Q: LLM calls are expensive?**
A: Reduce extraction frequency or use a cheaper model (e.g., DeepSeek distill). Future releases will support model selection per plugin.

## API Reference

### `build_semantic_context(entries, context, llm=None, store=None) → list[dict]`

Build context with semantic memory. Called by the framework automatically.

**Args:**
- `entries`: Iterable of TapeEntry objects
- `context`: TapeContext instance
- `llm`: republic.LLM instance (optional; if None, returns base context)
- `store`: SemanticStore instance (optional; if None, returns base context)

**Returns:** List of message dicts ready for model input

### `extract_semantics(entries, llm, tape_id, anchor_id=None, max_tokens=1000) → SemanticSnapshot`

Extract entities and relations from tape entries.

**Args:**
- `entries`: List of TapeEntry objects
- `llm`: republic.LLM instance for extraction
- `tape_id`: Session/tape identifier
- `anchor_id`: Optional anchor point identifier
- `max_tokens`: Max tokens for LLM response

**Returns:** SemanticSnapshot with extracted entities/relations

## Contributing

This plugin is part of Bub's extensibility model. To extend:

1. **Custom entity types**: Modify Entity.type enum in models.py
2. **Custom extractors**: Replace or wrap extractor.py
3. **Custom storage**: Implement SemanticStore interface
4. **Custom formatters**: Replace _format_snapshots in context.py

All without modifying Bub core.

## License

Same as Bub (Apache 2.0)

---

**Questions?** See [Bub documentation](https://bub.build) or open an issue.
