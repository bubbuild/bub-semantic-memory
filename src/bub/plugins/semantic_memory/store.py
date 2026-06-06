from __future__ import annotations

import json
from pathlib import Path

import aiofiles

from bub.plugins.semantic_memory.types import SemanticSnapshot


class SemanticStore:
    def __init__(self, storage_root: Path | None = None) -> None:
        if storage_root is None:
            storage_root = Path.home() / ".bub" / "tapes" / "semantic"
        self._storage_root = storage_root
        self._storage_root.mkdir(parents=True, exist_ok=True)

    def tape_file_path(self, tape_id: str) -> Path:
        return self._storage_root / f"{tape_id}.jsonl"

    async def append(self, tape_id: str, snapshot: SemanticSnapshot) -> None:
        path = self.tape_file_path(tape_id)
        async with aiofiles.open(path, mode="a", encoding="utf-8") as f:
            await f.write(snapshot.model_dump_json() + "\n")

    async def load(self, tape_id: str) -> list[SemanticSnapshot]:
        path = self.tape_file_path(tape_id)
        if not path.exists():
            return []
        async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
            lines = await f.readlines()
        snapshots: list[SemanticSnapshot] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                snapshots.append(SemanticSnapshot.model_validate(data))
            except (json.JSONDecodeError, Exception):
                continue
        return snapshots
