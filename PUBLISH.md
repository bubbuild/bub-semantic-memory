# Publish bub-semantic-memory to PyPI

## 📍 Location
```
~/Documents/playground/bub/packages/semantic-memory/
```

## 📦 What's Included
- `src/bub/plugins/semantic_memory/` - 7 modules (652 lines)
- `tests/` - 43 tests
- `README.md` - Documentation
- `pyproject.toml` - PyPI metadata

## 🚀 Quick Publish

### Step 1: Create GitHub Repo
```bash
cd ~/Documents/playground/bub
git add packages/semantic-memory
git commit -m "feat: add semantic-memory plugin for distribution"
git push

# Or create separate repo at https://github.com/bubbuild/bub-semantic-memory
```

### Step 2: Build & Publish to PyPI
```bash
cd ~/Documents/playground/bub/packages/semantic-memory
uv build
uv publish
```

### Step 3: Submit to hub.bub.build
Fork https://github.com/bubbuild/buildscape
Add `plugins/semantic-memory.json`:
```json
{
  "name": "semantic-memory",
  "title": "Semantic Memory",
  "description": "Extract and retain semantic entities and relations from conversations",
  "author": "Bub Community",
  "license": "Apache-2.0",
  "repository": "https://github.com/bubbuild/bub",
  "pypi": "bub-semantic-memory",
  "documentation": "https://github.com/bubbuild/bub#semantic-memory",
  "categories": ["memory", "context"]
}
```

## 📊 Package Info
- **PyPI Name**: bub-semantic-memory
- **Entry Point**: bub.plugins.semantic_memory.hook_impl:SemanticMemoryPlugin
- **Version**: 0.1.0
- **Python**: 3.12+
- **License**: Apache 2.0

## ✅ Verification
- [x] Code: 652 lines, 7 modules
- [x] Tests: 43 passing
- [x] Documentation: README.md
- [x] pyproject.toml: Configured for PyPI
- [x] Entry point: bub.plugins.semantic_memory

Ready to publish! 🎉
