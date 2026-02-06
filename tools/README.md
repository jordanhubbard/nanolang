# NanoLang Tools

## Module Index Generator

Generates `modules/index.json` from all `module.manifest.json` files in the modules directory.

### Python Version (Current)

**File:** `generate_module_index.py`  
**Status:** ✅ Production - fully functional

```bash
python3 tools/generate_module_index.py
```

Features:
- Scans all modules for manifests
- Builds reverse indices (capabilities, keywords, IO surfaces)
- Validates schema compliance
- Integrated into `make modules-index`

### NanoLang Version (Work-in-Progress)

**File:** `generate_module_index.nano`  
**Status:** 🚧 WIP - "eating our own dog food"

```bash
# Not yet working - module import issues to resolve
./bin/nanoc tools/generate_module_index.nano -o bin/generate_module_index
./bin/generate_module_index
```

**Why rewrite in NanoLang?**
1. Demonstrate NanoLang's capabilities (JSON, file I/O, traversal)
2. Practice what we preach - "eat our own dog food"
3. Reduce external dependencies (Python)
4. Provide real-world example of using NanoLang for tooling

**Blockers:**
- Module import path resolution needs fixing
- Need to verify JSON/filesystem module integration
- Reverse index building logic is simplified

**TODO:** Complete the NanoLang version and switch `make modules-index` to use it.

Related to user feedback: "wait why don't you write that tool in nanolang"

## NanoLang Linter

Checks NanoLang files for syntax errors and canonical style violations.

**File:** `nano_lint.nano`  
**Status:** ✅ Available

```bash
./bin/nanoc tools/nano_lint.nano -o bin/nano_lint
./bin/nano_lint path/to/file.nano
```

Current checks:
- Lexer + parser errors (syntax)
- Missing shadow tests for functions
