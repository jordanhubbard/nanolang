# Git Reconciliation Plan - NanoLang Repository

## Situation Analysis

**Status:** Local and remote have diverged significantly
- **Local:** At commit abd7fa0, with ~38 files of **uncommitted changes**
- **Remote:** 20 commits ahead at c58e813
- **Divergence Point:** abd7fa0 (local HEAD)

### Work Streams

#### Remote Changes (20 commits, ~5,400 insertions)
1. **Infrastructure:** Added logging API, debugging guides, property testing examples
2. **Organization:** Reorganized examples/ into subdirectories (advanced/, games/, audio/, graphics/, data/, debug/)
3. **Features:** Union variant construction, module introspection, compiler schema updates
4. **Quality:** Added json_diagnostics.c, error handling improvements
5. **CI/Automation:** Integrated autobeads.py for automatic issue tracking

#### Local Changes (uncommitted, ~2,600 insertions)
1. **Compiler Core:** Type system improvements, module-qualified calls
2. **Type Handling:** Support for TYPE_UNKNOWN in string contexts (nested concatenation fix)
3. **Module System:** Module introspection (exported struct tracking)
4. **Temp Files:** Added temp file helpers (tmp_dir, mktemp, mktemp_dir)
5. **Safety:** Added unsafe blocks to module FFI calls
6. **Examples:** Bug fixes to sdl_pong.nano (simplified conditionals)
7. **Testing:** Updates to test infrastructure

## Conflict Analysis

### HIGH PRIORITY CONFLICTS

#### 1. **Makefile** (Both modified autobeads integration)
- **Remote:** Added json_diagnostics.c to sources, basic autobeads integration
- **Local:** Modified test runner to call autobeads.py with different flags/modes
- **Resolution:** Merge both changes - need json_diagnostics.c + sophisticated autobeads flags

#### 2. **src/typechecker.c** (Different features added)
- **Remote:** Added union variant construction (UnionName.VariantName syntax support)
- **Local:** Added:
  - TYPE_UNKNOWN handling in string concatenation
  - Module introspection (exported struct tracking)
  - Temp file helper function registrations
- **Resolution:** Keep ALL changes - these are additive, non-conflicting features

#### 3. **examples/sdl_pong.nano** (Modified vs Relocated)
- **Remote:** Moved to examples/games/sdl_pong.nano
- **Local:** Modified in place (simplified conditional logic)
- **Resolution:** Apply local changes to new location (examples/games/)

#### 4. **modules/std/log/log.nano** (Style changes)
- **Remote:** Kept constants as `pub let` (public)
- **Local:** Changed to `let` (private) + added `unsafe` blocks around FFI calls
- **Resolution:** Favor local (safer with unsafe blocks + encapsulation)

### MEDIUM PRIORITY CONFLICTS

#### 5. **.beads/issues.jsonl** (Both modified extensively)
- **Remote:** 130 lines changed
- **Local:** 235 lines changed
- **Resolution:** Manual merge required - use `bd sync` after reconciliation

#### 6. **src/generated/compiler_schema.h** (Generated file, staged locally)
- **Local:** Staged changes (alphabetically sorted forward declarations)
- **Remote:** May have different generation
- **Resolution:** Regenerate after merge (this is a generated file)

### LOW PRIORITY (Additive Changes)

#### 7. **src_nano/** (Compiler improvements)
- Both modified, but different functions
- **Resolution:** Merge naturally, verify bootstrap passes

#### 8. **tests/** (Test additions/modifications)
- Remote added new test files
- Local modified existing tests
- **Resolution:** Keep both sets

## Reconciliation Strategy

### Phase 1: Backup and Prepare
```bash
# Create safety branch with local changes
git branch local-work-backup-$(date +%Y%m%d)

# Create stash of local work for reference
git stash push -u -m "Local uncommitted work before reconciliation"

# Verify stash worked
git stash list
```

### Phase 2: Merge Remote Changes
```bash
# Reset to clean state
git reset --hard HEAD

# Pull remote changes (fast-forward to c58e813)
git merge --ff-only origin/main
```

### Phase 3: Apply Local Changes Intelligently

#### 3.1 Restore Non-Conflicting Files First
```bash
# Apply local changes (will have conflicts)
git stash pop

# At this point, handle conflicts file by file
```

#### 3.2 Resolve Makefile
- Merge json_diagnostics.c addition from remote
- Keep sophisticated autobeads.py integration from local
- Manual edit required

#### 3.3 Resolve src/typechecker.c
- Keep remote's union variant construction
- Add local's TYPE_UNKNOWN string handling
- Add local's module introspection
- Add local's temp file helpers
- These are additive - combine all features

#### 3.4 Handle Relocated examples/
```bash
# examples/sdl_pong.nano was moved to examples/games/
# Apply local changes to new location
git checkout --theirs examples/games/sdl_pong.nano
# Then manually apply local logic fixes
```

#### 3.5 Resolve Module Changes
- modules/std/log/log.nano: Favor local (unsafe blocks)
- modules/std/json/: Merge changes
- modules/std/fs.nano: Merge changes

#### 3.6 Beads Database
```bash
# Keep remote's beads for now
git checkout --theirs .beads/issues.jsonl
git checkout --theirs .beads/last-touched

# After full reconciliation, run:
bd sync
```

### Phase 4: Regenerate Generated Files
```bash
# Regenerate compiler schema
make schema

# This should update src/generated/compiler_schema.h
```

### Phase 5: Verification
```bash
# Rebuild everything
make clean
make build

# Run tests
make test

# Run selfhost tests
make bootstrap

# Verify examples work
make examples
```

### Phase 6: Commit and Push
```bash
# Review all changes
git status
git diff --cached

# Commit with detailed message
git add -A
git commit -m "Reconcile local and remote work

Local changes (uncommitted work):
- Type system: TYPE_UNKNOWN string context handling
- Module introspection: exported struct tracking
- Temp file helpers: tmp_dir, mktemp, mktemp_dir
- Safety: unsafe blocks in module FFI calls
- Bug fixes: sdl_pong conditional logic

Remote changes (20 commits):
- Infrastructure: logging API, debugging guides
- Organization: examples subdirectories
- Features: union variants, module introspection
- Quality: json_diagnostics, error handling

Conflict resolutions:
- Makefile: merged autobeads integration
- typechecker.c: combined all features (union variants + local improvements)
- examples/sdl_pong.nano: applied local fixes to new location
- modules/std/log/log.nano: used local version (safer with unsafe blocks)
- Regenerated src/generated/compiler_schema.h

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"

# Push to remote
git push origin main
```

## Risk Mitigation

1. **Backup Branch:** Created before any changes
2. **Stash:** Full copy of local work preserved
3. **Incremental:** Resolve conflicts file-by-file
4. **Verification:** Full test suite + bootstrap before push
5. **Beads:** Use `bd sync` after manual merge to preserve issue tracking

## Files Requiring Manual Attention

### Critical (Must manually merge):
- [ ] Makefile
- [ ] src/typechecker.c
- [ ] examples/games/sdl_pong.nano (apply local changes)
- [ ] modules/std/log/log.nano
- [ ] .beads/issues.jsonl (resolve after main merge)

### Important (Verify after automatic merge):
- [ ] src/transpiler.c
- [ ] src/parser.c
- [ ] src_nano/typecheck.nano
- [ ] src_nano/transpiler.nano
- [ ] tests/* (ensure all local test updates preserved)

### Generated (Regenerate):
- [ ] src/generated/compiler_schema.h

## Success Criteria

✅ All remote commits integrated (at c58e813)
✅ All local improvements preserved
✅ `make build` succeeds
✅ `make test` passes all tests
✅ `make bootstrap` completes successfully
✅ Key examples compile and run (sdl_pong, others)
✅ No regression in existing functionality
✅ Beads database reconciled with `bd sync`
✅ Changes pushed to origin/main

## Notes

- The divergence is ADDITIVE, not conflicting work
- Remote focused on infrastructure and organization
- Local focused on compiler improvements and safety
- Most conflicts are in overlapping areas (Makefile, typechecker)
- Examples reorganization is straightforward to handle
- Beads database may need manual review after `bd sync`
