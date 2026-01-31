# Module Build System Integration

## Dependency Locator Integration

The module build system (`build_module.sh`) now integrates with the nanolang dependency locator (`dep_locator.nano`) to automatically discover compilation flags at build time.

### How It Works

1. **Module Metadata**: Add `library_name` to `module.json`:
   ```json
   {
     "compilation": {
       "library_name": "SDL2",
       "include_paths": { ... },
       "library_paths": { ... },
       "libraries": [ ... ]
     }
   }
   ```

2. **Runtime Discovery**: When building a module:
   - If `library_name` is specified, `build_module.sh` calls `dep_locator.sh`
   - The dependency locator searches for the library using:
     - pkg-config (if available)
     - Heuristic search in common prefixes
   - Returns JSON with `include_dirs`, `library_dirs`, and `libraries`

3. **Fallback**: If discovery fails or `library_name` is not specified:
   - Falls back to hardcoded paths in `module.json`
   - Uses platform-specific paths (`macos`/`linux`)

### Benefits

- **Platform Independence**: No need to hardcode paths for different systems
- **Automatic Discovery**: Finds libraries installed via Homebrew, apt, etc.
- **Backward Compatible**: Still works with hardcoded paths in `module.json`
- **Self-Documenting**: `library_name` clearly indicates what external library is needed

### Example

```bash
# Build SDL module - automatically discovers SDL2 installation
cd modules
./tools/build_module.sh sdl

# Output:
# Building module: sdl
# Discovering compilation flags for library: SDL2
# ✓ Discovered compilation flags via dependency locator
# Module is FFI-only, skipping compilation
# Creating package: sdl.nano.tar.zst
# ✓ Module package created: sdl/sdl.nano.tar.zst
```

### Module.json Format

```json
{
  "compilation": {
    "library_name": "SDL2",  // Library to locate (optional)
    "include_paths": {       // Fallback paths (optional)
      "macos": ["/opt/homebrew/include/SDL2"],
      "linux": ["/usr/include/SDL2"]
    },
    "library_paths": {       // Fallback paths (optional)
      "macos": ["/opt/homebrew/lib"],
      "linux": ["/usr/lib/x86_64-linux-gnu"]
    },
    "libraries": ["SDL2"]   // Library names (optional)
  }
}
```

### Future Enhancements

Tracked as beads:
- `nanolang-3jpj`: Parse pkg-config output to extract exact flags
- `nanolang-z0m9`: Support multiple library dependencies
- `nanolang-tija`: Cache discovery results
- `nanolang-rc4y`: Validate discovered paths before using them

View with: `bd ready` or `bd show <id>`

