# Module Tools

Tools for building, packaging, and managing nanolang modules with C/C++ dependencies.

## dep_locator.sh - Universal Dependency Locator

**Location:** `modules/tools/dep_locator.sh`

A portable shell script that finds C/C++ libraries on macOS and Linux. Outputs structured JSON for easy parsing in Makefiles and build scripts.

### Features

- **Cross-platform**: Works on macOS (Homebrew, MacPorts) and Linux (standard paths)
- **JSON output**: Structured data for easy parsing
- **Flexible**: Supports custom header names and library names
- **No dependencies**: Pure shell script, no external tools required
- **Fast**: Direct filesystem checks

### Usage

```bash
# Basic usage
./dep_locator.sh <library-name>

# With custom header/library names
./dep_locator.sh <library-name> --header-name <header> --lib-name <lib>

# Examples
./dep_locator.sh SDL2
./dep_locator.sh openssl --header-name openssl --lib-name ssl
./dep_locator.sh zlib --lib-name z
```

### Output Format

```json
{
  "name": "SDL2",
  "found": true,
  "origin": "heuristic",
  "include_dirs": [
    "/opt/homebrew/include/SDL2"
  ],
  "library_dirs": [
    "/opt/homebrew/lib"
  ],
  "libraries": [
    "SDL2"
  ]
}
```

### Integration with Makefiles

**Example 1: Simple Integration**

```makefile
DEP_LOCATOR := path/to/dep_locator.sh

# Find library
MYLIB_INFO := $(shell $(DEP_LOCATOR) mylib 2>/dev/null)
MYLIB_FOUND := $(shell echo '$(MYLIB_INFO)' | grep -q '"found": *true' && echo "yes" || echo "no")

ifeq ($(MYLIB_FOUND),yes)
    # Extract include and library directories
    MYLIB_INCLUDE := $(shell echo '$(MYLIB_INFO)' | sed -n 's/.*"include_dirs": *\[ *"\([^"]*\)".*/\1/p')
    MYLIB_LIBDIR := $(shell echo '$(MYLIB_INFO)' | sed -n 's/.*"library_dirs": *\[ *"\([^"]*\)".*/\1/p')
    MYLIB_LIB := $(shell echo '$(MYLIB_INFO)' | sed -n 's/.*"libraries": *\[ *"\([^"]*\)".*/\1/p')
    
    CFLAGS += -I$(MYLIB_INCLUDE)
    LDFLAGS += -L$(MYLIB_LIBDIR) -l$(MYLIB_LIB)
else
    $(error mylib not found. Please install mylib)
endif
```

**Example 2: pkg-config Fallback Pattern**

The recommended pattern is to try pkg-config first (most reliable), then fall back to dep_locator:

```makefile
DEP_LOCATOR := path/to/dep_locator.sh

# Try pkg-config first
MYLIB_PKG := $(shell pkg-config --exists mylib 2>/dev/null && echo "yes" || echo "no")

ifeq ($(MYLIB_PKG),yes)
    CFLAGS += $(shell pkg-config --cflags mylib)
    LDFLAGS += $(shell pkg-config --libs mylib)
else
    # Fall back to dep_locator
    MYLIB_INFO := $(shell $(DEP_LOCATOR) mylib 2>/dev/null)
    MYLIB_FOUND := $(shell echo '$(MYLIB_INFO)' | grep -q '"found": *true' && echo "yes" || echo "no")
    
    ifeq ($(MYLIB_FOUND),yes)
        MYLIB_INCLUDE := $(shell echo '$(MYLIB_INFO)' | sed -n 's/.*"include_dirs": *\[ *"\([^"]*\)".*/\1/p')
        MYLIB_LIBDIR := $(shell echo '$(MYLIB_INFO)' | sed -n 's/.*"library_dirs": *\[ *"\([^"]*\)".*/\1/p')
        CFLAGS += -I$(MYLIB_INCLUDE)
        LDFLAGS += -L$(MYLIB_LIBDIR) -lmylib
    else
        $(error mylib not found)
    endif
endif
```

### Search Strategy

`dep_locator.sh` searches the following prefixes in order:

1. `/usr` (standard Linux location)
2. `/usr/local` (common for manual installs)
3. `/opt/homebrew` (Homebrew on Apple Silicon Macs)
4. `/opt/local` (MacPorts on macOS)

For each prefix, it checks:
- **Include directory**: `$prefix/include/$header_name/`
- **Library directory**: `$prefix/lib/` (and verifies library file exists)

### Use Cases

**1. Building Examples with Dependencies**

See `examples/Makefile` for a complete real-world example of using `dep_locator.sh` to find SDL2.

**2. Module Build Scripts**

When building nanolang modules that depend on C libraries:

```bash
#!/bin/bash
# build_my_module.sh

# Find OpenGL
GL_INFO=$(./dep_locator.sh GL --header-name GL --lib-name GL)
if echo "$GL_INFO" | grep -q '"found": *true'; then
    GL_INCLUDE=$(echo "$GL_INFO" | sed -n 's/.*"include_dirs": *\[ *"\([^"]*\)".*/\1/p')
    GL_LIBDIR=$(echo "$GL_INFO" | sed -n 's/.*"library_dirs": *\[ *"\([^"]*\)".*/\1/p')
    
    gcc -I"$GL_INCLUDE" my_module.c -L"$GL_LIBDIR" -lGL -o my_module
else
    echo "OpenGL not found!" >&2
    exit 1
fi
```

**3. CI/CD and Build Verification**

Use `dep_locator.sh` to verify dependencies before building:

```bash
#!/bin/bash
# verify_deps.sh

REQUIRED_LIBS=("SDL2" "openssl" "zlib")

for lib in "${REQUIRED_LIBS[@]}"; do
    if ! ./dep_locator.sh "$lib" | grep -q '"found": *true'; then
        echo "Missing required library: $lib" >&2
        exit 1
    fi
done

echo "All dependencies found!"
```

### Advantages Over Hard-Coded Paths

| Hard-Coded Paths | dep_locator.sh |
|------------------|----------------|
| ❌ Breaks on different systems | ✅ Works across macOS/Linux |
| ❌ Requires manual updates | ✅ Auto-discovers locations |
| ❌ Not portable | ✅ Fully portable |
| ❌ Maintenance burden | ✅ Single source of truth |

### Future Enhancements

The current implementation is a standalone shell script. Future versions may:
- Add support for Windows (via Git Bash/WSL)
- Support custom search paths via environment variables
- Cache results for faster repeated lookups
- Integrate with nanolang's module system more tightly

### Related Files

- **dep_locator.nano**: Future nanolang implementation (currently has type system issues)
- **examples/Makefile**: Real-world usage example for finding SDL2
- **MODULE_BUILD_INTEGRATION.md**: Guide for integrating with module builds

### Contributing

When adding support for new libraries:
1. Test on both macOS and Linux
2. Verify the JSON output is valid
3. Update this README with examples
4. Add to `examples/Makefile` if applicable

### Troubleshooting

**Q: dep_locator.sh says library not found, but it's installed**

A: The library might be in a non-standard location. Check:
```bash
# Find where the library actually is
find /usr /opt -name "libmylib.*" 2>/dev/null

# Add custom search logic to dep_locator.sh if needed
```

**Q: How do I debug the JSON output?**

A: Run directly to see the full output:
```bash
./dep_locator.sh SDL2
```

**Q: Can I use this with other build systems (CMake, etc.)?**

A: Yes! Just call the script and parse the JSON output. Example for CMake:
```cmake
execute_process(
    COMMAND ${CMAKE_SOURCE_DIR}/modules/tools/dep_locator.sh SDL2
    OUTPUT_VARIABLE SDL2_INFO
)
```

