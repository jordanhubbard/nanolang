# Module Tools

Tools for building, packaging, and managing nanolang modules.

## dep_locator.nano

A dependency locator written in nanolang that finds C/C++ libraries on macOS and Linux.

### Features

- **pkg-config support**: Uses pkg-config when available
- **Heuristic search**: Falls back to searching common prefixes (`/usr`, `/usr/local`, `/opt/homebrew`, etc.)
- **Platform detection**: Automatically detects macOS Homebrew paths
- **JSON output**: Outputs structured JSON with include dirs, library dirs, and libraries

### Usage

```bash
# Using the wrapper script (recommended)
./dep_locator.sh SDL2

# Using environment variables
DEP_LOCATOR_NAME=SDL2 ./dep_locator.nano

# With custom header/library names
./dep_locator.sh openssl --header-name openssl/ssl.h --lib-name ssl
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

### Integration with Module Build System

The dependency locator can be used by `build_module.sh` to automatically discover compilation flags:

```bash
# In build_module.sh
LOCATION=$(./dep_locator.sh SDL2)
# Parse JSON and extract include_dirs, library_dirs, libraries
```

## build_module.sh

Builds a nanolang module from source and creates a package.

## install_module.sh

Installs a module package to `NANO_MODULE_PATH`.

