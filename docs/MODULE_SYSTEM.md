# My Module System

I use a module system to organize code, manage dependencies, and bridge the gap between my safe world and the C world. This document describes how I build modules, compile C sources automatically, and handle linking.

## Overview

My modules are self-contained packages. They can include:
- My source code (`.nano` files)
- C FFI implementations (`.c` files)
- Module metadata (`module.json`)

When you import a module, I automatically:
1. Compile any C sources if they have changed.
2. Cache the compiled objects in a local build directory.
3. Track dependencies between modules.
4. Generate the correct link commands for the final binary.

## Module Structure

I expect a typical module directory to look like this:

```
modules/sdl_ttf/
├── module.json              # Module metadata (required if C sources exist)
├── sdl_ttf.nano             # Main module file (required)
├── sdl_ttf_helpers.c        # C implementation (optional)
└── sdl_ttf_helpers.nano     # Additional nanolang helpers (optional)
```

## module.json Specification

I use the `module.json` file to understand how to build and link a module.

### Basic Structure

```json
{
  "name": "module_name",
  "version": "1.0.0",
  "description": "Short description of the module",
  
  "c_sources": [
    "helper.c",
    "another_file.c"
  ],
  
  "system_libs": [
    "SDL2_ttf",
    "m"
  ],
  
  "pkg_config": [
    "sdl2_ttf"
  ],
  
  "include_dirs": [
    "/usr/local/include"
  ],
  
  "cflags": [
    "-O2",
    "-Wall"
  ],
  
  "ldflags": [
    "-L/usr/local/lib"
  ],
  
  "dependencies": [
    "sdl"
  ]
}
```

### Fields

#### Required Fields

- **`name`** (string): The module name. This should match the directory name.

#### Optional Fields

- **`version`** (string): Semantic version (e.g., "1.0.0").
- **`description`** (string): A brief description of what the module does.
- **`c_sources`** (array of strings): C source files I should compile. These are relative to the module directory.
- **`system_libs`** (array of strings): System libraries I need to link (e.g., "SDL2", "m", "pthread").
- **`pkg_config`** (array of strings): pkg-config packages I should use for compile and link flags.
- **`include_dirs`** (array of strings): Additional directories I should search for headers.
- **`cflags`** (array of strings): Additional compiler flags I should pass to the C compiler.
- **`ldflags`** (array of strings): Additional linker flags I should use.
- **`dependencies`** (array of strings): Other modules this module depends on.

### Field Priority

When I determine build flags, I apply this priority. Later items override earlier ones:

1. Default system flags
2. `pkg_config` flags
3. `include_dirs` -> `-I` flags
4. `cflags` (custom compile flags)
5. `ldflags` (custom link flags)
6. `system_libs` -> `-l` flags

## Build Process

### When I Process an Import

1. **I parse module.json** if it exists in the module directory.
2. **I check dependencies** and recursively process any dependent modules.
3. **I check my build cache** using content and build-context evidence, not
   source timestamps alone. The cache root is local or selected by
   `NANO_BUILD_CACHE`; `current` names a complete immutable generation.
4. **I rebuild if necessary** in a private staging directory. Missing,
   damaged or changed dependency evidence prevents reuse. A completed build
   publishes a new generation without overwriting the previous one.
5. **I track the results for linking**:
   - I add the object file to my link list.
   - I add `system_libs` to my link list.
   - I add `ldflags` to my link list.

### Build Cache

Without a shared cache setting, I keep C artifacts beneath the module's
`.build` directory. A generation contains native objects, a shared library,
dependency evidence and, when reuse can be justified, `source_hashes.json`.
For example, on Darwin:

```
modules/example/.build/
├── .build.lock
├── current -> .nano-gen-ABC123
└── .nano-gen-ABC123/
    ├── example.o
    ├── libexample.dylib
    └── source_hashes.json
```

The abbreviated tree omits per-source objects and dependency records. On other
supported POSIX hosts the shared library uses `.so`. My current builder does
not use the historical `.build_info.json` layout. Shared roots use a versioned
SHA-256 key of the module directory's canonical path; generation names are
private publication identities, not content-addressed hashes.

### Incremental Compilation

I compare source, manifest, dependency and build-context evidence before reuse.
Same-size edits with restored timestamps still invalidate recorded content.
If I cannot establish the required evidence, I rebuild rather than treating a
timestamp as proof. Source snapshots and complete toolchain discovery remain
open; the detailed limitations are in my [NanoISA cache contract](NANOISA.md).

### Parallel Builds

I do not yet compile modules in parallel. This is a planned enhancement for when modules have no interdependencies.

## Using Modules

### Pure Modules

If a module contains no C code, you do not need a `module.json` file. You can simply import it:

```nano
import "modules/my_pure_module/my_module.nano"
```

### Modules with C FFI

I handle compilation and linking automatically when you import a module with C code:

```nano
import "modules/sdl_ttf/sdl_ttf.nano"

fn main() -> int {
    (TTF_Init)  # C function automatically available
    return 0
}
```

When you compile:
```bash
nanoc my_app.nano -o my_app
```

I perform these steps:
1. I detect the `sdl_ttf` module import.
2. I read `modules/sdl_ttf/module.json`.
3. I compile `sdl_ttf_helpers.c` if my cache indicates it is stale.
4. I link with the `SDL2_ttf` system library.
5. I produce the final binary.

### Module Dependencies

If module A depends on module B, I handle the ordering.

**modules/my_module/module.json**:
```json
{
  "name": "my_module",
  "dependencies": ["sdl", "sdl_helpers"],
  "c_sources": ["my_module.c"]
}
```

When you import `my_module`, I automatically:
1. Process the `sdl` module first.
2. Process the `sdl_helpers` module.
3. Process `my_module`.
4. Link the object files from all three modules.

## Environment Variables

### NANO_MODULE_PATH

I use this semicolon-separated list of directories to search for modules:

```bash
export NANO_MODULE_PATH="/usr/local/lib/nano/modules:./modules:~/nano/modules"
```

If you do not set this, I default to `modules` relative to the current directory.

### NANO_BUILD_CACHE

I use this directory for my module C/FFI build cache:

```bash
export NANO_BUILD_CACHE="/tmp/nano_build_cache"
```

If you do not set this, I default to `.build/` within each module directory.

When you build through my Makefile I export `NANO_BUILD_CACHE` as
`obj/module_cache`. Ordinary `make clean` and `make -C examples clean`
retain this cache, the configured cache, and local module `.build` directories.
Existing bytecode can contain absolute references to their immutable library
generations. Deleting them is not an ordinary compiler-object cleanup.
I remove unrelated compiler objects and binaries, preserving the directory
ancestors needed by retained caches. I conservatively retain earlier cache
locations containing my lock, generation or private-stage names even when the
current environment selects another cache. Those names authorize retention,
not collection. I reject cleanup trees outside the
workspace and paths through directory symlinks before removing anything.
The build workspace and Make configuration remain trusted inputs.

Cache retention has no automatic size or age limit. Before manually resetting
a cache, stop its builders and compiler descendants, retire every dependent
bytecode artifact, and accept that those artifacts will need rebuilding.
A released build lock or a dead builder PID does not establish quiescence.
My [NanoISA cache contract](NANOISA.md) records the tested generation,
publication and lifetime boundaries; automatic collection remains unfinished.

### NANO_CC

I use this C compiler:

```bash
export NANO_CC=clang
```

I default to `gcc` if you do not specify otherwise.

### NANO_VERBOSE_BUILD

If you enable this, I will provide verbose output during the build process:

```bash
export NANO_VERBOSE_BUILD=1
nanoc my_app.nano -o my_app
```

## Examples

### Example 1: SDL Window Module

**modules/sdl/module.json**:
```json
{
  "name": "sdl",
  "version": "1.0.0",
  "description": "SDL2 windowing, rendering, and events",
  "pkg_config": ["sdl2"]
}
```

This module has no C sources. I use the system SDL2 library directly.

### Example 2: SDL Helpers with C Code

**modules/sdl_helpers/module.json**:
```json
{
  "name": "sdl_helpers",
  "version": "1.0.0",
  "description": "Helper functions for SDL rendering",
  "c_sources": ["sdl_helpers.c"],
  "pkg_config": ["sdl2"],
  "dependencies": ["sdl"]
}
```

This module has a C implementation and depends on my `sdl` module.

### Example 3: SDL_ttf Extension

**modules/sdl_ttf/module.json**:
```json
{
  "name": "sdl_ttf",
  "version": "1.0.0",
  "description": "TrueType font rendering for SDL",
  "c_sources": ["sdl_ttf_helpers.c"],
  "pkg_config": ["sdl2_ttf"],
  "dependencies": ["sdl"]
}
```

## Troubleshooting

### "pkg-config: command not found"

I need `pkg-config` to find system libraries. You should install it:
- macOS: `brew install pkg-config`
- Linux: `sudo apt-get install pkg-config`

### "No package 'sdl2_ttf' found"

I cannot find the development files for this library. You should install the development package:
- macOS: `brew install sdl2_ttf`
- Linux: `sudo apt-get install libsdl2-ttf-dev`

### Module not rebuilding

If you think I should rebuild a module but I am not doing so, you can clear my build cache:
```bash
rm -rf modules/*/build/
```

Alternatively, you can update the timestamp on the source file:
```bash
touch modules/sdl_ttf/sdl_ttf_helpers.c
```

### Verbose build output

If you want to see exactly what I am doing during a build, set this environment variable:

```bash
NANO_VERBOSE_BUILD=1 nanoc my_app.nano -o my_app
```

## Best Practices

1. **Keep module.json minimal.** Only specify what I need to know.
2. **Use pkg-config when available.** This is more reliable than hardcoded paths.
3. **Version your modules.** I recommend semantic versioning.
4. **Document dependencies.** List all other modules your module depends on.
5. **Test incremental builds.** Ensure my cache works as you expect.
6. **Provide examples.** Show others how to use your module.

## Future Enhancements

- [ ] Parallel module compilation
- [ ] Cross-compilation support
- [ ] Module registry and package manager
- [ ] Automatic dependency installation
- [ ] Binary module distribution (.a/.so files)
- [ ] Module versioning and compatibility checking
