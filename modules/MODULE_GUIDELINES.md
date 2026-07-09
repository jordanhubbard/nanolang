# Nanolang Module System Guidelines

## Cross-Platform Module Best Practices

### Core Principle
**The module system completely decouples third-party library implementation details from nanolang code.**

Users should be able to `import "modules/foo/foo.nano"` without knowing or caring about:
- Where the library is installed
- Whether it's from Homebrew, apt, or compiled from source
- Whether they're on Intel vs Apple Silicon macOS
- Whether they're on macOS, Linux, or Windows

### Use pkg-config Universally

**ALWAYS use pkg-config as the primary discovery mechanism:**

```json
{
  "name": "example",
  "pkg_config": ["libexample"]
}
```

**Why pkg-config?**
- Automatically finds libraries in `/usr/lib`, `/usr/local/lib`, `/opt/homebrew/lib`, etc.
- Handles different Homebrew paths (Intel: `/usr/local`, Apple Silicon: `/opt/homebrew`)
- Works identically on Linux regardless of installation method
- Provides correct `-I`, `-L`, and `-l` flags automatically
- Supported by virtually all modern C libraries

### What NOT to Do

❌ **Never hardcode paths:**
```json
{
  "ldflags": ["-L/opt/homebrew/lib"],      // WRONG - Intel Macs won't work
  "include_dirs": ["/usr/local/include"]   // WRONG - Other systems won't work
}
```

❌ **Never hardcode framework flags:**
```json
{
  "ldflags": ["-framework", "OpenGL"]  // WRONG - Linux doesn't have frameworks
}
```

❌ **Never use `system_libs` when pkg-config is available:**
```json
{
  "system_libs": ["sqlite3"]  // WRONG - Use pkg_config instead
}
```

### Cross-Platform Header Handling

For libraries with platform-specific header locations, create a platform wrapper:

**Good example - modules/glut/glut_platform.h:**
```c
#ifndef NANOLANG_GLUT_PLATFORM_H
#define NANOLANG_GLUT_PLATFORM_H

#ifdef __APPLE__
    #include <GLUT/glut.h>    // macOS framework path
#else
    #include <GL/glut.h>      // Linux pkg-config path
#endif

#endif
```

Then reference the wrapper in module.json:
```json
{
  "headers": ["glut_platform.h"],
  "cflags": ["-Imodules/glut"]
}
```

### Module.json Structure

**Complete cross-platform module:**
```json
{
  "name": "example",
  "version": "1.0.0",
  "description": "Example library with proper cross-platform support",
  "headers": ["example.h"],
  "c_sources": ["example_helpers.c"],
  "pkg_config": ["libexample"],
  "dependencies": [],
  "install": {
    "macos": {
      "brew": "example",
      "command": "brew install example"
    },
    "linux": {
      "apt": "libexample-dev",
      "command": "sudo apt install libexample-dev",
      "yum": "example-devel",
      "pacman": "example"
    }
  },
  "notes": "Additional platform-specific notes if needed"
}
```

### System Libraries Without pkg-config

For system libraries that are always available (like libm):
```json
{
  "name": "math_ext",
  "ldflags": ["-lm"]  // OK - libm is universally available at standard paths
}
```

### When pkg-config Isn't Available

Some libraries don't provide pkg-config (legacy systems, proprietary software). In these cases:

1. **First choice:** Encourage users to use versions that do provide pkg-config
2. **Fallback:** Document manual installation with explicit path requirements
3. **Last resort:** Create a configure script that searches common paths

### Testing Cross-Platform Support

Before committing a module, verify:

1. **macOS Intel** (if available):
   ```bash
   brew install libfoo  # Uses /usr/local
   ```

2. **macOS Apple Silicon**:
   ```bash
   brew install libfoo  # Uses /opt/homebrew
   ```

3. **Linux**:
   ```bash
   sudo apt install libfoo-dev  # Or yum, pacman, etc.
   ```

4. **Verify pkg-config works**:
   ```bash
   pkg-config --cflags --libs libfoo
   ```

### Common Libraries and Their pkg-config Names

| Library | pkg-config name | Notes |
|---------|----------------|-------|
| SDL2 | sdl2 | ✓ Universal |
| SDL2_ttf | SDL2_ttf | ✓ Universal |
| SDL2_mixer | SDL2_mixer | ✓ Universal |
| GLFW | glfw3 | ✓ Universal |
| GLEW | glew | ✓ Includes OpenGL automatically |
| FreeGLUT | glut | ✓ Universal, replaces GLUT.framework |
| SQLite | sqlite3 | ✓ Universal |
| libcurl | libcurl | ✓ Universal |
| libuv | libuv | ✓ Universal |
| libevent | libevent | ✓ Universal |

### Summary

**The Golden Rule:** If a library provides pkg-config support, ALWAYS use it. If it doesn't, consider whether it's the right library to use.

This ensures nanolang modules work seamlessly across all platforms without modification.
