# NanoLang Module System Audit Report

**Last refreshed:** 2026-05-21
**Previous audit:** 2024-11-23 (covered 19 modules; superseded by this revision)
**Scope:** All 50 modules under `modules/` validated for cross-platform auto-install metadata.

## Summary

✅ All modules wrapping external native libraries now use the canonical `system_packages` field driven by the [`packages.json`](../packages.json) registry. Auto-install fires during `module_build` when system metadata is present, so importing a module from a NanoLang program also installs the underlying native dep on macOS, Debian/Ubuntu, Fedora/RHEL, Arch, openSUSE, Alpine, FreeBSD, and Windows (chocolatey/winget/scoop).

## Module classification

50 modules, three categories:

### 1. Canonical — declares `system_packages` (23)

These wrap a native library and let `install_system_packages` resolve the platform package via `packages.json`:

`audio_viz`, `bullet`, `curl`, `dispatch`, `event`, `glew`, `glfw`, `glut`, `http_server`, `mujoco`, `ncurses`, `pt2_audio`, `readline`, `sdl`, `sdl_helpers`, `sdl_image`, `sdl_mixer`, `sdl_term`, `sdl_ttf`, `sqlite`, `ui_widgets`, `unicode`, `uv`.

### 2. Transitive — inherits via `dependencies` (2)

These declare `dependencies: ["curl", ...]` so the `curl` module's `system_packages` is processed before they build:

`github`, `openai` — both `#include <curl/curl.h>` and rely on the `curl` module's auto-install to make libcurl available.

### 3. Pure / internal — no external build-time deps (25)

Either pure NanoLang, libc-only, or runtime-only Python via socket IPC. Correctly need no `system_packages`:

`audio_helpers`, `filesystem`, `forth_see`, `gpu`, `libc`, `log`, `math_ext`, `nano_highlight`, `nano_tools`, `preferences`, `proptest`, `pt2_module`, `pt2_state`, `pty`, `pybridge`, `pybridge_matplotlib`, `pybridge_warp`, `std`, `std/collections`, `std/json`, `std/peg`, `std/peg2`, `stdio`, `vector2d`, `websocket`.

Notes:
- **`websocket`** explicitly declares `system_packages: []` — RFC 6455 implementation over plain POSIX sockets; no system dep.
- **`pybridge_*`** spawn a Python subprocess over a socket — at build time they need only `<dlfcn.h>` and `<stdio.h>`. Their runtime Python package requirements (matplotlib, warp) are handled inside the subprocess.

## How auto-install works

```
NanoLang source           module.json                packages.json                    package manager
─────────────────         ─────────────              ──────────────                   ───────────────
unsafe module "..."  ───► system_packages: ["mujoco"] ──► packages.mujoco.install.brew ──► brew install --cask mujoco
                                                                                  .apt ──► sudo apt install ...
                                                                                  .pkg ──► sudo pkg install ...
```

1. `src/module_builder.c:module_build` runs when the module needs rebuild.
2. `install_system_packages` collects logical names from `system_packages`.
3. `lookup_package_name(name, pm)` finds the platform-specific package in `packages.json`.
4. `lookup_install_command(name, pm)` and `lookup_test_command(name, pm)` (since 2026-05-21) check for custom overrides — used when the default `<pm> install <pkg>` doesn't fit (e.g., brew casks, pip user installs, manual fallbacks).
5. `install_single_package_ex` executes the test command; if it returns non-zero, runs the install command.

### Schema reference

| Field | Where | Purpose |
|---|---|---|
| `system_packages: ["name"]` | `modules/<m>/module.json` | Logical package names to resolve via registry |
| `packages.<name>.install.<pm>.package` | `packages.json` | Platform-native package name |
| `packages.<name>.install.<pm>.install_command` | `packages.json` | Optional shell command overriding the default install incantation |
| `packages.<name>.install.<pm>.test_command` | `packages.json` | Optional shell command returning 0 when the package is already present |
| `packages.<name>.detection.headers` | `packages.json` | Header files used to detect prior installs (advisory) |
| `cflags_macos` / `cflags_linux` | `modules/<m>/module.json` | Platform-specific compile flags (e.g., `-F<framework_dir>`) |
| `ldflags_macos` / `ldflags_linux` | `modules/<m>/module.json` | Platform-specific link flags |
| `dependencies: ["m"]` | `modules/<m>/module.json` | Other NanoLang modules whose `system_packages` should be installed first |

## Changes since the 2024-11 audit

| Module | Change |
|---|---|
| `mujoco` | Added 2026-05-20. Brew cask install via `install_command`; framework header path via `cflags_macos`; `mujoco.c` scans `mujoco.framework/Versions/{Current,A}` for `libmujoco.*.dylib`. |
| `dispatch` | Added `system_packages: ["libdispatch"]` (2026-05-21). `packages.json` brew entry no-ops with `test_command: "true"` since libdispatch ships with libSystem on macOS. |
| `module_builder.c` | Honors `install_command`/`test_command` per platform (2026-05-21). Previously the fields existed in `packages.json` for chocolatey but were never read — they're now active for all package managers. |
| `audio_viz`, `bullet`, `http_server`, `pt2_audio`, `sdl_*`, `sdl_term`, `ui_widgets`, `unicode` | Migrated to `system_packages` since 2024-11. |

## Verification

- `bin/mujoco_opengl_cartpole` runs from a clean `modules/mujoco/.build` checkout: triggers `brew install --cask mujoco`, picks up the framework header, dlopen's the versioned dylib, renders an OpenGL window.
- `make examples` builds the full set on macOS arm64 with `glfw`/`glew` already present; OpenGL/freeglut examples remain gated on `pkg-config --exists glut` (skipped, not broken).

## Open follow-ups

- `pybridge_matplotlib` / `pybridge_warp` have runtime pip dependencies (`matplotlib`, `warp`) that aren't surfaced through the module system. Worth adding a `pip_packages` field analogous to `system_packages` if we want a truly one-shot install.
- The `packages.json` schema isn't formally validated. Adding a draft-07 schema file would catch missing fields at PR time.
- `modules/MODULE_SCHEMA.md` correctly notes that the per-module `install` block in `module.json` is documentation-only; that field has been superseded by `system_packages` plus the packages.json registry. Modules that still carry an `install` block (mujoco, glfw, glew, …) keep it for human readers but the auto-installer ignores it.
