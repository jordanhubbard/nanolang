# NanoLang API Reference

*Auto-generated module documentation from reflection*

This directory contains complete API references for all NanoLang modules, automatically generated from the module source code using the compiler's reflection capabilities.

## Standard Library

- [log](log.md) - Structured logging with log levels (TRACE, DEBUG, INFO, WARN, ERROR, FATAL)
- [coverage](coverage.md) - Code coverage tracking and reporting
- [StringBuilder](StringBuilder.md) - Efficient string building and manipulation
- ~~regex~~ - Regular expression support *(documentation pending)*

## SDL Family (Graphics & Audio)

- [sdl](sdl.md) - SDL2 bindings for window management, rendering, and input
- [sdl_image](sdl_image.md) - SDL_image for loading various image formats
- [sdl_mixer](sdl_mixer.md) - SDL_mixer for audio playback
- [sdl_ttf](sdl_ttf.md) - SDL_ttf for TrueType font rendering

## Terminal

- [ncurses](ncurses.md) - Terminal UI library for creating text-based interfaces

## Network

- [curl](curl.md) - HTTP client using libcurl
- [http_server](http_server.html) - Simple HTTP server
- [uv](uv.md) - libuv bindings for async I/O and event loop

## Data

- [sqlite](sqlite.md) - SQLite database bindings

## Graphics

- [opengl](opengl.md) - OpenGL bindings for 3D graphics
- [glew](glew.md) - GLEW for OpenGL extension loading
- [glfw](glfw.md) - GLFW for window and input management
- [glut](glut.md) - GLUT for simple OpenGL applications

## Physics

- [bullet](bullet.md) - Bullet Physics engine bindings

## Utilities

- [filesystem](filesystem.md) - File system operations
- [preferences](preferences.md) - User preferences management
- [event](event.md) - Event handling system
- [vector2d](vector2d.md) - 2D vector math
- [proptest](proptest.md) - Property-based testing framework

---

## About This Documentation

These API references are generated automatically using the NanoLang compiler's `--reflect` flag and a **NanoLang-based generator**, which extracts complete module metadata including:

- Function signatures with parameter types
- Struct definitions with field information
- Enum variants
- Union types with variants
- Opaque types (C pointer types)
- Module constants

To regenerate this documentation:

```bash
perl -e 'alarm 60; exec @ARGV' ./bin/nanoc scripts/generate_module_api_docs.nano -o build/userguide/generate_module_api_docs
bash scripts/generate_all_api_docs.sh
```

Last updated: 2026-01-18
