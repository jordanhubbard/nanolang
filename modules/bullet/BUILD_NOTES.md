# Bullet Physics Module - Build Notes

## Current Status

The Bullet Physics module follows the standard NanoLang module format:
- `module.json` with `c_sources` and `pkg_config`
- `c_compiler: "c++"` to specify C++ compilation
- No custom build scripts

## Known Limitation: C++ Headers in Generated Code

**Issue**: Bullet is a C++ library. When NanoLang code imports the bullet module, the generated C code includes Bullet's C++ headers, but nanoc compiles the main program with `cc -std=c99`, causing compilation errors.

**Current Workaround**: The module C sources (bullet_ffi.c) compile correctly with c++, but programs using the module need manual compilation.

**Manual Build Command**:
```bash
# 1. Transpile to C
./bin/nanoc examples/bullet_beads_simple.nano -S -o /tmp/dummy

# 2. Compile with C++
clang++ -O2 \
    -I./src -I./src/runtime \
    $(pkg-config --cflags bullet) \
    examples/bullet_beads_simple.nano.genC \
    modules/bullet/.build/bullet.o \
    src/runtime/*.c \
    $(pkg-config --libs bullet) \
    -o bin/bullet_test
```

## Future Solution

The compiler needs to detect when any module uses `c_compiler: "c++"` and:
1. Switch the main compilation to use `c++` instead of `cc`
2. Remove `-std=c99` flag
3. Ensure runtime C files compile correctly as C++ (they do)

This would allow seamless use of C++ libraries like Bullet, OpenCV, etc.

## Alternative Approach

For now, C++ modules can provide a helper script that wraps the manual build process, or use the standard `build_module.sh` for just the module compilation and document the manual linking step.

