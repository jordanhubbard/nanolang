# My Examples Index

Last audited: 2026-08-23.

I have 241 `.nano` files under `examples/`: 8 at the directory root and 233 below category directories. This count includes the generated OPL output fixture. I count files on disk; deleted sketches do not remain examples by reputation alone.

This file is the source of truth for my public examples. `examples/README.md` and `docs/LEARNING_PATH.md` point here instead of keeping their own stale catalogs.

## Tracks

I sort examples by what they are for, not only by directory.

| Track | Meaning | Rule |
| --- | --- | --- |
| Learn | Small deterministic programs that teach my syntax. | No API keys, no network, no GUI loop, no hidden service dependency. |
| Modules | Focused examples for one module or integration surface. | State external libraries, build mode, and runtime assumptions. |
| Showcases | Larger programs that prove I can build complete tools, games, or systems. | Keep pure helpers tested even when rendering or I/O is hard to assert. |
| Internal | Shared helper code, generated output, runners, fixtures, and support modules. | Do not present these as teaching examples. |
| Experimental | Concept sketches, expected-failure diagnostics, or features not ready for a beginner path. | Label the boundary plainly. |

## Metadata

Every new example should use this header shape.

```nano
# Example: Name
# Purpose: One sentence.
# Features: arrays, shadow tests
# Difficulty: Beginner
# Category: language
# Prerequisites: none
# Track: learn
# Build: local
# Dependencies: none
# Tags: shadow-tested, native-compatible
# Expected Output: Hello from NanoLang!
```

`Track`, `Build`, `Dependencies`, and `Tags` are now parsed by `modules/examples/meta.nano` and `examples/lib/example_discovery.nano`. Old examples may still lack those fields. They should be normalized when touched.

Use these build labels:

| Label | Meaning |
| --- | --- |
| `local` | I compile and run without external services. |
| `external-deps` | I need a C library, native package, or module build artifact. |
| `network` | I open sockets or require a local server. |
| `api-key` | I call a remote authenticated service. |
| `graphical` | I open a window. |
| `audio` | I need audio device support. |
| `gpu` | I need CUDA, OpenCL, or graphics drivers. |
| `generated` | I am generated output or a compiler artifact. |
| `concept` | I explain a design that is not fully compiler-backed. |
| `expected-failure` | I am supposed to fail so diagnostics can be inspected. |

## Directory Counts

| Directory | `.nano` files | Track |
| --- | ---: | --- |
| `examples/` | 8 | tools, launchers, root showcases |
| `examples/advanced/` | 25 | modules and advanced language demos |
| `examples/api_lessons/` | 41 | focused API lessons |
| `examples/audio/` | 6 | audio modules and showcases |
| `examples/cross_backend/` | 1 | cross-backend learn fixture |
| `examples/data/` | 1 | data module demos |
| `examples/debug/` | 4 | diagnostics, logging, property tests |
| `examples/diagnostics/` | 1 | expected-failure diagnostics |
| `examples/games/` | 3 | SDL game showcases |
| `examples/gpu/` | 6 | GPU modules |
| `examples/graphics/` | 19 | SDL and OpenGL-adjacent visual examples |
| `examples/hello_pkg/` | 1 | package manifest example |
| `examples/integration/` | 1 | deterministic integration fixture |
| `examples/language/` | 69 | core language learning path |
| `examples/large_project/` | 3 | real multi-file package example |
| `examples/lib/` | 5 | internal launcher/editor support |
| `examples/mujoco/` | 6 | MuJoCo modules |
| `examples/network/` | 2 | HTTP server modules |
| `examples/opengl/` | 6 | OpenGL modules |
| `examples/opl/` | 12 | parser/compiler showcase |
| `examples/opl/examples/output/` | 1 | generated OPL output fixture |
| `examples/physics/` | 6 | Bullet physics modules |
| `examples/playground/` | 1 | playground server |
| `examples/properties/` | 1 | formal property illustration |
| `examples/terminal/` | 3 | ncurses modules |
| `examples/verified/` | 9 | verified-subset examples |

## Learn Track

Start here. These examples are deterministic and useful for learning the language itself.

1. `examples/language/nl_hello.nano`
2. `examples/language/nl_operators.nano`
3. `examples/language/nl_comparisons.nano`
4. `examples/language/nl_logical.nano`
5. `examples/language/nl_types.nano`
6. `examples/language/nl_mutable.nano`
7. `examples/language/nl_functions_basic.nano`
8. `examples/language/nl_function_return_values.nano`
9. `examples/language/nl_factorial.nano`
10. `examples/language/nl_fibonacci.nano`
11. `examples/language/nl_control_if_while.nano`
12. `examples/language/nl_control_for.nano`
13. `examples/language/nl_array_complete.nano`
14. `examples/language/nl_array_bounds.nano`
15. `examples/language/nl_for_in_array.nano`
16. `examples/language/nl_array_functional.nano`
17. `examples/language/nl_struct.nano`
18. `examples/language/nl_enum.nano`
19. `examples/language/nl_union_types.nano`
20. `examples/language/nl_types_union_construct.nano`
21. `examples/language/nl_generics_demo.nano`
22. `examples/language/nl_hashmap.nano`
23. `examples/language/nl_types_tuple.nano`
24. `examples/language/nl_result_propagation.nano`
25. `examples/language/nl_affine_resource_demo.nano`
26. `examples/advanced/testing_strategies.nano`
27. `examples/cross_backend/hello_cross_backend.nano`
28. `examples/integration/file_pipeline.nano`

## Progressions

### Arrays

Use this order:

1. `nl_array_complete.nano` teaches literals, indexing, length, and simple operations.
2. `nl_array_bounds.nano` focuses on checked access.
3. `nl_array_infer.nano` shows inference around array literals.
4. `nl_for_in_array.nano` shows iteration.
5. `nl_array_functional.nano` shows map/filter/fold-style helpers.

The old catalog treated array examples as peers. They are not peers. They are a progression.

### Functions

Use this order:

1. `nl_functions_basic.nano`
2. `nl_function_return_values.nano`
3. `nl_function_variables.nano`
4. `nl_first_class_functions.nano`
5. `nl_function_factories_v2.nano`
6. `nl_filter_map_fold.nano`

The beginner path uses the first two. The rest belong to advanced function values.

### Algorithms

Use this order:

1. `nl_factorial.nano`
2. `nl_fibonacci.nano`
3. `nl_primes.nano`
4. `nl_primes_sieve.nano`
5. `nl_pi_calculator.nano`
6. `nl_pi_chudnovsky.nano`

This turns several overlapping math examples into increasing algorithmic weight.

### REPL

`examples/language/full_repl.nano` is the canonical REPL example. The staged compile-and-execute variants were removed after their useful behavior was consolidated into it.

### Forth

`examples/language/nl_forth_interpreter.nano` is the canonical Forth interpreter. It has the larger test harness and the module integration. The legacy root sketch was removed.

## Modules Track

These examples demonstrate module surfaces. They are useful, but they are not all beginner material.

| Module Area | Examples | Build |
| --- | --- | --- |
| JSON | `examples/data/json_demo.nano`, `examples/opl/opl_json.nano` | local |
| regex | `examples/advanced/regex_demo_simple.nano`, `examples/advanced/regex_demo.nano` | missing or external module status must be checked before claiming local |
| datetime | `examples/advanced/datetime_demo.nano` | local; included in native and NanoVM example selection |
| checked math | `examples/language/nl_checked_math_demo.nano` | local |
| vector2d | `examples/advanced/module_introspection_demo.nano` | external SDL dependency because the demo also imports SDL |
| filesystem/process/env | `examples/run_examples.nano`, `examples/opl/opl_cli.nano` | local with compiler/tool assumptions |
| logging/coverage/proptest | `examples/debug/logging_demo.nano`, `coverage_demo.nano`, `property_test_*.nano` | local |
| HTTP server | `examples/network/http_hello_world.nano`, `examples/network/http_static_server.nano` | external-deps, network, native-only; static serving only |
| curl | No current example. The module remains available, but I do not claim deleted client demos as coverage. | external-deps, network |
| issue triage | `examples/ai_github_agent.nano` | local, deterministic; prompt construction and classification only, with no GitHub or LLM call |
| SDL | `examples/graphics/sdl_*.nano`, `examples/games/sdl_*.nano`, `examples/audio/sdl_*.nano` | graphical/audio/external-deps |
| ncurses | `examples/terminal/ncurses_*.nano` | external-deps |
| OpenGL | `examples/opengl/*.nano` | graphical/external-deps |
| Bullet | `examples/physics/*.nano` | external-deps |
| MuJoCo | `examples/mujoco/mujoco_state_audit.nano`, `examples/mujoco/mujoco_cartpole_balance.nano`, `examples/mujoco/mujoco_opengl_cartpole.nano`, `examples/mujoco/mujoco_opengl_drop_lab.nano`, `examples/mujoco/mujoco_opengl_geom_browser.nano` | external-deps, graphical for `mujoco_opengl_*` |
| GPU | `examples/gpu/*.nano` | gpu/external-deps |
| OPL | `examples/opl/*.nano` | local showcase with generated output |
| packages | `examples/hello_pkg/`, `examples/large_project/` | local package layout |
| API lessons | `examples/api_lessons/*_api.nano` | one focused lesson per module surface (see table below) |

### API lessons (`examples/api_lessons/`)

| File | Module | Build |
| --- | --- | --- |
| `array_utils_api.nano` | `modules/std/collections/array_utils` | local |
| `audio_helpers_api.nano` | `modules/audio_helpers` | local |
| `binary_api.nano` | `modules/std/binary`, `modules/std/result` | local |
| `collections_hashmap_api.nano` | builtin `HashMap<K,V>` | local |
| `collections_set_api.nano` | `modules/std/collections/set` | local |
| `diagnostics_api.nano` | `modules/std/diagnostics` | local |
| `dispatch_api.nano` | `modules/dispatch` (libdispatch / GCD) | local on macOS |
| `env_api.nano` | `modules/std/env` | local |
| `examples_diag_api.nano` | `modules/examples/diag` | local |
| `fs_api.nano` | `modules/std/fs` | local |
| `glut_helpers_api.nano` | `modules/glut/glut_helpers` | external-deps, graphical |
| `json_api.nano` | `modules/std/json` | local |
| `log_api.nano` | `modules/std/log` | local |
| `math_array_ops_api.nano` | `modules/std/math/array_ops` | local |
| `math_complex_api.nano` | `modules/std/math/complex` | local |
| `math_ext_api.nano` | `modules/math_ext` | local |
| `math_extended_api.nano` | `modules/std/math/extended` | local |
| `math_matrix4_api.nano` | `modules/std/math/matrix4` | local |
| `math_quaternion_api.nano` | `modules/std/math/quaternion` | local |
| `math_vector2d_api.nano` | `modules/std/math/vector2d` | local |
| `math_vector3d_api.nano` | `modules/std/math/vector3d` | local |
| `math_vector4d_api.nano` | `modules/std/math/vector4d` | local |
| `nano_highlight_api.nano` | `modules/nano_highlight` | local |
| `peg_api.nano` | `modules/std/peg` | local |
| `peg2_api.nano` | `modules/std/peg2` | local |
| `preferences_api.nano` | `modules/preferences` | external-deps |
| `process_api.nano` | `modules/std/process` | local |
| `proptest_api.nano` | `modules/proptest` | local |
| `pt2_audio_api.nano` | `modules/pt2_audio` | external-deps, audio |
| `pt2_module_api.nano` | `modules/pt2_module` | external-deps |
| `pt2_state_api.nano` | `modules/pt2_state` | local |
| `pybridge_api.nano` | `modules/pybridge` | external-deps (Python) |
| `pybridge_matplotlib_api.nano` | `modules/pybridge_matplotlib` | external-deps (Python + matplotlib) |
| `result_api.nano` | `modules/std/result` | local |
| `sdl_image_safe_api.nano` | `modules/sdl_image/sdl_image_safe` | external-deps, graphical |
| `std_lib_api.nano` | `modules/std/lib` | local |
| `stringbuilder_api.nano` | `modules/std/collections/stringbuilder` | local |
| `vector2d_api.nano` | `modules/vector2d` | local |
| `stdio_api.nano` | `modules/std/io/stdio` | local |
| `websocket_api.nano` | `modules/websocket` | local |
| `tidy_api.nano` | `stdlib/tidy` | local |

## Showcases

These have real value because they show that I can hold a larger design together.

| Showcase | Why it matters |
| --- | --- |
| `examples/opl/` | Lexer, parser, validator, compiler, JSON IR, and driver split across files. |
| `examples/language/nl_forth_interpreter.nano` | A substantial interpreter with a real test vocabulary. |
| `examples/games/sdl_checkers.nano` | Board state, move rules, UI, and simple AI. |
| `examples/games/sdl_asteroids.nano` | Real-time loop, collision logic, wrapping, scoring, and rendering. |
| `examples/audio/sdl_nanoamp.nano` | Audio, UI, visualization, and preferences. |
| `examples/graphics/sdl_forth_ide.nano` | Terminal-like UI, process interaction, and editor behavior. |
| `examples/playground/playground_server.nano` | Browser-facing tooling around compilation. |
| `examples/large_project/` | A small package with a manifest and multiple imported source files. |

## Redundancy Decisions

These are the overlaps found in the audit and how I now classify them.

| Area | Decision |
| --- | --- |
| Arrays | Keep all five, but teach them as a sequence. |
| Functions | Keep the basic path and move function values/factories/fold examples to advanced. |
| Algorithms | Keep factorial, Fibonacci, primes, sieve, and pi examples as a progression. |
| REPL | Keep `full_repl.nano` as the canonical example; the staged variants are removed. |
| Forth | Keep `nl_forth_interpreter.nano` as the canonical implementation; the root sketch is removed. |
| project structure | Use `examples/large_project/` as the real multi-file package; the competing single-file sketch is removed. |
| HashMap API | Keep `collections_hashmap_api.nano`, which teaches the builtin `HashMap<K,V>` surface. |
| logging | Keep one `logging_demo.nano` that covers the retained logging surface. |
| SDL_image | Keep the demo, sprite animation, and tiled-background progression; the test and effects duplicates are removed. |
| row polymorphism | Mark conceptual files as `concept` until the compiler accepts them as ordinary examples. |
| network and AI | Keep the two local HTTP server examples in Modules. Use the rewritten offline issue-triage example for deterministic prompt construction; there is no current live curl, REST API, GitHub, or OpenAI example. |

## Missing Coverage

These gaps remain visible so future work does not rediscover them.

| Gap | Current state | Next useful example |
| --- | --- | --- |
| resource and affine types | Added `nl_affine_resource_demo.nano`. | Add negative expected-failure fixtures for leak and use-after-consume diagnostics. |
| LLM diagnostics | Added `examples/diagnostics/llm_diags_type_mismatch.nano`. | Add scripted golden JSON tests for `--llm-diags-json`. |
| package manifests | Fixed `hello_pkg` and added `examples/large_project/`. | Add registry publish/install dry-run once the registry CLI is stable. |
| cross-backend execution | Added `hello_cross_backend.nano`. | Add a small runner that executes native, VM, and WASM when those backends are present. |
| deterministic integrations | Added `file_pipeline.nano`. | Add local HTTP fixture with fixed input and no internet dependency. |
| stdlib modules | `examples/api_lessons/` covers supported module paths including `modules/std/result.nano`, `modules/std/collections/set.nano`, binary, env, fs, stdio, stringbuilder, array utilities, JSON, logging, math, PEG, and process surfaces. | The deleted root `stdlib/` option, list, iter, map, set, string, result, and async files are not supported modules. Add examples only after a supported module exists under `modules/std/` or the capability is built in. |
| internal module tools | `modules/tools/dep_locator.nano` is an internal CLI/tool implementation sketch. The supported dependency locator is `modules/tools/dep_locator.sh`, documented in `modules/tools/README.md` and used by the module build flow. | Do not add it to the sample browser as an API lesson unless it becomes an importable module API. |
| side-effect shadows | Many graphical and I/O demos still use `assert true`. | Extract pure helpers and test those helpers with concrete inputs. |
| structured headers | Parser support exists now. | Normalize old headers opportunistically rather than churning every file at once. |

## Build Truth

`examples/Makefile` is still the build truth for compiled examples. It already excludes some files because of missing libraries, live services, or compiler/backend limits.

The metadata does not replace the Makefile. It explains why an example is or is not suitable for a learning path, a launcher, or a backend test.
