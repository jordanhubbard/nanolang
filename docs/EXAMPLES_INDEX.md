# My Examples Index

Last audited: 2026-05-21.

I have 211 `.nano` files under `examples/`. The audit began at 204 files; this update adds missing package, resource, diagnostics, cross-backend, and deterministic integration examples.

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
| `examples/` | 9 | tools, launchers, root showcases |
| `examples/advanced/` | 26 | modules and advanced language demos |
| `examples/audio/` | 5 | audio modules and showcases |
| `examples/cross_backend/` | 1 | cross-backend learn fixture |
| `examples/data/` | 1 | data module demos |
| `examples/debug/` | 6 | diagnostics, logging, property tests |
| `examples/diagnostics/` | 1 | expected-failure diagnostics |
| `examples/games/` | 3 | SDL game showcases |
| `examples/gpu/` | 6 | GPU modules |
| `examples/graphics/` | 21 | SDL and OpenGL-adjacent visual examples |
| `examples/hello_pkg/` | 1 | package manifest example |
| `examples/integration/` | 1 | deterministic integration fixture |
| `examples/language/` | 75 | core language learning path |
| `examples/large_project/` | 3 | real multi-file package example |
| `examples/lib/` | 5 | internal launcher/editor support |
| `examples/mujoco/` | 2 | MuJoCo modules |
| `examples/network/` | 5 | HTTP and curl modules |
| `examples/opengl/` | 6 | OpenGL modules |
| `examples/opl/` | 12 | parser/compiler showcase |
| `examples/physics/` | 7 | Bullet physics modules |
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
4. `nl_primes_trial_division.nano`
5. `nl_primes_sieve.nano`
6. `nl_pi_calculator.nano`
7. `nl_pi_chudnovsky.nano`

This turns several overlapping math examples into increasing algorithmic weight.

### REPL

The canonical teaching pair is:

1. `simple_repl.nano`
2. `full_repl.nano`

`vars_repl.nano`, `multi_type_repl.nano`, `multiline_repl.nano`, and `readline_repl.nano` are staged internal variants. Keep them if they are useful while developing the final REPL, but do not put all of them in the main learning path.

### Forth

`examples/language/nl_forth_interpreter.nano` is the canonical Forth interpreter. It has the larger test harness and the module integration.

`examples/forth.nano` is a legacy root example. Keep it only as a smaller historical sketch or remove it after checking that no docs or scripts depend on it.

## Modules Track

These examples demonstrate module surfaces. They are useful, but they are not all beginner material.

| Module Area | Examples | Build |
| --- | --- | --- |
| JSON | `examples/data/json_demo.nano`, `examples/opl/opl_json.nano` | local |
| regex | `examples/advanced/regex_demo_simple.nano`, `examples/advanced/regex_demo.nano` | missing or external module status must be checked before claiming local |
| datetime | `examples/advanced/datetime_demo.nano` | local or excluded by Makefile pending build check |
| checked math | `examples/language/nl_checked_math_demo.nano` | local |
| vector2d | `examples/advanced/module_introspection_demo.nano` | external SDL dependency because the demo also imports SDL |
| filesystem/process/env | `examples/run_examples.nano`, `examples/opl/opl_cli.nano` | local with compiler/tool assumptions |
| logging/coverage/proptest | `examples/debug/*.nano` | local |
| HTTP server | `examples/network/http_hello_world.nano`, `http_static_server.nano`, `http_rest_api.nano` | network, native-only |
| curl | `examples/network/curl_example.nano` | external-deps, network |
| GitHub/OpenAI | `examples/ai_github_agent.nano`, `examples/autonomous_github_agent.nano` | api-key, network |
| SDL | `examples/graphics/sdl_*.nano`, `examples/games/sdl_*.nano`, `examples/audio/sdl_*.nano` | graphical/audio/external-deps |
| ncurses | `examples/terminal/ncurses_*.nano` | external-deps |
| OpenGL | `examples/opengl/*.nano` | graphical/external-deps |
| Bullet | `examples/physics/*.nano` | external-deps |
| MuJoCo | `examples/mujoco/*.nano` | external-deps |
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
| `fs_api.nano` | `modules/std/fs` | local |
| `json_api.nano` | `modules/std/json` | local |
| `log_api.nano` | `modules/std/log` | local |
| `math_ext_api.nano` | `modules/math_ext` | local |
| `math_extended_api.nano` | `modules/std/math/extended` | local |
| `math_vector2d_api.nano` | `modules/std/math/vector2d` | local |
| `nano_highlight_api.nano` | `modules/nano_highlight` | local |
| `peg_api.nano` | `modules/std/peg` | local |
| `peg2_api.nano` | `modules/std/peg2` | local |
| `preferences_api.nano` | `modules/preferences` | external-deps |
| `process_api.nano` | `modules/std/process` | local |
| `proptest_api.nano` | `modules/proptest` | local |
| `pt2_module_api.nano` | `modules/pt2_module` | external-deps |
| `pt2_state_api.nano` | `modules/pt2_state` | local |
| `pybridge_api.nano` | `modules/pybridge` | external-deps (Python) |
| `pybridge_matplotlib_api.nano` | `modules/pybridge_matplotlib` | external-deps (Python + matplotlib) |
| `result_api.nano` | `modules/std/result` | local |
| `stringbuilder_api.nano` | `modules/std/collections/stringbuilder` | local |
| `vector2d_api.nano` | `modules/vector2d` | local |
| `stdio_api.nano` | `modules/std/io/stdio` | broken — upstream extern signatures (`fgetc`, `clearerr`) clash with system `<stdio.h>` |
| `websocket_api.nano` | `modules/websocket` | broken — compiler segfaults on the websocket FFI surface |
| `tidy_api.nano` | `stdlib/tidy` | broken — `stdlib/tidy.nano` defines its own `fn main()` that clashes with importers |

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
| Algorithms | Keep factorial, Fibonacci, primes, sieve, trial division, and pi examples as a progression. |
| REPL | Teach `simple_repl` then `full_repl`; classify the rest as staged internals. |
| Forth | Make `nl_forth_interpreter.nano` canonical; mark root `forth.nano` legacy. |
| `large_project_structure.nano` | Treat as a sketch. Use `examples/large_project/` as the real multi-file package. |
| row polymorphism | Mark conceptual files as `concept` until the compiler accepts them as ordinary examples. |
| network and AI | Keep live examples, but require deterministic dry-run or local fixtures before they enter Learn. |

## Missing Coverage

These gaps remain visible so future work does not rediscover them.

| Gap | Current state | Next useful example |
| --- | --- | --- |
| resource and affine types | Added `nl_affine_resource_demo.nano`. | Add negative expected-failure fixtures for leak and use-after-consume diagnostics. |
| LLM diagnostics | Added `examples/diagnostics/llm_diags_type_mismatch.nano`. | Add scripted golden JSON tests for `--llm-diags-json`. |
| package manifests | Fixed `hello_pkg` and added `examples/large_project/`. | Add registry publish/install dry-run once the registry CLI is stable. |
| cross-backend execution | Added `hello_cross_backend.nano`. | Add a small runner that executes native, VM, and WASM when those backends are present. |
| deterministic integrations | Added `file_pipeline.nano`. | Add local HTTP fixture with fixed input and no internet dependency. |
| stdlib modules | `examples/api_lessons/` now covers result, set, binary, env, fs, preferences, websocket (broken), pybridge, nano_highlight, audio_helpers, pt2_state, stdio (broken), stringbuilder, array_utils, diagnostics, dispatch, json, log, math_ext, math/extended, math/vector2d, peg, peg2, process, proptest, pt2_module, vector2d, hashmap. | Outstanding gaps: option/list/iter/map/string lessons require `stdlib/*.nano` rewrites (currently use a non-supported dialect); math/vector3d, vector4d, matrix4, quaternion, complex still need lessons. |
| side-effect shadows | Many graphical and I/O demos still use `assert true`. | Extract pure helpers and test those helpers with concrete inputs. |
| structured headers | Parser support exists now. | Normalize old headers opportunistically rather than churning every file at once. |

## Build Truth

`examples/Makefile` is still the build truth for compiled examples. It already excludes some files because of missing libraries, live services, or compiler/backend limits.

The metadata does not replace the Makefile. It explains why an example is or is not suitable for a learning path, a launcher, or a backend test.
