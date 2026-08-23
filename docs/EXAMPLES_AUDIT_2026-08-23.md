# My Examples Audit, 2026-08-23

I keep examples for teaching, API coverage, regression pressure, and finished
programs. Those purposes are different. This audit separates them and records
what I know without pretending that a successful compile is a visual review.

## Method and Limits

I inventoried the current `examples/` tree, grouped all 241 `.nano` files by
family, read the catalog and build rules, and inspected the completed example
changes in the current git diff. I judged each family on four questions:

The total includes `examples/opl/examples/output/hello.nano`, a generated
fixture. That leaves 240 authored sources and 241 files on disk. This audit and
the public catalogs use the on-disk count.

1. Does it teach one clear thing?
2. Does its code and metadata state its real dependency and trust boundary?
3. Does its shadow test exercise behavior rather than merely say `assert true`?
4. Does it earn a distinct place beside nearby examples?

My evidence labels are strict:

- **Observed** means I inspected source, metadata, assets, build selection, or
  the current diff.
- **Tested** means a compiler, test runner, or runtime check was actually run.
- **Assumed** means the source suggests a result that was not exercised here.

For this documentation pass, I tested only document/worktree validation. I did
not rerun the example suite, open GUI windows, listen to audio, contact network
services, use a CUDA device, or inspect generated images frame by frame. The
earlier code audit repaired examples; this file records those findings. It does
not turn source inspection into runtime evidence.

## Family Decisions

| Family | Classification | Decision | Evidence and boundary |
| --- | --- | --- | --- |
| Root `.nano` files | applications, experiments, tools | **Keep** the eight current files. `nl_forth_interpreter.nano` replaces the removed root Forth sketch. | Observed in the tree and diff. `agentos_ipc_schema.nano` and `ai_github_agent.nano` are deterministic local examples. `fluid_sim.nano` still has external Python/Warp boundaries. |
| `advanced/` (25) | advanced language and module demonstrations | **Keep**, but split conceptually into language patterns, FFI/module demos, and host-tool demos. Keep `large_project/` as the package example and remove the competing structure sketch. Admit repaired `datetime_demo.nano` to native and NanoVM build selection. | Observed in source, diff, and Makefile selection. Regex examples remain excluded because their module status is unresolved. |
| `api_lessons/` (41) | focused module API lessons | **Keep as the canonical module-surface family.** One file should answer one API question. Keep `collections_hashmap_api.nano` for the builtin surface and remove the overlapping std collections lesson. The PEG and math pairs remain distinct module surfaces. | Observed. External-dependency labels are build truth, not optional decoration. |
| `audio/` (6 plus MOD assets) | SDL audio lessons and showcases | **Keep** the player, visualizer, tracker shell, NanoAmp, WAV lesson, and playlist harness as distinct layers. **Repair** lifecycle/default-asset consistency. Keep `dangerzone_-_windowsrpg.mod` and `gabba-studies-12.mod` only as intentional redistributable fixtures. | Observed. WAV cleanup and an asset path were repaired. Audio output was not heard. |
| `cross_backend/` (1) | backend smoke fixture | **Keep** `hello_cross_backend.nano` small and deterministic. **Replace** manual claims with a runner that compares available native, NanoISA, and Wasm execution. | Observed; backend parity was not tested here. |
| `data/` (1 `.nano`, one C fixture) | JSON module lesson and support fixture | **Keep** `json_demo.nano`; keep `json_errors_demo.c` internal. Prefer `api_lessons/json_api.nano` for a narrow API lesson and this file for an end-to-end JSON narrative. | Observed. JSON shadows were strengthened in the diff. |
| `debug/` (4) | logging, coverage, and property-testing demonstrations | **Keep** one logging demo, one coverage demo, and two property lessons. The removed category and level demos duplicated the retained logging surface. | Observed. Runtime diagnostics were not re-run here. |
| `diagnostics/` (1) | expected-failure fixture | **Keep** `llm_diags_type_mismatch.nano` isolated and labeled expected failure. Never include it in an undifferentiated success build. | Observed from catalog/build classification. |
| `games/` (3) | SDL showcases | **Keep** Asteroids, Checkers, and Pong: they exercise different state models. **Repair** interaction, layering, and pure game-rule shadows before adding games. | Observed. Pong quit handling and control draw order were repaired. No game window was observed. |
| `gpu/` (6) | GPU kernels and driver examples | **Keep** the operation progression: add, scale, SAXPY, reduction, matrix work, then ocean. **Repair** `vector_add.nano` before calling its partial dot-product a complete reduction. **Replace** stale README syntax and architecture claims with commands verified against current compiler behavior. | Observed. Allocation cleanup and real load/store/result checks were added. No CUDA/OpenCL execution or ocean image was observed. |
| `graphics/` (19) | SDL rendering lessons and visual showcases | **Keep**, divided into primitives/input, image/texture, simulation, and larger applications. The SDL_image test and effects duplicates were removed; the demo, sprite animation, and tiled background remain the progression. | Observed. Game-of-Life shadows, NanoViz claims/cleanup, and widget layering were repaired. No graphical output was observed. |
| `hello_pkg/` (1) | minimal package fixture | **Keep** as the smallest manifest example. Do not grow it into a second `large_project/`. | Observed. Package commands were not tested here. |
| `icons/` (0 `.nano`) | launcher assets | **Keep only icons for discoverable, retained examples. Remove** orphan icons such as `icons/bullet_soft_body_beads.png` when their source is removed, unless the launcher has an explicit historical use. | Observed inventory; image content was not visually audited. |
| `integration/` (1) | deterministic integration fixture | **Keep** `file_pipeline.nano`. Add integrations only when they cross a real boundary with fixed local inputs. | Observed; execution was not rerun here. |
| `language/` (69 plus Forth fixtures/C helper) | primary learning path and language stress examples | **Keep the family, not a flat list.** Preserve ordered progressions for arrays, functions, control flow, records/unions, and algorithms. Keep `full_repl.nano` and `nl_forth_interpreter.nano` as the canonical implementations; the staged REPLs and trial-division duplicate are removed. | Observed. Missing behavioral shadows were added to functions, tuples, match, for-in, and prime examples. |
| `large_project/` (3) | executable multi-file package | **Keep as canonical** for package structure. The count includes `main.nano` and two source modules. | Observed. README/compiler path and manifest authorship were repaired. |
| `lib/` (5) | internal launcher/editor support | **Keep internal.** Do not count these as public teaching examples or launch them independently. Extract reusable modules out of `examples/` only when another product consumes them. | Observed from role and build exclusions. |
| `mujoco/` (6 plus XML assets) | simulation/control and OpenGL showcases | **Keep** headless state/control examples separately from the four OpenGL scenes. **Repair** metadata and tests around deterministic state before adding scenes. | Observed inventory. MuJoCo simulation and rendered output were not observed. |
| `network/` (2 plus static fixture) | local HTTP server lessons | **Keep** `http_hello_world.nano` as the minimal server and `http_static_server.nano` only if it clearly advances static-root configuration. **Remove** the false REST API and obsolete client examples until real supported APIs replace them. | Observed. The retained servers now point at `network/hello_public/` and state libuv/interactive requirements. No server was started here. |
| `obj/` (0 source files) | build artifact directory | **Remove from the authored example taxonomy.** Keep ignored only if a build still writes there. | Observed as empty in the current inventory. |
| `opengl/` (6) | OpenGL progression and showcases | **Keep** triangle, cube, teapot, solar system, particles, and postprocess as an increasing visual sequence. **Repair** resource cleanup and descriptions to match actual render passes. | Observed. Postprocess no longer claims an FBO pass it does not perform. No OpenGL frame was observed. |
| `opl/` (12 `.nano`, DSL examples, bundle, generated fixtures) | compiler/toolchain showcase | **Keep as a showcase.** Keep library stages internal, drivers as focused entry points, and `opl_cli.nano` canonical. **Keep** `bundle/` and `opl/examples/` as fixtures, not separate NanoLang families. **Remove generated output** when reproducible unless it is a checked golden fixture. | Observed. CLI paths, docs, metadata, and a process shadow were repaired. The full golden workflow was not rerun here. |
| `physics/` (6) | Bullet rigid/soft-body progression | **Keep** one simple bead lesson, one visual bead lesson, bouncy balls, megastacks, hourglass, and one modern soft-body showcase. **Remove** `bullet_soft_body_beads.nano`: it overlapped the retained bead and modern soft-body examples and carried weak tests. | Observed. Modern soft-body quit handling and projection shadows were repaired. No simulation was observed. |
| `playground/` (1 `.nano` plus web assets) | browser tooling showcase | **Keep** as a product-like showcase, not a beginner example. **Repair** documentation whenever execution, CDN, Wasm, or server fallback changes. | Observed. README claims and server shadows were repaired. Browser/Wasm behavior was not observed. |
| `properties/` (1) | passing property-test lesson | **Keep** `sort_idempotent.nano`, but rename it if it continues to teach several algebraic laws rather than sorting. Put deliberately false properties in test fixtures, not this green example. | Observed. The broken property was removed and concrete shadows were added. |
| `terminal/` (3) | ncurses applications | **Keep** Snake, Life, and Matrix Rain because their interaction and display models differ. **Repair** key contracts and label output as terminal animation, not graphical output. | Observed. Matrix Rain now quits only on ESC/Q; no terminal animation was observed. |
| `verified/` (9) | trust-boundary demonstrations | **Keep**, but do not call an entire file formally verified without a current `--trust-report`. **Repair** `verified/README.md`, which still makes blanket claims contradicted by the more careful file headers. Keep `proof_trace.nano` as an illustration of the model, not an extracted Coq trace. | Observed. Three file headers and a missing shadow were corrected. Trust reports were not run here. |

## Concrete Overlap Policy

- **Keep as progressions:** arrays, function values, prime algorithms, GPU
  kernels, OpenGL scenes, and the retained SDL image stages. Similar subject matter is not
  duplication when each file has a named next lesson.
- **Removed overlaps:** staged REPLs, the root Forth sketch, the package-layout
  sketch, the overlapping HashMap API lesson, separate logging category/level
  demos, SDL_image test/effects demos, and trial-division primes. The retained
  examples now own those learning paths.
- **Remove:** unsupported service theater, examples that advertise behavior
  they do not implement, and a visual duplicate with weaker tests. The current
  diff applies this rule to the autonomous agent, obsolete HTTP/curl demos,
  false REST server, and duplicate soft-body beads.
- **Replace:** removed network clients with one local deterministic HTTP client
  fixture when the supported client API exists; replace live AI automation with
  offline prompt/classification lessons plus a separately guarded integration
  test; replace conceptual row-polymorphism syntax with compiler-backed syntax
  or an explicitly non-runnable design document.

## Completed Work in the Current Diff

The following is **observed in git diff**, not a claim that every changed
program was executed:

- Removed `examples/autonomous_github_agent.nano` and its README/environment
  template; replaced the surviving `examples/ai_github_agent.nano` with local,
  deterministic issue triage.
- Replaced conceptual, non-runnable row-record sketches in
  `examples/row_poly_records.nano` and `examples/agentos_ipc_schema.nano` with
  explicit compiler-backed records and behavioral shadows.
- Removed `examples/network/curl_example.nano`, `http_demo.nano`, and
  `http_rest_api.nano`; repaired static paths, metadata, voice, and boundaries
  in the two retained HTTP servers.
- Removed `examples/physics/bullet_soft_body_beads.nano`; repaired input and
  projection checks in `bullet_soft_body_modern.nano`.
- Strengthened shadows in language, API, JSON, OPL, playground, property, and
  graphical examples. This includes concrete cases in
  `nl_functions_basic.nano`, `nl_for_in_array.nano`, `nl_control_match.nano`,
  `nl_match_int.nano`, both prime variants, `nl_types_tuple.nano`, PEG/JSON API
  lessons, and `sdl_game_of_life.nano`.
- Repaired interactive behavior and draw ordering in `games/sdl_pong.nano`,
  `graphics/sdl_ui_widgets_extended.nano`, `terminal/ncurses_matrix_rain.nano`,
  and `physics/bullet_soft_body_modern.nano`.
- Repaired GPU allocation cleanup and made `gpu/vector_add.nano` perform and
  check actual transfers/load/store work while leaving its incomplete reduction
  claim explicit.
- Corrected overstated visual or verification descriptions in
  `opengl/opengl_modern_postprocess.nano`, `graphics/sdl_nanoviz.nano`, and the
  touched `verified/*.nano` files.
- Corrected paths, metadata, and dependency statements in the example
  Makefile, launchers, OPL, playground, package, audio, fluid, terminal, and
  runner files.
- Removed the staged REPLs, root Forth sketch, package-layout sketch,
  overlapping HashMap lesson, redundant logging demos, redundant SDL_image
  demos, and trial-division primes example. Repaired `datetime_demo.nano` is no
  longer excluded from native or NanoVM example selection.

These edits reduce false claims and dead overlap. They do not prove that the
remaining 241 NanoLang sources all compile, run, render correctly, or belong in
the default build.

## Ranked Remaining Backlog

1. **Make the verified boundary truthful.** Run current trust reports for every
   `examples/verified/*.nano`, then rewrite `examples/verified/README.md` from
   those results. Do not infer whole-file proof coverage from the directory
   name.
2. **Settle excluded advanced examples.** Repair, relabel, or remove
   `regex_demo.nano`,
   `regex_demo_simple.nano`, `event_example.nano`, `uv_example.nano`,
   `libc_demo.nano`, `sqlite_simple.nano`, `unicode_demo.nano`,
   `module_introspection_demo.nano`, and `ui_code_display_demo.nano`.
3. **Validate visual families without claiming blanket observation.** Capture a
   short, named smoke checklist for `examples/games/*.nano`,
   `examples/graphics/*.nano`, `examples/opengl/*.nano`,
   `examples/physics/*.nano`, and `examples/mujoco/mujoco_opengl_*.nano`.
   Record which windows were actually inspected and on what host.
4. **Validate device and media behavior.** Exercise
   `examples/gpu/vector_add.nano`, `reduce_sum.nano`, `saxpy.nano`,
   `vector_scale.nano`, `matmul.nano`, and `ocean.nano` on a supported GPU;
   separately exercise `examples/audio/*.nano` with both MOD fixtures. Keep
   compile, device execution, audible output, and visual output as separate
   results.
5. **Repair stale family documentation.** Bring `examples/gpu/README.md` into
   my first-person voice and current import/compiler syntax; verify the nine
   built-ins claimed by `examples/playground/README.md`; check generated versus
   golden status under `examples/opl/bundle/` and `examples/opl/examples/`.
6. **Clean non-source residue.** Audit every `examples/icons/*.png` against
   launcher discovery, and remove `examples/obj/` if no build writes to it.
7. **Add the missing deterministic integration fixtures.** Add a local HTTP
    client/server test beside `examples/network/hello_public/index.html`, a
    scripted cross-backend runner for
    `examples/cross_backend/hello_cross_backend.nano`, and golden JSON checks for
    `examples/diagnostics/llm_diags_type_mismatch.nano`.

## Audit State

- **Observed:** all current families are classified above; the listed removals,
  repairs, and replacements are present in the worktree diff.
- **Tested:** reference/path validation and Makefile dry runs are recorded by
  this documentation pass.
- **Assumed pending tests:** compile success, runtime correctness, dependency
  availability, graphical composition, audio quality, GPU numerical results,
  network behavior, Wasm behavior, and whole-file formal trust status.

I will not call an example correct because it has a window, verified because it
lives in a directory, or distinct because it has a different filename. Each
claim needs its own evidence.
