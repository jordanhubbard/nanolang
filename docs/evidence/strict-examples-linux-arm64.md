# My strict Linux example gate

I tested the release candidate at `4373abc5`, followed by fixture repairs
`b9c9854d` and `8b8822bd`, in a fresh isolated worktree on Linux ARM64 with
Ubuntu GCC 13.3.0. My CI Linux strict-example job uses x64; this local result
establishes ARM64 coverage.

I ran the strict CI commands with its installed Linux dependencies: SDL2 and
its image, mixer and TTF libraries, Bullet, GLUT, libuv, readline, ncurses,
OpenSSL and libffi. I kept the example selection and exclusions unchanged.

- `make -j8 stage1` passed.
- `make test-examples-regressions` passed all five compilations: the original
  graphics and audio regressions, NanoAmp from `examples/`, and OPL CLI from
  both the repository root and `examples/`.
- `make examples EXAMPLES_TIMEOUT=2400` passed. All 185 selected example
  artifacts exist. I also rebuilt the OPL AST-builder artifact after updating
  its shared helper, then reran the strict target successfully.

The strict build found two fixture defects. OPL compiler/codegen shadows and
NanoAmp's audio-list shadow assumed the repository root was the working
directory. My strict example Makefile runs from `examples/`. I now resolve
those fixtures from either directory and retain their assertions. My OPL
codegen shadow additionally checks the fixture's expected plan version, so an
empty fixture cannot pass merely because the generated template is long.

These results cover compilation, linking and compiler-selected shadows. They
do not establish interactive graphics, sound playback or simulation behavior.
