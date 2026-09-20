# I obtain example seeds through my typed timing module

My random-sentence, fire, starfield and terminal-rain examples declared C time with an integer parameter although the native function takes a pointer. Argument staging exposed that mismatch. I replace those declarations and calls with my existing Timing.get_microseconds divided by1000000, preserving epoch-second seed granularity.

At28ba6c213, all four changed examples compile on Linux and Darwin, and the five unchanged strict-example regressions pass. The full `make examples EXAMPLES_TIMEOUT=2400` gate passes in194.573 seconds on Linux and146.250 seconds on Darwin using the normal CI stage1 prerequisite and default per-module compiler selection. Existing shadows remain enabled. I did not run graphical or interactive interfaces.

I retain [seven qualification histories](evidence/example-time-ffi/seal.json), their exact runners, commands, logs and source/tool/product hashes. Larger reports are losslessly compressed and carry original and stored hashes. Original reports/CAS remain at the recorded paths; the downloaded Darwin archive hash matches the remote archive.

My first external runner accidentally replaced its source inventory variable while assembling the example list. Both builds exited0, but inventory equality correctly stopped the run. I renamed only that local and preserved both failed records. The next full run exposed a missing bin/nanoc alias in my C-seed-only setup, then a global CC override that displaced Bullet's declared C++ compiler. Using the actual CI stage1 prerequisite and removing that override resolved those setup failures. No REPL, Bullet, compiler capture, or shadow assertion changed. Linux's intermediate stage1 run and every first terminal remain retained.

The final CI drivers record their initial selection in environment.json, then explicitly remove CC before running the listed commands. The retained drivers and command lists define that transition. Named executable hashes are not a complete system-library inventory. Earlier failed-run products may later be rebuilt; the retained phase objects preserve their historical bytes.

These checks qualify the example repair at its recorded source pin. Full integrated5.1 compiler, platform and release acceptance remain separate.
