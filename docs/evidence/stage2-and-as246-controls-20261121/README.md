# My follow-up CPU and same-host assembler controls

I retain two completed diagnostic observations under #1161 and #1157.
Neither replaces complete compiler generation or general tool admission.

My actual packed-leaf VM candidate passes the independently checked CPU
sampler's raw-equal hello control, records a complete 30-second typecheck
window, and stops automatically after that window. All 2998 main-thread
samples resolve. Of these, 2823 (94.2%) still lie in `for_each_child`,
`scan_black_child` or `mark_gray_child`; 1623 resolve to the field-name loop.
The packed-leaf exclusion therefore has not removed this sampled hotspot.
The full uninstrumented generation remains a separate run. I next test a
private child-walker inlining candidate without changing graph edges,
reference-count operations, the collection trigger or original assertions.

For the assembler comparison, I download the published upstream
[2.46.0 source archive](https://sourceware.org/pub/binutils/releases/binutils-2.46.0.tar.xz),
retain its actual bytes/hash and build a private native AArch64 assembler.
My initial guessed `binutils-2.46.tar.xz` path returns HTTP404 before any
build; I correct it to the published archive name. I retain configure/build
logs, arguments and actual resulting images. Upstream's missing Makeinfo
warning applies to its unbuilt Info documentation. No private tool replaces
a selected product compiler or assembler.

I reuse the saved hosted preprocessing witness, changing only its assembler
include pathname to the identical local macro file. Both assemblers receive
identical bytes on this host through direct literal argv with `--gdwarf-5`.
Installed 2.42 preserves the original `payload.S`; the fresh upstream 2.46.0
instead records the physical replay `.s` file. Removing either the directory
marker or standard-predefinition marker, or using a single source marker,
restores identity under both tools. This controls the host, driver and
preprocessor differences from my earlier hosted/local comparison.

My source comparison identifies an added include-state restoration adjustment
in `gas/input-scrub.c::input_scrub_pop`. A private one-condition candidate
restricts its logical-line adjustment to expansions, excluding ordinary file
includes. Rebuilding that candidate restores `payload.S` for the original
witness and preserves all three neighbor controls: twelve direct assembler
comparisons across installed, upstream-original and upstream-candidate tools.
I retain both exact affected source versions, all source bytes in the upstream
archive, the private candidate source, build records, actual objects, tool
hashes and complete debug reports.

This is a bounded upstream regression localization and private patch control.
I have not run a complete upstream assembler regression suite, published an
upstream patch, qualified a product tool replacement or passed the unchanged
full source-capture/platform gate. General assembler 2.46 admission remains
open. My manifest retains 96 logical inputs, 56,923,876 raw bytes and
40,899,898 unique compressed bytes with checked raw/compressed SHA-256 hashes.
