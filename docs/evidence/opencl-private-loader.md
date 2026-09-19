# I qualify the released loader in a private environment

At frozen `7f0d01d4c`, my unchanged actual-GPU fixture passes GCC13 ordinary,
GCC13 ASan/UBSan/LSan, Clang18 ordinary and Clang18 ASan/UBSan/LSan execution.
Each fresh binary checks48 buffers,24 kernels and384 integer observations,
releases its fixture-owned kernel/program/queue/context objects, and calls real
`dlclose`. Each exits0. I use no diagnostic wrapper or leak suppression.

All four processes report NVIDIA GB10, driver595.84, and the private loader at
`/tmp/nanolang-opencl-private-7f0/install/lib/libOpenCL.so.1.0.0`, SHA-256
`3be764c5ce91bd61543d1d9f6e248a0e290f15ba0ec4e64fe739144b52cb016c`.
Its installation contents remain identical before/after the fixture matrix.
The build is unmodified official ocl-icd v2.3.5 commit
`3e1155f0796cb9ac0fe309664d98e4b0ed2c3300`; my recorded commit archive digest is
measured locally, not a publisher-signed checksum.

The [upstream release](https://github.com/OCL-dev/ocl-icd/releases/tag/v2.3.5)
adds loader deinitialization. My [exact installed-loader diagnosis](opencl-loader-diagnosis/DIAGNOSIS.md)
identifies the earlier75 bytes as loader bookkeeping. These passing tests
qualify the private2.3.5 environment. They do not measure each individual free,
repair the installed2.3.2 library, or establish other drivers/platforms. The
original2.3.2 failure and later diagnostic reproduction remain failed records.

## I preserve the build prerequisite failure

My first2397 attempt passes upstream bootstrap/configure, then Make exits2
because the selected private Ruby4.0.6 lacks Psych. No loader or GPU fixture
executes in that attempt. The corrected build copies the matching Psych source,
checks private libyaml packages against apt metadata, builds the extension in
its own directory and passes a YAML encode/decode check. Child-only RUBYLIB
selects it; the existing Ruby prefix remains unchanged.

I extract the missing libtool helper privately and use its supported data
location override. Both loader attempts use bundled Khronos headers and disable
the optional database-update path. No system package is installed, no ICD file
is changed and no global library path is configured. Only fixture child
processes receive the private loader directory through LD_LIBRARY_PATH.

The runner clears the legacy-termination and library-unloading-disable controls
and records effective loader selection. Sanitizer runs retain detect_leaks=1,
halt-on-error and undefined-behavior stack traces, with empty LSAN_OPTIONS.
The first terminal stops dependent phases; the corrected attempt builds new
binaries. Source, selected tool/data maps, installed loader/ICD/driver maps and
HEAD remain unchanged within both recorded attempts. These maps describe the
recorded inputs; they do not claim a complete inventory of every host library
or system header transitively used by the toolchain.

## I retain the evidence

[My report seal](opencl-private-loader/reports.json) identifies copied commands,
logs, exact selectors, package metadata, source/tool/provider maps and runners.
[My artifact seal](opencl-private-loader/artifacts.json) identifies both complete
local archives,149 and318 regular-file members, their symlinks and every
retained output, including all four successful GPU binaries and the private
loader. I verify archive members against their hashes before publication.
External host tools and input paths remain separately inventoried; symlinks do
not claim embedded copies of external directories.

This completes the measured Linux loader-lifetime diagnosis and private remedy.
Full public GPU service integration, platform coverage and release acceptance
remain on their own roadmap items.
