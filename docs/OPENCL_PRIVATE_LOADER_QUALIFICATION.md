# I qualify a private loader with released teardown support

I retain the installed 2.3.2 failure and exact-build diagnosis under task315caf.
Official ocl-icd release v2.3.5 adds loader deinitialization; I pin commit
`3e1155f0796cb9ac0fe309664d98e4b0ed2c3300` and its official commit archive,
measured SHA-256 `c94648dbe9d2b3f72a9e2710406ac295fd729787cc4ab715bec1993bf996478a`.
This is a measured download digest, not a publisher-signed checksum. The
[release](https://github.com/OCL-dev/ocl-icd/releases/tag/v2.3.5) and exact source
are the basis for a candidate remedy; successful qualification remains unrun.

I build the unmodified pinned source in a new isolated directory, with private
installation prefix. I use explicit GCC13, bundled Khronos headers and
`--disable-update-database`; no system package, loader path, vendor registration,
NVIDIA driver or ICD changes. The optional dummy-ICD update path is excluded.
I inventory the source archive, build tools and their data, Ruby, generated
configuration, compiler commands and resulting library. The missing libtool
helper may come from an apt-metadata-checksummed Ubuntu package extracted
privately, with its supported `_lt_pkgdatadir` override; no package installation
or helper source edit is required. Existing private host Ruby may supply the
upstream dispatch generator. I record these exact selections.

After build succeeds, I compile new binaries of my unchanged actual-GPU fixture:
GCC ordinary, GCC ASan/UBSan/LSan, Clang18 ordinary and Clang18 sanitizer, in
that order. Each retains 48 buffers, 24 kernels, 384 integer observations and
explicit fixture cache/context cleanup followed by real `dlclose`. I do not
link the diagnostic wrapper, retain the loader, suppress leaks, or enable the
upstream legacy-termination or library-unloading-disable controls.

Only these child processes receive the private loader directory through
`LD_LIBRARY_PATH`. I require the fixture's observed loader path to resolve to
the hashed private library, and retain the actual NVIDIA GB10/device/driver
identity. I keep the installed ICD and driver hashes unchanged. Each build/run
has a 120-second parent bound; upstream configure/build phases have a bounded
600-second allowance. I preserve each first terminal and stop dependent gates
on failure. A corrected setup requires a recorded reason; no failed executable
is replayed. Source/tool/provider inventories and all fresh binaries remain
retained, with before/after distinctions for generated build outputs.

A passing matrix qualifies this exact private-loader environment. It does not
repair the installed old loader or prove all drivers/platforms. I document the
required loader provenance for reproducing the gate. Full public GPU service,
platform coverage and release acceptance remain separate. Any runtime dependency
policy or wider installation requires a separate concrete implementation.
