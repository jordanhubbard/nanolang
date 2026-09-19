# I identify the installed OpenCL loader allocations

I inspect retained diagnostic `/tmp/nanolang-opencl-unload-4b71`, not a new GPU execution. The same-process mapping places every unknown LSan frame within installed `libOpenCL.so.1.0.0`. Its first executable ELF LOAD has file offset and virtual address zero, so the mapped file offsets are also ELF symbolization addresses.

I downloaded Ubuntu's official `ocl-icd-libopencl1-dbgsym_2.3.2-1build1_arm64.ddeb` to this directory and extracted it without installation. Its debug ELF build ID is `d2e11dd86dc02194317b8b27a72e8ccdd08abc57`, equal to the installed library; its GNU debuglink CRC is `f566d74d`, equal to the installed debuglink. This establishes exact-build symbolization, not nearest-public-symbol guessing. Ubuntu debuginfod returned HTTP404; the official ddebs archive supplied the matching file.

| Observed offset | Exact debug attribution | Retained allocation |
|---|---|---|
| b24c | `_initClIcd_real`, loader.c318 (inlined `_allocate_platforms`) | `_picds`, platform records; actual instruction multiplies count by48; LSan direct48 bytes |
| b0ac | `_initClIcd_real`, loader.c866 (inlined `__initClIcd`) | `_icds`, vendor records; actual instruction multiplies count by24; LSan indirect24 bytes |
| a8cc | `_malloc_clGetPlatformInfo.isra.0`, loader.c338 | platform property allocation; LSan indirect3 bytes |
| b50c | `_initClIcd_real`, loader.c564 | caller requesting `CL_PLATFORM_ICD_SUFFIX_KHR`, stored in `p->extension_suffix` at569 |
| c1b4 | `clGetPlatformIDs_hid`, loader.c970 | common inlined initialization call |

I verify source archives against the exact Ubuntu `.dsc` SHA256 values. The Debian source overlay contains no patch directory; its build rules configure `--disable-debug` (logging, not absence of separate debug symbols), `--disable-update-database`, and official Khronos headers. The `.dsc` was obtained over official HTTPS; I did not independently authenticate its OpenPGP signature. Archive checksums and debug build identity are separate verified facts.

The matching source's generated platform record holds its suffix and a pointer to its vendor record (icd_generator.rb539–555). This fits the observed direct48 plus indirect24+3 graph. The3-byte suffix content itself was not captured, so I do not assert its literal contents.

On successful initialization, loader.c903–904 returns with these global tables retained. `_icds` is freed at908 only on the initialization abort path. I find no successful teardown freeing `_picds` or its suffix, no loader-specific destructor/atexit registration, and no documented loader shutdown call in the shipped manpage/export contract. The installed `.fini_array` contains only compiler `__do_global_dtors_aux`; `.fini` does not perform loader cleanup. The ordinary OpenCL compiler-unload API is not loader-global-table shutdown.

Thus the observed allocations are exact installed-loader bookkeeping, not NanoLang device-buffer allocations or unexplained NVIDIA allocation PCs. Loss of loader-global roots when the fixture unloads the library is consistent with the source and exit-time report. This does not establish a vendor-object leak, prove an upstream violation of a promised unload contract, or satisfy the sanitizer acceptance requirement. I do not privately free opaque loader state, retain the library, suppress LSan, or modify fixture lifetime.

## I distinguish current upstream

I downloaded official upstream source at immutable commit `3e1155f0796cb9ac0fe309664d98e4b0ed2c3300` (API date2026-06-02). Its NEWS lists loader deinitialization under2.3.5. Its loader.c1495–1551 implements a destructor through `_deinitClIcd_no_inline`: it frees each platform suffix and dispatch data, platform storage and vendor storage, and conditionally unloads vendor libraries according to unloadable-platform accounting. Its manpage documents `OCL_ICD_FORCE_LEGACY_TERMINATION` to disable this scheme, plus a separate diagnostic library-unloading disable option. I do not enable either.

This is documented newer behavior absent from installed2.3.2. No new version is built, installed or qualified here; no claim that it passes the unchanged actual-GPU sanitizer gate follows from static review. Task315caf remains open pending an explicitly reviewed acceptance remedy.

## I preserve provenance

`provenance.json` records original official URLs, SHA256 and sizes. `report-sha256.json` seals all downloaded metadata, source/debug archives, extracted source/debug files and static inspection reports in this directory. I changed no installed package or project source and ran no GPU operation. The parent retains the original failed run and same-process map evidence separately.

## I pin the actual release and its prerequisites

The official GitHub release API confirms `v2.3.5`, published2026-06-02T18:23:29Z. Its tag resolves directly to commit `3e1155f0796cb9ac0fe309664d98e4b0ed2c3300`, equal to the inspected master snapshot. Thus the inspected cleanup is released2.3.5, not only unreleased master. The release description explicitly announces loader deinitialization and links upstream PR43. Its asset list is empty: no separately uploaded release tarball or publisher checksum is offered there. I retain the official GitHub commit archive, locally SHA256 `c94648dbe9d2b3f72a9e2710406ac295fd729787cc4ab715bec1993bf996478a`; this is my measured checksum, not a separately signed upstream digest.

Upstream CI builds this source with `./bootstrap`, `./configure --enable-official-khronos-headers`, and `make`; bootstrap invokes autoreconf, autoconf/autoheader, automake/aclocal and libtoolize. A C compiler, linker, make, shell utilities and Ruby are needed; Makefile.am invokes Ruby for generated dispatch sources. Bundled Khronos headers avoid system header-version dependence. Configure treats asciidoc/a2x/xmlto as optional documentation tools, disabling documentation when unavailable. It checks libdl. For an isolated proposed build, preserve explicit `--disable-update-database`: upstream README warns the optional database-update path installs a dummy ICD system-wide. Do not invoke that target or install system-wide. None of these build steps has been executed by this audit.
