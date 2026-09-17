# My compiler artifact support

I expose one compiler host operation: `module_artifact(source_path)` builds the
source file's manifest-backed C support and returns the absolute shared-library
path in the immutable generation returned by that build. I return an empty
string when the source, manifest or build is unavailable. I do not compile
NanoLang expressions or execute bytecode here.

My C export `nlc_module_artifact(const char *)` returns a per-thread snapshot
that the next call replaces. Foreign adapters must copy that result when they
retain it. The native adapter uses the existing string-snapshot contract.

I reuse the module builder's source capture, cache and publication machinery.
On Linux, a compiler moved away from its installed tools supplies the existing
`NANO_AS_CAPTURE_HELPER` toolchain setting. I retain the builder's refusal when
that helper is unavailable; I do not fall back to uncaptured compilation.

`tests/test_compiler_artifact_support.py` checks stable generation reuse,
changed-source generations, preservation of old artifact bytes, failed-build
refusal, missing metadata and non-file input. The fixture calls this module
through a compiler-built NanoLang driver.
