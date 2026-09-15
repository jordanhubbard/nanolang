# Owned walk-result release

I provide `fs_walkdir_release` as an opt-in C operation for an unmodified,
exclusively owned result from `fs_walkdir` in the same library. I free its
copied path strings before releasing the array. Callers must copy escaping
strings first; no borrowed element pointer may survive a successful release.

I check GC membership, array kind, reference count and string storage layout.
These checks do not establish provenance: a caller must not pass an arbitrary
string array or a modified walk result. This is a trusted native ABI contract,
not a safe operation exposed to ordinary NanoLang programs. Existing callers
and ordinary array collection retain their previous ownership behavior.

## Verification

On Darwin, `make test-walk-result-release` passes with assertions enabled.
I check a populated result, an escaped copy after release, refusal while an
extra GC owner exists, empty results, null and unmanaged pointers, and refusal
of an integer array. GC object counts return to baseline; those counts do not
measure individual string allocations.

`make test-directory-walk` passes the release probe and six Python cases in
5.140 seconds, with one host path-limit skip. The final release probe also
passes after adding unmanaged-pointer and integer-array refusal checks.
`git diff --check` passes.

I have not implemented the AOT exact-artifact loader or typed array conversion.
That work remains under MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
General native string-array ownership remains separate. This checkpoint does
not establish release readiness or general leak freedom.
