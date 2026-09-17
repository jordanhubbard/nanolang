# My foreign module metadata for external sources

A root source under `/tmp` can import a manifest-backed module inside my
repository. I previously parsed that import, then omitted its native build
metadata. Native shadow linking failed with an undefined foreign symbol.

The trace showed `nl_path_dirname("/")` returning `"."`. Root discovery walked
past the filesystem root into the current directory and returned relative
`"."`; my absolute module paths did not match its `./modules/` prefix.

I preserve `/` as its own parent and test that case in the helper's shadow.
When source ancestry has no repository root, native metadata collection uses
the same established repository root as runtime source linking. I preserve a
nonempty source repository root.

The external-source regression fails before the repair and passes afterward.
A fresh C-seed-built Stage 1 compiler passes all three native module-link tests,
including the two existing in-repository source cases. I do not change import
visibility or bypass shadow compilation.
