# Main through PR #334 on Linux

I reconcile main `ded3ee5a` with integration `5cf21fc1` on Ubuntu ARM64.
I retain the independent ownership pass in `resource_flow.c`; main's older
identifier-only pass does not replace its lexical, branch and exit checks.
I combine checker visibility with source-bounded emitter metadata and keep
VM slot allocation separate from live lexical bindings.

I retain dynamic native frames, owned arrays and scalar strings, tagged
locals/globals and recursive shape facts. Main carries older snapshots of
these same subsystems. I preserve the newer implementations and existing
acceptance cases. Main's current-frame map reclamation does not account for
caller/global/aggregate roots, so I retain execution-owned map storage pending
the existing lifetime task `task_d3310bef8bd541ba9e1e267ee213eb9e`.

I retain shared path normalization in native helpers and import main's dynamic
interpreter path normalization and 600-component regression. I retain the
newer filter emitter rather than duplicate its boolean helper and array
inference. Incoming scalar-filter tests remain in the language fixture.

## Linux findings

- Fortified GCC rejects a discarded diagnostic `write` result; I consume it
  explicitly without changing the failed-shell exit status.
- GCC diagnoses potentially truncated composed test assembly. I size those
  buffers for both input bounds and headers.
- Strict GCC rejects four generated runtime statement forms as misleadingly
  indented. I brace those conditions without disabling warnings.
- A shell wrapper turns child signals into ordinary exit codes on Linux.
  I make it execute its supervisor directly. A focused probe distinguishes
  ordinary exit 134 from `SIGABRT` (-6 in Python).

## Verified checkpoint

- C build and self-hosted bootstrap smoke checks pass.
- The shared ownership/record-pattern suite passes all 33 methods after both
  self-hosted stages and VM binaries are built.
- VM code generation passes all 77 cases, including incoming lexical tests.
- The 600-component path-normalization regression passes.
- Process-capture regression passes across its native/VM and module paths.

An early direct ownership invocation preceded the VM build and reported 20
missing-executable errors. The dependency-ordered run above passes; I do not
count the premature invocation as a product regression.

Translator, native-compiler, broad quick and full-release validation remain
separate gates. This checkpoint is not a published release.
