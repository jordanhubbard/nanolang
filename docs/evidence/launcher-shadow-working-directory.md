# My launcher shadow working directory

Darwin strict-example CI job `104900668619` failed the imported `build_example`
shadow while compiling `sdl_example_launcher.nano` from the examples directory.
The fixture wrote to `bin/launcher_prebuilt_artifact_probe` without ensuring
that `bin` existed or checking the write result. The launcher correctly refused
the absent artifact; all three behavior assertions failed.

I now create and check the fixture's `bin` directory. I reserve a unique name
with `mktemp_dir`, derive the artifact name through the same `get_binary_name`
function as the launcher, and check write and cleanup results. I retain the
exit-code, prebuilt-message and artifact-existence assertions. I do not alter
`build_example`, add compilation to launch behavior or skip dependency shadows.

On 2026-09-16 the complete launcher compiles, including imported shadows, from
both repository and examples working directories on Linux ARM64 and Darwin
arm64. Linux uses a fresh compiler on the isolated documentation branch before
the effect integration; Darwin uses the isolated `7a6ac4e8` checkout plus the
union-lifetime repair. The final integrated compiler needs its own rerun.
The Darwin linker emits its existing SDK text-stub/duplicate-library warnings;
I do not describe that output as warning-free. I compiled the GUI launcher;
I did not claim a visual GUI execution test.

MAC: `task_d43b4d7f2ede47788cbeb2b94db32db3`.
