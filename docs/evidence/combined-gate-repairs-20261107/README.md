# My combined 5.1 gate repairs

I retain the component commits in `components.json` as actual merge ancestors of
PR #1068. I require a merge commit into main after the combined source revision
passes its required checks; squashing would discard the component ancestry.
My separate original-source metadata and retained lifecycle still require their
own custody, authority and acceptance evidence before release.

My combined local checks pass:

- 32 workflow, container, classifier, prerequisite, runtime-installer and capture
  controls, with the unavailable pinned-Xcode SDK check explicitly skipped.
- The real two-build compiler version regression, six release-version/package
  tests and release workflow controls.
- All 16 managed-record eligibility methods and the complete ordinary/resource
  authority gate, including normal and sanitized fixture routes.
- 223 strict Clang and 223 strict GCC compilations plus both prerequisite audits
  and the bootstrap-tool control in `make check-fast`.

I preserve all component assertions while resolving the shared Make recipe and
roadmap insertions. Existing full release and native-lifecycle requirements remain
open. These local gates do not establish complete hosted or release acceptance.

## My image tool closure correction (#1075)

Image build job [128431125271](https://github.com/jordanhubbard/nanolang/actions/runs/42381763074/job/128431125271)
installs clang-18 and llvm-18-linker-tools but no llvm-18, lld-18 or nodejs. My
runtime packager invokes opt; Wasm publication invokes llc/wasm-ld; unit tests
require LLVM tools and Node. Ubuntu lists opt in [llvm](https://packages.ubuntu.com/noble/amd64/llvm/filelist)
and wasm-ld in [lld](https://packages.ubuntu.com/noble/amd64/lld/filelist).
I add llvm/lld/nodejs from the existing signed timestamp-pinned repository and
require all actual version commands during image construction.

Before bootstrap my verified-container smoke now compiles and verifies LLVM IR,
assembles bitcode, links Wasm and executes the result in Node and Wasmtime.
I execute that exact added pipeline with the installed local tools as a script
control. The new hosted Ubuntu image and its actual smoke remain required;
local tool success does not establish image acceptance.
