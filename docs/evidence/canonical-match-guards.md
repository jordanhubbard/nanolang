# My canonical guard checkpoint

I qualify the first corrective a18 guard checkpoint at `eafcd1e52651996b1960a21ef5a3af57baf7a3b1`. I preserve guard AST identity, exact BOOL and effect checks, ordered native statement/value selection, lexical payload scope, retained NanoISA scrutinees, terminal backstops and enclosing control flow. I do not qualify concrete generic-union instance maps or affine guard joins here. Those remain required under a18 and the full 5.1 scope.

## My actual outcomes

| Host | Fresh bootstrap | Six guard methods | Shared policy + C-seed totality | Two migrated owner refusals |
| --- | --- | --- | --- | --- |
| sparky, Linux ARM64 | PASS 284.542s | PASS 96.514s phase / 95.590s unittest | 10 PASS, 2.510s phase | 2 PASS, 1.005s phase |
| puck, macOS26.6.2 ARM64 | PASS 306.557s | PASS 151.922s phase / 150.193s unittest | 10 PASS, separate 2.816s phase | 2 PASS, separate 0.891s phase |

Native source controls and the two unchanged-source owner refusal methods use C seed, Stage1 and Stage2. Named scalar-union module controls use nano_virt, Stage1 and Stage2, followed by verification, NanoVM and generated native execution. Parser877 runs unchanged through the three native source compilers. Integer and conditional-wildcard source controls retain their native scope; unsupported ordinary NanoISA match profiles remain refused. Both unchecked generated backstop forms succeed on the represented input and terminate with the required diagnostic on a miss.

I use GCC13 for Linux setup and explicit generated-C sanitizer probes. On puck, setup uses Apple Clang17, while generated-C sanitizer probes use Homebrew LLVM23.1.1. The commands retain ASan/UBSan and leak detection with empty LSAN_OPTIONS. Ordinary source-driver executables and compiler providers are not relabeled as sanitizer-instrumented. Runtime-library inventories were taken before focused probes and afterward, not before bootstrap. Both inventories match.

## My retained first terminals

At `2ad92f457`, both fresh bootstraps stop before Stage1 publication on my new purity shadow: it expected mutation mask4 for an existing mutable-read mask2. Linux stops67.203s; puck87.334s. The exact shadow correction is `e75a719bf`, with the immutable guard expectation0 retained.

At `e75a719bf`, both fresh bootstraps pass, then each complete six-method suite reports four passes and two failures. Ordered native statement emission places `#line` mid-line; the raw fixture separately lacks explicit argc/argv imports. The next adjacency phases do not run. I correct the newline plus its shadow and add only the missing imports in `eafcd1e52`. Historical failed generated C was deleted by existing cleanup, so I retain its source and compiler output but do not claim historical C bytes. The corrected fixture retains generated C before asserting command success.

These earlier six-method phases also rebuild compiler module objects. Their original unequal maps remain intact. I classify `obj/nano_modules` as generated outputs in corrected per-command snapshots; fixed compiler/provider maps remain separate.

Puck's corrected six-method suite passes, then its inventory guard stops on four newly generated `obj/module_cache` artifacts. No prior provider or compiler changes or disappears. Retained timestamps place those std module fs.c/process.c products in parser877's C-seed compilation, not the raw-driver compilation initially inferred. The original outer command returns0 for the passing unittest, but its manifest has only three phases and is not full acceptance. I retain that manifest unchanged. A separately reviewed continuation runs only previously unreached adjacency, with source/fixed-provider equality and generated-cache before/after archives. I do not repeat the successful bootstrap or six methods.

## My evidence and integration boundary

My [ready input map](canonical-match-guards/ready-inputs.json) verifies exact production/schema/fixture equality against the qualified tree. I integrate canonical File885/887/888 separately: private File APIs and additive recipes change; no guard frontend, schema or fixture changes. I do not attribute a new bootstrap to this additive integration.

My Linux final seal is `/tmp/nanolang-guards-eaf-linux/final-seal.json`:42 top-level reports,8062 unique archived files including the tracked source snapshot,741 fixture files. Puck's final seal is `/private/tmp/nanolang-guards-eaf-darwin/final-seal.json`:53 reports,8162 unique archived files including sources,715 fixture files. These counts are artifact inventories, not executed test counts. Current source/fixed-provider hashes and every stored content hash were checked. I retain all original and corrected trees and stores.

The copied reports below are committed Git blobs; their original absolute artifact references identify Linux-local or puck-local stores. My [report digest map](canonical-match-guards/report-sha256.json) seals the published reports. The original first-bootstrap report copies remain under `docs/evidence/canonical-match-guards-first`. The second partial seals retain all six outcomes, and the corrected Darwin seal links its separate adjacency continuation. Temporary files automatically removed by existing adjacent test helpers are not claimed as retained binaries; their completed assertions and logs remain evidence.

This is a qualified guard milestone. Neither a18, its remaining union/affine criteria, nor the full release closes from this report.
