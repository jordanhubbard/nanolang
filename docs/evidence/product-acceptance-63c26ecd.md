# My frozen candidate acceptance

I freeze production `63c26ecd2723f8c480eaba4b25ebdf797f4764bd` through PR697 while later canonical work continues separately. This candidate is not ready for release.

My fresh Linux bootstrap passes in189.669 seconds; tools build in25.129 seconds. All65 executed focused methods pass, but borrowed-source setup fails before its26 methods execute: Stage2 compiling `src_nano/nanoisa_emit.nano` exceeds the ordinary180-second deadline. The focused command exits1 in1357.081 seconds. My fail-closed runner does not start full `make test-quick`. I preserve the [initial manifest](product-gates-63c26ecd-initial.json) and [complete focused log](product-gates-63c26ecd-initial-focused.log), SHA-256 `9601791a3cce0207870a6209b262892d3b2829b967954fbc7fc57db42a211183`.

Task `task_ef9261a163f64e0397a93d377d623ed7` measures exact current setup separately before a setup-only budget correction. Ordinary assertion deadlines stay unchanged. The standalone runner also omitted the existing local-binding probe prerequisite; I will build that probe before qualifying the corrected suite. These are test preparation findings, not a full product pass.

My fresh isolated VM fixed-point run passes at63c26ecd. Initial, first-generation and second-generation compiler modules are exactly407992bytes with SHA-256 `d4f5dfb18fc63b8f689403c16c159c9636cd0acfce1215ce743fac1375d06011`. Both generations verify and retain identical three-library closure; Stage2 compiles, verifies and executes hello. Stage1 takes884.566seconds and Stage2 takes885.655seconds under their original1200-second limits. [Manifest](product-vm-fixedpoint-63c26ecd-manifest.json) and [post-run integrity](product-vm-fixedpoint-63c26ecd-post-run-integrity.json) retain exact source, tool and library identities; tracked source is clean and unchanged. The manifest top-level1800-second field is inherited runner metadata; each VM stage records and enforces1200seconds. Native second generation is still running and remains separately pinned. Raw fixed points do not establish compiler semantic equivalence or release readiness.

## Corrected Darwin comparison at the earlier compiler source

The peer completes full `make -j8 test-quick` under PTY at `c9c747858ee4c557563f0b3c3ca9e86a7407dec0`, exit0 in985.04 seconds. I independently compared Makefile.gnu, src, src_nano and modules against Linux-qualified `ece241ba`: there is no difference. Relative to evidence-only d021ad99, the corrected pin changes only the roadmap and Forth harness stdin isolation.

I retain the peer's [sealed report](product-acceptance-ece241ba-darwin-corrected.txt), published in evidence-only commit `cd142d78`. I copied and independently hashed its192219-byte full log at `/tmp/nanolang-product-ece-forth-corrected-c9c74785-test-quick.log`: `47ef1a62b9004841cffea574998f783b02177ddd01b7fe5e00670c49562c1d26`. Bootstrap,17core examples,244eligible VM examples, all Jackson wordsets,280Forth cases, production PTY REPL and IDE build/smoke pass. The report's older component-driver validation means that those drivers compiled and exited successfully; it does not establish all imported shadows or the new PR691 entry assertions.

This closes the corrected earlier-source platform comparison. It does not qualify the later63c production integration. Current-candidate Darwin acceptance is separately requested; all broader release parents and the publication hold remain open.

## Bounded Sail model refresh

At documentation-only f38b6409, the existing `scripts/check_sail_container.sh --rocq-check` completes on the peer in 529.47 seconds. I independently verify the [report](sail-rocq-f38b6409-evidence.txt) and [log](sail-rocq-f38b6409.log) hashes against MAC evidence, count all nine closed-assumption reports and the independent-check marker, and match the model/proof/script/checker inputs to this63c source. Taskf352 is completed. This proof-only gate does not run decoder/VM corpora and does not close full formal foundations or production correspondence.
