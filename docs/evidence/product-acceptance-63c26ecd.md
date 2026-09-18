# My frozen candidate acceptance

I freeze production `63c26ecd2723f8c480eaba4b25ebdf797f4764bd` through PR697 while later canonical work continues separately. This candidate is not ready for release.

My fresh Linux bootstrap passes in189.669 seconds; tools build in25.129 seconds. All65 executed focused methods pass, but borrowed-source setup fails before its26 methods execute: Stage2 compiling `src_nano/nanoisa_emit.nano` exceeds the ordinary180-second deadline. The focused command exits1 in1357.081 seconds. My fail-closed runner does not start full `make test-quick`. I preserve the [initial manifest](product-gates-63c26ecd-initial.json) and [complete focused log](product-gates-63c26ecd-initial-focused.log), SHA-256 `9601791a3cce0207870a6209b262892d3b2829b967954fbc7fc57db42a211183`.

Task `task_ef9261a163f64e0397a93d377d623ed7` measures exact current setup separately before a setup-only budget correction. Ordinary assertion deadlines stay unchanged. The standalone runner also omitted the existing local-binding probe prerequisite; I will build that probe before qualifying the corrected suite. These are test preparation findings, not a full product pass.

Fresh isolated VM and native raw fixed-point runs are still in progress. Both first self-compilation generations pass; second generations, final raw byte comparison and closure integrity remain pending. Earlier84e fixed points do not qualify this pin.

## Corrected Darwin comparison at the earlier compiler source

The peer completes full `make -j8 test-quick` under PTY at `c9c747858ee4c557563f0b3c3ca9e86a7407dec0`, exit0 in985.04 seconds. I independently compared Makefile.gnu, src, src_nano and modules against Linux-qualified `ece241ba`: there is no difference. Relative to evidence-only d021ad99, the corrected pin changes only the roadmap and Forth harness stdin isolation.

I retain the peer's [sealed report](product-acceptance-ece241ba-darwin-corrected.txt), published in evidence-only commit `cd142d78`. I copied and independently hashed its192219-byte full log at `/tmp/nanolang-product-ece-forth-corrected-c9c74785-test-quick.log`: `47ef1a62b9004841cffea574998f783b02177ddd01b7fe5e00670c49562c1d26`. Bootstrap,17core examples,244eligible VM examples, all Jackson wordsets,280Forth cases, production PTY REPL and IDE build/smoke pass. The report's older component-driver validation means that those drivers compiled and exited successfully; it does not establish all imported shadows or the new PR691 entry assertions.

This closes the corrected earlier-source platform comparison. It does not qualify the later63c production integration. Current-candidate Darwin acceptance is separately requested; all broader release parents and the publication hold remain open.
