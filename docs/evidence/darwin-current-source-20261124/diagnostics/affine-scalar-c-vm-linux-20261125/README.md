# My scalar affine-transfer C/VM checkpoint

I implement an internal42-opcode transfer leaf in NanoLang at9d5ace7053b6904ec7dd3f98225b4fa4b72cb998. I model scalar/void literal tags, guarded DUP/POP/SWAP, integer/float/Boolean operations and comparisons, assertion/branch consumption and graph-gated immutable-string output. My caller supplies checked operands and graph mode. I neither establish graph authority nor admit metadata-dependent instructions, CFGs, functions, modules or executable selection.

My separate C fixture includes the unchanged production affine step. All3837 commands pass through the actual C reference and NanoVM, with stable recorded source/tool/object guards. All1001 original frame reports retain SHA256 `f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0`. All25642 scalar reports agree completely and meet the independent expectations, SHA256 `e710c83d560f1d3e60ff5d908fbd0260d64eef4fe71616446ad92c7764b427df`. I retain every old8-case/600-second batch; new cases use the same60-second limit with at most32 cases/8KiB per batch, running larger frames singly. No case or assertion is removed.

I preserve both unsuccessful precursors. Source1c67455c8c maps U8/FLOAT/BOOL incorrectly in both port and oracle; a separate read-only three-literal VM witness confirms old VM tags4/2/3 against actual C2/3/4. Source8902042 fixes those mappings, and all25642 actual-C expectations pass, but a full-stack refusal exposes my wrapper forwarding a borrowed failed frame. I clone only that failed result. The exact caller-isolation driver and assertions remain unchanged across all three versions; mutating the result now leaves the caller intact. Existing pure frame/profile implementations and original methods/corpus remain unchanged.

A separate test-harness cleanup is qualified atba718b39a0. The old C frame-reference zero-function path leaks a successfully loaded model; its sanitizer failure retains empty stdout and the leak diagnostic. The scalar sibling already passes. Releasing the model makes both identical inputs emit D/exit0 with empty ASan/UBSan streams. This changes no production analyzer or protocol bound. The cleanup commit's added malformed case is separately native-sanitizer qualified, beyond this9d corpus; I do not claim current-head whole-suite acceptance from that control.

My independent reader verifies105843 records, exact source pins, all streams and3837 commands, all1001/25642 complete reports, both retained failures, unchanged isolation assertions, the literal red witness and identical-input native sanitizer cleanup. Manifest SHA256 `e935d8db72480bd294dc34eceec760bd627f9f5d33b8716a13472f6878060a1c`.

Complete original frame/operand and scalar five-route/sanitizer/paired gates remain required. Local/array/reference/call/effect/global transfer, handler/CFG convergence, private admission, actual C emission/cutover #1142 and all69 release requirements remain open. Puck retains #1252/#1204/#1256 and C/cache/profile ownership. I preserve every frozen root and operational pause.

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/affine-scalar-c-vm-linux-20261125/read_archive.py
```
