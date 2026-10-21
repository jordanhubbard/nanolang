# Native map concat-key compatibility prerequisite

## Scope and actual evidence boundary

The public consumer component admits counted-STRING→INT authority to VM while its native target accepts only NUL-free executable literals and immutable transports. Automatic checked-type publishers must not make previously native-accepted source fail merely by adding authority. This checkpoint proposes a baseline and one bounded native closure audit; it makes no universal STRING or publisher-completion claim.

The unchanged paired anchor is tests/nanoisa/fixtures/map_record_fields.nano (CollectResult with array<string>, HashMap<string,int>, bool). Its existing source test is tests.test_nanoisa_flat_records.test_map_record_fields_match_and_execute. The retained four producer baselines already execute raw original bytes and explicitly annotated consumer bytes. Existing tests/test_hashmap_set_advanced.nano uses dynamic concatenation/to_string keys but also imports and operations outside this authority profile; it is evidence to map, not an exact paired profile qualification.

Current C seed codegen.c:3255 emits AGG_PACK AGG_RECORD with struct def_idx; Nano nanoisa_codegen.nano:4816/4873 emits AGG_PACK with its registered record index. Both ordinary publishers build record layout rows in that same order. STRUCT and nonidentity ordinal/layout cases currently appear in synthetic import/public consumer controls. No current source-produced regression from those representations has been demonstrated. Keep their native refusal explicit; do not rewrite producers or claim they are impossible generally.

## Bounded baseline, before any target edit

Retain the unchanged anchor byte-for-byte. Create a separately named variant by replacing the single main invocation `(remember alias "second" 2)` with `(remember alias (+ "sec" "ond") 2)`. Assert that exactly one occurrence changes and that every original assertion remains byte-for-byte. Preserve source hashes and both complete files. The variant remains a proposal until actual producers accept it.

Use only retained seed, current C-hosted Nano emitter and installed Stage1/Stage2, with source/provider hashes before and after. No frontend build or bootstrap. For each accepted output retain canonical opcode/operand evidence, actual verification and ordinary VM success, legacy native translation plus O0/O2 execution, then independently annotate the exact module and verify/public-VM it. Preserve selected native concat refusal before changing the gate. A producer refusal is a baseline result; do not rewrite the fixture to conceal it. Keep raw and annotated modules distinct.

Initial baseline uses unchanged and variant ordinary inputs. Original-shadow selection, if needed for the later permanent qualification, is separately produced using the retained existing driver; it is not inferred from main-module metadata.

## Smallest candidate target change, still unapproved

Keep the current closed-module, root/initializer, exact pair, ordinal/layout and STRUCT boundaries. Keep the executable PUSH_STR embedded-NUL rejection. Audit STR_CONCAT as the sole newly admitted dynamic producer; every other ROOT_STRING producer remains refused. Do not permit the whole recipe category: it includes CAST_STRING, substring, trim, case conversion, replace, numeric conversions and split.

The intended induction is: executable literal leaves contain no embedded NUL; closed typed locals/globals/arguments/results/fields/array transport preserve string bytes; no external input or unaudited producer introduces bytes; concatenating two NUL-free values copies their complete byte sequences and appends only the terminal NUL. This is a native-target eligibility proof, not a change to generic STRING metadata or counted VM semantics. STRING operations that merely observe values do not create a new string. Empty and UTF-8 literal bytes remain unchanged. Existing NUL inputs still refuse native translation atomically.

Current nvm2c.c emit_nstr_concat uses strlen on each operand, checked na+nb, nstr_allocate, two memcpy calls and a terminal zero. Under the induction its lengths equal the VM counted lengths. The old native runtime already implements concat, so the smallest eventual change may be a target whitelist exception with no new runtime implementation. That remains contingent on baseline and exact lifetime evidence.

## Lifetime and failure obligations

nstr_concat creates a tracked native string allocation. HM_SET borrows the key; nmap_set hashes/finds it and copies the complete key only for insertion. Equal-content replacement preserves the existing stored key, while integer values need no text ownership. Thus a stored key is independent of later nstr collection; the temporary concat operands/results must nevertheless remain rooted across all safepoints until the set completes. HM_GET/HAS similarly borrow the dynamic key during lookup. Missing GET remains tagged VOID, never invented INT zero.

Audit emitted safepoint placement and local/operand root frames, not only map helper code. Qualify dynamic key creation in a helper, use after caller temporaries disappear, replacement from distinct equal-content concatenations, map growth, aliases through record fields and repeated calls, empty/UTF-8 keys, missing GET, O0/O2 ASan/UBSan and forced collection. Keep allocation-failure compiler sweeps distinct from generated abort-process sweeps; exact allocation abort and sanitizer diagnostics must distinguish intended failure. Compare old native output bytes and private/public boundaries. No shared nmap byte-domain rewrite is proposed.

## Gates and remaining scope

No target edit before actual raw-source baseline and this lifetime audit are reviewed. No publisher encoding/admission changes in this slice. Other accepted dynamic producers require their own demonstrated source baseline and closure audit before broad publisher compatibility can be claimed. STRUCT/nonidentity remain consumer restrictions without demonstrated current producer need. Counted-NUL VM behavior stays qualified independently and does not imply C-string native parity. Keep old budgets, limits, generated code behavior outside the explicit target exception and full malformed refusal controls unchanged.

## Actual baseline and lifetime review

All8 unchanged/concat inputs from seed, retained generic-field C-hosted canonical and installed stages compile, verify and execute raw VM/native.32 native runs (O0/O2, ordinary/forcedGC, ASan/UBSan) pass exact0/empty stderr. Every concat module contains exactly1 STR_CONCAT and no STRUCT opcodes; authority remains absent. Separate explicit annotations pass837 public verification/transport/VM checks and preserve the expected native concat refusal for all4 variants. Eight raw tool/compiler pins and current aggregate consumer object pins remain unchanged. This demonstrates the metadata-selected native regression boundary; it does not assert general dynamic STRING compatibility.

Independent lifetime read confirms emit_map_roots enumerates all non-scalar locals and operand roots before CALL/TAIL_CALL collection and argument consumption. Self-tail restarts also collect only after those roots are published. RET assigns the result then reaches L_return; frame unlink/destruction performs no collection, so the returned string transfer has no intervening collector. STR_CONCAT and nmap_set allocate/copy without collection; map insertion owns an independent key copy, while replacement retains its original key. Actual helper-return/tail/local/replacement forced-GC controls remain required after the reviewed exact STR_CONCAT-only target exception. No runtime redesign is authorized.
