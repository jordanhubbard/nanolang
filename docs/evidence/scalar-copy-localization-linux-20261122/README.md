# I localize scalar-copy inference without claiming a repair

I publish source `0b81a823608e923d7c989be064a40d8e54d24a63` under #1202/#1142. My [12-line source](reproduce.nano) reduces the retained frame C AOT barrier to an uncalled helper with a Boolean array read, array write and short-circuit copy. My production translator remains unchanged; its image is still SHA256 `729820b30caaca7f5d3897b403d20e3b7cdaac34a5d12963119c501eadd90005` from the preceding [frame checkpoint](../affine-cfg-foundation-linux-20261122/README.md).

| Linux ARM64 check | Result |
| --- | --- |
| Basic direct/record source variants | All 16 translate |
| Additional and/or/not variants | All 48 produce NanoISA; eight and/or C translations refuse |
| Focused64-source regression | All 64 source/shadow gates pass; 56 C translations pass and eight valid translations refuse |
| Valid raw PICK0 | Verification passes; C translation supplies the ninth retained failure |
| Invalid PICK1 | Assembler and downstream verifier/C refuse; prior outputs remain unchanged |
| Unchanged graph API | 312 checks pass with strict flags, and again under ASan/UBSan |
| Identifier controls | Five methods pass; 3446 fragments lexed; 135 existing lexer refusals retained |

My read-only traces establish the failing sequence. ARR_GET registers unknown element/result IDs 1600/1675. STORE_LOCAL unifies the unknown read result 1675 with local 1662. A later DUP reaches sim_push_slot and seeds a fresh exact Boolean node 1688 onto that shared local. The deferred array-read solver then discovers the nullable scalar read and correctly refuses its conversion to an exact Boolean destination. PICK0 follows the same copy classifier and reproduces the refusal. All unary-not neighbors translate. These controls localize the copy path; the shape engine's deliberate exact-target rejection remains correct.

My [one-line caller proposal](proposed-scalar-copy.patch) extends LOAD_LOCAL's existing inferred-scalar handling to DUP/PICK. It is **unapplied and unqualified**: I have not compiled a modified translator. The 312 API checks characterize this proposed graph treatment using the unchanged public graph API, including exact-target rejection, mismatched payloads, immutable exact producer kinds, distinct nullable storage and stable repeated solves. They do not establish that the caller patch passes the actual compiler or full frame gate.

I retain the first focused test's negative-fixture error: I expected my assembler to publish an invalid PICK1 module, but it correctly refused stack underflow. The corrected test additionally checks assembler prior-output preservation, then changes only PICK0's depth in an already checked module and recomputes the CRC32 to exercise the downstream guards. The corrected suite retains **nine valid translation failures**, with no errors or altered success assertions. It remains a red product regression. I also retain a supplemental observer's unsupported --verify probe and give it no verification credit; the separately recorded --verify-only commands establish the actual positive/negative decisions.

My capsule contains 2790 records/101770256 logical bytes and 1366 exact Git source blobs, reusing hash-bound immutable objects from the preceding checkpoint. It includes all 64 diagnostic sources, both formal regression attempts, actual modules/generated C/streams, both debugger traces, optimized/sanitized graph checks, native tool images and the original complete frame module. The human-readable source and patch above are checked against their archived bytes.

```sh
/usr/bin/python3 docs/evidence/scalar-copy-localization-linux-20261122/read_archive.py
```

My [reader result](verification/linux.json) must say `LOCALIZATION_INTEGRITY_PASS_PRODUCTION_REGRESSION_STILL_RED`. I preserve the protected C/shape/affine/regex source bytes, all original checks/deadlines, frozen roots, the primary dirty audit and stashes. I make no NAS/migration/TCC recovery or cutover action and hold no native Linux/Darwin reservation after these bounded checks.

The [existing source-only grant](https://github.com/jordanhubbard/nanolang/issues/1158#issuecomment-7083671875) covers #1193/#1194 in Puck-owned nvm2c.c; the new #1202 copy hunk needs its own ownership handoff. My complete frame five-route requirement, original shape/refusal guards, paired Darwin, whole-function CFG, pure C emission and the full 69-requirement release remain open. No executable selector uses the new pure verifier components.
