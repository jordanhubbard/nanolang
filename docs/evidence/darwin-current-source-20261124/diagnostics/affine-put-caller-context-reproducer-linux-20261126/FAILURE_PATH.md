# My observed nonfinal refusal path

I bind both existing nonrecursive traces to `67d99e5121521428740d385d10071b9679e66013:src/nanoisa/nvm2c.c`. The selected source and debugger source file currently match that Git snapshot byte for byte (SHA-256 `994d83e225343cfee19f5674cc30259fcea6bc3db44d65911764fd63489f31c0`). I execute no compiler and make no new reduction here.

At `put` byte42, both recorded producer traces have receiver kind `UNK=2`, value kind `ARR=3`, `written_array=ARR=3`, `facts.final=0` and `canonical_records=false`. My source explains that immediate refusal:

1. Known nested-array storage has its own branch at line2753, but this receiver is still unknown.
2. The array-valued-write deferral at line2807 requires `canonical_records`; it does not run here.
3. The next unknown-receiver deferral at line2824 covers a tagged value or a nonfinal integer value. This value is an array, so it does not run either.
4. The fallback at lines2873–2879 selects `ARR` storage, marks the receiver's local origin and pushes that representation. This agrees with the captured `written_array=3` and local kind3 while the popped `arr.kind` remains2.
5. The scalar-payload check at lines2884–2892 interprets that storage as integer elements (`INT=0`). The known array-valued element (`ARR=3`) then triggers the recorded refusal.

The ordinary inference loop at lines8926–8961 aborts when classification returns false. Both traces show this happens before the final pass. These source and runtime observations explain the immediate refusal; they do not show how a repaired fixed point would converge or establish final storage/admission correctness.

The C owner retains the repair. It must preserve unknown storage until sufficient facts or a checked refusal is available, while retaining exact representation checks, malformed-input rejection, ownership/profile authority and prior-output protection. Enabling canonical mode or accepting an unresolved final write is not established by this receipt. Private/public consumers and the original forward-factory/cache cases remain acceptance requirements under #1262.

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/affine-put-caller-context-reproducer-linux-20261126/read_failure_path.py
```

My checker binds exact source spans and both archived traces, including the previously unreported `written_array=3`. It does not simulate a corrected classifier or prove compiler correctness. My original 100-record reproducer manifest remains unchanged; `failure-path.json` is sealed separately by the checker.
