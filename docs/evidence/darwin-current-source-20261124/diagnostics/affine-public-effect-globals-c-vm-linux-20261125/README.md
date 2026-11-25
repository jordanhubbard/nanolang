# My public affine effect/global checker

I compose my private body walk with a separate pure NanoLang effect/global checker and both public analyzer wrappers at source `05b1dc70bb4e1474efe15bc5828b9e931da96c4d`. I preserve exact activation/handler/point identities, FIFO joins and requeue, lexical return/resumption, global initialization relations, runtime depth traps, offsets and failure messages. Public instruction analysis stages its facts until both checks pass. I retain the original 262144 storage bound and 65536 hash buckets; lowered diagnostic bounds never enter my public wrappers. I change no product C implementation.

My baseline verifies842 complete C/VM reports,1,001 byte-identical original C reports and673 public controls against an observer-disabled C twin. The845-case permanent corpus at `d1687ded5ca551eb102b66b32a8eb4eafa6019e7` adds three immutable-store controls without changing any earlier carrier or expectation. Those controls reuse the exact baseline C binaries and VM artifact: canonical-point input survives STORE/JMP requeue, and repeated resumed stores fail both public wrappers while retaining prior facts. Both additional public controls match the unhooked C twin. Production source is identical across these two pins.

My combined reader verifies13,429 records and2,493 successful commands,845 complete C/VM reports and675 unhooked public controls. Independent criteria cover wrapper precedence/capacity/fact rollback, effect handlers and aliases, lexical returns/resumption, global joins, immutable/mutable stores, bounded control storage, depth traps and131 exact hash vectors. Strict C, typed AST, shadow presence, complete ordinary imported shadows and the actual additive Make shadow target pass. Original600-second builds, default10-second ordinary shadows and60-second new carriers remain unchanged. Source/Git/tool/object/host and immutable artifact guards match.

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/affine-public-effect-globals-c-vm-linux-20261125/read_archive.py
```

My parent manifest binds the frozen842-case checkpoint; `immutable-stores/` separately binds the three-case supplement and its parent result. The reader reconstructs complete reports, checks independent expectations and source pins, verifies that the only container change exports the existing xor32 helper, and confirms unchanged product C.

I have not executed the new method's full original-frame VM, genuine-source, sanitized-native or paired routes. Existing #1204 still blocks ordered full global/array and leaf qualification. Public profile admission, runtime C emission, executable cutover and all69 release requirements remain open. This checkpoint grants no release admission.
