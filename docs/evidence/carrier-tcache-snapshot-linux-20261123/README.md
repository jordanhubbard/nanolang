# I account for one observed allocator-cache delta

I continue #1213's [earlier accounting diagnostic](../carrier-byte-accounting-linux-20261123/README.md) with the unchanged original native observer, SHA256 `01164f0cec7feee171ad7e1ef6affc31d7f2728b970687f5262af20f79d2d0b6`. I edit no product or fixture, make no inferior allocation calls, retain ordinary ASLR, and install no allocator setting. My parent limit is 90 seconds; the completed debugger observation takes 0.141 seconds.

| Existing sample | After 64 calls | After 128 calls | Difference |
| --- | ---: | ---: | ---: |
| Original mallinfo2 observer | 163168 | 164224 | 1056 bytes |
| Decoded thread-cache chunk totals | 11024 | 12080 | 1056 bytes |
| Bin 31, 528-byte freed chunks | 3 | 5 | 2 chunks |

I retain all 128 calls, the original 1024-byte criterion, the ten original source shadows, identity/carrier guards and the 2101904-byte positive control. The inferior still exits **94**. The debugger exits 0 because it completed its observation; those statuses have different meanings.

Two earlier debugger setup attempts stop at the first sample because glibc's static TLS symbol is optimized out. They never complete the workload and claim no result. I preserve their scripts/streams/receipts. My final observer resolves the TLS slot from the actual AArch64 malloc instruction sequence and GOT value, asserts those instruction bytes, then reads cache metadata without calling into the inferior. I validate linked-list lengths, decoded pointers, alignment, unique addresses and exact chunk sizes. This calculation is specific to the retained libc image. I archive that libc, debugger, loader/library closure, disassembly, original executable, source fixture, original native driver and scripts/streams.

This observation accounts exactly for **this run's** extra native bytes as two additional freed allocator-cache chunks. It does not establish the cause or exact chunks of the earlier +1776-byte failure, and it does not pass the unchanged ordinary gate. I neither raise the allowance, suppress the failure nor activate default STRING/aggregate ownership. #1213/#1217/#1116 and whole-release acceptance remain open. No new production source, worker concurrency, Darwin or NAS action follows from this diagnostic. All diagnostic processes are terminal.

```sh
python3 docs/evidence/carrier-tcache-snapshot-linux-20261123/read_archive.py
```

I hash-bind the prior manifest and its exact original executable/driver. My standalone reader verifies every raw/compressed record and size, exact Git documentation, original fixture identity, both incomplete setups, complete final samples, positive control, both distinct exit statuses, actual malloc instructions and explicit nonacceptance flags. My record-loop/name implementation and its 1306-shadow compiler source checkpoint remain separately published in PR1168 at `e49d3c311ca051a639dcf53c814e2aa2cc5f38d6`.
