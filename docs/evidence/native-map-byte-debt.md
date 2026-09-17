# My native map byte-debt batching

MAC `task_c7931f1c22db473682d077b119b9c87d` follows the preserved full compiler timeout. I changed a measured local cost, without rerunning full compilation or attributing the whole timeout to collection.

## My contract

I account retained owned map storage: headers, bucket capacity, entries, keys, copied result strings and ownership nodes. Checked additions reject live/debt counter overflow; checked release rejects underflow. Successful positive bucket growth contributes its net capacity increase. Replacement/delete/destroy release accounted bytes. I keep the existing published-root safe points, full graph closure, alias rules, return handoff and forced collection.

At an eligible safe point, I collect once map allocation debt reaches a 64 KiB minimum budget. After collection, the budget becomes the larger of that floor and surviving map bytes. Dropped objects can wait for new debt or forced collection. A large allocation and allocations between safe points can exceed the budget; it is a trigger, not a hard allocation limit. Reachable application data is not bounded by this policy.

My counters do not include allocator overhead or the temporary simultaneous old and replacement bucket arrays during growth. Consequently, reported peaks are retained owned storage, not instantaneous allocation or RSS bounds. Net-capacity accounting preserves the existing checked resize and introduces no collection inside allocation.

## My measured ordinary workload

Both versions translate identical bytecode SHA-256 `e8c2629f03c85b67115586bf226248e9b4eec0759285dd73d4b05ee8064cfde5`. An ordinary generated-runtime harness retains 10,000 scalar records and a string map, performs 5,000 successful copied reads, verifies values and final cleanup, and runs at `-O0`:

| Runtime | Collections | Root visits | CPU seconds |
| --- | ---: | ---: | ---: |
| Prior per-allocation trigger | 5000 | 50015000 | 3.609566 |
| Byte-debt trigger | 3 | 30009 | 0.002529 |

A separate sanitizer control also retains an escaped map-result string. It performs three scans and 30,012 visits, with a 65,838-byte retained-storage peak and zero final owners/bytes. The harness evidence is `/tmp/nanolang-native-map-byte-{before,after}/`; this small ordinary workload is not a rerun or reproduction of the full compiler failure.

## My validation boundary

Thirty-two lifetime/semantic regression methods passed in 43.607 seconds. Five final focused methods passed in 5.716 seconds, including three new map-byte methods with ASan, UBSan and LSan. They cover the large live graph, copied aliases, map growth, entry replacement/delete, forced cleanup, and checked counter overflow/underflow. Existing caller/global/operand/record/array roots, mutable edges, returns and tail calls remain checked.

The old map tests required immediate collection after every allocation and tiny owner-count peaks. I retain those initial failures in `/tmp/nanolang-map-byte-original-assertions.log`; I replaced scheduling-specific assertions with the new explicit byte contract, not only an exit-leak check. Ordinary small-root churn must stay below 69,632 retained bytes (64 KiB budget plus 4 KiB for retained fixture roots and one allocation event), and both owner count and retained bytes must reach zero at exit. Forced-collection tests still require exact scans and immediate release; allocation-free loops remain scan-free. An initial test-edit mistake called forced collection at the wrong checkpoint; its failure remains in `/tmp/nanolang-map-byte-focused.log` and the corrected intended checkpoint passes.

The first native translator run retained four strict-C failures because a collector-only budget variable was initially declared in the standalone map-runtime fragment. I moved it into the emitter's collector scope; I did not suppress warnings. The original and final logs remain `/tmp/nanolang-map-byte-nvm2c{,-final}.log`.

Full native self-compilation and raw fixed-point acceptance remain open. No source or binary from the preserved timed-out attempt was changed by this repair.

After restacking onto `c3287dba`, I rebuilt the tools and passed all 15 map/debt/lifetime methods in 35.743 seconds. The source merge was clean; I retained additive roadmap entries.

The corrected native translator gate passed 2,412 checks with zero failures, alongside 1,092 shape checks and the opcode/sanitizer-driver checks. The final overflow-control rerun requires the intended SIGABRT for each checked overflow/underflow boundary and passed.
