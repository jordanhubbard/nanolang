# My directed argument checker baseline

I retain failed [run 42479674224](https://github.com/jordanhubbard/nanolang/actions/runs/42479674224) at `ffca5d21374dce1d0ee6fa632f61dff5f2dd308f` under GCC and Clang. Both actual checker-driver builds fail their new production-control shadow: the direct and indirect U8-to-INT cases each violate the three expected acceptance/error-count/diagnostic assertions. I retain the six failures, unchanged inputs and actual build refusal.

No checker executable is published and no native checker run occurs. The success marker printed later by main's shadow does not change that result. I require successful compilation, an actual native product, matching hashes and a separate successful native invocation before accepting a component checkpoint. Focused source execution is skipped after this failed prerequisite.

Both original C parser/typechecker suites, independent aliases and complete translator suites pass. Those results do not qualify the skipped source execution. Both ZIPs pass CRC and have complete inventories and exact deterministic text. Embedded TAR hashes and empty successful inventories independently verify; empty inventories reflect cleanup. Original ZIPs remain private, and direct ZIP custody does not establish original modes or symlinks.
