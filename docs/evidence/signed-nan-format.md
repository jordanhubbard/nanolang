# My signed nonfinite VM/native formatting checkpoint

I qualify reviewed production8b9594e9 and frozen testsbcbe5ed047bdc9436ec59274fc6e56015744053f.
My shared integer classifier preserves signed NaN/Inf spelling before libc or
floating comparison. My VM string conversion, display and generated native C
use that policy. Managed conversion already follows it and remains unchanged.
Finite libc reference bytes and all2077 corpus values remain checked. Every
nonfinite generated input uses exact integer bits; host `%a` is not its oracle.

Atbcbe5ed0, Linux passes six focused methods in7.395 seconds. Darwin builds in
4.240 seconds and passes five focused methods in12.959 seconds. Those methods
cover exact positive/negative quiet/signaling NaNs, infinities, signed zeros,
extreme finite/subnormal values, actual VM/native/LLVM/Node/Wasmtime conversion,
and direct VM display/helper execution under ASan/UBSan/float-cast-overflow.
The direct C harness runs under C99 and C11. Truncation, zero-capacity formatting,
heap cleanup and unchanged input bits are checked. Darwin uses LLVM23 with
LeakSanitizer enabled. Every selected method runs; none is skipped.

The one Linux-only method is the pre-existing GNU malloc-wrapper control, whose
portable replacement is independently staged in PR739. I do not run the known
incompatible wrapper here. The full71-method Darwin corpus remains required
once this checkpoint and PR739 are integrated; this scoped pass does not erase
the retained failed full-corpus reports on PR739.

My Linux adjacent gate passes274541 VM checks,2422 native checks and1365 shape
checks, including heap/stack/callback allocation recovery. After terminal gates,
I merged canonical PR743, resolved only the roadmap append conflict and rebuilt
tools. At integratedb5644cce28524090f15b51d1920c2882a553a8d3, the six Linux methods
pass in7.592 seconds with unchanged before/after source/tool hashes. Reviewed
formatting production and tests are unchanged by integration. The Darwin
qualification remains assigned to its originalbcbe5ed0 pin.

I independently compare every copied Darwin report with its remote SHA256.
The [sealed reports](signed-nan-format/) preserve exact commands, hashes, tool
inventory and outcomes. I corrected the separate VM print range-before-cast
issue e48563e1f62c4bd89648596c8c55849b; nonfinite/extreme direct VM printing now
passes the explicit float-cast-overflow sanitizer gate on both hosts.

Parent e92a45b66a104e9ba3854cd5f994df8b stays open for legacy interpreter/array/
formatting and paired/public C source routes. Full managed, product and release
acceptance remain open. Product PR522 stays held.
