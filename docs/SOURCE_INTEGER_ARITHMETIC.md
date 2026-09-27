# My source integer arithmetic contract

I use 64-bit modular addition, subtraction and multiplication. Division truncates toward zero; a zero divisor returns zero. The minimum integer divided by minus one returns the minimum integer, and its remainder is zero. These are my existing NanoVM results. I select unsigned arithmetic and a range-checked reconstruction of the signed value, so the C implementation never evaluates signed overflow or an out-of-range unsigned-to-signed conversion.

My static route inventory covers scalar prefix evaluation, static integer arrays (pairwise and both broadcast directions), dynamic integer arrays (pairwise and both broadcast directions, including recursive nested dispatch), and the one- and two-parameter optimized integer callback evaluators. Each route will use the same five total scalar helpers. Operand evaluation and loop order remain unchanged. Floating-point and string routes remain separate.

I require independent expected endpoint cases across the actual interpreter routes, including zero divisors and minimum-integer division, with unchanged inputs and caller-owned fixture cleanup. Ordinary and address/undefined/leak sanitizer gates run only after the corrected source and fixture are reviewed. Existing unary-negation qualification and the full final compiler matrix remain required. No old overflowing case is executed to demonstrate the static finding.

This is task `task_0c20ce4e77b742b8bf8cba9b51c56ba0`.
