# My calls in a bound program

I resolve ordinary calls through `mb_resolve` with their source owner and local
names. Qualified calls use `mb_lookup` with their source owner and qualified
name. Both call forms share argument lowering, result inference, void handling,
and tail-return selection. I consume the canonical frontend's existing binding
context; I do not change its module binder. My raw-source wrapper clears that
context so an earlier parsed-program invocation cannot redirect its calls.

On Linux ARM64, `make -j8 test-nanoisa-src-nano` passes 86 baseline comparisons
and 13 focused Python cases. My new parsed-program fixture models two source
modules that each call their own `value`, plus root qualified calls, an imported
alias, and qualified void statements. Sixteen bytecode/function checks match a
C-seed reference with explicit bound names. Both modules verify and execute in
NanoVM and strict C11 native output, producing `A` and `B` on separate lines.
Shadows also cover a local name taking precedence and a bound-then-raw invocation
sequence. The separate canonical frontend integration exercises actual module
binding; my focused fixture supplies that context explicitly.

Task `task_39dd3479c5174b299fb8555c6da8b0af` records this fix. Full compiler
emission still needs the recorded scalar conversion and later lowering slices;
this call fix does not establish bootstrap equality.
