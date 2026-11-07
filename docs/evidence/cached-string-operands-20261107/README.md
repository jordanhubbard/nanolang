# My cached STRING operand contract

I track this repair in #1084, under #1081/#1082/#976. My
[existing-provider probe](existing-provider-probe.json) publishes a NanoISA
module for [a known wrong trim operand](trim-int.nano.txt). I retain the actual
provider/input/source hashes and diagnostics. This local provider is not fresh
qualification, and I do not attribute other focused-suite failures to this
single probe without their original logs.

My current environment caches language builtin Function objects with no
parameter metadata, even when my registry declares exact STRING operands.
My C checker therefore skips those type constraints. I now borrow immutable
cached parameter records for those exact STRING positions. Other positions
keep their prior unknown contract; builtins without STRING operands retain
their prior numeric/polymorphic path. I bound the static cache index and fill
only the registry's bounded four-position parameter representation. I leave
my existing arity and unknown-type compatibility rules unchanged.

I also resolve a real current-module body/extern declaration before builtin
cache metadata. My lexical function-value and qualified namespace paths retain
their existing resolution. My tests require real declaration parameter/result
identity rather than treating every same-named call as a builtin.

My six source-level methods retain valid empty/whitespace/UTF-8/nested strings,
lexical binding, a real same-named declaration, a qualified imported wrapper,
known wrong STRING operands and wrong arities. Each selected producer must
preserve prior output after refusal. Successful cases execute through VM and
sanitized C AOT, with the C-native seed counterpart. My ordinary target requires
all four fresh seed/VM/stage providers; a separate explicitly scoped fast
hosted diagnosis selects only the two C frontends and also runs the complete
unchanged canonical string suite. The full bootstrap diagnosis requires all
four providers and still collects the original complete JUnit routes.

My local strict C99 syntax, Python AST, workflow YAML, actual fixture lexer
(six static fragments) and whitespace checks pass. These are source checks;
fresh source/runtime/installed-stage and platform gates remain required. My
focused diagnostic artifacts now publish before long corpus collection so a
failed focused step does not hide its results until the corpus finishes.
I do not remove the original trim refusal or any complete canonical fixture.
