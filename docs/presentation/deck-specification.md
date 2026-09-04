# NanoLang developer overview

## Purpose

Explain NanoLang to software developers and compiler engineers. Show the
language contract, the compiler pipeline, NanoISA, NanoVM, tests, diagnostics,
and what my 4.0 verifier actually proves.

**4.0 edition.** The 3.5 edition ended at the boundary between measurement and
a verified v2. That boundary is now behind me, so the deck's job changes: it
must show what the verifier proves, it must be honest that the previous
verifier proved less than it appeared to, and it must show the hardening
underneath the verifier rather than presenting the verifier as the whole of my
defence.

The deck is also read by people who have never encountered me. It must
introduce what I am before it argues about what I prove.

## Narrative stance

I speak in the first person. I describe current code and tested behavior. I
label roadmap work as future work. I do not make productivity, speed, or
adoption claims that my repository cannot support.

## Core message

I make intent explicit in syntax, carry it into inspectable NanoISA bytecode,
and require executable evidence before I call a change complete. My C
transpiler, NanoVM, FFI boundary, and formal core are different trust surfaces;
the deck must show where each one begins and ends.

For 4.0 the message sharpens: a check that cannot say what it failed to check
is not a check. My verifier skipped every instruction whose stack effect it did
not know -- and skipped its successors, so it stopped early and still returned
ok. The deck presents that as the central engineering lesson of the release,
because the fix is what makes every other 4.0 claim mean something.

## Slide sequence

1. I am NanoLang 4.0.
2. My design refuses ambiguity.
3. One source language, two execution paths.
4. NanoISA is readable bytecode, not a hidden intermediate.
5. My module format carries the instruction set.
6. My verifier proves a program before it runs.
7. What my verifier used to miss.
8. I treat every module as hostile input.
9. NanoVM dispatches through a label table, and keeps a portable fallback.
10. Every function carries a shadow test.
11. FFI is an explicit unsafe boundary and can be isolated.
12. I collect cycles, so my two backends agree about leaks.
13. What I measured, and what I declined because of it.
14. What 4.0 shipped, and what I have not done.
15. Start with the code, then run the gates.

Slide 1 must say what I am before it says what I prove. A reader may never
have heard of me, and a reader who has will have met me at 3.5; neither is
served by a cover that assumes the answer. The orientation for the second
reader is one line: my bytecode used to be well-formed, and now it is verified.

Slide 8 exists because slides 6 and 7 are about one layer and the release was
about more than one. A correct verifier still sits on top of a decoder, a
loader, an assembler, a disassembler and a wire protocol, and every one of
those parses input that a hostile module controls. The slide names all six as
fuzzed, and shows the bounds form -- `size > total - offset`, never
`offset + size > total` -- because the wrapping version passes exactly the case
it exists to reject. Do not reduce this to a list of test counts: the claim is
that the arithmetic was changed, not that more tests were added around it.

Slide 7 is the one that must not be softened. It shows the six-instruction
program that passed verification, and it names the count: effects declared for
32 of 161 instructions. A deck that presents only the fix teaches less than one
that presents the failure and the fix together.

Slide 12 must show a noise band beside every number. A measurement without its
spread is the thing the 4.0 benchmark work exists to stop.

## Visual direction

Use the NanoLang mascot from `assets/nanolang-mascot.png` on the cover and
closing slide. Use native shapes for compiler and VM flows. Use dark graphite,
warm white, NanoLang green, and orange diagnostic paths. Keep diagrams sparse,
and leave space for the mascot rather than importing unrelated project imagery.

Every slide carries speaker notes with source paths and limitations.
