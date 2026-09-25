# NanoLang developer overview

## Purpose

Explain NanoLang to software developers and compiler engineers. Show the
language contract, the compiler pipeline, NanoISA, NanoVM, tests, diagnostics,
what my verifier checks, and what 4.1–4.5 added on top of that.

**5.1 candidate edition; local until publication is explicitly initiated.** I
retain the historical 4.0 and 4.1–4.5 account while presenting my One IR
implementation, pinned evidence and unfinished release qualification.
4.0 added bytecode verification beyond structural loading.
4.5 keeps that contract and adds Forth Core evidence, NSI, a POSIX capability
fabric, an isolated editor walker, effects-to-policy, and a trap journal.
I am a language with a runtime under development. The deck must not promote those as a
Standard System, GNU Emacs, a kernel, or an internationalized product.

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

For 4.5 the message keeps the 4.0 lesson and adds a second one: a service
fabric on POSIX is not a kernel, a Forth suite that passes is not a Standard
System, six catalog languages are not an internationalized compiler, and a
trap journal library is not a hook on every VM trap.

## Slide sequence

1. I am NanoLang: my 5.1 One IR candidate scope.
2. My design refuses ambiguity.
3. One verified compiler product, several consumers.
4. NanoISA is readable bytecode, not a hidden intermediate.
5. My module format carries the instruction set.
6. My verifier checks modeled invariants before execution.
7. What my verifier used to miss.
8. I treat every module as hostile input.
9. NanoVM dispatches through a label table, and keeps a portable fallback.
10. My policy requires shadows; my compiler has exemptions.
11. FFI is an explicit unsafe boundary and can be isolated.
12. I collect tested reference cycles; this is not complete leak freedom.
13. What I measured, and what I declined because of it.
14. My runtime foundations: contracts, capabilities, fabric, journal.
15. My 5.1 One IR contract, and the boundaries I retain.
16. Start with the code, then run the gates.

Slide 1 must say what I am before it says what I prove. A reader may never
have heard of me, and a reader who has will have met me at 3.5; neither is
served by a cover that assumes the answer. The orientation for the second
reader is one line: my bytecode used to be well-formed, and now it is verified.

Slide 8 exists because slides 6 and 7 are about one layer and the release was
about more than one. A correct verifier still sits on top of a decoder, a
loader, an assembler, a disassembler and a wire protocol, and every one of
those parses input that a hostile module controls. The slide names all six as
tested with malformed inputs, and shows the bounds form -- first bound the
offset, then check `size > total - offset`, never
`offset + size > total` -- because the wrapping version passes exactly the case
it exists to reject. Do not reduce this to a list of test counts: the claim is
that the arithmetic was changed, not that more tests were added around it.

Slide 7 is the one that must not be softened. It shows the six-instruction
program that passed verification, and it names the count: effects declared for
32 of 161 instructions. A deck that presents only the fix teaches less than one
that presents the failure and the fix together.

Slide 14 is the runtime mechanism. Five named layers — source effect, module
requirement, NanoISA trap, NSI method, capability — and a journal that records
at the trap boundary. Do not collapse them into one word. POSIX is the host.

Slide 15 names the 5.1 One IR candidate contract, the verified `d56d15ff6` VM
and standalone-native fixed points and translator boundary. It retains
complete hosted acceptance as a pending gate. Retained callbacks execute through the owner-thread
bridge; they do not establish isolated callback support. Private mixed graphs
and conservative refusals do not become public authority. Slide 7 still shows
the six-instruction program that passed verification; that lesson did not
expire.

Slides 9 and 13 retain the recorded historical noise bands beside measurements.
A measurement without its
spread is the thing the 4.0 benchmark work exists to stop.

Slides 2 and 10 distinguish required project policy from warning/exemption
behavior. Ordinary compilation runs dependency and root shadows in separate
test processes; source-only emission does not execute them. A shadow tests
selected assertions, not every input. Both members use `examples/gcd.nano`,
and acceptance compiles the example extracted from the narrative.
Slide 6 distinguishes abstract reference balance from object ownership and
whole-program proof. Runtime laboratory evidence does not establish production
isolation. Historical test counts and measurements must be labeled as historical.

## Visual direction

Use the NanoLang mascot from `assets/nanolang-mascot.png` on the cover and
closing slide. Use native shapes for compiler and VM flows. Use dark graphite,
warm white, NanoLang green, and orange diagnostic paths. Keep diagrams sparse,
and leave space for the mascot rather than importing unrelated project imagery.

Every slide carries speaker notes with source paths and limitations.
