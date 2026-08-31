# NanoLang developer overview

## Purpose

Explain NanoLang to software developers and compiler engineers. Show the
language contract, the compiler pipeline, NanoISA, NanoVM, tests, diagnostics,
and the boundary between what I ship and what remains 4.0 work.

## Narrative stance

I speak in the first person. I describe current code and tested behavior. I
label roadmap work as future work. I do not make productivity, speed, or
adoption claims that my repository cannot support.

## Core message

I make intent explicit in syntax, carry it into inspectable NanoISA bytecode,
and require executable evidence before I call a change complete. My C
transpiler, NanoVM, FFI boundary, and formal core are different trust surfaces;
the deck must show where each one begins and ends.

## Slide sequence

1. I am NanoLang.
2. My design refuses ambiguity.
3. One source language, two execution paths.
4. The compiler pipeline from `.nano` to C or NanoISA.
5. NanoISA is readable bytecode, not a hidden intermediate.
6. NanoVM keeps decoded instructions and typed runtime values.
7. Every function carries a shadow test.
8. FFI is an explicit unsafe boundary and can be isolated.
9. Diagnostics are optional and cheap when disabled.
10. What I measured for the 3.5 release.
11. What 3.5 shipped and what 4.0 still requires.
12. Start with the code, then run the gates.

## Visual direction

Use the NanoLang mascot from `assets/nanolang-mascot.png` on the cover and
closing slide. Use native shapes for compiler and VM flows. Use dark graphite,
warm white, NanoLang green, and orange diagnostic paths. Keep diagrams sparse,
and leave space for the mascot rather than importing unrelated project imagery.

Every slide carries speaker notes with source paths and limitations.
