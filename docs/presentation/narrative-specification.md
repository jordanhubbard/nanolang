# NanoLang developer overview narrative

## Purpose

Give a developer the technical account behind the companion deck. Explain my
syntax, compiler and VM pipeline, runtime boundaries, tests, diagnostics, and
release evidence without turning future roadmap items into current features.

## Heading hierarchy

1. NanoLang: the language, compiler, and VM
   1.1. What I am
   1.2. Who this is for
2. The language contract
   2.1. Explicit syntax and types
   2.2. Shadow tests
   2.3. The formal core and its boundary
3. The compilation paths
   3.1. C transpilation
   3.2. NanoISA generation
   3.3. NanoVM execution
   3.4. Side-table debug data
4. Runtime boundaries
   4.1. Heap values and ownership
   4.2. FFI and the co-process boundary
   4.3. Modules and serialized bytecode
5. Evidence and diagnostics
   5.1. Unit, integration, and shadow tests
   5.2. NanoISA profiles and opcode traces
   5.3. Generated-C profiling
6. Release 4.0
   6.1. What I shipped
   6.2. What the verifier used to miss
   6.3. What I measured
7. Release 4.1–4.5
   7.1. Forth Core evidence
   7.2. Catalogs and guide drafts
   7.3. NSI, capabilities, and POSIX fabric
   7.4. Isolated Nano Emacs walker
   7.5. Effects, policy, journal, and provenance
8. What I have not done
9. How to work on me
   9.1. Read the source and roadmap
   9.2. Run the gates

## Style

Use continuous first-person prose. Quote real NanoLang and NanoISA excerpts.
Every implementation claim cites `source-notes.md`; every forward-looking claim
is labeled as roadmap work.
