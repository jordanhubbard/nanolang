# I measure a private interpreter prefix

I retain one diagnostic at production source `221255c7a56b903b5fd34472a9109972230e6ab7`. My current variable and function lookups already use name indexes. Software `perf` recording is denied by the existing host policy, so I make no lookup change from the earlier static lead and change no system setting or package.

I first admit actual GNU gprof on Linux ARM64. I then link a private C seed against guarded original normal providers, the exact current checker, and private `-pg` versions of `eval.c` and `env.c`. A separate observer disables profiling in the parent, starts it at shadow dispatch and dumps after four child CPU seconds. It delegates function lookup unchanged and does not alter the supervisor, selected source, assertions or original ten-second deadline. Baseline and profiling compilers both compile and run the same small source/native control; the profile contains actual interpreter calls in the shadow child.

The single original compiler-source **diagnostic** exits **1 after 125.818 seconds**, inside its smaller 150-second outer bound. It records 1,011 shadow starts and 1,010 completions, with zero completed assertion failures, then the original ten-second supervisor reports signal14 and `parent_timeout=true`. No seed is published. This instrumented observation cannot replace the original 600/default10 qualification.

One retained dump covers **4.001467 child CPU seconds / 4.013999 wall seconds**, with 2,339,840 observed function-lookups. Mapped executable samples total 1.80 seconds:

| Symbol | Mapped self seconds |
| --- | ---: |
| gc_alloc | 0.41 |
| eval_expression_impl | 0.30 |
| env_define_var_impl | 0.21 |
| symbol_lookup | 0.18 |

Only eval/env objects carry gprof call arcs. Shared-library work and profiling overhead remain unassigned to application symbols. The observer changes execution cost and completed-prefix coverage. I claim no benchmark gain, historical timeout cause, complete heap accounting or renewed source/platform/release acceptance.

My separate uninstrumented C API witness measures a fresh record with one STRING field. Binding it makes a distinct record and adds **one managed owner / 46 GC-accounted bytes**. Identifier binding still copies independently; a borrowed binding retains identity. Those bytes are not a whole-process allocation or RSS count. This witness links the earlier qualified894 provider closure; its actual env/GC sources match221255 byte-for-byte. I record and correct an initial postmortem guard comparison against the wrong, newer provider root without changing native bytes or repeating the witness.

Fresh-literal adoption remains proposed and unchecked. Before changing it I require plain/identifier/call/borrowed/control-flow boundaries, nested fields, callbacks, arrays, lifetime checks, original GC allocation-failure assertions and complete original qualification. I change no GC allocator, registry layout, ownership default or compiler pin here.

Run `python3 read_archive.py`. I verify **6,527 file records / 3,522 selected Git sources**, raw/compressed chunks, parent archive bindings, actual observer/compiler/tools, linked provider identities, source/native controls, raw child gmon/report identity, original supervisor failure and the independent allocation witness. For the witness's older providers I read and verify their committed compressed artifacts through the local Git object store. My manifest SHA256 is `e21ff5b5d52ab42f547dc5ec6310f9a1c6cbae6f48a4b9b021de1f9d77354fb8`.

Every native job is terminal. I preserve every earlier root/audit/stash and the NAS/TCC/migration pause. #1229/#1247/#1116/#976 and the complete 5.1 release remain open.
