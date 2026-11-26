# My current Stage2 stack windows

I observe my actual qualified Stage2 image `5a537570a4f6708212fd22c7ccedd8be69207e53a11db4e5bc579a443126b1c3` at exact source `7b9f08ee0941c23810d18fa3f8ebfbb759c968de`. I bind its original passing bootstrap `cac73fa60c22d33e45afa3dd8b8ced7220f8b804`, all 3,911 selected source inputs, current compiler closure, image/tool/host identities, fresh local admission and actual debugger and adopted-child cleanup controls.

I collect eight one-second-spaced stacks over 8.000216 seconds after typecheck-start, then eight over 8.000295 seconds after program-lowering-start. Both windows are reached. My debugger exits zero and kills/reaps its inferior at 267.378903 seconds, within my original 300-second outer limit and ordinary ten-second shadow default. No inferior or adopted child remains; all final guards match. My full MI transcript, markers and controls are retained, including an actual nonterminal collector refusal.

In my first eight stacks, seven include `__memcpy_sve`, three include `env_copy_symbols`, four include `env_lookup`, and one includes layout validation. In my second eight, two include `parser_get_struct_def_count` through `nb_record`/`nanoisa_has_borrows`, two include `nisa_call_index`, one includes `nisa_enum_index`, and two include shape/type checking. A path may include several of these names. These counts describe sampled stacks in two selected windows; they are not whole-run cost fractions, a causal diagnosis or a speedup measurement.

My perturbed run reaches typecheck completion at 258.329 seconds and program lowering at 258.352. I make no timing comparison with my separate [300-second phase observation](../lookup-current-stage2-phase-linux-20261126/README.md). My original emitter12 remains failed in setup and has executed zero methods. I claim no source repair, factory/native12 acceptance, paired qualification, pure translator cutover or release acceptance.

Run `python3 read_archive.py` to verify my archived bytes, exact source/image origin, phase ordering, original limits, bounded sample windows, complete transcript, controls, cleanup and guards.
