# My bounded forward-factory cache controls

I reduce #1262 to two small fixtures. Both return a record containing nested integer arrays, write a row and return3. My forward fixture obtains the record from a factory; my inline fixture constructs the same record locally. The factory and exercise functions have useful shadows.

My established C producer (`nano_virt`) compiles both fixtures; both modules return3 in NanoVM and translate to C. My unchanged genuine Stage2 producer compiles both fixtures and both modules return3 in NanoVM. Only the forward genuine module refuses translation: `I require the matching scalar shape at this array write`. The inline genuine module translates. Both C-seed native builds pass ordinary shadows and return3.

I preserve16 commands with original600-second build/translation limits,60-second executions and default10-second ordinary shadows. All recorded source/tool guards match. I generate C but do not link or execute generated C in this bounded control.

My first diagnostic draft incorrectly requested C-seed `--emit-nvm` for these non-File fixtures. I retain both checked-File-only refusals and correct the observer to the established `nano_virt` producer. I assign no product defect to those invocation errors.

I replay the archive from this checkout with:

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/affine-cache-forward-factory-controls-linux-20261125/read_controls.py
```

My manifest SHA-256 is `90e8f738272ecf7f2ec36730250981d56a46ff5e8ccc5f56800c8aafc8474cd6`, covering86 records. I verify all command outcomes, exact construction-only source difference, VM/native results, guards and retained invocation errors. This control establishes a bounded producer/construction difference. It does not establish a C repair, complete cache qualification, paired-platform behavior, runtime C emission, product cutover or release acceptance. #1262 remains open.
