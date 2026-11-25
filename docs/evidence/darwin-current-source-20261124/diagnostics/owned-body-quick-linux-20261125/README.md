# My owned-function scope comparison

I retain source `00025ff76979b4009b84f10f8cb8f07492c62210` against baseline `ebe6386487a9b22075c1e607c1ca6d23794237d3`. I reuse only the private symbol table already created for a defined function. Statement, block and value wrappers still isolate their inputs, and nested scope calls remain unchanged. My statement-checking loop is byte-identical after the helper-name and initial-table substitutions.

My corrected comparison passes 108 commands at the original 600-second build, 60-second runtime and ten-second ordinary shadow limits. The baseline and candidate each pass C-native and VM controls and all 24 complete original diagnostic records. I preserve all 839 prior typechecker assertion lines, all 14 copy sizes and the original fixture bytes outside the added helper and calls. New controls exercise caller aliases and metadata, nested scopes, parameter/global separation, float recursion, direct value checking and enclosing-function state restoration.

My retained C confirms that defined functions call the owned-table core and the existing block helper stays byte-identical. The one remaining core copy call belongs to its unchanged for-loop scope. These are static call-site observations, not a whole-compiler performance measurement.

I preserve the first external fixture refusal: private typechecker globals cannot be read by that importing fixture. I move those checks to a same-module shadow in the corrected source. I also retain the first observer's ambiguous suffix selector, its exact-name correction and the final version that excludes function declarations from call counts.

Run `python3 read_archive.py` here. The reader verifies the archive, Git source, products, complete reports, selected guards and C excerpts. Selected producer host guards are distinct from a complete before/after runtime module-closure observation. Full five-route sanitizer qualification, an updated compiler, the original factory/native/emitter targets, paired qualification, pure translator cutover and full release acceptance remain separate.
