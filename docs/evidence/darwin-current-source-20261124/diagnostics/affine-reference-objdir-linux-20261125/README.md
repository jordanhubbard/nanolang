I repair #1258 in `3408c58c65`: my four affine frame/scalar/local/global reference recipes now exclude `$(OBJ_DIR)/nanoisa/affine_bytecode.o`. Each oracle embeds that production source, so retaining the ordinary object produces duplicate definitions. Default object selection remains identical; a private object directory now excludes the same implementation correctly. I change no other input or flag.

I retain both a duplicate-link control and the failing actual private-directory Make link. All four corrected actual Make links then pass at the original strict-C99/O3/PIC flags. I mark the retained prerequisite objects as assumed old and verify their bytes before and after: these controls exercise the actual recipes without rebuilding or editing frozen inputs. Each successful Make command emits exactly one compiler command and has empty stderr.

All693 report commands pass: the original1001 complete reports agree across all four linked references, and the global reference preserves all3688 new reports from my separate actual-C audit. Independent reader PASS7686 records binds both Makefile states to Git, verifies exactly four substitutions, complete outputs, command budgets, default/private selection and unchanged source/compiler/object hashes.

This establishes the scoped reference-link repair. I do not claim a fresh complete custom-directory build, private-directory support for older allocation targets with literal output paths, NanoLang execution, paired qualification or release acceptance. My frozen exactb430 component run remains separate and unchanged.

I verify my archive with `python3 read_archive.py`.
