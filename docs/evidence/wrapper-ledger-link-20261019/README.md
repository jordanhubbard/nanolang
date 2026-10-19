# Packaged-interpreter ledger linkage

I reproduce the missing ledger provider on candidate `1f06e46ff`: hosted Linux job118149401316 and the current Darwin packaged-interpreter test fail with unresolved checked-borrow symbols. The current Darwin objects also require field-read/write capture symbols.

I add the existing `checked_borrow_facts.o` provider to the explicit packaged-interpreter common-object list beside `resource_flow.o`. I change no runtime admission or publication checks. The daemon client route retains its separate object list.

I rebuild with `make -f Makefile.gnu -j1 CC=/opt/homebrew/opt/llvm/bin/clang nano_virt`, then run `python3 -m unittest -v tests.test_wrapper_packaged_interpreter tests.test_wrapper_publication`: all 12 methods pass in 6.403s. They include actual packaged execution, killed/failed compilation, overlapping publication and prior-output preservation. Independent read-only review finds no further missing provider in these lists.

I retain the failing local terminal, hosted excerpt, corrected build/test terminals and exact provider hashes. This is local Darwin correction evidence; the exact hosted Linux/platform rerun remains pending under #976.
