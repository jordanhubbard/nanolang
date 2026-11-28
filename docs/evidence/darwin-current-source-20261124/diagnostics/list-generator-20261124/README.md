# I remove unused module list generation

My module-object compiler scanned every `List_` substring in emitted C, including plain string literals, and launched `scripts/generate_list.sh` into global `/tmp`. I ignored its status. The module compile command does not include that directory; schema lists use committed runtime headers and other emitted list definitions are inline. I remove only this obsolete block. I preserve compiler arguments, metadata, output publication and capture code.

My red fixture supplies a private generator that records arguments and exits 97 without generating files. The original module object still compiles a literal containing `List_UnusedMarker ` and the actual compiler schema's `List<LexerToken>`, then preserves a prior object when a separate source is rejected. It fails only the final assertion that no generator was launched. The marker retains both spurious literal and schema matches.

My candidate passes the same assertions in normal, sanitizer and production profiles with only `module.o` replaced in independently guarded original provider closures. Original module-rejection rollback cases pass in all three profiles. Sanitizer cases retain `detect_leaks=1:halt_on_error=1` and UBSan halt behavior. A separate default-mode run verifies fixture cleanup. Two comment/output-text corrections distinguish the final fixture from the retained red source; all fixture sources and assertions remain available.

I also preserve setup failures (initial wrong helper casing; missing private source-root link) and actual private **and public** record-list C identity failures under #1248. These are not successful inline-record-list qualification. I preserve every failed product, generated C, command and log. The generator removal does not repair that separate defect or establish a cause for Darwin's timeout.

Run `python3 read_archive.py` from this checkout after integration. I verify 211 file records, compressed/raw hashes, original source deletion boundary, exact candidate source/test/Make bytes, original provider identities and flags, red/green/native results, rollback neighbors, cleanup and the retained record-list failures. Native products and raw journals are compressed in content-addressed chunks. My manifest SHA256 is `96d5f9b7d4b2ea3fe880717c7f52f4cb8dcd697164a1934902d72312f9b6150b`.

Full current-source ordinary/private compiler gates on Linux and Darwin remain required. I keep #1247/#1248 and the complete 5.1 release open; I change no assertion, deadline, compiler pin or old qualification root.
