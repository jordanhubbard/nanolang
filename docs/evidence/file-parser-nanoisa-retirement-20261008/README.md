# My File parser fixture through NanoISA

I remove this fixture's legacy `transpiler.nano` dependency. My ordinary-match
controls now call `nanoisa_emit_program_nasm`: accepted zero-field binders emit
programs; omitted payload binders and unknown variants refuse. After each case,
I emit the accepted program twice and require identical, nonempty output. My
parser/checker coverage, malformed syntax, service strings, byte counts,
publisher metadata and five selected source shadows remain asserted.

I build the actual `nsi-file-binding` publisher and use its generated binding
as the fixture input. The C seed, installed Stage1 and installed Stage2 each
compile and execute the resulting fixture successfully, producing exactly
`publisher:1:5:retained`. I retain the generated input, publisher files, tool
hashes, compile logs, commands and outcomes. When the original runner handle
was lost after the C-seed build, I separately executed that product and resumed
only the unattempted Stage1/Stage2 builds. The source hash is checked on resume.

This is fixture qualification through installed compilers. The broader existing
`tests.test_file_service_parser` corpus, including its independent schema
regeneration, exact selected-shadow multisets, C ownership/fault injection and
ordinary-source products, still requires a full run. I have not replaced those
tests or claimed public File service execution. Final candidate qualification
requires fresh compilers at the candidate pin.
