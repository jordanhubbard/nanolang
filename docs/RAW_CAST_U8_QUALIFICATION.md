# My raw byte conversion checkpoint

I qualify source `565a48d3faa2d68eb49a81809846cbc5ad61ec3d` and the isolated
native-result guard fixture from `8caba839c`. My [seal](evidence/raw-cast-u8/seal.json)
retains nine qualification histories, including every first failure. This closes
only the raw VM prerequisite `task_b9424348ce3643b582627c159621ce2c` after merge;
computed source conversion parent `task_c6b2a040c1434fc784a9d46c02a4981e` remains open.

Both switch and computed-goto dispatches pass 2,742 assertions per executable:
one-byte catalog/encoding, canonical assembly round trip, INT boundary values,
all256 byte identities, definite wrong-tag verification, and actual owned STRING
argument refusal/cleanup followed by valid recovery. Linux GCC/Clang ordinary
and ASan/UBSan plus Darwin Apple ordinary/Homebrew ordinary/ASan/UBSan pass.
Instrumentation covers the rebuilt fixture and VM translation unit; linked
providers retain their inventoried ordinary flags. Leak detection stays enabled.

My unchanged seven scalar-byte neighbors pass on Linux. Darwin passes six;
the seventh originally hardcodes Apple cc and aborts on unsupported leak detection
before its intended invariant. The isolated corrected Git fixture honors my
existing NANO_NATIVE_TEST_CC selector and passes on both hosts. Its SIGABRT,
native diagnostic and no-sanitizer-error assertions remain unchanged. I do not
claim a fresh complete module run at that corrected fixture pin.

Native C, LLVM, Wasm and reconstructed C/Nano reject a verified CAST_U8 program
and preserve existing output sentinels on both hosts. Those implementations,
source destination lowering, the original failing source corpus and full5.1
bootstrap/platform/release gates remain required. No release is authorized by
this checkpoint.

I retain the initial missing-PyYAML Darwin schema terminal, missing LLVM PATH
neighbor preflight, Linux Clang GCC-installation diagnostic and Apple sanitizer
terminal. Corrected runs select installed dependency-capable Python, explicit
GCC13 support for Linux Clang, actual LLVM tools and privately verified Wasmtime43.
The original external refusal text's DROP typo was corrected to POP before it
executed. No product assertion was weakened.

My frozen trees remain `/home/jkh/Src/nanolang-cast-u8-qualification` and
`/tmp/nanolang-cast-u8-565a4` on puck. Initial setup has no obj/bin products;
later phases reuse the inventoried ordinary providers and rebuild each selected
fixture/VM. Source inventories exclude historical docs/evidence; selected tool
inventories expand for final neighbor/refusal runs. Original neighbor temporary
programs were deleted by their existing harness and are not claimed retained.
I retain the new fixture's actual executable/command/output artifacts. Product
maps cover bin/obj/build/tests; they do not establish historical lib/ archive
immutability. Source/tool pairs remain equal through all recorded phases.

Darwin reports and artifacts are preserved in
`/tmp/nanolang-cast-u8-puck-evidence.tar.gz`, SHA256
`a582f8f32dd34cd54900df8734e56df04914f5de275bca4a363cac96076524be`.
The corrected guard's source blob is inventoried separately from frozen providers.
My bounded command runner fails a gate if descendants remain after its first
completion probe; that failed state would not establish completed cleanup.
