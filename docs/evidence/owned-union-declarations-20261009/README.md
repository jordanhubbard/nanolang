# My resource-union declaration checkpoint

I extend version-3 ownership declaration validation under #981 to retain complete
union layouts with exact earlier record or union payload layouts. A resource
child requires a resource parent. Unknown child authority, mismatched nested
kinds, incomplete flags, malformed variant slices and legacy-version misuse
remain refused. Zero-flag unions retain their scalar-only contract. No new
opcode or unchecked source admission is introduced.

My new fixture describes Handle, Box<Handle> and Box<Box<Handle>>, including
empty variants. It checks resource classification, exact variant ranges,
binary and canonical assembly byte preservation, malformed declarations and
unchanged query outputs. Public verified assembly, affine state creation,
verification and native emission refuse the as-yet unsupported executable
contract. Unverified assembly is used only for the metadata round-trip test;
its result is explicitly required to fail verification.

My first build found an omitted declaration-projection reader argument. The
next test correctly reached the verified assembler's refusal; I retained that
assertion and added a separate unverified transport round trip. The private
mixed-declaration suite then caught a broader COMPLETE-union admission than
its existing contract. I preserved that profile and its original assertions.
All failures are retained here.

Final ownership transport and private declaration gates pass. Adjacent checks
pass: 314 affine-state checks and 346 allocation checks, affine bytecode and
allocation checks, and the existing sanitized scalar-union VM/native runtime
case. The private suite passes its 10,960 mixed-declaration checks and all 13
allocation failure positions. This is Darwin metadata qualification, not full
ownership execution, source acceptance, or Linux qualification.

Next I must implement selected union ownership in affine state and stack/CFG
verification; checked construction, transfers, projections and joins; paired
source lowering; and VM/native destruction. I retain every selected/generic
source acceptance fixture and the C-seed resource-payload failure. This
checkpoint does not close #981 or the full 5.1 ownership requirement.
