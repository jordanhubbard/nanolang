# My derived native payload checkpoint

I derive private native payload C text mechanically from explicit regions in my existing nl_string header and implementation. I change names and linkage, preserve function bodies and use the existing UTF-8 runtime symbol. I do not add an object to my four-object AOT closure.

My first standalone compile refuses five unused private helpers under -Werror. I retain that diagnostic and mark generated private inline helpers as intentionally optional without disabling warnings. The corrected actual standalone program passes all sixteen original string unit controls under ASan/UBSan and warnings-as-errors. I retain build/run commands, complete stdout/stderr, stable inputs and the actual product hash. The local executable remains in the test output; I do not copy its bytes into this receipt.

I have not connected this text to native BSTRING import/owner/container lowering. Hosted qualification and that implementation remain open; this unit result is a payload foundation only.
