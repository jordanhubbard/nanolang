# My imported-global baseline

At source revision 9ed23b86f, I exercise qualified and selectively renamed global reads through nano_virt and both installed self-hosted stages. Their executable hashes are retained in results.json; the installed stages come from the qualified b2aa10a5a bootstrap.

All three producers compile and execute the private-global function-access control in NanoVM. All three refuse direct qualified and selective reads of that plain global and preserve prior output. I retain those as private-access controls, not as a reason to export private declarations.

The public/ variant declares pub let explicitly. My C frontend rejects that declaration at parsing. Both self-hosted stages accept the declaration and compile the function-access control, but refuse the qualified and selective global reads. I retain all logs; compilation of the public function-access control alone is not runtime evidence.

The implementation must preserve explicit export visibility, declaring-module identity, alias storage and ordered initialization across both frontends and the One IR products. I have not implemented or qualified that contract yet.
