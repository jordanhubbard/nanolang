# My qualified global assignment checkpoint

I distinguish namespace-qualified global assignment from field mutation only when the receiver has no visible variable binding. Both checkers resolve the imported declaration and retain its mutability and declared type. Both bytecode emitters store into the existing module-owned slot; local and captured receivers keep precedence.

My fourteen-method shared suite passes through the rebuilt C seed in 3.947 seconds and the final freshly built source-compiler component in 7.835 seconds. Successful products execute checked dependency/root shadows, verified NanoVM and strict C11 ASan/UBSan/LSan native products. The new write control observes shared storage through qualified and selective aliases. Immutable/private/missing/wrong-type writes preserve prior output. The suite also retains conflict, purity, capture, wildcard and relative-path controls.

My direct checker shadow tests qualified writes and local receiver precedence. Its first fixture uses the reserved `module` keyword as an identifier, so no assignment AST is produced; I retain that failure in final-compiler.log. Correcting the fixture to `first` allows the full source compiler's shadows to pass. The unchanged suite then passes through this final component. The adjacent gate passes 52 environment assertions, ten lexical-scope methods and the C typechecker suite; 90 bytecode checks pass. The adjacent C gate uses `make -o stage1` after rebuilding C tools and does not establish a new bootstrap.

Installed-stage, Linux/Darwin, full fixed-point, aggregate/callable ownership and broader initialization acceptance remain open under #986. This is not full release qualification.
