# My C-header integer lowering checkpoint

I reproduce `undefined variable HEADER_VALUE` while compiling shadows with the unchanged C-seed bytecode compiler at `191a825b8`. My private header exercises the same module metadata import as SQLite without requiring a system SQLite header. I retain that baseline before changing lowering.

I now emit an integer literal for an immutable, global, header-origin integer only after local, global, function and captured-variable resolution. My checker retains locals from other functions, so I search the header-origin symbols rather than letting those unrelated locals mask the fallback. I do not materialize arbitrary environment values.

My corrected Darwin test executes dependency/root shadows, verifies and runs the module in NanoVM, then translates and executes strict C11 with Homebrew Clang ASan/UBSan/LSan. It checks positive, negative and hexadecimal header values, a global initializer, function precedence, local and parameter shadowing, and a captured local. Unknown names and mutation of the imported constant still refuse without replacing prior output. The corrected test adds those controls after the initial baseline. Four real SQLite tests honestly skip because this host lacks sqlite3.h in the compiler's header search directories.

My adjacent `make test-nanovirt test-global-initializer-context` passes 90 bytecode tests and four initializer methods. The source build succeeds with strict warnings. I have not qualified this source change on Linux: Docker socket access is denied in this session. Hosted CI and self-hosted C-header transport remain open under #982/#976. This checkpoint does not establish the correctness of every numeric macro parsed by the existing C-header scanner.
