# Public pure module facts

My parser accepts an optional `pub` modifier followed by `pure fn` or `pure extern fn`. My source facts scanner previously cleared the public flag at `TOKEN_PURE`, so those public functions were missing from export counts and indexed names. I now preserve modifiers through `pure`, as I already do for `resource`.

My scanner shadow checks declaration-order names for public pure, public pure extern and ordinary public functions; private pure functions stay private. It also retains FFI facts, public resource structs, declared module identity and isolation from function-body tokens. My canonical positive fixture checks ordered indexed exports and calls the public pure function through bytecode and native publication.

This is MAC `task_0c5b59e1670049729c5216eadc444213`, stacked PR #573 on the product branch. I retain the task open until actual main integration. The active parent build was not edited, and I neither replayed historical incident artifacts nor attributed incident `dd74` to this independent scanner defect.

At source `7577f505`, `make bootstrap nano_vm nvm2c nvm2c-runtime` completed successfully in the isolated tree. This executes the new scanner shadow during fresh compilation. Three ordinary canonical methods passed through both Stage1 (0.529 seconds) and Stage2 (0.648 seconds): public pure export metadata, all introspection operations with once-only index evaluation, and empty export sets. The adjacent operations tests include strict sanitizer-backed native execution. Logs: `/tmp/nanolang-module-facts-pure-build.log`, `/tmp/nanolang-module-facts-pure-stage1-tests.log`, and `/tmp/nanolang-module-facts-pure-tests.log`. The historical export-shadow case was not selected.
