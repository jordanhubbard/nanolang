# My SQLite declaration-priority checkpoint

I skip implicit C header constants when this module explicitly declares a function of the same name. I retain lexical variable priority.

My Linux ARM64 Docker source copy (python:3.12.12-bookworm) builds with GCC warnings as errors and passes all three regression methods: original prepared CRUD with dependency shadows, callable local binding, and noncallable local refusal with previous output preserved in both frontends. My Darwin build passes; all three header-specific methods skip because the importer search paths contain no SQLite header. This does not qualify the triggering condition on Darwin.

I wire this regression into test-units and test-vm-examples. Source hashes pin the checked implementation and harness. The retained nvm2c diagnostic is an outstanding refusal, not a passing native result; #984 tracks typed SQLite artifact support. Header constant value lowering and the separate legacy C macro collision remain recorded in my roadmap. Full candidate platform and bootstrap qualification remains open under #976/#982.
