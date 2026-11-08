# My latest standalone runtime checkpoint

Run 42419811890 succeeds at 7a320e0851bcdb9a8dd023d9159101757914279f on Linux GCC, Linux Clang and macOS 15. Each job executes all four actual-runtime methods under ASan/UBSan, including all scalar pairs, the unchanged original growth/collision fixture, eleven invalid boundaries and 136 allocation-denial prefixes. Successful teardown checks zero live storage. The standalone refusal hook exits 70; this does not replace the production SIGABRT checks in my generated C AOT allocation checkpoint.

I retain all original logs, compiler/worker details and input hashes. This standalone runtime checkpoint does not establish full Darwin source or installed-stage acceptance.
