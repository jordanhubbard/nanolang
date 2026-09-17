# My standalone VM shadow supervisor

`nano_vm --check-shadows module.nvm` runs my existing verified standalone
execution path under the shared shadow supervisor. The parent accepts a
successful result only after the child returns normally and sends the
completion byte. A foreign `exit(0)` or `_exit(0)` does not send that byte.
The default deadline is ten seconds, with the same validated bounded
`NANO_SHADOW_TIMEOUT_SECONDS` override as my other shadow runners. This
bounds execution time; it does not restrict the child process authority.

I reject daemon, verification-only, repeat, profiling and guest-argument
combinations in this mode. My ordinary standalone execution path remains
available. Shadow output follows the existing supervisor convention and
goes to stderr.

My strict VM and NanoVirt builds pass. Fourteen methods pass in 12.809
seconds: five VM supervision methods, the existing supervisor protocol
method, three verification-only methods and five guest-argument methods.
They check normal completion, nonzero results, assertions, a bounded loop,
invalid deadlines/modules/options, actual foreign early termination,
verification without execution and ordinary guest argument handling.
The retained log is `/tmp/nanolang-vm-shadow-supervision-integrated.log`.

This closes host prerequisite `task_457e55fa59e146878cee92cabc201b6f` after
merge. My canonical self-hosted driver does not select this mode yet; full
shadow-module lowering and driver cutover remain separate acceptance work.
