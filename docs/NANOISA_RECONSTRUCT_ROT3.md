# My exact scalar ROT3 reconstruction

I record `task_866bb67bb82f43e6ad996e741286189f` before code. For the
existing exact INT/BOOL stack, bottom-to-top `a b c` becomes `c a b`,
as specified by `src/nanovm/vm.c` ROT3. I permute immutable expressions,
not their computations. Calls and local loads retain their snapshots;
pure loop conditions retain existing effect restrictions. I require three
operands and refuse underflow before output publication or execution.
No additional tags, heap ownership or signatures are admitted.

I require fresh small distinct-value/mixed-tag, call/local/loop and
output-preserving underflow controls through VM, native C and both
reconstructed surfaces. I reuse the isolated successful `f38b6409` compiler
copies and retained original/copy host-library hash manifest. This is not
hermetic relocation or fresh current-product bootstrap. Historical carry679
inputs/tools remain excluded. Full reconstruction parent
`task_4bd034f6029b7458201db74e2c3aeb32` stays open.

My first ordinary fixture assembles but native C classification refuses ROT3
(opcode0x0A at function0 offset50). No native binary was produced or executed.
I preserve `/tmp/nanolang-reconstruct-rot3-gcc.log` and record native companion
`task_18ca25c8fc294c81b1a662fd22fc411b` before any native implementation.
