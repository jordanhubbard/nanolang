# Functional-array driver enum leak

I retain CI run 35967867927, job 107530646541, at `e70c0de46`.
Instrumented C-seed driver compilation fails with 10,136 leaked bytes in 77
allocations. `type_check_module` allocates enum variant values twice and loses
the first allocation. The original functional-array gate must pass with both
compiler and generated-program leak detection enabled before I close
`task_b3580edb93ad45c9a2233c6667ba0798`.
