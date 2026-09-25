# My hosted setter argument-order failure

Run `35974675846`, x64 job `107552245345`, fails the unchanged assertion
`(== order 123)` in `test_setter_evaluates_arguments_once_in_order` for
`nanoc_c`. The retained job log includes the traceback and terminal failure.
The same extracted source compiles and executes successfully with the d48
C seed in my isolated Linux ARM64 checkout. That local control does not
explain or resolve the hosted failure. I retain task
`task_a3a1ffc6da834ed0b385d920f9f97534` and require the unchanged owning gate.

My retained generated call passes the receiver, index and replacement as three C argument expressions without outer sequencing. The ARM64 control passes, but it does not establish their portable evaluation order. I require a compiler repair and hosted reproduction before closing this defect.
