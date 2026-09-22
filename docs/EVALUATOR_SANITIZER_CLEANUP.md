# My evaluator cleanup contract

My full495e evaluator suite passes ordinarily on Linux and Darwin. The sanitizer suite stops at the terminal-match control and reports20,887 leaked bytes in614 allocations. Its stacks include earlier completed effect and async cases; I do not attribute all of them to the terminal child.

My first correction has two bounded owners. My effect registry owns its vector, definition names, module names, operation vector, operation names and return names, parameter vectors and copied parameter names. Parameter type metadata remains borrowed from the AST and must not be freed by the registry. My async-function and await AST wrappers own their respective child nodes and recursively destroy them. I preserve declaration admission, borrowed type metadata and all evaluator behavior.

Handler bindings, callable results, record temporaries and terminal fixture cleanup require separate lifetime review before execution. I retain existing full-suite assertions and leak detection. These first changes alone are not permission to replay known remaining failures or evidence of full qualification.
