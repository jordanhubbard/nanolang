# Nano Emacs: a live frame and an isolated walker

I ship a windowed SDL editor in `examples/emacs/nano_emacs.nano`. It is
an Emacs-shaped text frame: panes, a minibuffer, C-x / M-x keys. It is
not GNU Emacs. I do not claim compatibility.

## Two processes

The frame is the SDL client. It never `dlopen`s my interpreter. Eval
goes through `modules/nano_eval/nano_eval_bridge.c`, which is an RPC
client of `bin/nano_emacs_worker`.

The worker is a child process. It links the in-process tree-walker in
`modules/nano_eval/nano_eval.c` and speaks a length-prefixed pipe
protocol (create, destroy, bind buffer, eval string, drain `ed_*`
commands). That protocol is not COP. I do not overload
`COP_MSG_FFI_REQ` as eval.

Build the worker with `make nano_emacs_worker`. Override the path with
`NANO_EMACS_WORKER` if you need to. The child never links SDL and never
re-enters the frame.

`make test-nano-eval` still links `nano_eval.c` directly. That in-process
session is the walker unit test. The live editor does not use it.

## Crash restart

If the walker dies mid-eval, the frame stays up. I echo that the walker
crashed, restart `bin/nano_emacs_worker`, and re-bind the buffers I
still hold. The next C-x C-e talks to the new child.

## freeze-defun

`C-x C-z` and `M-x freeze-defun` extract the top-level `fn` at point,
timeout-compile it with `nano_virt --emit-nvm`, and run `nano_vm` as a
grandchild of the frame. `C-x C-e` stays walker eval.

Frozen v1 is pure: a result or an error in the echo line. I refuse a
fn that calls `ed_*`. A freeze-child crash or timeout does not kill the
frame.

## Tests

- `make test-nano-eval` — in-process walker.
- `make test-nano-emacs-worker` — pipe protocol, kill-mid-eval, freeze
  of a pure fn.
- `make test-nano-emacs` — timeout compile of the editor.

Capability-supervised isolation of the same children is documented in
`docs/NSI_FABRIC.md`. Those 4.4 tests use fabric stand-ins. This
document is the dedicated-pipe cut.
