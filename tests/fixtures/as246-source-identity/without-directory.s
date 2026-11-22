# 0 "/home/runner/work/nanolang/nanolang/.test_output/source-capture-diagnostic/0/native-nop-include-step/payload.S"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 0 "<command-line>" 2
# 1 "/home/runner/work/nanolang/nanolang/.test_output/source-capture-diagnostic/0/native-nop-include-step/payload.S"

.include "debug macro.s"
.text
step nop
.data
.globl snapshot_payload
snapshot_payload:
.byte 42
