# 0 "/home/runner/work/nanolang/nanolang/.test_output/source-capture-diagnostic/0/native-nop-include-step/payload.S"
# 1 "/home/runner/work/nanolang/nanolang/.test_output/source-capture-diagnostic/0/native-nop-include-step//"
# 0 "<built-in>"
# 0 "<command-line>"
# 0 "<command-line>" 2
# 1 "/home/runner/work/nanolang/nanolang/.test_output/source-capture-diagnostic/0/native-nop-include-step/payload.S"

.include "debug macro.s"
.text
step nop
.data
.globl snapshot_payload
snapshot_payload:
.byte 42
