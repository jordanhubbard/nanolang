# 1 "payload.S"

.include "debug macro.s"
.text
step nop
.data
.globl snapshot_payload
snapshot_payload:
.byte 42
