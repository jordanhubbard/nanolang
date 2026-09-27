	.arch armv8-a
	.file	"value_return_probe.c"
	.text
.Ltext0:
	.file 0 "/work" "/tmp/value_return_probe.c"
	.global	__asan_stack_malloc_1
	.section	.rodata
	.align	3
.LC0:
	.string	"1 32 32 3 v:3"
	.text
	.align	2
	.global	named_void
	.type	named_void, %function
named_void:
.LASANPC21:
.LFB21:
	.file 1 "/tmp/value_return_probe.c"
	.loc 1 2 24
	.cfi_startproc
	stp	x29, x30, [sp, -144]!
	.cfi_def_cfa_offset 144
	.cfi_offset 29, -144
	.cfi_offset 30, -136
	mov	x29, sp
	stp	x19, x20, [sp, 16]
	str	x21, [sp, 32]
	.cfi_offset 19, -128
	.cfi_offset 20, -120
	.cfi_offset 21, -112
	mov	x21, x8
	add	x19, sp, 48
	mov	x20, x19
	adrp	x0, :got:__asan_option_detect_stack_use_after_return
	ldr	x0, [x0, :got_lo12:__asan_option_detect_stack_use_after_return]
	ldr	w0, [x0]
	cmp	w0, 0
	beq	.L1
	mov	x0, 96
	bl	__asan_stack_malloc_1
	cmp	x0, 0
	beq	.L1
	mov	x19, x0
.L1:
	add	x0, x19, 96
	mov	x1, 35507
	movk	x1, 0x41b5, lsl 16
	str	x1, [x19]
	adrp	x1, .LC0
	add	x1, x1, :lo12:.LC0
	str	x1, [x19, 8]
	adrp	x1, .LASANPC21
	add	x1, x1, :lo12:.LASANPC21
	str	x1, [x19, 16]
	lsr	x1, x19, 3
	mov	x2, 68719476736
	add	x2, x1, x2
	mov	w3, -235802127
	str	w3, [x2]
	mov	w3, -202116109
	str	w3, [x2, 8]
	.loc 1 4 11
	sub	x2, x0, #64
	mov	w3, 12
	str	w3, [x2]
	.loc 1 4 32
	sub	x2, x0, #64
	strb	wzr, [x2, 4]
	.loc 1 4 49
	sub	x2, x0, #64
	strb	wzr, [x2, 5]
	.loc 1 4 69
	sub	x2, x0, #64
	strb	wzr, [x2, 6]
	.loc 1 5 12
	sub	x2, x0, #64
	mov	x0, x21
	ldp	q0, q1, [x2]
	stp	q0, q1, [x0]
	.loc 1 2 24
	cmp	x20, x19
	beq	.L2
	mov	x0, 13838
	movk	x0, 0x45e0, lsl 16
	str	x0, [x19]
	mov	x0, 68719476736
	add	x0, x1, x0
	movi	v0.16b, 0xfffffffffffffff5
	fmov	x1, d0
	str	x1, [x0]
	fmov	w1, s0
	str	w1, [x0, 8]
	b	.L3
.L2:
	mov	x0, 68719476736
	add	x0, x1, x0
	str	wzr, [x0]
	add	x0, x0, 8
	str	wzr, [x0]
.L3:
	.loc 1 6 1
	ldp	x19, x20, [sp, 16]
	ldr	x21, [sp, 32]
	ldp	x29, x30, [sp], 144
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 21
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE21:
	.size	named_void, .-named_void
	.align	2
	.global	literal_void
	.type	literal_void, %function
literal_void:
.LASANPC22:
.LFB22:
	.loc 1 7 26
	.cfi_startproc
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	mov	x0, x8
	.loc 1 7 35
	stp	xzr, xzr, [x0]
	stp	xzr, xzr, [x0, 16]
	mov	w1, 12
	str	w1, [x0]
	.loc 1 7 60
	add	sp, sp, 32
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE22:
	.size	literal_void, .-literal_void
	.section	.rodata
	.align	3
.LC1:
	.string	"1 32 32 3 v:9"
	.text
	.align	2
	.global	named_int
	.type	named_int, %function
named_int:
.LASANPC23:
.LFB23:
	.loc 1 8 28
	.cfi_startproc
	stp	x29, x30, [sp, -160]!
	.cfi_def_cfa_offset 160
	.cfi_offset 29, -160
	.cfi_offset 30, -152
	mov	x29, sp
	stp	x19, x20, [sp, 16]
	str	x21, [sp, 32]
	.cfi_offset 19, -144
	.cfi_offset 20, -136
	.cfi_offset 21, -128
	mov	x21, x8
	str	x0, [sp, 56]
	add	x19, sp, 64
	mov	x20, x19
	adrp	x0, :got:__asan_option_detect_stack_use_after_return
	ldr	x0, [x0, :got_lo12:__asan_option_detect_stack_use_after_return]
	ldr	w0, [x0]
	cmp	w0, 0
	beq	.L8
	mov	x0, 96
	bl	__asan_stack_malloc_1
	cmp	x0, 0
	beq	.L8
	mov	x19, x0
.L8:
	add	x0, x19, 96
	mov	x1, 35507
	movk	x1, 0x41b5, lsl 16
	str	x1, [x19]
	adrp	x1, .LC1
	add	x1, x1, :lo12:.LC1
	str	x1, [x19, 8]
	adrp	x1, .LASANPC23
	add	x1, x1, :lo12:.LASANPC23
	str	x1, [x19, 16]
	lsr	x1, x19, 3
	mov	x2, 68719476736
	add	x2, x1, x2
	mov	w3, -235802127
	str	w3, [x2]
	mov	w3, -202116109
	str	w3, [x2, 8]
	.loc 1 10 11
	sub	x2, x0, #64
	str	wzr, [x2]
	.loc 1 10 31
	sub	x2, x0, #64
	strb	wzr, [x2, 4]
	.loc 1 10 48
	sub	x2, x0, #64
	strb	wzr, [x2, 5]
	.loc 1 10 68
	sub	x2, x0, #64
	strb	wzr, [x2, 6]
	.loc 1 10 87
	sub	x2, x0, #64
	ldr	x3, [sp, 56]
	str	x3, [x2, 16]
	.loc 1 11 12
	sub	x2, x0, #64
	mov	x0, x21
	ldp	q0, q1, [x2]
	stp	q0, q1, [x0]
	.loc 1 8 28
	cmp	x20, x19
	beq	.L9
	mov	x0, 13838
	movk	x0, 0x45e0, lsl 16
	str	x0, [x19]
	mov	x0, 68719476736
	add	x0, x1, x0
	movi	v0.16b, 0xfffffffffffffff5
	fmov	x1, d0
	str	x1, [x0]
	fmov	w1, s0
	str	w1, [x0, 8]
	b	.L10
.L9:
	mov	x0, 68719476736
	add	x0, x1, x0
	str	wzr, [x0]
	add	x0, x0, 8
	str	wzr, [x0]
.L10:
	.loc 1 12 1
	ldp	x19, x20, [sp, 16]
	ldr	x21, [sp, 32]
	ldp	x29, x30, [sp], 160
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 21
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE23:
	.size	named_int, .-named_int
	.align	2
	.global	literal_int
	.type	literal_int, %function
literal_int:
.LASANPC24:
.LFB24:
	.loc 1 13 30
	.cfi_startproc
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	mov	x1, x8
	str	x0, [sp, 8]
	.loc 1 13 39
	stp	xzr, xzr, [x1]
	stp	xzr, xzr, [x1, 16]
	ldr	x0, [sp, 8]
	str	x0, [x1, 16]
	.loc 1 13 77
	add	sp, sp, 48
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE24:
	.size	literal_int, .-literal_int
	.align	2
	.type	_sub_I_00099_0, %function
_sub_I_00099_0:
.LFB25:
	.cfi_startproc
	.loc 1 13 1
	stp	x29, x30, [sp, -16]!
	.cfi_def_cfa_offset 16
	.cfi_offset 29, -16
	.cfi_offset 30, -8
	mov	x29, sp
	bl	__asan_init
	bl	__asan_version_mismatch_check_v8
	ldp	x29, x30, [sp], 16
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE25:
	.size	_sub_I_00099_0, .-_sub_I_00099_0
	.section	.init_array.00099,"aw"
	.align	3
	.xword	_sub_I_00099_0
	.text
.Letext0:
	.file 2 "/usr/include/aarch64-linux-gnu/bits/types.h"
	.file 3 "/usr/include/aarch64-linux-gnu/bits/stdint-intn.h"
	.file 4 "/usr/include/aarch64-linux-gnu/bits/stdint-uintn.h"
	.file 5 "src/runtime/dyn_array.h"
	.file 6 "src/nanolang.h"
	.file 7 "src/runtime/gc_struct.h"
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.4byte	0x77c
	.2byte	0x5
	.byte	0x1
	.byte	0x8
	.4byte	.Ldebug_abbrev0
	.uleb128 0xc
	.4byte	.LASF137
	.byte	0xc
	.4byte	.LASF0
	.4byte	.LASF1
	.8byte	.Ltext0
	.8byte	.Letext0-.Ltext0
	.4byte	.Ldebug_line0
	.uleb128 0x5
	.byte	0x8
	.byte	0x7
	.4byte	.LASF2
	.uleb128 0xd
	.byte	0x8
	.uleb128 0xe
	.byte	0x4
	.byte	0x5
	.string	"int"
	.uleb128 0x5
	.byte	0x1
	.byte	0x8
	.4byte	.LASF3
	.uleb128 0x5
	.byte	0x2
	.byte	0x7
	.4byte	.LASF4
	.uleb128 0x5
	.byte	0x4
	.byte	0x7
	.4byte	.LASF5
	.uleb128 0x5
	.byte	0x1
	.byte	0x6
	.4byte	.LASF6
	.uleb128 0x4
	.4byte	.LASF8
	.byte	0x2
	.byte	0x26
	.byte	0x17
	.4byte	0x3e
	.uleb128 0x5
	.byte	0x2
	.byte	0x5
	.4byte	.LASF7
	.uleb128 0x4
	.4byte	.LASF9
	.byte	0x2
	.byte	0x2c
	.byte	0x19
	.4byte	0x79
	.uleb128 0x5
	.byte	0x8
	.byte	0x5
	.4byte	.LASF10
	.uleb128 0x3
	.4byte	0x85
	.uleb128 0x5
	.byte	0x1
	.byte	0x8
	.4byte	.LASF11
	.uleb128 0x5
	.byte	0x8
	.byte	0x5
	.4byte	.LASF12
	.uleb128 0x4
	.4byte	.LASF13
	.byte	0x3
	.byte	0x1b
	.byte	0x13
	.4byte	0x6d
	.uleb128 0x5
	.byte	0x8
	.byte	0x7
	.4byte	.LASF14
	.uleb128 0x3
	.4byte	0xab
	.uleb128 0xf
	.uleb128 0x4
	.4byte	.LASF15
	.byte	0x4
	.byte	0x18
	.byte	0x13
	.4byte	0x5a
	.uleb128 0x8
	.4byte	0x4c
	.byte	0x5
	.byte	0x13
	.4byte	0xf4
	.uleb128 0x2
	.4byte	.LASF16
	.byte	0x1
	.uleb128 0x2
	.4byte	.LASF17
	.byte	0x8
	.uleb128 0x2
	.4byte	.LASF18
	.byte	0x2
	.uleb128 0x2
	.4byte	.LASF19
	.byte	0x3
	.uleb128 0x2
	.4byte	.LASF20
	.byte	0x4
	.uleb128 0x2
	.4byte	.LASF21
	.byte	0x5
	.uleb128 0x2
	.4byte	.LASF22
	.byte	0x6
	.uleb128 0x2
	.4byte	.LASF23
	.byte	0x7
	.byte	0
	.uleb128 0x4
	.4byte	.LASF24
	.byte	0x5
	.byte	0x1c
	.byte	0x3
	.4byte	0xb8
	.uleb128 0x7
	.byte	0x20
	.byte	0x5
	.byte	0x1f
	.4byte	0x14a
	.uleb128 0x1
	.4byte	.LASF25
	.byte	0x5
	.byte	0x20
	.byte	0xd
	.4byte	0x93
	.byte	0
	.uleb128 0x1
	.4byte	.LASF26
	.byte	0x5
	.byte	0x21
	.byte	0xd
	.4byte	0x93
	.byte	0x8
	.uleb128 0x1
	.4byte	.LASF27
	.byte	0x5
	.byte	0x22
	.byte	0x11
	.4byte	0xf4
	.byte	0x10
	.uleb128 0x1
	.4byte	.LASF28
	.byte	0x5
	.byte	0x23
	.byte	0xd
	.4byte	0xac
	.byte	0x14
	.uleb128 0x1
	.4byte	.LASF29
	.byte	0x5
	.byte	0x24
	.byte	0xb
	.4byte	0x35
	.byte	0x18
	.byte	0
	.uleb128 0x4
	.4byte	.LASF30
	.byte	0x5
	.byte	0x25
	.byte	0x3
	.4byte	0x100
	.uleb128 0x5
	.byte	0x1
	.byte	0x2
	.4byte	.LASF31
	.uleb128 0x3
	.4byte	0x14a
	.uleb128 0x3
	.4byte	0x80
	.uleb128 0x4
	.4byte	.LASF32
	.byte	0x6
	.byte	0x1e
	.byte	0x16
	.4byte	0x173
	.uleb128 0x9
	.4byte	.LASF32
	.byte	0x20
	.byte	0x9e
	.byte	0x8
	.4byte	0x1cd
	.uleb128 0x1
	.4byte	.LASF33
	.byte	0x6
	.byte	0x9f
	.byte	0xf
	.4byte	0x294
	.byte	0
	.uleb128 0x1
	.4byte	.LASF34
	.byte	0x6
	.byte	0xa0
	.byte	0xa
	.4byte	0x156
	.byte	0x4
	.uleb128 0x1
	.4byte	.LASF35
	.byte	0x6
	.byte	0xa1
	.byte	0xa
	.4byte	0x156
	.byte	0x5
	.uleb128 0x1
	.4byte	.LASF36
	.byte	0x6
	.byte	0xa2
	.byte	0xa
	.4byte	0x156
	.byte	0x6
	.uleb128 0x1
	.4byte	.LASF37
	.byte	0x6
	.byte	0xa3
	.byte	0x11
	.4byte	0xa6
	.byte	0x8
	.uleb128 0x10
	.string	"as"
	.byte	0x6
	.byte	0xb3
	.byte	0x7
	.4byte	0x620
	.byte	0x10
	.byte	0
	.uleb128 0x7
	.byte	0x30
	.byte	0x7
	.byte	0xf
	.4byte	0x224
	.uleb128 0x1
	.4byte	.LASF38
	.byte	0x7
	.byte	0x10
	.byte	0xb
	.4byte	0x80
	.byte	0
	.uleb128 0x1
	.4byte	.LASF39
	.byte	0x7
	.byte	0x11
	.byte	0x9
	.4byte	0x37
	.byte	0x8
	.uleb128 0x1
	.4byte	.LASF40
	.byte	0x7
	.byte	0x12
	.byte	0xc
	.4byte	0x162
	.byte	0x10
	.uleb128 0x1
	.4byte	.LASF41
	.byte	0x7
	.byte	0x13
	.byte	0xc
	.4byte	0x224
	.byte	0x18
	.uleb128 0x1
	.4byte	.LASF42
	.byte	0x7
	.byte	0x14
	.byte	0xe
	.4byte	0x229
	.byte	0x20
	.uleb128 0x1
	.4byte	.LASF43
	.byte	0x7
	.byte	0x15
	.byte	0xe
	.4byte	0x229
	.byte	0x28
	.byte	0
	.uleb128 0x3
	.4byte	0x35
	.uleb128 0x3
	.4byte	0xac
	.uleb128 0x4
	.4byte	.LASF44
	.byte	0x7
	.byte	0x16
	.byte	0x3
	.4byte	0x1cd
	.uleb128 0x8
	.4byte	0x4c
	.byte	0x6
	.byte	0x26
	.4byte	0x294
	.uleb128 0x2
	.4byte	.LASF45
	.byte	0
	.uleb128 0x2
	.4byte	.LASF46
	.byte	0x1
	.uleb128 0x2
	.4byte	.LASF47
	.byte	0x2
	.uleb128 0x2
	.4byte	.LASF48
	.byte	0x3
	.uleb128 0x2
	.4byte	.LASF49
	.byte	0x4
	.uleb128 0x2
	.4byte	.LASF50
	.byte	0x5
	.uleb128 0x2
	.4byte	.LASF51
	.byte	0x6
	.uleb128 0x2
	.4byte	.LASF52
	.byte	0x7
	.uleb128 0x2
	.4byte	.LASF53
	.byte	0x8
	.uleb128 0x2
	.4byte	.LASF54
	.byte	0x9
	.uleb128 0x2
	.4byte	.LASF55
	.byte	0xa
	.uleb128 0x2
	.4byte	.LASF56
	.byte	0xb
	.uleb128 0x2
	.4byte	.LASF57
	.byte	0xc
	.byte	0
	.uleb128 0x4
	.4byte	.LASF58
	.byte	0x6
	.byte	0x34
	.byte	0x3
	.4byte	0x23a
	.uleb128 0x7
	.byte	0x18
	.byte	0x6
	.byte	0x37
	.4byte	0x2dd
	.uleb128 0x1
	.4byte	.LASF59
	.byte	0x6
	.byte	0x38
	.byte	0xf
	.4byte	0x294
	.byte	0
	.uleb128 0x1
	.4byte	.LASF25
	.byte	0x6
	.byte	0x39
	.byte	0x9
	.4byte	0x37
	.byte	0x4
	.uleb128 0x1
	.4byte	.LASF26
	.byte	0x6
	.byte	0x3a
	.byte	0x9
	.4byte	0x37
	.byte	0x8
	.uleb128 0x1
	.4byte	.LASF29
	.byte	0x6
	.byte	0x3b
	.byte	0xb
	.4byte	0x35
	.byte	0x10
	.byte	0
	.uleb128 0x4
	.4byte	.LASF60
	.byte	0x6
	.byte	0x3c
	.byte	0x3
	.4byte	0x2a0
	.uleb128 0x7
	.byte	0x20
	.byte	0x6
	.byte	0x3f
	.4byte	0x326
	.uleb128 0x1
	.4byte	.LASF38
	.byte	0x6
	.byte	0x40
	.byte	0xb
	.4byte	0x80
	.byte	0
	.uleb128 0x1
	.4byte	.LASF40
	.byte	0x6
	.byte	0x41
	.byte	0xc
	.4byte	0x162
	.byte	0x8
	.uleb128 0x1
	.4byte	.LASF41
	.byte	0x6
	.byte	0x42
	.byte	0xc
	.4byte	0x326
	.byte	0x10
	.uleb128 0x1
	.4byte	.LASF39
	.byte	0x6
	.byte	0x43
	.byte	0x9
	.4byte	0x37
	.byte	0x18
	.byte	0
	.uleb128 0x3
	.4byte	0x167
	.uleb128 0x4
	.4byte	.LASF61
	.byte	0x6
	.byte	0x44
	.byte	0x3
	.4byte	0x2e9
	.uleb128 0x7
	.byte	0x30
	.byte	0x6
	.byte	0x47
	.4byte	0x38e
	.uleb128 0x1
	.4byte	.LASF62
	.byte	0x6
	.byte	0x48
	.byte	0xb
	.4byte	0x80
	.byte	0
	.uleb128 0x1
	.4byte	.LASF63
	.byte	0x6
	.byte	0x49
	.byte	0x9
	.4byte	0x37
	.byte	0x8
	.uleb128 0x1
	.4byte	.LASF64
	.byte	0x6
	.byte	0x4a
	.byte	0xb
	.4byte	0x80
	.byte	0x10
	.uleb128 0x1
	.4byte	.LASF40
	.byte	0x6
	.byte	0x4b
	.byte	0xc
	.4byte	0x162
	.byte	0x18
	.uleb128 0x1
	.4byte	.LASF41
	.byte	0x6
	.byte	0x4c
	.byte	0xc
	.4byte	0x326
	.byte	0x20
	.uleb128 0x1
	.4byte	.LASF39
	.byte	0x6
	.byte	0x4d
	.byte	0x9
	.4byte	0x37
	.byte	0x28
	.byte	0
	.uleb128 0x4
	.4byte	.LASF65
	.byte	0x6
	.byte	0x4e
	.byte	0x3
	.4byte	0x337
	.uleb128 0x7
	.byte	0x10
	.byte	0x6
	.byte	0x51
	.4byte	0x3bd
	.uleb128 0x1
	.4byte	.LASF66
	.byte	0x6
	.byte	0x52
	.byte	0xc
	.4byte	0x326
	.byte	0
	.uleb128 0x1
	.4byte	.LASF67
	.byte	0x6
	.byte	0x53
	.byte	0x9
	.4byte	0x37
	.byte	0x8
	.byte	0
	.uleb128 0x4
	.4byte	.LASF68
	.byte	0x6
	.byte	0x54
	.byte	0x3
	.4byte	0x39a
	.uleb128 0x8
	.4byte	0x4c
	.byte	0x6
	.byte	0x57
	.4byte	0x465
	.uleb128 0x2
	.4byte	.LASF69
	.byte	0
	.uleb128 0x2
	.4byte	.LASF70
	.byte	0x1
	.uleb128 0x2
	.4byte	.LASF71
	.byte	0x2
	.uleb128 0x2
	.4byte	.LASF72
	.byte	0x3
	.uleb128 0x2
	.4byte	.LASF73
	.byte	0x4
	.uleb128 0x2
	.4byte	.LASF74
	.byte	0x5
	.uleb128 0x2
	.4byte	.LASF75
	.byte	0x6
	.uleb128 0x2
	.4byte	.LASF76
	.byte	0x7
	.uleb128 0x2
	.4byte	.LASF77
	.byte	0x8
	.uleb128 0x2
	.4byte	.LASF78
	.byte	0x9
	.uleb128 0x2
	.4byte	.LASF79
	.byte	0xa
	.uleb128 0x2
	.4byte	.LASF80
	.byte	0xb
	.uleb128 0x2
	.4byte	.LASF81
	.byte	0xc
	.uleb128 0x2
	.4byte	.LASF82
	.byte	0xd
	.uleb128 0x2
	.4byte	.LASF83
	.byte	0xe
	.uleb128 0x2
	.4byte	.LASF84
	.byte	0xf
	.uleb128 0x2
	.4byte	.LASF85
	.byte	0x10
	.uleb128 0x2
	.4byte	.LASF86
	.byte	0x11
	.uleb128 0x2
	.4byte	.LASF87
	.byte	0x12
	.uleb128 0x2
	.4byte	.LASF88
	.byte	0x13
	.uleb128 0x2
	.4byte	.LASF89
	.byte	0x14
	.uleb128 0x2
	.4byte	.LASF90
	.byte	0x15
	.uleb128 0x2
	.4byte	.LASF91
	.byte	0x16
	.uleb128 0x2
	.4byte	.LASF92
	.byte	0x17
	.byte	0
	.uleb128 0x4
	.4byte	.LASF93
	.byte	0x6
	.byte	0x70
	.byte	0x3
	.4byte	0x3c9
	.uleb128 0x9
	.4byte	.LASF94
	.byte	0x90
	.byte	0x73
	.byte	0x10
	.4byte	0x568
	.uleb128 0x1
	.4byte	.LASF95
	.byte	0x6
	.byte	0x74
	.byte	0xa
	.4byte	0x465
	.byte	0
	.uleb128 0x1
	.4byte	.LASF59
	.byte	0x6
	.byte	0x75
	.byte	0x16
	.4byte	0x568
	.byte	0x8
	.uleb128 0x1
	.4byte	.LASF96
	.byte	0x6
	.byte	0x78
	.byte	0xb
	.4byte	0x80
	.byte	0x10
	.uleb128 0x1
	.4byte	.LASF97
	.byte	0x6
	.byte	0x79
	.byte	0x17
	.4byte	0x56d
	.byte	0x18
	.uleb128 0x1
	.4byte	.LASF98
	.byte	0x6
	.byte	0x7a
	.byte	0x9
	.4byte	0x37
	.byte	0x20
	.uleb128 0x1
	.4byte	.LASF99
	.byte	0x6
	.byte	0x7d
	.byte	0xb
	.4byte	0x572
	.byte	0x28
	.uleb128 0x1
	.4byte	.LASF100
	.byte	0x6
	.byte	0x7e
	.byte	0xc
	.4byte	0x162
	.byte	0x30
	.uleb128 0x1
	.4byte	.LASF101
	.byte	0x6
	.byte	0x7f
	.byte	0x9
	.4byte	0x37
	.byte	0x38
	.uleb128 0x1
	.4byte	.LASF102
	.byte	0x6
	.byte	0x82
	.byte	0xb
	.4byte	0x80
	.byte	0x40
	.uleb128 0x1
	.4byte	.LASF103
	.byte	0x6
	.byte	0x85
	.byte	0x1f
	.4byte	0x5ec
	.byte	0x48
	.uleb128 0x1
	.4byte	.LASF104
	.byte	0x6
	.byte	0x8d
	.byte	0xc
	.4byte	0x156
	.byte	0x50
	.uleb128 0x1
	.4byte	.LASF105
	.byte	0x6
	.byte	0x8e
	.byte	0xc
	.4byte	0x80
	.byte	0x58
	.uleb128 0x1
	.4byte	.LASF106
	.byte	0x6
	.byte	0x8f
	.byte	0xc
	.4byte	0x162
	.byte	0x60
	.uleb128 0x1
	.4byte	.LASF107
	.byte	0x6
	.byte	0x90
	.byte	0xc
	.4byte	0x572
	.byte	0x68
	.uleb128 0x1
	.4byte	.LASF108
	.byte	0x6
	.byte	0x91
	.byte	0xc
	.4byte	0x162
	.byte	0x70
	.uleb128 0x1
	.4byte	.LASF109
	.byte	0x6
	.byte	0x92
	.byte	0xc
	.4byte	0x37
	.byte	0x78
	.uleb128 0x1
	.4byte	.LASF110
	.byte	0x6
	.byte	0x99
	.byte	0xc
	.4byte	0x162
	.byte	0x80
	.uleb128 0x1
	.4byte	.LASF111
	.byte	0x6
	.byte	0x9a
	.byte	0xc
	.4byte	0x37
	.byte	0x88
	.byte	0
	.uleb128 0x3
	.4byte	0x471
	.uleb128 0x3
	.4byte	0x568
	.uleb128 0x3
	.4byte	0x465
	.uleb128 0x9
	.4byte	.LASF112
	.byte	0x40
	.byte	0xf0
	.byte	0x10
	.4byte	0x5ec
	.uleb128 0x1
	.4byte	.LASF113
	.byte	0x6
	.byte	0xf1
	.byte	0xb
	.4byte	0x572
	.byte	0
	.uleb128 0x1
	.4byte	.LASF114
	.byte	0x6
	.byte	0xf2
	.byte	0x9
	.4byte	0x37
	.byte	0x8
	.uleb128 0x1
	.4byte	.LASF115
	.byte	0x6
	.byte	0xf3
	.byte	0xc
	.4byte	0x162
	.byte	0x10
	.uleb128 0x1
	.4byte	.LASF116
	.byte	0x6
	.byte	0xf4
	.byte	0x10
	.4byte	0x6c3
	.byte	0x18
	.uleb128 0x1
	.4byte	.LASF117
	.byte	0x6
	.byte	0xf5
	.byte	0xf
	.4byte	0x6c8
	.byte	0x20
	.uleb128 0x1
	.4byte	.LASF118
	.byte	0x6
	.byte	0xf6
	.byte	0xa
	.4byte	0x465
	.byte	0x28
	.uleb128 0x1
	.4byte	.LASF119
	.byte	0x6
	.byte	0xf7
	.byte	0xb
	.4byte	0x80
	.byte	0x30
	.uleb128 0x1
	.4byte	.LASF120
	.byte	0x6
	.byte	0xf8
	.byte	0x1f
	.4byte	0x5ec
	.byte	0x38
	.byte	0
	.uleb128 0x3
	.4byte	0x577
	.uleb128 0x4
	.4byte	.LASF94
	.byte	0x6
	.byte	0x9b
	.byte	0x3
	.4byte	0x471
	.uleb128 0x7
	.byte	0x10
	.byte	0x6
	.byte	0xae
	.4byte	0x620
	.uleb128 0x1
	.4byte	.LASF121
	.byte	0x6
	.byte	0xaf
	.byte	0x13
	.4byte	0x80
	.byte	0
	.uleb128 0x1
	.4byte	.LASF122
	.byte	0x6
	.byte	0xb0
	.byte	0x27
	.4byte	0x5ec
	.byte	0x8
	.byte	0
	.uleb128 0x11
	.byte	0x10
	.byte	0x6
	.byte	0xa4
	.byte	0x5
	.4byte	0x6a3
	.uleb128 0x6
	.4byte	.LASF123
	.byte	0xa5
	.byte	0x13
	.4byte	0x8c
	.uleb128 0x6
	.4byte	.LASF124
	.byte	0xa6
	.byte	0x10
	.4byte	0x6a3
	.uleb128 0x6
	.4byte	.LASF125
	.byte	0xa7
	.byte	0xe
	.4byte	0x156
	.uleb128 0x6
	.4byte	.LASF126
	.byte	0xa8
	.byte	0xf
	.4byte	0x80
	.uleb128 0x6
	.4byte	.LASF127
	.byte	0xa9
	.byte	0x10
	.4byte	0x6aa
	.uleb128 0x6
	.4byte	.LASF128
	.byte	0xaa
	.byte	0x13
	.4byte	0x15d
	.uleb128 0x6
	.4byte	.LASF129
	.byte	0xab
	.byte	0x16
	.4byte	0x6af
	.uleb128 0x6
	.4byte	.LASF130
	.byte	0xac
	.byte	0x13
	.4byte	0x6b4
	.uleb128 0x6
	.4byte	.LASF131
	.byte	0xad
	.byte	0x15
	.4byte	0x6b9
	.uleb128 0x6
	.4byte	.LASF132
	.byte	0xb1
	.byte	0xb
	.4byte	0x5fd
	.uleb128 0x6
	.4byte	.LASF133
	.byte	0xb2
	.byte	0x15
	.4byte	0x6be
	.byte	0
	.uleb128 0x5
	.byte	0x8
	.byte	0x4
	.4byte	.LASF134
	.uleb128 0x3
	.4byte	0x2dd
	.uleb128 0x3
	.4byte	0x32b
	.uleb128 0x3
	.4byte	0x22e
	.uleb128 0x3
	.4byte	0x38e
	.uleb128 0x3
	.4byte	0x3bd
	.uleb128 0x3
	.4byte	0x6c8
	.uleb128 0x3
	.4byte	0x5f1
	.uleb128 0x12
	.4byte	.LASF135
	.byte	0x1
	.byte	0xd
	.byte	0x7
	.4byte	0x167
	.8byte	.LFB24
	.8byte	.LFE24-.LFB24
	.uleb128 0x1
	.byte	0x9c
	.4byte	0x6fc
	.uleb128 0xa
	.string	"n"
	.byte	0xd
	.byte	0x1b
	.4byte	0x93
	.uleb128 0x2
	.byte	0x91
	.sleb128 -40
	.byte	0
	.uleb128 0x13
	.4byte	.LASF136
	.byte	0x1
	.byte	0x8
	.byte	0x7
	.4byte	0x167
	.8byte	.LFB23
	.8byte	.LFE23-.LFB23
	.uleb128 0x1
	.byte	0x9c
	.4byte	0x737
	.uleb128 0xa
	.string	"n"
	.byte	0x8
	.byte	0x19
	.4byte	0x93
	.uleb128 0x3
	.byte	0x91
	.sleb128 -104
	.uleb128 0xb
	.string	"v"
	.byte	0x9
	.4byte	0x167
	.uleb128 0x2
	.byte	0x70
	.sleb128 -64
	.byte	0
	.uleb128 0x14
	.4byte	.LASF138
	.byte	0x1
	.byte	0x7
	.byte	0x7
	.4byte	0x167
	.8byte	.LFB22
	.8byte	.LFE22-.LFB22
	.uleb128 0x1
	.byte	0x9c
	.uleb128 0x15
	.4byte	.LASF139
	.byte	0x1
	.byte	0x2
	.byte	0x7
	.4byte	0x167
	.8byte	.LFB21
	.8byte	.LFE21-.LFB21
	.uleb128 0x1
	.byte	0x9c
	.uleb128 0xb
	.string	"v"
	.byte	0x3
	.4byte	0x167
	.uleb128 0x2
	.byte	0x70
	.sleb128 -64
	.byte	0
	.byte	0
	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev0:
	.uleb128 0x1
	.uleb128 0xd
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x38
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x2
	.uleb128 0x28
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x1c
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x3
	.uleb128 0xf
	.byte	0
	.uleb128 0xb
	.uleb128 0x21
	.sleb128 8
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x4
	.uleb128 0x16
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x5
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0xe
	.byte	0
	.byte	0
	.uleb128 0x6
	.uleb128 0xd
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 6
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x7
	.uleb128 0x13
	.byte	0x1
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0x21
	.sleb128 9
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x8
	.uleb128 0x4
	.byte	0x1
	.uleb128 0x3e
	.uleb128 0x21
	.sleb128 7
	.uleb128 0xb
	.uleb128 0x21
	.sleb128 4
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0x21
	.sleb128 14
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x9
	.uleb128 0x13
	.byte	0x1
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 6
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0xa
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0xb
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0x21
	.sleb128 11
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0xc
	.uleb128 0x11
	.byte	0x1
	.uleb128 0x25
	.uleb128 0xe
	.uleb128 0x13
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x1f
	.uleb128 0x1b
	.uleb128 0x1f
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x10
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0xd
	.uleb128 0xf
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0xe
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x8
	.byte	0
	.byte	0
	.uleb128 0xf
	.uleb128 0x26
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x10
	.uleb128 0xd
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x38
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x11
	.uleb128 0x17
	.byte	0x1
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x12
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x7a
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x13
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x7c
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x14
	.uleb128 0x2e
	.byte	0
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x7a
	.uleb128 0x19
	.byte	0
	.byte	0
	.uleb128 0x15
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x7c
	.uleb128 0x19
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_aranges,"",@progbits
	.4byte	0x2c
	.2byte	0x2
	.4byte	.Ldebug_info0
	.byte	0x8
	.byte	0
	.2byte	0
	.2byte	0
	.8byte	.Ltext0
	.8byte	.Letext0-.Ltext0
	.8byte	0
	.8byte	0
	.section	.debug_line,"",@progbits
.Ldebug_line0:
	.section	.debug_str,"MS",@progbits,1
.LASF26:
	.string	"capacity"
.LASF74:
	.string	"TYPE_BSTRING"
.LASF62:
	.string	"union_name"
.LASF83:
	.string	"TYPE_LIST_TOKEN"
.LASF37:
	.string	"return_target"
.LASF121:
	.string	"function_name"
.LASF8:
	.string	"__uint8_t"
.LASF80:
	.string	"TYPE_GENERIC"
.LASF17:
	.string	"ELEM_U8"
.LASF42:
	.string	"field_gc_flags"
.LASF64:
	.string	"variant_name"
.LASF33:
	.string	"type"
.LASF136:
	.string	"named_int"
.LASF132:
	.string	"function_val"
.LASF14:
	.string	"long long unsigned int"
.LASF90:
	.string	"TYPE_UNKNOWN"
.LASF9:
	.string	"__int64_t"
.LASF95:
	.string	"base_type"
.LASF103:
	.string	"fn_sig"
.LASF52:
	.string	"VAL_GC_STRUCT"
.LASF19:
	.string	"ELEM_STRING"
.LASF119:
	.string	"return_struct_name"
.LASF60:
	.string	"Array"
.LASF73:
	.string	"TYPE_STRING"
.LASF12:
	.string	"long long int"
.LASF44:
	.string	"GCStruct"
.LASF28:
	.string	"elem_size"
.LASF39:
	.string	"field_count"
.LASF16:
	.string	"ELEM_INT"
.LASF125:
	.string	"bool_val"
.LASF91:
	.string	"TYPE_BORROW_SHARED"
.LASF13:
	.string	"int64_t"
.LASF21:
	.string	"ELEM_ARRAY"
.LASF10:
	.string	"long int"
.LASF51:
	.string	"VAL_STRUCT"
.LASF101:
	.string	"tuple_element_count"
.LASF98:
	.string	"type_param_count"
.LASF5:
	.string	"unsigned int"
.LASF110:
	.string	"type_var_names"
.LASF134:
	.string	"double"
.LASF59:
	.string	"element_type"
.LASF89:
	.string	"TYPE_OPEN_RECORD"
.LASF87:
	.string	"TYPE_TUPLE"
.LASF96:
	.string	"generic_name"
.LASF75:
	.string	"TYPE_VOID"
.LASF93:
	.string	"Type"
.LASF72:
	.string	"TYPE_BOOL"
.LASF122:
	.string	"signature"
.LASF56:
	.string	"VAL_COROUTINE"
.LASF100:
	.string	"tuple_type_names"
.LASF63:
	.string	"variant_index"
.LASF137:
	.string	"GNU C99 13.3.0 -funwind-tables -mlittle-endian -mabi=lp64 -g -O0 -std=c99 -fsanitize=address,undefined -fno-omit-frame-pointer -fasynchronous-unwind-tables -fstack-protector-strong -fstack-clash-protection"
.LASF68:
	.string	"TupleValue"
.LASF115:
	.string	"param_struct_names"
.LASF127:
	.string	"array_val"
.LASF79:
	.string	"TYPE_UNION"
.LASF38:
	.string	"struct_name"
.LASF130:
	.string	"gc_struct_val"
.LASF29:
	.string	"data"
.LASF4:
	.string	"short unsigned int"
.LASF45:
	.string	"VAL_INT"
.LASF106:
	.string	"row_field_names"
.LASF84:
	.string	"TYPE_LIST_GENERIC"
.LASF114:
	.string	"param_count"
.LASF69:
	.string	"TYPE_INT"
.LASF43:
	.string	"field_types"
.LASF58:
	.string	"ValueType"
.LASF85:
	.string	"TYPE_HASHMAP"
.LASF92:
	.string	"TYPE_BORROW_MUT"
.LASF49:
	.string	"VAL_ARRAY"
.LASF111:
	.string	"type_var_count"
.LASF94:
	.string	"TypeInfo"
.LASF66:
	.string	"elements"
.LASF20:
	.string	"ELEM_BOOL"
.LASF118:
	.string	"return_type"
.LASF77:
	.string	"TYPE_STRUCT"
.LASF138:
	.string	"literal_void"
.LASF36:
	.string	"is_continue"
.LASF117:
	.string	"return_type_info"
.LASF50:
	.string	"VAL_DYN_ARRAY"
.LASF76:
	.string	"TYPE_ARRAY"
.LASF131:
	.string	"union_val"
.LASF86:
	.string	"TYPE_FUNCTION"
.LASF18:
	.string	"ELEM_FLOAT"
.LASF31:
	.string	"_Bool"
.LASF3:
	.string	"unsigned char"
.LASF139:
	.string	"named_void"
.LASF67:
	.string	"element_count"
.LASF135:
	.string	"literal_int"
.LASF65:
	.string	"UnionValue"
.LASF7:
	.string	"short int"
.LASF70:
	.string	"TYPE_U8"
.LASF108:
	.string	"row_field_type_names"
.LASF109:
	.string	"row_field_count"
.LASF116:
	.string	"param_type_info"
.LASF34:
	.string	"is_return"
.LASF113:
	.string	"param_types"
.LASF120:
	.string	"return_fn_sig"
.LASF126:
	.string	"string_val"
.LASF2:
	.string	"long unsigned int"
.LASF25:
	.string	"length"
.LASF123:
	.string	"int_val"
.LASF11:
	.string	"char"
.LASF97:
	.string	"type_params"
.LASF41:
	.string	"field_values"
.LASF35:
	.string	"is_break"
.LASF24:
	.string	"ElementType"
.LASF48:
	.string	"VAL_STRING"
.LASF15:
	.string	"uint8_t"
.LASF54:
	.string	"VAL_FUNCTION"
.LASF22:
	.string	"ELEM_STRUCT"
.LASF46:
	.string	"VAL_FLOAT"
.LASF128:
	.string	"dyn_array_val"
.LASF61:
	.string	"StructValue"
.LASF129:
	.string	"struct_val"
.LASF99:
	.string	"tuple_types"
.LASF40:
	.string	"field_names"
.LASF88:
	.string	"TYPE_OPAQUE"
.LASF32:
	.string	"Value"
.LASF112:
	.string	"FunctionSignature"
.LASF81:
	.string	"TYPE_LIST_INT"
.LASF27:
	.string	"elem_type"
.LASF104:
	.string	"is_open_row"
.LASF82:
	.string	"TYPE_LIST_STRING"
.LASF102:
	.string	"opaque_type_name"
.LASF124:
	.string	"float_val"
.LASF71:
	.string	"TYPE_FLOAT"
.LASF105:
	.string	"row_var_name"
.LASF78:
	.string	"TYPE_ENUM"
.LASF53:
	.string	"VAL_UNION"
.LASF57:
	.string	"VAL_VOID"
.LASF6:
	.string	"signed char"
.LASF30:
	.string	"DynArray"
.LASF47:
	.string	"VAL_BOOL"
.LASF55:
	.string	"VAL_TUPLE"
.LASF23:
	.string	"ELEM_POINTER"
.LASF133:
	.string	"tuple_val"
.LASF107:
	.string	"row_field_types"
	.section	.debug_line_str,"MS",@progbits,1
.LASF0:
	.string	"/tmp/value_return_probe.c"
.LASF1:
	.string	"/work"
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
