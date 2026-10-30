	.build_version macos, 14, 0	sdk_version 14, 5
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_nano_build_answer
	.p2align	2	; -- Begin function nano_build_answer
_nano_build_answer:	; @nano_build_answer
Lfunc_begin0:
	.cfi_startproc
	; %bb.0:
	.file	1 "/Users/runner/work/nanolang/nanolang" ".test_output/source-capture-diagnostic/2/answer.c"
	.loc	1 1 37 prologue_end             ; :1:37	; .test_output/source-capture-diagnostic/2/answer.c:1:37
	mov	x0, #42	; =0x2a                 ; =0x2a
	ret
Ltmp0:
Lfunc_end0:
	.cfi_endproc
	; -- End function
	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
Lsection_abbrev0:
	.byte	1
	.byte	17	; Abbreviation Code
	.byte	1	; DW_TAG_compile_unit
	.byte	37	; DW_CHILDREN_yes
	.byte	14	; DW_AT_producer
	.byte	19	; DW_FORM_strp
	.byte	5	; DW_AT_language
	.byte	3	; DW_FORM_data2
	.byte	14	; DW_AT_name
	.ascii	"\202|"	; DW_FORM_strp
	.byte	14	; DW_AT_LLVM_sysroot
	.ascii	"\357\177"	; DW_FORM_strp
	.byte	14	; DW_AT_APPLE_sdk
	.byte	16	; DW_FORM_strp
	.byte	23	; DW_AT_stmt_list
	.byte	27	; DW_FORM_sec_offset
	.byte	14	; DW_AT_comp_dir
	.byte	17	; DW_FORM_strp
	.byte	1	; DW_AT_low_pc
	.byte	18	; DW_FORM_addr
	.byte	6	; DW_AT_high_pc
	.byte	0	; DW_FORM_data4
	.byte	0	; EOM(1)
	.byte	2	; EOM(2)
	.byte	46	; Abbreviation Code
	.byte	0	; DW_TAG_subprogram
	.byte	17	; DW_CHILDREN_no
	.byte	1	; DW_AT_low_pc
	.byte	18	; DW_FORM_addr
	.byte	6	; DW_AT_high_pc
	.ascii	"\347\177"	; DW_FORM_data4
	.byte	25	; DW_AT_APPLE_omit_frame_ptr
	.byte	64	; DW_FORM_flag_present
	.byte	24	; DW_AT_frame_base
	.byte	3	; DW_FORM_exprloc
	.byte	14	; DW_AT_name
	.byte	58	; DW_FORM_strp
	.byte	11	; DW_AT_decl_file
	.byte	59	; DW_FORM_data1
	.byte	11	; DW_AT_decl_line
	.byte	39	; DW_FORM_data1
	.byte	25	; DW_AT_prototyped
	.byte	73	; DW_FORM_flag_present
	.byte	19	; DW_AT_type
	.byte	63	; DW_FORM_ref4
	.byte	25	; DW_AT_external
	.byte	0	; DW_FORM_flag_present
	.byte	0	; EOM(1)
	.byte	3	; EOM(2)
	.byte	36	; Abbreviation Code
	.byte	0	; DW_TAG_base_type
	.byte	3	; DW_CHILDREN_no
	.byte	14	; DW_AT_name
	.byte	62	; DW_FORM_strp
	.byte	11	; DW_AT_encoding
	.byte	11	; DW_FORM_data1
	.byte	11	; DW_AT_byte_size
	.byte	0	; DW_FORM_data1
	.byte	0	; EOM(1)
	.byte	0	; EOM(2)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:	; EOM(3)
Lsection_info1:
Lcu_begin0:
Lset0 = Ldebug_info_end0-Ldebug_info_start0	; Length of Unit
	.long	Lset0
Ldebug_info_start0:
	.short	4
Lset1 = Lsection_abbrev0-Lsection_abbrev0	; DWARF version number	; Offset Into Abbrev. Section
	.long	Lset1
	.byte	8
	.byte	1	; Address Size (in bytes)
	.long	0	; Abbrev [1] 0xb:0x48 DW_TAG_compile_unit
	.short	29	; DW_AT_producer
	.long	30	; DW_AT_language
	.long	117	; DW_AT_name
	.long	171	; DW_AT_LLVM_sysroot
Lset2 = Lline_table_start0-Lsection_line0	; DW_AT_APPLE_sdk	; DW_AT_stmt_list
	.long	Lset2
	.long	184
	.quad	Lfunc_begin0	; DW_AT_comp_dir
Lset3 = Lfunc_end0-Lfunc_begin0	; DW_AT_low_pc	; DW_AT_high_pc
	.long	Lset3
	.byte	2
	.quad	Lfunc_begin0	; Abbrev [2] 0x32:0x19 DW_TAG_subprogram
Lset4 = Lfunc_end0-Lfunc_begin0	; DW_AT_low_pc	; DW_AT_high_pc
	.long	Lset4
	; DW_AT_APPLE_omit_frame_ptr
	.byte	1
	.byte	111	; DW_AT_frame_base
	.long	221
	.byte	1	; DW_AT_name
	.byte	1	; DW_AT_decl_file
	; DW_AT_decl_line	; DW_AT_prototyped
	.long	75
	; DW_AT_type	; DW_AT_external
	.byte	3
	.long	239	; Abbrev [3] 0x4b:0x7 DW_TAG_base_type
	.byte	5	; DW_AT_name
	.byte	8	; DW_AT_encoding
	.byte	0	; DW_AT_byte_size
Ldebug_info_end0:	; End Of Children Mark
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
Linfo_string0:
	.ascii	"Homebrew clang version 23.1.0"
	.byte	0
	.ascii	"/Users/runner/work/nanolang/nanolang/.test_output/source-capture-diagnostic/2/answer.c"	; string offset=0 ; Homebrew clang version 23.1.0
	.byte	0
	.ascii	"/Library/Developer/CommandLineTools/SDKs/MacOSX14.sdk"	; string offset=30 ; /Users/runner/work/nanolang/nanolang/.test_output/source-capture-diagnostic/2/answer.c
	.byte	0
	.ascii	"MacOSX14.sdk"	; string offset=117 ; /Library/Developer/CommandLineTools/SDKs/MacOSX14.sdk
	.byte	0
	.ascii	"/Users/runner/work/nanolang/nanolang"	; string offset=171 ; MacOSX14.sdk
	.byte	0
	.ascii	"nano_build_answer"	; string offset=184 ; /Users/runner/work/nanolang/nanolang
	.byte	0
	.ascii	"long long"	; string offset=221 ; nano_build_answer
	.byte	0
	.section	__DWARF,__apple_names,regular,debug
Lnames_begin:	; string offset=239 ; long long
Lnames_begin0:
	.long	1212240712
	.short	1	; Header Magic
	.short	0	; Header Version
	.long	1	; Header Hash Function
	.long	1	; Header Bucket Count
	.long	12	; Header Hash Count
	.long	0	; Header Data Length
	.long	1	; HeaderData Die Offset Base
	.short	1	; HeaderData Atom Count
	.short	6	; DW_ATOM_die_offset
	.long	0	; DW_FORM_data4
	.long	-1056308497	; Bucket 0
Lset5 = LNames0-Lnames_begin0	; Hash in Bucket 0	; Offset in Bucket 0
	.long	Lset5
LNames0:
	.long	221
	.long	1	; nano_build_answer
	.long	50	; Num DIEs
	.long	0
	.section	__DWARF,__apple_objc,regular,debug
Lobjc_begin:
Lobjc_begin0:
	.long	1212240712
	.short	1	; Header Magic
	.short	0	; Header Version
	.long	1	; Header Hash Function
	.long	0	; Header Bucket Count
	.long	12	; Header Hash Count
	.long	0	; Header Data Length
	.long	1	; HeaderData Die Offset Base
	.short	1	; HeaderData Atom Count
	.short	6	; DW_ATOM_die_offset
	.long	-1	; DW_FORM_data4
	.section	__DWARF,__apple_namespac,regular,debug
Lnamespac_begin:	; Bucket 0
Lnamespac_begin0:
	.long	1212240712
	.short	1	; Header Magic
	.short	0	; Header Version
	.long	1	; Header Hash Function
	.long	0	; Header Bucket Count
	.long	12	; Header Hash Count
	.long	0	; Header Data Length
	.long	1	; HeaderData Die Offset Base
	.short	1	; HeaderData Atom Count
	.short	6	; DW_ATOM_die_offset
	.long	-1	; DW_FORM_data4
	.section	__DWARF,__apple_types,regular,debug
Ltypes_begin:	; Bucket 0
Ltypes_begin0:
	.long	1212240712
	.short	1	; Header Magic
	.short	0	; Header Version
	.long	1	; Header Hash Function
	.long	1	; Header Bucket Count
	.long	20	; Header Hash Count
	.long	0	; Header Data Length
	.long	3	; HeaderData Die Offset Base
	.short	1	; HeaderData Atom Count
	.short	6	; DW_ATOM_die_offset
	.short	3	; DW_FORM_data4
	.short	5	; DW_ATOM_die_tag
	.short	4	; DW_FORM_data2
	.short	11	; DW_ATOM_type_flags
	.long	0	; DW_FORM_data1
	.long	-1921963995	; Bucket 0
Lset6 = Ltypes0-Ltypes_begin0	; Hash in Bucket 0	; Offset in Bucket 0
	.long	Lset6
Ltypes0:
	.long	239
	.long	1	; long long
	.long	75	; Num DIEs
	.short	36
	.byte	0
	.long	0
.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lsection_line0:
Lline_table_start0:
