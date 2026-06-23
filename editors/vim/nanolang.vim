" Vim syntax file for Nanolang
" Language: Nanolang
" Maintainer: Nanolang Team
" Latest Revision: 2024-12-17

if exists("b:current_syntax")
  finish
endif

" Keywords
syn keyword nanoKeyword fn let mut set return if else while for break continue
syn keyword nanoKeyword match case default pub mod import as extern module
syn keyword nanoKeyword struct union enum typedef const shadow assert opaque
syn keyword nanoKeyword nextgroup=nanoFunction skipwhite

" Types
syn keyword nanoType int float bool string void array Result Option
syn match nanoType "\<[A-Z][a-zA-Z0-9_]*\>"

" Constants
syn keyword nanoBoolean true false
syn keyword nanoNull null

" Numbers
syn match nanoNumber "\<[0-9]\+\>"
syn match nanoFloat "\<[0-9]\+\.[0-9]\+\>"
syn match nanoHex "\<0x[0-9a-fA-F]\+\>"

" Strings
syn region nanoString start=+"+ skip=+\\"+ end=+"+
syn match nanoEscape +\\[ntr"\\]+ contained containedin=nanoString

" Comments
syn match nanoComment "//.*$"
syn region nanoBlockComment start="/\*" end="\*/"

" Functions
syn match nanoFunction "\<[a-z_][a-zA-Z0-9_]*\>\s*("me=e-1

" Built-in Functions
syn keyword nanoBuiltin println print len get set push pop
syn keyword nanoBuiltin map reduce filter fold cast sizeof

" Operators
syn match nanoOperator "[-+*/%]"
syn match nanoOperator "[<>=!]=\?"
syn match nanoOperator "&&\|||"

" S-expressions
syn region nanoSexp start="(" end=")" transparent fold

" Highlighting
hi def link nanoKeyword Keyword
hi def link nanoType Type
hi def link nanoBoolean Boolean
hi def link nanoNull Constant
hi def link nanoNumber Number
hi def link nanoFloat Float
hi def link nanoHex Number
hi def link nanoString String
hi def link nanoEscape SpecialChar
hi def link nanoComment Comment
hi def link nanoBlockComment Comment
hi def link nanoFunction Function
hi def link nanoBuiltin Function
hi def link nanoOperator Operator

let b:current_syntax = "nanolang"

