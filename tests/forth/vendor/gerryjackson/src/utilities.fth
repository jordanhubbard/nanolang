( The ANS/Forth 2012 test suite is being modified so that the test programs  )
( for the optional word sets only use standard words from the Core word set. )
( This file, which is included *after* the Core test programs, contains      )
( various definitions for use by the optional word set test programs to      )
( remove any dependencies between word sets.                                 )

DECIMAL

( First a definition to see if a word is already defined. Note that          )
( [DEFINED] [IF] [ELSE] and [THEN] are in the optional Programming Tools     )
( word set.                                                                  )

VARIABLE (\?) 0 (\?) !     ( Flag: Word defined = 0 | word undefined = -1 )

( [?DEF]  followed by [?IF] cannot be used again until after [THEN] )
: [?DEF]  ( "name" -- )
   BL WORD FIND SWAP DROP 0= (\?) !
;

\ Test [?DEF]
T{ 0 (\?) ! [?DEF] ?DEFTEST1 (\?) @ -> -1 }T
: ?DEFTEST1 1 ;
T{ -1 (\?) ! [?DEF] ?DEFTEST1 (\?) @ -> 0 }T

: [?UNDEF] [?DEF] (\?) @ 0= (\?) ! ;

\ Equivalents of [IF] [ELSE] [THEN], these must not be nested
: [?IF]  ( f -- )  (\?) ! ; IMMEDIATE
: [?ELSE]  ( -- )  (\?) @ 0= (\?) ! ; IMMEDIATE
: [?THEN]  ( -- )  0 (\?) ! ; IMMEDIATE

( A conditional comment and \ will be defined. Note that these definitions )
( are inadequate for use in Forth blocks. If needed in the blocks test     )
( program they will need to be modified here or redefined there )

( \? is a conditional comment )
: \?  ( "..." -- )  (\?) @ IF EXIT THEN SOURCE >IN ! DROP ; IMMEDIATE

\ Test \?
T{ [?DEF] ?DEFTEST1 \? : ?DEFTEST1 2 ;    \ Should not be redefined
          ?DEFTEST1 -> 1 }T
T{ [?DEF] ?DEFTEST2 \? : ?DEFTEST1 2 ;    \ Should be redefined
          ?DEFTEST1 -> 2 }T

[?DEF] TRUE  \? -1 CONSTANT TRUE
[?DEF] FALSE \?  0 CONSTANT FALSE
[?DEF] NIP   \?  : NIP SWAP DROP ;
[?DEF] TUCK  \?  : TUCK SWAP OVER ;

[?DEF] PARSE
\? : BUMP  ( caddr u n -- caddr+n u-n )
\?    TUCK - >R CHARS + R>
\? ;

\? : PARSE  ( ch "ccc<ch>" -- caddr u )
\?    >R SOURCE >IN @ BUMP
\?    OVER R> SWAP >R >R         ( -- start u1 ) ( R: -- start ch )
\?    BEGIN
\?       DUP
\?    WHILE
\?       OVER C@ R@ = 0=
\?    WHILE
\?       1 BUMP
\?    REPEAT
\?       1-                      ( end u2 )  \ delimiter found
\?    THEN
\?    SOURCE NIP SWAP - >IN !    ( -- end )
\?    R> DROP R>                 ( -- end start )
\?    TUCK - 1 CHARS /           ( -- start u )
\? ;

[?DEF] .(  \? : .(  [CHAR] ) PARSE TYPE ; IMMEDIATE

\ S=  to compare (case sensitive) two strings to avoid use of COMPARE from
\ the String word set. It is defined in core.fr and conditionally defined
\ here if core.fr has not been included by the user

[?DEF] S=
\? : S=  ( caddr1 u1 caddr2 u2 -- f )   \ f = TRUE if strings are equal
\?    ROT OVER = 0= IF DROP 2DROP FALSE EXIT THEN
\?    DUP 0= IF DROP 2DROP TRUE EXIT THEN 
\?    0 DO
\?         OVER C@ OVER C@ = 0= IF 2DROP FALSE UNLOOP EXIT THEN
\?         CHAR+ SWAP CHAR+
\?      LOOP 2DROP TRUE
\? ;

\ Buffer for strings in interpretive mode since S" only valid in compilation
\ mode when File-Access word set is defined

64 CONSTANT SBUF-SIZE
CREATE SBUF1 SBUF-SIZE CHARS ALLOT
CREATE SBUF2 SBUF-SIZE CHARS ALLOT

\ ($") saves a counted string at (caddr)
: ($")  ( caddr "ccc" -- caddr' u )
   [CHAR] " PARSE ROT 2DUP C!       ( -- ca2 u2 ca)
   CHAR+ SWAP 2DUP 2>R CHARS MOVE   ( -- )  ( R: -- ca' u2 )
   2R>
;

: $"   ( "ccc" -- caddr u )  SBUF1 ($") ;
: $2"  ( "ccc" -- caddr u )  SBUF2 ($") ;
: $CLEAR  ( caddr -- ) SBUF-SIZE BL FILL ;
: CLEAR-SBUFS  ( -- )  SBUF1 $CLEAR SBUF2 $CLEAR ;

\ More definitions in core.fr used in other test programs, conditionally
\ defined here if core.fr has not been loaded

[?DEF] MAX-UINT   \? 0 INVERT                 CONSTANT MAX-UINT
[?DEF] MAX-INT    \? 0 INVERT 1 RSHIFT        CONSTANT MAX-INT
[?DEF] MIN-INT    \? 0 INVERT 1 RSHIFT INVERT CONSTANT MIN-INT
[?DEF] MID-UINT   \? 0 INVERT 1 RSHIFT        CONSTANT MID-UINT
[?DEF] MID-UINT+1 \? 0 INVERT 1 RSHIFT INVERT CONSTANT MID-UINT+1

[?DEF] 2CONSTANT \? : 2CONSTANT  CREATE , , DOES> 2@ ;

BASE @ 2 BASE ! -1 0 <# #S #> SWAP DROP CONSTANT BITS/CELL BASE !


\ ------------------------------------------------------------------------------
\ Tests

: STR1  S" abcd" ;  : STR2  S" abcde" ;
: STR3  S" abCd" ;  : STR4  S" wbcd"  ;
: S"" S" " ;

T{ STR1 2DUP S= -> TRUE }T
T{ STR2 2DUP S= -> TRUE }T
T{ S""  2DUP S= -> TRUE }T
T{ STR1 STR2 S= -> FALSE }T
T{ STR1 STR3 S= -> FALSE }T
T{ STR1 STR4 S= -> FALSE }T

T{ CLEAR-SBUFS -> }T
T{ $" abcdefghijklm"  SBUF1 COUNT S= -> TRUE  }T
T{ $" nopqrstuvwxyz"  SBUF2 OVER  S= -> FALSE }T
T{ $2" abcdefghijklm" SBUF1 COUNT S= -> FALSE }T
T{ $2" nopqrstuvwxyz" SBUF1 COUNT S= -> TRUE  }T

\ ------------------------------------------------------------------------------

CR $" Test utilities loaded" TYPE CR
