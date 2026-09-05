\ To test the ANS Forth Block word set and extension words

\ This program was written by Steve Palmer in 2015, with contributions from
\ others where indicated, and is in the public domain - it can be distributed
\ and/or modified in any way but please retain this notice.

\ This program is distributed in the hope that it will be useful,
\ but WITHOUT ANY WARRANTY; without even the implied warranty of
\ MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

\ The tests are not claimed to be comprehensive or correct

\ ------------------------------------------------------------------------------
\ Version 0.1 23 October 2015  First Version
\ Version 0.2 15 November 2015 Updated after feedback from Gerry Jackson

\ ------------------------------------------------------------------------------
\ The tests are based on John Hayes test program for the core word set
\
\ Words tested in this file are:
\     BLK BLOCK BUFFER EVALUATE FLUSH LOAD SAVE-BUFFERS UPDATE
\     EMPTY-BUFFERS LIST SCR THRU REFILL SAVE-INPUT RESTORE-INPUT \
\
\ ------------------------------------------------------------------------------
\ Assumptions and dependencies:
\     - tester.fr or ttester.fs has been loaded prior to this file
\     - errorreport.fth has been loaded prior to this file
\     - utilities.fth has been loaded prioir to this file
\ ------------------------------------------------------------------------------
TESTING Block word set

DECIMAL

\ Define these constants from the system documentation provided.
\ WARNING: The contents of the test blocks will be destroyed by this test.
\ The blocks tested will be in the range
\    FIRST-TEST-BLOCK <= u < LIMIT-TEST-BLOCK
\ The tests need at least 2 test blocks in the range to complete.
20 CONSTANT FIRST-TEST-BLOCK
30 CONSTANT LIMIT-TEST-BLOCK \ one beyond the last

FIRST-TEST-BLOCK LIMIT-TEST-BLOCK U< 0= [?IF]
\?  .( Error: Test Block range not identified ) CR ABORT
[?THEN]

LIMIT-TEST-BLOCK FIRST-TEST-BLOCK - CONSTANT TEST-BLOCK-COUNT
TEST-BLOCK-COUNT 2 U< [?IF]
\?  .( Error: At least 2 Test Blocks are required to run the tests ) CR ABORT
[?THEN]

\ ------------------------------------------------------------------------------
TESTING Random Number Utilities

\ The block tests make extensive use of random numbers to select blocks to test
\ and to set the contents of the block.  It also makes use of a Hash code to
\ ensure the integrity of the blocks against unexpected changes.

\ == Memory Walk tools ==

: @++ ( a-addr -- a-addr+4 a-addr@ )
    DUP CELL+ SWAP @ ;

: !++ ( x a-addr -- a-addr+4 )
    TUCK ! CELL+ ;

: C@++ ( c-addr -- c-addr;char+ c-addr@ )
    DUP CHAR+ SWAP C@ ;

: C!++ ( char c-addr -- c-addr+1 )
    TUCK ! CHAR+ ;

\ == Random Numbers ==
\ Based on "Xorshift" PRNG wikipedia page
\ reporting on results by George Marsaglia
\ https://en.wikipedia.org/wiki/Xorshift
\ Note: THIS IS NOT CRYPTOGRAPHIC QUALITY

: PRNG
    CREATE ( "name" -- )
        4 CELLS ALLOT
    DOES> ( -- prng )
;

: PRNG-ERROR-CODE ( prng -- errcode | 0 )
    0 4 0 DO          \ prng acc
        >R @++ R> OR  \ prng acc'
    LOOP              \ prng xORyORzORw
    NIP 0= ;          \ xORyORzORw=0

: PRNG-COPY ( src-prng dst-prng -- )
    4 CELLS MOVE ;

: PRNG-SET-SEED ( prng w z y x -- )
    4 PICK                 \ prng w z y x prng
    4 0 DO !++ LOOP DROP   \ prng
    DUP PRNG-ERROR-CODE IF \ prng
        1 OVER +!          \ prng
    THEN                   \ prng
    DROP ;                 \

BITS/CELL 64 = [?IF]
\?  : PRNG-RND ( prng -- rnd )
\?      DUP @
\?      DUP 21 LSHIFT XOR
\?      DUP 35 RSHIFT XOR
\?      DUP  4 LSHIFT XOR
\?      TUCK SWAP ! ;
[?THEN]

BITS/CELL 32 = [?IF]
\?  : PRNG-RND ( prng -- rnd )
\?      DUP @                            \ prng x
\?      DUP 11 LSHIFT XOR                \ prng t=x^(x<<11)
\?      DUP 8 RSHIFT XOR                 \ prng t'=t^(t>>8)
\?      OVER DUP CELL+ SWAP 3 CELLS MOVE \ prng t'
\?      OVER 3 CELLS + @                 \ prng t' w
\?      DUP 19 RSHIFT XOR                \ prng t' w'=w^(w>>19)
\?      XOR                              \ prng rnd=w'^t'
\?      TUCK SWAP 3 CELLS + ! ;          \ rnd
[?THEN]

BITS/CELL 16 = [?IF]
\?  .( === NOT TESTED === )
\?  \ From http://b2d-f9r.blogspot.co.uk/2010/08/16-bit-xorshift-rng-now-with-more.html
\?  : PRNG-RND ( prng -- rnd )
\?      DUP @                        \ prng x
\?      DUP 5 LSHIFT XOR             \ prng t=x^(x<<5)
\?      DUP 3 RSHIFT XOR             \ prng t'=t^(t>>3)
\?      OVER DUP CELL+ @ TUCK SWAP ! \ prng t' y
\?      DUP 1 RSHIFT XOR             \ prng t' y'=y^(y>>1)
\?      XOR                          \ prng rnd=y'^t'
\?      TUCK SWAP CELL+ ! ;          \ rnd
[?THEN]

[?DEF] PRNG-RND
\?  .( You need to add a Psuedo Random Number Generator for your cell size: )
\?  BITS/CELL U. CR
\?  ABORT
[?THEN]

: PRNG-RANDOM ( lower upper prng -- rnd )
    >R OVER - R> PRNG-RND UM* NIP + ;
\ PostCondition: T{ lower upper 2DUP 2>R prng PRNG-RANDOM 2R> WITHIN -> TRUE }T

PRNG BLOCK-PRNG
\ Generated by Random.org
BLOCK-PRNG -1865266521 188896058 -2021545234 -1456609962 PRNG-SET-SEED
: BLOCK-RND ( -- rnd )                BLOCK-PRNG PRNG-RND ;
: BLOCK-RANDOM ( lower upper -- rnd ) BLOCK-PRNG PRNG-RANDOM ;

: RND-TEST-BLOCK ( -- blk )
    FIRST-TEST-BLOCK LIMIT-TEST-BLOCK BLOCK-RANDOM ;
\ PostCondition: T{ RND-TEST-BLOCK FIRST-TEST-BLOCK LIMIT-TEST-BLOCK WITHIN -> TRUE }T

\ Two distinct random test blocks
: 2RND-TEST-BLOCKS ( -- blk1 blk2 )
    RND-TEST-BLOCK BEGIN  \ blk1
        RND-TEST-BLOCK    \ blk1 blk2
        2DUP =            \ blk1 blk2 blk1==blk2
    WHILE                 \ blk1 blk1
        DROP              \ blk1
    REPEAT ;              \ blk1 blk2
\ PostCondition: T{ 2RND-TEST-BLOCKS = -> FALSE }T

\ first random test block in a sequence of length u
: RND-TEST-BLOCK-SEQ ( u -- blks )
    FIRST-TEST-BLOCK LIMIT-TEST-BLOCK ROT 1- - BLOCK-RANDOM ;

\ I'm not sure if this algorithm is correct if " 1 CHARS 1 <> ".
: ELF-HASH-ACCUMULATE ( hash c-addr u -- hash )
    >R SWAP R> 0 DO                          \ c-addr h
        4 LSHIFT                             \ c-addr h<<=4
        SWAP C@++ ROT +                      \ c-addr' h+=*s
        DUP [ HEX ] F0000000 [ DECIMAL ] AND \ c-addr' h high=h&0xF0000000
        DUP IF                               \ c-addr' h high
            DUP >R 24 RSHIFT XOR R>          \ c-addr' h^=high>>24 high
        THEN                                 \ c-addr' h high
        INVERT AND                           \ c-addr' h&=~high
    LOOP NIP ;

: ELF-HASH ( c-addr u -- hash )
    0 ROT ROT ELF-HASH-ACCUMULATE ;

\ ------------------------------------------------------------------------------
TESTING BLOCK ( read-only mode )

\ BLOCK signature
T{ RND-TEST-BLOCK BLOCK DUP ALIGNED = -> TRUE }T

\ BLOCK accepts all blocks in the test range
: BLOCK-ALL ( blk2 blk1 -- )
    DO
        I BLOCK DROP
    LOOP ;
T{ LIMIT-TEST-BLOCK FIRST-TEST-BLOCK BLOCK-ALL -> }T

\ BLOCK twice on same block returns the same value
T{ RND-TEST-BLOCK DUP BLOCK SWAP BLOCK = -> TRUE }T

\ BLOCK twice on distinct block numbers
\ may or may not return the same value!
\ Nothing to test

\ ------------------------------------------------------------------------------
TESTING BUFFER ( read-only mode )

\ Although it is not in the spirit of the specification,
\ a compliant definition of BUFFER would be
\ : BUFFER BLOCK ;
\ So we can only repeat the tests for BLOCK ...

\ BUFFER signature
T{ RND-TEST-BLOCK BUFFER DUP ALIGNED = -> TRUE }T

\ BUFFER accepts all blocks in the test range
: BUFFER-ALL ( blk2 blk1 -- )
    DO
        I BUFFER DROP
    LOOP ;
T{ LIMIT-TEST-BLOCK FIRST-TEST-BLOCK BUFFER-ALL -> }T

\ BUFFER twice on the same block returns the same value
T{ RND-TEST-BLOCK DUP BUFFER SWAP BUFFER = -> TRUE }T

\ BUFFER twice on distinct block numbers
\ may or may not return the same value!
\ Nothing to test

\ Combinations with BUFFER
T{ RND-TEST-BLOCK DUP BLOCK SWAP BUFFER = -> TRUE }T
T{ RND-TEST-BLOCK DUP BUFFER SWAP BLOCK = -> TRUE }T

\ ------------------------------------------------------------------------------
TESTING Read and Write access with UPDATE and FLUSH

\ Ideally, we'd like to be able to test the persistence across power cycles
\ of the writes, but we can't do that in a simple test.
\ The tests below could be fooled by a large buffers store and a tricky FLUSH
\ but what else are you going to do?

\ Signatures
T{ RND-TEST-BLOCK BLOCK DROP UPDATE -> }T
T{ FLUSH -> }T

: BLANK-BUFFER ( blk -- blk-addr )
    BUFFER DUP 1024 BL FILL ;

\ Test R/W of a Simple Blank Random Block
T{ RND-TEST-BLOCK                 \ blk
   DUP BLANK-BUFFER               \ blk blk-addr1
   1024 ELF-HASH                  \ blk hash
   UPDATE FLUSH                   \ blk hash
   SWAP BLOCK                     \ hash blk-addr2
   1024 ELF-HASH = -> TRUE }T

\ Boundary Test: Modify first character
T{ RND-TEST-BLOCK                  \ blk
   DUP BLANK-BUFFER                \ blk blk-addr1
   CHAR \ OVER C!                  \ blk blk-addr1
   1024 ELF-HASH                   \ blk hash
   UPDATE FLUSH                    \ blk hash
   SWAP BLOCK                      \ hash blk-addr2
   1024 ELF-HASH = -> TRUE }T

\ Boundary Test: Modify last character
T{ RND-TEST-BLOCK                  \ blk
   DUP BLANK-BUFFER                \ blk blk-addr1
   CHAR \ OVER 1023 CHARS + C!     \ blk blk-addr1
   1024 ELF-HASH                   \ blk hash
   UPDATE FLUSH                    \ blk hash
   SWAP BLOCK                      \ hash blk-addr2
   1024 ELF-HASH = -> TRUE }T

\ Boundary Test: First and Last (and all other) blocks in the test range
1024 8 * BITS/CELL / CONSTANT CELLS/BLOCK

: PREPARE-RND-BLOCK ( hash blk -- hash' )
    BUFFER DUP                     \ hash blk-addr blk-addr
    CELLS/BLOCK 0 DO               \ hash blk-addr blk-addr[i]
        BLOCK-RND OVER ! CELL+     \ hash blk-addr blk-addr[i+1]
    LOOP DROP                      \ hash blk-addr
    1024 ELF-HASH-ACCUMULATE ;     \ hash'

: WRITE-RND-BLOCKS-WITH-HASH ( blk2 blk1 -- hash )
    0 ROT ROT DO                   \ hash
        I PREPARE-RND-BLOCK UPDATE \ hash'
    LOOP ;                         \ hash'

: READ-BLOCKS-AND-HASH ( blk2 blk1 -- hash )
    0 ROT ROT DO                         \ hash(i)
        I BLOCK 1024 ELF-HASH-ACCUMULATE \ hash(i+1)
    LOOP ;                               \ hash

T{ LIMIT-TEST-BLOCK FIRST-TEST-BLOCK WRITE-RND-BLOCKS-WITH-HASH FLUSH
   LIMIT-TEST-BLOCK FIRST-TEST-BLOCK READ-BLOCKS-AND-HASH = -> TRUE }T

: TUF1 ( xt blk -- hash )
    DUP BLANK-BUFFER               \ xt blk blk-addr1
    1024 ELF-HASH                  \ xt blk hash
    ROT EXECUTE                    \ blk hash
    SWAP BLOCK                     \ hash blk-addr2
    1024 ELF-HASH = ;              \ TRUE

\ Double UPDATE make no difference
: TUF1-1 ( -- ) UPDATE UPDATE FLUSH ;
T{ ' TUF1-1 RND-TEST-BLOCK TUF1 -> TRUE }T

\ Double FLUSH make no difference
: TUF1-2 ( -- ) UPDATE FLUSH FLUSH ;
T{ ' TUF1-2 RND-TEST-BLOCK TUF1 -> TRUE }T

\ FLUSH only saves UPDATEd buffers
T{ RND-TEST-BLOCK                      \ blk
   0 OVER PREPARE-RND-BLOCK            \ blk hash
   UPDATE FLUSH                        \ blk hash
   OVER 0 SWAP PREPARE-RND-BLOCK DROP  \ blk hash
   FLUSH ( with no preliminary UPDATE) \ blk hash
   SWAP BLOCK 1024 ELF-HASH = -> TRUE }T

\ UPDATE only marks the current block buffer
\ This test needs at least 2 distinct buffers, though this is not a
\ requirement of the language specification.  If 2 distinct buffers
\ are not returned, then the tests quits with a trivial Pass
: TUF2 ( xt blk1 blk2 -- hash1'' hash2'' hash1' hash2' hash1 hash2 )
    OVER BUFFER OVER BUFFER = IF             \ test needs 2 distinct buffers
        2DROP DROP 0 0 0 0 0 0               \ Dummy result
    ELSE
        OVER 0 SWAP PREPARE-RND-BLOCK UPDATE \ xt blk1 blk2 hash1
        OVER 0 SWAP PREPARE-RND-BLOCK UPDATE \ xt blk1 blk2 hash1 hash2
        2>R                                  \ xt blk1 blk2
        FLUSH                                \ xt blk1 blk2
        OVER 0 SWAP PREPARE-RND-BLOCK        \ xt blk1 blk2 hash1'
        OVER 0 SWAP PREPARE-RND-BLOCK        \ xt blk1 blk2 hash1' hash2'
        2>R                                  \ xt blk1 blk2
        ROT EXECUTE                          \ blk1 blk2
        FLUSH                                \ blk1 blk2
        SWAP BLOCK 1024 ELF-HASH             \ blk2 hash1''
        SWAP BLOCK 1024 ELF-HASH             \ hash1'' hash2''
        2R> 2R> \ hash1'' hash2'' hash1' hash2' hash1 hash2
    THEN ;

: 2= ( x1 x2 x3 x4 -- flag )
    ROT = ROT ROT = AND ;

: TUF2-0 ( blk1 blk2 -- blk1 blk2 ) ;   \ no updates
T{ ' TUF2-0 2RND-TEST-BLOCKS TUF2       \ run test procedure
   2SWAP 2DROP 2= -> TRUE }T            \ compare expected and actual

: TUF2-1 ( blk1 blk2 -- blk1 blk2 )     \ update blk1 only
    OVER BLOCK DROP UPDATE ;
T{ ' TUF2-1 2RND-TEST-BLOCKS TUF2       \ run test procedure
   SWAP DROP SWAP DROP 2= -> TRUE }T

: TUF2-2 ( blk1 blk2 -- blk1 blk2 )     \ update blk2 only
    DUP BUFFER DROP UPDATE ;
T{ ' TUF2-2 2RND-TEST-BLOCKS TUF2       \ run test procedure
   DROP ROT DROP SWAP 2= -> TRUE }T

: TUF2-3 ( blk1 blk2 -- blk1 blk2 )     \ update blk1 and blk2
    TUF2-1 TUF2-2 ;
T{ ' TUF2-3 2RND-TEST-BLOCKS TUF2       \ run test procedure
   2DROP 2= -> TRUE }T

\ FLUSH and then UPDATE is ambiguous and untestable

\ ------------------------------------------------------------------------------
TESTING SAVE-BUFFERS

\ In principle, all the tests above can be repeated with SAVE-BUFFERS instead of
\ FLUSH.  However, only the full random test is repeated...

T{ LIMIT-TEST-BLOCK FIRST-TEST-BLOCK WRITE-RND-BLOCKS-WITH-HASH SAVE-BUFFERS
   LIMIT-TEST-BLOCK FIRST-TEST-BLOCK READ-BLOCKS-AND-HASH = -> TRUE }T

\ FLUSH and then SAVE-BUFFERS is harmless but undetectable
\ SAVE-BUFFERS and then FLUSH is undetectable

\ Unlike FLUSH, SAVE-BUFFERS then BUFFER/BLOCK
\ returns the original buffer address
T{ RND-TEST-BLOCK DUP BLANK-BUFFER
   SAVE-BUFFERS        SWAP BUFFER = -> TRUE }T
T{ RND-TEST-BLOCK DUP BLANK-BUFFER
   UPDATE SAVE-BUFFERS SWAP BUFFER = -> TRUE }T
T{ RND-TEST-BLOCK DUP BLANK-BUFFER
   SAVE-BUFFERS        SWAP BLOCK  = -> TRUE }T
T{ RND-TEST-BLOCK DUP BLANK-BUFFER
   UPDATE SAVE-BUFFERS SWAP BLOCK  = -> TRUE }T

\ ------------------------------------------------------------------------------
TESTING BLK

\ Signature
T{ BLK DUP ALIGNED = -> TRUE }T

\ None of the words considered so far effect BLK
T{ BLK @ RND-TEST-BLOCK BUFFER DROP BLK @ = -> TRUE }T
T{ BLK @ RND-TEST-BLOCK BLOCK  DROP BLK @ = -> TRUE }T
T{ BLK @ UPDATE                     BLK @ = -> TRUE }T

T{ BLK @ FLUSH        BLK @ = -> TRUE }T
T{ BLK @ SAVE-BUFFERS BLK @ = -> TRUE }T

\ ------------------------------------------------------------------------------
TESTING LOAD and EVALUATE

\ Signature: n LOAD --> blank screen
T{ RND-TEST-BLOCK DUP BLANK-BUFFER DROP UPDATE FLUSH LOAD -> }T

T{ BLK @ RND-TEST-BLOCK DUP BLANK-BUFFER DROP UPDATE FLUSH LOAD BLK @ = -> TRUE }T

: WRITE-BLOCK ( blk c-addr u -- )
    ROT BLANK-BUFFER SWAP CHARS MOVE UPDATE FLUSH ;

\ blk: u; blk LOAD
: TL1 ( u blk -- )
    SWAP 0 <# #S #> WRITE-BLOCK ;
T{ BLOCK-RND RND-TEST-BLOCK 2DUP TL1 LOAD = -> TRUE }T

\ Boundary Test: FIRST-TEST-BLOCK
T{ BLOCK-RND FIRST-TEST-BLOCK 2DUP TL1 LOAD = -> TRUE }T

\ Boundary Test: LIMIT-TEST-BLOCK-1
T{ BLOCK-RND LIMIT-TEST-BLOCK 1- 2DUP TL1 LOAD = -> TRUE }T

: WRITE-AT-END-OF-BLOCK ( blk c-addr u -- )
    ROT BLANK-BUFFER
    OVER 1024 SWAP - CHARS +
    SWAP CHARS MOVE UPDATE FLUSH ;

\ Boundary Test: End of Buffer
: TL2 ( u blk -- )
    SWAP 0 <# #S #> WRITE-AT-END-OF-BLOCK ;
T{ BLOCK-RND RND-TEST-BLOCK 2DUP TL2 LOAD = -> TRUE }T

\ LOAD updates BLK
\ u: "BLK @"; u LOAD
: TL3 ( blk -- )
    S" BLK @" WRITE-BLOCK ;
T{ RND-TEST-BLOCK DUP TL3 DUP LOAD = -> TRUE }T

\ EVALUATE resets BLK
\ u: "EVALUATE-BLK@"; u LOAD
: EVALUATE-BLK@ ( -- BLK@ )
    S" BLK @" EVALUATE ;
: TL4 ( blk -- )
    S" EVALUATE-BLK@" WRITE-BLOCK ;
T{ RND-TEST-BLOCK DUP TL4 LOAD -> 0 }T

\ EVALUTE can nest with LOAD
\ u: "BLK @"; S" u LOAD" EVALUATE
: TL5 ( blk -- c-addr u )
    0 <#                       \ blk 0
         [CHAR] D HOLD
         [CHAR] A HOLD
         [CHAR] O HOLD
         [CHAR] L HOLD
         BL HOLD
    #S #> ;                    \ c-addr u
T{ RND-TEST-BLOCK DUP TL3 DUP TL5 EVALUATE = -> TRUE }T

\ Nested LOADs
\ u2: "BLK @"; u1: "LOAD u2"; u1 LOAD
: TL6 ( blk1 blk2 -- )
    DUP TL3                    \ blk1 blk2
    TL5 WRITE-BLOCK ;
T{ 2RND-TEST-BLOCKS 2DUP TL6 SWAP LOAD = -> TRUE }T

\ A test of the effect of LOAD on UPDATE has been deleted
\ as unjustified by the Forth Standard - See Issue #37

\ I would expect LOAD to work on the contents of the buffer cache
\ and not the block device, but the specification doesn't say.
\ Similarly, I would not expect LOAD to FLUSH the buffer cache,
\ but the specification doesn't say so.

\ ------------------------------------------------------------------------------
TESTING LIST and SCR

\ Signatures
T{ SCR DUP ALIGNED = -> TRUE }T
\ LIST signature is test implicitly in the following tests...

: TLS1 ( blk -- )
    S" Should show a (mostly) blank screen" WRITE-BLOCK ;
T{ RND-TEST-BLOCK DUP TLS1 DUP LIST SCR @ = -> TRUE }T

\ Boundary Test: FIRST-TEST-BLOCK
: TLS2 ( blk -- )
    S" List of the First test block" WRITE-BLOCK ;
T{ FIRST-TEST-BLOCK DUP TLS2 LIST -> }T

\ Boundary Test: LIMIT-TEST-BLOCK
: TLS3 ( blk -- )
    S" List of the Last test block" WRITE-BLOCK ;
T{ LIMIT-TEST-BLOCK 1- DUP TLS3 LIST -> }T

\ Boundary Test: End of Screen
: TLS4 ( blk -- )
    S" End of Screen" WRITE-AT-END-OF-BLOCK ;
T{ RND-TEST-BLOCK DUP TLS4 LIST -> }T

\ BLOCK, BUFFER, UPDATE et al don't change SCR
: TLS5 ( blk -- )
    S" Should show another (mostly) blank screen" WRITE-BLOCK ;
\ the first test below sets the scenario for the subsequent tests
\ BLK is unchanged by LIST
T{ BLK @ RND-TEST-BLOCK DUP TLS5 LIST                BLK @ = -> TRUE }T
\ SCR is unchanged by Earlier words
T{ SCR @ FLUSH                                       SCR @ = -> TRUE }T
T{ SCR @ FLUSH DUP 1+ BUFFER DROP                    SCR @ = -> TRUE }T
T{ SCR @ FLUSH DUP 1+ BLOCK DROP                     SCR @ = -> TRUE }T
T{ SCR @ FLUSH DUP 1+ BLOCK DROP UPDATE              SCR @ = -> TRUE }T
T{ SCR @ FLUSH DUP 1+ BLOCK DROP UPDATE SAVE-BUFFERS SCR @ = -> TRUE }T
: TLS6 ( blk -- )
    S" SCR @" WRITE-BLOCK ;
T{ SCR @ RND-TEST-BLOCK DUP TLS6 LOAD                SCR @ OVER 2= -> TRUE }T

\ ------------------------------------------------------------------------------
TESTING EMPTY-BUFFERS

T{ EMPTY-BUFFERS -> }T
T{ BLK @ EMPTY-BUFFERS BLK @ = -> TRUE }T
T{ SCR @ EMPTY-BUFFERS SCR @ = -> TRUE }T

\ Test R/W, but discarded changes with EMPTY-BUFFERS
T{ RND-TEST-BLOCK                    \ blk
   DUP BLANK-BUFFER                  \ blk blk-addr1
   1024 ELF-HASH                     \ blk hash
   UPDATE FLUSH                      \ blk hash
   OVER BLOCK CHAR \ SWAP C!         \ blk hash
   UPDATE EMPTY-BUFFERS FLUSH        \ blk hash
   SWAP BLOCK                        \ hash blk-addr2
   1024 ELF-HASH = -> TRUE }T

\ EMPTY-BUFFERS discards all buffers
: TUF2-EB ( blk1 blk2 -- blk1 blk2 )
    TUF2-1 TUF2-2 EMPTY-BUFFERS ;  \ c.f. TUF2-3
T{ ' TUF2-EB 2RND-TEST-BLOCKS TUF2
   2SWAP 2DROP 2= -> TRUE }T

\ FLUSH and then EMPTY-BUFFERS is acceptable but untestable
\ EMPTY-BUFFERS and then UPDATE is ambiguous and untestable

\ ------------------------------------------------------------------------------
TESTING >IN manipulation from a block source

: TIN ( blk -- )
    S" 1 8 >IN +!     2        3" WRITE-BLOCK ;
T{ RND-TEST-BLOCK DUP TIN LOAD -> 1 3 }T

\ ------------------------------------------------------------------------------
TESTING \, SAVE-INPUT, RESTORE-INPUT and REFILL from a block source

\ Try to determine the number of charaters per line
\ Assumes an even number of characters per line
: | ( u -- u-2 ) 2 - ;
: C/L-CALC ( blk -- c/l )
    DUP BLANK-BUFFER                 \ blk blk-addr
    [CHAR] \ OVER C!                 \ blk blk-addr  blk:"\"
    511 0 DO                         \ blk c-addr[i]
        CHAR+ CHAR+ [CHAR] | OVER C! \ blk c-addr[i+1]
    LOOP DROP                        \ blk   blk:"\ | | | | ... |"
    UPDATE SAVE-BUFFERS FLUSH        \ blk
    1024 SWAP LOAD ;                 \ c/l
[?DEF] C/L
[?ELSE]
\? .( Given Characters per Line: ) C/L U. CR
[?ELSE]
\? RND-TEST-BLOCK C/L-CALC CONSTANT C/L
\? C/L 1024 U< [?IF]
\? .( Calculated Characters per Line: ) C/L U. CR
[?THEN]

: WRITE-BLOCK-LINE ( lin-addr[i] c-addr u -- lin-addr[i+1] )
    2>R DUP C/L CHARS + SWAP 2R> ROT SWAP MOVE ;

\ Discards to the end of the line
: TCSIRIR1 ( blk -- )
    BLANK-BUFFER
    C/L 1024 U< IF
        S" 2222 \ 3333" WRITE-BLOCK-LINE
        S" 4444"        WRITE-BLOCK-LINE
    THEN
    DROP UPDATE SAVE-BUFFERS ;
T{ RND-TEST-BLOCK DUP TCSIRIR1 LOAD -> 2222 4444 }T

VARIABLE T-CNT 0 T-CNT !

: MARK ( "<char>" -- ) \ Use between <# and #>
    CHAR HOLD ; IMMEDIATE

: ?EXECUTE ( xt f -- )
    IF EXECUTE ELSE DROP THEN ;

\ SAVE-INPUT and RESTORE-INPUT within a single block
: TCSIRIR2-EXPECTED S" EDCBCBA" ; \ Remember that the string comes out backwards
: TCSIRIR2 ( blk -- )
    C/L 1024 U< IF
        BLANK-BUFFER
        S" 0 T-CNT !"                   WRITE-BLOCK-LINE
        S" <# MARK A SAVE-INPUT MARK B" WRITE-BLOCK-LINE
S" 1 T-CNT +! MARK C ' RESTORE-INPUT T-CNT @ 2 < ?EXECUTE MARK D" WRITE-BLOCK-LINE
        S" MARK E 0 0 #>"               WRITE-BLOCK-LINE
        UPDATE SAVE-BUFFERS DROP
    ELSE
        S" 0 TCSIRIR2-EXPECTED"         WRITE-BLOCK
    THEN ;
T{ RND-TEST-BLOCK DUP TCSIRIR2 LOAD TCSIRIR2-EXPECTED S= -> 0 TRUE }T

\ REFILL across 2 blocks
: TCSIRIR3 ( blks -- )
    DUP S" 1 2 3 REFILL 4 5 6" WRITE-BLOCK
    1+  S" 10 11 12"           WRITE-BLOCK ;
T{ 2 RND-TEST-BLOCK-SEQ DUP TCSIRIR3 LOAD -> 1 2 3 -1 10 11 12 }T

\ SAVE-INPUT and RESTORE-INPUT across 2 blocks
: TCSIRIR4-EXPECTED S" HGF1ECBF1ECBA" ; \ Remember that the string comes out backwards
: TCSIRIR4 ( blks -- )
    C/L 1024 U< IF
        DUP BLANK-BUFFER
        S" 0 T-CNT !"                   WRITE-BLOCK-LINE
        S" <# MARK A SAVE-INPUT MARK B" WRITE-BLOCK-LINE
        S" MARK C REFILL MARK D"        WRITE-BLOCK-LINE
        DROP UPDATE 1+ BLANK-BUFFER
        S" MARK E ABS CHAR 0 + HOLD"    WRITE-BLOCK-LINE
S" 1 T-CNT +! MARK F ' RESTORE-INPUT T-CNT @ 2 < ?EXECUTE MARK G" WRITE-BLOCK-LINE
        S" MARK H 0 0 #>"               WRITE-BLOCK-LINE
        DROP UPDATE SAVE-BUFFERS
    ELSE
        S" 0 TCSIRIR4-EXPECTED"         WRITE-BLOCK
    THEN ;
T{ 2 RND-TEST-BLOCK-SEQ DUP TCSIRIR4 LOAD TCSIRIR4-EXPECTED S= -> 0 TRUE }T

\ ------------------------------------------------------------------------------
TESTING THRU

: TT1 ( blks -- )
    DUP S" BLK" WRITE-BLOCK
    1+  S" @"   WRITE-BLOCK ;
T{ 2 RND-TEST-BLOCK-SEQ DUP TT1 DUP DUP 1+ THRU 1- = -> TRUE }T

\ ------------------------------------------------------------------------------

BLOCK-ERRORS SET-ERROR-COUNT

CR .( End of Block word tests) CR
