\ fpio-test.fs
\
\ Evaluate the floating point input/output number conversion of a 
\ Forth system which uses IEEE floating point format.
\
\ Copyright (c) 2010, Krishna Myneni
\
\ Permission is granted to use this code for any purpose, 
\ provided the copyright notice above is preserved.
\
\ Revisions:
\    2010-11-28  km  created.
\    2010-11-30  km  revised comments; added rounding tests provided
\                    by Andrew P. Haley, and others [1].
\    2010-12-01  km  added further examples from [1--3]; separated
\                    section 2 tests based on system fp precision
\                    to avoid double rounding.
\
\ Notes:
\   0. The "tests" performed by this module are intended to assess
\      the behavior of fp number conversion for a Forth system.
\      Failure of certain tests by a given Forth system does NOT 
\      imply that the system does not conform to any present Forth
\      standard, e.g. Forth-94. 

\   1. Tests are valid only for systems which use IEEE floating point 
\      formats to represent fp numbers. "Round to nearest" mode
\      is assumed (IEEE 754 "nearest ties to even" rounding mode).
\
\   2. The Forth system's floating point number conversion
\      should be capable of processing at least 55 digits
\      excluding the exponent field, in order to run all of these
\      tests.
\
\   3. Currently, only tests for floating point number input 
\      at single and and double precision are performed; need
\      to add extended precision tests for those systems which
\      support the 10 byte format. Also need to test >FLOAT
\      and REPRESENT.
\
\   4. Uses the enhanced ttester-xf test harness by David N.
\      Williams; however, the older ttester.fs may also be
\      used.
\
\ References:
\   1. http://sourceware.org/bugzilla/show_bug.cgi?id=3479
\
\   2. R. Regan, "Incorrectly Rounded Conversions in GCC and GLIBC",
\      http://www.exploringbinary.com/incorrectly-rounded-conversions-in-gcc-and-glibc/
\      June 3, 2010.
\
\   3. R. Regan, "Incorrectly Rounded Conversions in Visual C++",
\      http://www.exploringbinary.com/incorrectly-rounded-conversions-in-visual-c-plus-plus/
\      May 28, 2010.

CR .( Running fpio-test.4th)
CR .( ---------------------)

CR .( FPIO-TEST         V1.1      01 Dec     2010 )
BASE @
[undefined] T{ [if] s" ttester" included [then]

HEX
  4  constant  SINGLE_PREC
  8  constant  DOUBLE_PREC
  A  constant  EXT_PREC
 10  constant  QUAD_PREC

1 FLOATS constant SYSTEM_PREC

\ The following definitions are taken from the reference implementation 
\ of the memory access words Rfd (v. 20100621), for Forth 200x.

: B!    ( x addr --    ) SWAP FF AND SWAP C! ;
: B@    (   addr -- x  ) C@ FF AND ;
: BYTES CHARS ( n1 -- n2 ) ;

: b@+ ( x1 addr1 -- x2 addr2 )  SWAP 8 LSHIFT OVER B@ + SWAP 1 BYTES + ;
: b@- ( x1 addr1 -- x2 addr2 )  1 BYTES - DUP B@ ROT 8 LSHIFT + SWAP ;

: BE-L@ ( addr -- x )  0 SWAP  b@+ b@+ b@+ b@+    DROP ;
: LE-L@ ( addr -- x )  0 SWAP 4 BYTES + b@- b@- b@- b@- DROP ;

1234 PAD !
PAD B@ 34 = [IF]
\ Little-endian systems
: L@   ( a -- u ) LE-L@ ;
: lDF@ ( a -- u ) L@ ;
: uDF@ ( a -- u ) 4 BYTES + L@ ;
[ELSE]
\ Big-endian systems
: L@   ( a -- u ) BE-L@ ;
: lDF@ ( a -- u ) 4 BYTES + L@ ;
: uDF@ ( a -- u ) L@ ;
[THEN] 

: 2L@ dup uDF@ swap lDF@ ;

create r4   4 bytes allot
create r8   8 bytes allot

: !r ( a -- ) ( F: r -- ) fdup r4 sf! r8 df! ;

: dec_t{  decimal t{ ;
: hex_t{  hex     t{ ;

\ Section 1.
cr
TESTING Conversion of Exactly Representable Numbers
dec_t{  0.000000000000000000000000e0 !r -> }t
hex_t{  r4 L@  ->  00000000 }t 
hex_t{  r8 2L@ ->  00000000 00000000 }t

dec_t{  9.99999935045640392457461415399766451285519391957298315801212e-39 !r -> }t
hex_t{  r4 L@  ->  006ce3ee }t
hex_t{  r8 2L@ ->  380b38fb 80000000 }t

dec_t{ -1.00000001335143196001808973960578441619873046875e-10 !r -> }t
hex_t{  r4 L@  ->  aedbe6ff }t
hex_t{  r8 2L@ ->  bddb7cdf e0000000 }t

dec_t{  9.99999974737875163555145263671875e-05 !r -> }t
hex_t{  r4 L@  ->  38d1b717 }t
hex_t{  r8 2L@ ->  3f1a36e2 e0000000 }t

dec_t{  0.100000001490116119384765625e0 !r -> }t
hex_t{  r4 L@  ->  3dcccccd }t
hex_t{  r8 2L@ ->  3fb99999 a0000000 }t

dec_t{  1.0e0 !r -> }t
hex_t{  r4 L@  ->  3f800000 }t
hex_t{  r8 2L@ ->  3ff00000 00000000 }t

dec_t{ -1.0e0 !r -> }t
hex_t{  r4 L@  ->  bf800000 }t
hex_t{  r8 2L@ ->  bff00000 00000000 }t

dec_t{  3.926990926265716552734375e-1 !r -> }t
hex_t{  r4 L@  ->  3ec90fdb  }t
hex_t{  r8 2L@ ->  3fd921fb 60000000 }t

dec_t{  5.235987901687622070312500e-1 !r -> }t
hex_t{  r4 L@  ->  3f060a92 }t
hex_t{  r8 2L@ ->  3fe0c152 40000000 }t

dec_t{  7.853981852531433105468750e-1 !r -> }t
hex_t{  r4 L@  ->  3f490fdb }t
hex_t{  r8 2L@ ->  3fe921fb 60000000 }t

dec_t{  1.047197580337524414062500e0 !r -> }t
hex_t{  r4 L@  ->  3f860a92 }t
hex_t{  r8 2L@ ->  3ff0c152 40000000 }t

dec_t{  1.178097248077392578125000e0 !r -> }t
hex_t{  r4 L@  ->  3f96cbe4 }t
hex_t{  r8 2L@ ->  3ff2d97c 80000000 }t

dec_t{  1.570796370506286621093750e0 !r -> }t
hex_t{  r4 L@  ->  3fc90fdb }t
hex_t{  r8 2L@ ->  3ff921fb 60000000 }t

dec_t{  1.963495373725891113281250e0 !r -> }t
hex_t{  r4 L@  ->  3ffb53d1 }t
hex_t{  r8 2L@ ->  3fff6a7a 20000000 }t

dec_t{  2.094395160675048828125000e0 !r -> }t
hex_t{  r4 L@  ->  40060a92 }t
hex_t{  r8 2L@ ->  4000c152 40000000 }t

dec_t{  2.356194496154785156250000e0 !r -> }t
hex_t{  r4 L@  ->  4016cbe4 }t
hex_t{  r8 2L@ ->  4002d97c 80000000 }t

dec_t{  2.617993831634521484375000e0 !r -> }t
hex_t{  r4 L@  ->  40278d36 }t
hex_t{  r8 2L@ ->  4004f1a6 c0000000 }t

dec_t{  2.748893499374389648437500e0 !r -> }t
hex_t{  r4 L@  ->  402feddf }t
hex_t{  r8 2L@ ->  4005fdbb e0000000 }t

dec_t{  3.141592741012573242187500e0 !r -> }t
hex_t{  r4 L@  ->  40490fdb }t
hex_t{  r8 2L@ ->  400921fb 60000000 }t

dec_t{  10e0 !r -> }t
hex_t{  r4 L@  ->  41200000 }t
hex_t{  r8 2L@ ->  40240000 00000000 }t

dec_t{  1.0e1 !r -> }t
hex_t{  r4 L@  ->  41200000 }t
hex_t{  r8 2L@ ->  40240000 00000000 }t

dec_t{  0.10e2 !r -> }t
hex_t{  r4 L@  ->  41200000 }t
hex_t{  r8 2L@ ->  40240000 00000000 }t

dec_t{  0.010e3 !r -> }t
hex_t{  r4 L@  ->  41200000 }t
hex_t{  r8 2L@ ->  40240000 00000000 }t

dec_t{  0.0000010e7 !r -> }t
hex_t{  r4 L@  ->  41200000 }t
hex_t{  r8 2L@ ->  40240000 00000000 }t

dec_t{  0.000000000000010e15 !r -> }t
hex_t{  r4 L@  ->  41200000 }t
hex_t{  r8 2L@ ->  40240000 00000000 }t

dec_t{  0.0000000000000000000000000000000000010e37 !r -> }t
hex_t{  r4 L@  ->  41200000 }t
hex_t{  r8 2L@ ->  40240000 00000000 }t

dec_t{  1.0e10  !r -> }t
hex_t{  r4 L@  ->  501502f9 }t
hex_t{  r8 2L@ ->  4202a05f 20000000 }t

dec_t{  9999999933815812510711506376257961984e0 !r -> }t
hex_t{  r4 L@  ->  7cf0bdc2  }t
hex_t{  r8 2L@ ->  479e17b8 40000000 }t

\ Section 2.
cr
SYSTEM_PREC DOUBLE_PREC > [IF]
.( System FP precision is not supported for the rounding tests. ) cr
[ELSE]
TESTING Rounding of Numbers

SYSTEM_PREC SINGLE_PREC = [IF]
dec_t{  1.0e-10 !r -> }t
hex_t{  r4 L@  -> 2edbe6ff }t

dec_t{  2.71828182845904523536e0 !r -> }t
hex_t{  r4 L@  -> 402df854 }t

dec_t{  3.14159265358979323846e0 !r -> }t
hex_t{  r4 L@  -> 40490fdb }t

dec_t{  3.518437208883201171875E+013 !r -> }t
hex_t{  r4 L@  -> 56000000 }t

dec_t{  1.00000005960464477550e0 !r -> }t
hex_t{  r4 L@  -> 3f800001 }t

dec_t{  5.00000000000000166533453693773481063544750213623046875e-1 !r -> }t
hex_t{  r4 L@  -> 3f000000 }t

dec_t{  62.5364939768271845828e0 !r -> }t
hex_t{  r4 L@  -> 427a255f }t

dec_t{  8.10109172351e-10 !r ->  }t
hex_t{  r4 L@  -> 305eae5d }t

dec_t{  1.50000000000000011102230246251565404236316680908203125e0 !r ->  }t
hex_t{  r4 L@  -> 3fc00000 }t

dec_t{  9007199254740991.4999999999999999999999999999999995e0 !r ->  }t
hex_t{  r4 L@  -> 5a000000 }t

dec_t{  1.000000000000000111022302462515654042363166809082031250e+00 !r -> }t
hex_t{  r4 L@  -> 3f800000 }t

dec_t{  1.000000000000000111022302462515654042363166809082031251e+00 !r -> }t
hex_t{  r4 L@  -> 3f800000 }t

dec_t{  1.000000000000000111022302462515654042363166809082031251e+00 !r -> }t
hex_t{  r4 L@  -> 3f800000 }t
[ELSE]

dec_t{  1.0e-10 !r -> }t
hex_t{  r8 2L@ -> 3ddb7cdf d9d7bdbb }t

dec_t{  2.71828182845904523536e0 !r -> }t
hex_t{  r8 2L@ -> 4005bf0a 8b145769 }t

dec_t{  3.14159265358979323846e0 !r -> }t
hex_t{  r8 2L@ -> 400921fb 54442d18 }t

dec_t{  3.518437208883201171875E+013 !r -> }t
hex_t{  r8 2L@ -> 42c00000 00000002 }t

dec_t{  1.00000005960464477550e0 !r -> }t
hex_t{  r8 2L@ -> 3ff00000 10000000 }t

dec_t{  5.00000000000000166533453693773481063544750213623046875e-1 !r -> }t
hex_t{  r8 2L@ -> 3fe00000 00000002 }t

dec_t{  62.5364939768271845828e0 !r -> }t
hex_t{  r8 2L@ -> 404f44ab d5aa7ca4 }t

dec_t{  8.10109172351e-10 !r ->  }t
hex_t{  r8 2L@ -> 3e0bd5cb aef0fd0c }t

dec_t{  9214843084008499e0 !r -> }t
hex_t{  r8 2L@ -> 43405e6c ec57761a }t

dec_t{  1.50000000000000011102230246251565404236316680908203125e0 !r ->  }t
hex_t{  r8 2L@ ->  3ff80000 0 }t

dec_t{  9007199254740991.4999999999999999999999999999999995e0 !r ->  }t
hex_t{  r8 2L@ -> 433fffff ffffffff }t

dec_t{  1.000000000000000111022302462515654042363166809082031250e+00 !r -> }t
hex_t{  r8 2L@ -> 3ff00000 00000000 }t

dec_t{  1.000000000000000111022302462515654042363166809082031251e+00 !r -> }t
hex_t{  r8 2L@ -> 3ff00000 00000001 }t

[THEN]

[THEN]
BASE !

CR .( End of fpio-test.4th) CR

