\ paranoia
\
\ A Forth version of William Kahan's Floating Point test program, paranoia.
\ See detailed notes below.
\
\ Ported by Krishna Myneni
\
\ This version is based on the C program, paranoia.c at
\ 
\    http://www.math.utah.edu/~beebe/software/ieee/
\
\ See copying guidelines in the original comments below. Please notify 
\
\    krishna.myneni@ccreweb.org 
\
\ if you find any errors in this Forth translation.
\
\ Revisions:
\    2009-05-19  km;  v1.0 created.
\    2009-05-20  dnw; changed FE. to FS.; added support for PFE
\                     (initialization, case-sensitivity problem),
\                     fixed typo.
\                km; system-specific initialization for VFX Forth
\                    and bigforth.
\   2009-05-25   km; incorporated changes present in the netlib
\                    version of paranoia.c at 
\		     http://www.netlib.org/paranoia/
\                    The description at the original source page
\                    indicates the netlib version is more up to date;
\                    added conditional defn. of <=  as suggested by "Ed"
\                    on comp.lang.forth; added pause logic for testing
\                    1/0 and 0/0 in part8 -- pausing behavior throughout
\                    the program may be set with the constant NOPAUSE
\   2010-04-21   km; updated kForth-specific initialization for
\                    conditional defn. of FLOATS
\ Notes (by km):
\
\   1) This Forth program should run on standard Forth-94 systems with 
\      floating point extensions. See system-specific initialization
\      below these comments.
\  
\   2) Either a separate FP stack system, or an integrated data/FP stack
\      system may be used. Stack diagrams for words are specified for a 
\      separate FP system.
\
\   3) Signal handling has been commented out. The count of floating point
\      exceptions, fpecount, is not valid.
\
\   4) No attempt has been made to factor the code according to good Forth
\      coding practice. This Forth port is a "raw" translation, intended to
\      be directly comparable to the C code from which it was translated. 
\      Since the program examines arithmetic precision, care was taken to 
\      avoid modifying the arithmetic expressions, i.e. changing the order of 
\      operations that commute mathematically, or modifying them for 
\      Forth-readable appearance. Even where it would have made sense to use
\      floating point constants rather than variables, the temptation to
\      do so was resisted in the fear that it could change the behavior of
\      the program.
\
\   5) Other, possibly more up to date, C and Fortran versions of paranoia 
\      may be found at the following site:
\
\	http://orion.math.iastate.edu/burkardt/c_src/paranoia/paranoia.html
\
\      The output from the C version at the above page has been compared
\      to the output for this Forth version, and found to be the same for
\      two different Forth systems by David N. Williams.

\ Add/uncomment your system-specific initialization below; examples for 
\ kForth, PFE, VFX Forth, and bigforth

\ For kForth

\ include ans-words

CR .( Running paranoia.4th)
CR .( --------------------) CR
[undefined] FLOATS [IF] : FLOATS  DFLOATS ; [THEN]


\ For pfe
\ s" FLOATING-EXT" environment? 0= [IF]
\  cr .( ** Floating-point extension words not available **) cr ABORT
\ [THEN] ( flag) drop 

\ For VFX Forth
(
include Hfp387.fth
)

\ For bigforth
(
import float  also float
)


0 [IF]
	A C/C++ version of Kahan's Floating Point Test "Paranoia"

			Thos Sumner, UCSF, Feb. 1985
			David Gay, BTL, Jan. 1986

	This is a rewrite from the Pascal version by

			B. A. Wichmann, 18 Jan. 1985

	(and does NOT exhibit good C programming style).

(C) Apr 19 1983 in BASIC version by:
	Professor W. M. Kahan,
	567 Evans Hall
	Electrical Engineering & Computer Science Dept.
	University of California
	Berkeley, California 94720
	USA

converted to Pascal by:
	B. A. Wichmann
	National Physical Laboratory
	Teddington Middx
	TW11 OLW
	UK

converted to C by:

	David M. Gay		and	Thos Sumner
	AT&T Bell Labs			Computer Center, Rm. U-76
	600 Mountain Avenue		University of California
	Murray Hill, NJ 07974		San Francisco, CA 94143
	USA				USA

converted for K&R, Standard C, and Standard C++ compilation [with loss
of support for the split script below] by:

 	Nelson H. F. Beebe
 	Center for Scientific Computing
 	University of Utah
 	Department of Mathematics, 322 INSCC
 	155 S 1400 E RM 233
 	Salt Lake City, UT 84112-0090
 	USA
 	Email: beebe@math.utah.edu, beebe@acm.org, beebe@computer.org, beebe@ieee.org (Internet)
 	WWW URL: http://www.math.utah.edu/~beebe
 	Telephone: +1 801 581 5254
 	FAX: +1 801 585 1640, +1 801 581 4148

with simultaneous corrections to the Pascal source (reflected
in the Pascal source available over netlib).
[A couple of bug fixes from dgh = sun!dhough incorporated 31 July 1986.]

Reports of results on various systems from all the versions
of Paranoia are being collected by Richard Karpinski at the
same address as Thos Sumner.  This includes sample outputs,
bug reports, and criticisms.

You may copy this program freely if you acknowledge its source.
Comments on the Pascal version to NPL, please.


The C version catches signals from floating-point exceptions.
If signal(SIGFPE,...) is unavailable in your environment, you may
#define NOSIGNAL to comment out the invocations of signal.

This source file is too big for some C compilers, but may be split
into pieces.  Comments containing "SPLIT" suggest convenient places
for this splitting.  At the end of these comments is an "ed script"
(for the UNIX(tm) editor ed) that will do this splitting.

By #defining Single when you compile this source, you may obtain
a single-precision C version of Paranoia.


The following is from the introductory commentary from Wichmann's work:

The BASIC program of Kahan is written in Microsoft BASIC using many
facilities which have no exact analogy in Pascal.  The Pascal
version below cannot therefore be exactly the same.  Rather than be
a minimal transcription of the BASIC program, the Pascal coding
follows the conventional style of block-structured languages.  Hence
the Pascal version could be useful in producing versions in other
structured languages.

Rather than use identifiers of minimal length (which therefore have
little mnemonic significance), the Pascal version uses meaningful
identifiers as follows [Note: A few changes have been made for C;
I and J and Precision have been changed for Forth]:


BASIC   Forth           BASIC   Forth          BASIC   Forth

   A                       J    Jvar               S    StickyBit
   A1   AInverse           J0   NoErrors           T
   B    Radix                    [Failure]         T0   Underflow
   B1   BInverse           J1   NoErrors           T2   ThirtyTwo
   B2   RadixD2                  [SeriousDefect]   T5   OneAndHalf
   B9   BMinusU2           J2   NoErrors           T7   TwentySeven
   C                             [Defect]          T8   TwoForty
   C1   CInverse           J3   NoErrors           U    OneUlp
   D                             [Flaw]            U0   UnderflowThreshold
   D4   FourD              K    PageNo             U1
   E0                      L    Milestone          U2
   E1                      M                       V
   E2   Exp2               N                       V0
   E3                      N1                      V8
   E5   MinSqEr            O    Zero               V9
   E6   SqEr               O1   One                W
   E7   MaxSqEr            O2   Two                X
   E8                      O3   Three              X1
   E9                      O4   Four               X8
   F1   MinusOne           O5   Five               X9   Random1
   F2   Half               O8   Eight              Y
   F3   Third              O9   Nine               Y1
   F6                      P    PrecisionF         Y2
   F9                      Q                       Y9   Random2
   G1   GMult              Q8                      Z
   G2   GDiv               Q9                      Z0   PseudoZero
   G3   GAddSub            R                       Z1
   H                       R1   RMult              Z2
   H1   HInverse           R2   RDiv               Z9
   I    Ivar               R3   RAddSub
   IO   NoTrials           R4   RSqrt
   I3   IEEE               R9   Random9

   SqRWrng

All the variables in BASIC are true variables and in consequence,
the program is more difficult to follow since the "constants" must
be determined (the glossary is very helpful).  The Pascal version
uses Real constants, but checks are added to ensure that the values
are correctly converted by the compiler.

The major textual change to the Pascal version apart from the
identifiersis that named procedures are used, inserting parameters
wherehelpful.  New procedures are also introduced.  The
correspondence is as follows:


BASIC       Pascal
lines

  90- 140   Pause
 170- 250   Instructions
 380- 460   Heading
 480- 670   Characteristics
 690- 870   History
2940-2950   Random
3710-3740   NewD
4040-4080   DoesYequalX
4090-4110   PrintIfNPositive
4640-4850   TestPartialUnderflow

[THEN]

\ 	BadCond 		( n a u --  )
\ 	Characteristics 	(  --  )
\ 	Heading 		(  --  )
\ 	History 		(  --  )
\ 	Instructions 		(  --  )
\ 	IsYeqX 			(  --  )
\ 	NewD 			(  --  )
\ 	Pause 			(  --  )
\ 	PrintIfNPositive 	(  --  )
\ 	Random 			( F: -- r )
\ 	SR3750 			(  --  )
\ 	SR3980 			(  --  )
\ 	Sign 			( F: x -- ) ( -- n )
\ 	SqXMinX 		( n -- )
\ 	TstCond 		( n  Valid  a u -- )
\ 	TstPtUf 		(  --  )
\ 	main 			(  --  )
\ 	msglist 		** Not used **
\ 	notify 			( a u -- )
\ 	pow 			( F: x y -- u )


DECIMAL

s" [UNDEFINED]" pad c! pad char+ pad c@ move 
pad find nip 0=
[IF]
: [UNDEFINED]  ( "name" -- flag )
  bl word find nip 0= ; immediate
[THEN]

s" [DEFINED]" pad c! pad char+ pad c@ move 
pad find nip 0=
[IF]
: [DEFINED]  postpone [UNDEFINED] 0= ; immediate
[THEN]


[UNDEFINED] F~ 
[UNDEFINED] F<  or
[UNDEFINED] F0= or
[UNDEFINED] F** or
[IF]
	.( **  Requires  F**  and  F~  and  F<  and  F0=  ** ) cr ABORT
[THEN] 

[UNDEFINED] FS. [IF]
[DEFINED] F. [IF]  
		: FS. F. ; 
	[ELSE] 
		.( **  Requires  FS.  or  F.  for output  ** ) ABORT
	[THEN]
[THEN]


[UNDEFINED] F= [IF]
	: F= ( F: r1 r2 -- ) ( -- flag )
	    FDUP   F0= IF FABS THEN  FSWAP  
	    FDUP   F0= IF FABS THEN 
	    0E F~ ;
[THEN]

[UNDEFINED] F<> [IF] : F<> ( F: r1 r2 -- ) ( -- flag )   F= invert ;  [THEN]
[UNDEFINED] F>  [IF] 
	: F>  ( F: r1 r2 -- ) ( -- flag )   
	    FOVER FOVER F< >R F= R> or invert ; 
[THEN]
[UNDEFINED] F<= [IF] : F<= ( F: r1 r2 -- ) ( -- flag )   F> invert ;  [THEN]
[UNDEFINED] F>= [IF] : F>= ( F: r1 r2 -- ) ( -- flag )   F< invert ;  [THEN]
[UNDEFINED] S>F [IF] : S>F ( n -- ) ( F: -- r )      	 S>D D>F ;    [THEN]

[UNDEFINED] <=     [IF] : <=  ( n1 n2 -- flag )  2DUP < >R = R> OR ;  [THEN]
[UNDEFINED] ?allot [IF] : ?allot ( u -- a ) HERE SWAP ALLOT ; [THEN]
[UNDEFINED] cell-  [IF] : cell- ( a1 -- a2 ) 1 CELLS - ;      [THEN]

\ --- Arrays (FSL-style) ---

1 cells   constant  INTEGER

\ defining word for 1-d array
: ARRAY ( n cell_size -- | -- addr )
    create 2dup swap 1+ * ?allot ! drop does> cell+ ;

: }   ( addr n -- addr[n] | fetch 1-D array address)
    over cell- @ * swap + ;

True  constant NOPAUSE
True  constant NOSIGNAL

VARIABLE sigsave

FVARIABLE Radix
FVARIABLE BInvrse
FVARIABLE RadixD2
FVARIABLE BMinusU2


VARIABLE NoTrials  20 NoTrials !  \ Number of tests for commutativity.
	  
1  CONSTANT  Yes
0  CONSTANT  No
2  CONSTANT  Chopped
1  CONSTANT  Rounded 
0  CONSTANT  Other
3  CONSTANT  Flaw
2  CONSTANT  Defect
1  CONSTANT  Serious
0  CONSTANT  Failure


FVARIABLE  Zero
FVARIABLE  Half
FVARIABLE  One
FVARIABLE  Two
FVARIABLE  Three
FVARIABLE  Four
FVARIABLE  Five
FVARIABLE  Eight
FVARIABLE  Nine
FVARIABLE  TwentySeven
FVARIABLE  ThirtyTwo
FVARIABLE  TwoForty
FVARIABLE  MinusOne
FVARIABLE  OneAndHalf

VARIABLE Indx
CREATE ch 8 ALLOT
FVARIABLE AInvrse
FVARIABLE A1
FVARIABLE C
FVARIABLE CInvrse
FVARIABLE D
FVARIABLE FourD
FVARIABLE E0
FVARIABLE E1
FVARIABLE Exp2
FVARIABLE E3
FVARIABLE MinSqEr
FVARIABLE SqEr
FVARIABLE MaxSqEr
FVARIABLE E9
FVARIABLE Third
FVARIABLE F6
FVARIABLE F9
FVARIABLE H
FVARIABLE HInvrse
VARIABLE Ivar  		\ int I;
FVARIABLE StickyBit
FVARIABLE Jvar 		\ FLOAT J;
FVARIABLE MyZero
FVARIABLE PrecisionF
FVARIABLE Q
FVARIABLE Q9
FVARIABLE R
FVARIABLE Random9
FVARIABLE T
FVARIABLE Underflow
FVARIABLE S
FVARIABLE OneUlp
FVARIABLE UfThold
FVARIABLE U1
FVARIABLE U2
FVARIABLE V
FVARIABLE V0
FVARIABLE V9
FVARIABLE W
FVARIABLE X
FVARIABLE X1
FVARIABLE X2
FVARIABLE X8
FVARIABLE Random1
FVARIABLE Y
FVARIABLE Y1
FVARIABLE Y2
FVARIABLE Random2
FVARIABLE Z
FVARIABLE PseudoZero
FVARIABLE Z1
FVARIABLE Z2
FVARIABLE Z9
4 INTEGER ARRAY ErrCnt{ 
VARIABLE fpecount
VARIABLE Milestone
VARIABLE PageNo
VARIABLE M
VARIABLE N
VARIABLE N1

VARIABLE GMult
VARIABLE GDiv
VARIABLE GAddSub

VARIABLE RMult
VARIABLE RDiv
VARIABLE RAddSub
VARIABLE RSqrt

VARIABLE Break
VARIABLE Done
VARIABLE NotMonot
VARIABLE Monot
VARIABLE Anomaly
VARIABLE IEEE
VARIABLE SqRWrng
VARIABLE UfNGrad

\ Computed constants.
\ U1  gap below 1.0, i.e, 1.0-U1 is next number below 1.0 
\ U2  gap above 1.0, i.e, 1.0+U2 is next number above 1.0 

\ floating point exception receiver 
: Sigfpe ( i --  )
	1 fpecount +!
	cr ." * * * FLOATING-POINT ERROR * * *" cr
	ABORT 
;

: BadCond ( n a u -- )
	rot dup
	ErrCnt{ swap } 1 swap +!
	CASE
	  Failure OF ." FAILURE " ENDOF
	  Serious OF ." SERIOUS " ENDOF
	  Defect  OF ." DEFECT "  ENDOF
	  Flaw    OF ." FLAW "    ENDOF
	ENDCASE
	type
;


: TstCond ( n  Valid  a u -- )
	rot 0=  IF  BadCond ." ." cr  ELSE 2drop drop THEN
;

: Instructions ( -- )
	cr
	." Lest this program stop prematurely, i.e. before displaying" cr cr
	."    `END OF TEST'," cr cr
	." try to persuade the computer NOT to terminate execution when an" cr
	." error like Over/Underflow or Division by Zero occurs, but rather" cr
	." to persevere with a surrogate value after, perhaps, displaying some" cr
	." warning.  If persuasion avails naught, don't despair but run this" cr
	." program anyway to see how many milestones it passes, and then" cr
	." amend it to make further progress." cr cr
	." Answer questions with Y, y, N or n (unless otherwise indicated)." cr
;


: Heading ( -- )
	." Users are invited to help debug and augment this program so it will" cr
	." cope with unanticipated and newly uncovered arithmetic pathologies." cr
	." Please send suggestions and interesting results to" cr cr
	." Richard Karpinski" cr
	." Computer Center U-76" cr
	." University of California" cr
	." San Francisco, CA 94143-0704, USA" cr cr
	." In doing so, please include the following information:" cr cr
	." Precision: " 
	1 FLOATS 
	CASE
		4 OF ." single" ENDOF
		8 OF ." double" ENDOF
	       10 OF ." long double" ENDOF
		1 FLOATS . ." bytes" 
	ENDCASE
	cr
	." Version: 10 February 1989; Forth" cr
	." Computer:" cr
	." Compiler:" cr
	." Optimization level:" cr
	." Other relevant compiler options:" cr
;


: Characteristics ( -- )
	." Running this program should reveal these characteristics:" cr cr
	."     Radix = 1, 2, 4, 8, 10, 16, 100, 256 ..." cr
	."     Precision = number of significant digits carried." cr
	."     U2 = Radix/Radix^Precision = One Ulp" cr
	." (OneUlpnit in the Last Place) of 1.000xxx ." cr
	."     U1 = 1/Radix^Precision = One Ulp of numbers a little less than 1.0 ." cr
	."     Adequacy of guard digits for Mult., Div. and Subt." cr
	."     Whether arithmetic is chopped, correctly rounded, or something else" cr
	." for Mult., Div., Add/Subt. and Sqrt." cr
	."     Whether a Sticky Bit used correctly for rounding." cr
	."     UnderflowThreshold = an underflow threshold." cr
	."     E0 and PseudoZero tell whether underflow is abrupt, gradual, or fuzzy." cr
	."     V = an overflow threshold, roughly." cr
	."     V0  tells, roughly, whether  Infinity  is represented." cr
	."     Comparisions are checked for consistency with subtraction" cr
	." and for contamination with pseudo-zeros." cr
	."     Sqrt is tested.  Y^X is not tested." cr
	."     Extra-precise subexpressions are revealed but NOT YET tested." cr
	."     Decimal-Binary conversion is NOT YET tested for accuracy." cr
;

: History ( -- )
  \ History 
  \ Converted from Brian Wichmann's Pascal version to C by Thos Sumner,
  \	with further massaging by David M. Gay. 

	." The program attempts to discriminate among" cr cr
	."   FLAWs, like lack of a sticky bit," cr
	."   Serious DEFECTs, like lack of a guard digit, and" cr
	."   FAILUREs, like 2+2 == 5 ." cr cr
	." Failures may confound subsequent diagnoses." cr cr
	." The diagnostic capabilities of this program go beyond an earlier" cr
	." program called `MACHAR', which can be found at the end of the" cr
	." book  `Software Manual for the Elementary Functions' (1980) by" cr
	." W. J. Cody and W. Waite. Although both programs try to discover" cr
	." the Radix, Precision and range (over/underflow thresholds)" cr
	." of the arithmetic, this program tries to cope with a wider variety" cr
	." of pathologies, and to say how well the arithmetic is implemented." cr
	." The program is based upon a conventional radix representation for" cr
	." floating-point numbers, but also allows logarithmic encoding" cr
	." as used by certain early WANG machines." cr cr
	." BASIC version of this program (C) 1983 by Prof. W. M. Kahan;" cr
	." see source comments for more history." cr
;

: notify ( a u -- )
	type ."  test appears to be inconsistent..." cr
	."   PLEASE NOTIFY KARPINKSI!" cr
;



: Pause ( -- )
	NOPAUSE 0= IF
		cr ." To continue, press RETURN"
		KEY drop cr
	THEN
	." Diagnosis resumes after milestone Number " Milestone ? cr
	."          Page: " PageNo ? cr cr
	1 Milestone +!
	1 PageNo +!
;



: Sign  ( F: x -- y )
	0E F>= IF 1.0E ELSE -1.0E THEN
;



\ Random 
\  Random computes
\     X = (Random1 + Random9)^5
\     Random1 = X - FLOOR(X) + 0.000005 * X;
\   and returns the new value of Random1
\


: Random ( F: -- r )
	Random1 F@ Random9 F@ F+  \ F: -- r 
	FDUP FDUP F*  FDUP F* F*  \ F: -- x
	FDUP FDUP FLOOR F-        \ F: -- x  x-floor(x) 
	FSWAP 0.000005E F* F+
;



FVARIABLE XA
FVARIABLE XB

: SqXMinX ( nErrKind -- )

	X F@ BInvrse F@ F*  XB F!
	X F@ XB F@ F- XA F!
	X F@ X F@ F* FSQRT XB F@ F-  XA F@ F- OneUlp F@ F/ SqEr F!
	SqEr F@ Zero F@ F<>  IF
		SqEr F@ MinSqEr F@ F<  IF  SqEr F@ MinSqEr F!  THEN
		SqEr F@ MaxSqEr F@ F>  IF  SqEr F@ MaxSqEr F!  THEN
		Jvar F@ 1.0E F+ Jvar F!
		s" " BadCond
		." sqrt( " X F@ X F@ F* FS. ( %.17e) ."  - " X F@ FS. ( %.17e) ."  = "
		OneUlp F@ * SqEr F@ F* FS. ( %.17e) cr 
		." instead of correct value 0." cr
	ELSE drop
	THEN
;


: NewD ( -- )
	Z1 F@ Q F@ F* X F!
	Half F@  X F@ Radix F@ F/ F- FLOOR  Radix F@ F*  X F@ F+  X F!
	Q F@ X F@ Z F@ F* F- Radix F@ F/  X F@ X F@ F*  D F@ Radix F@ F/ F* F+ Q F!
	Z F@  Two F@ X F@ F* D F@ F*  F-  Z F!
	Z F@ Zero F@ F<=  IF
		Z F@ FNEGATE Z F!
		Z1 F@ FNEGATE Z1 F!
	THEN
	Radix F@ D F@ F* D F!
;



: SR3750 ( -- )
	X F@ Radix F@ F-  Z2 F@ Radix F@ F- F< 
	X F@ Z2 F@ F-  W F@ Z2 F@ F- F>  or invert IF
		1 Ivar +!
		X F@ D F@ F* FSQRT  X2 F!
		X2 F@ Z2 F@ F-  Y F@ Z2 F@ F-  F-  Y2 F!
		X8 F@  Y F@ Half F@ F-  F/  X2 F! 
		X2 F@  Half F@ X2 F@ F* X2 F@ F*  F-  X2 F!
		Y2 F@ Half F@ F+  Half F@ X2 F@ F-  F+  SqEr F!
		SqEr F@ MinSqEr F@ F< IF  SqEr F@  MinSqEr F! THEN
		Y2 F@ X2 F@ F-  SqEr F!
		SqEr F@ MaxSqEr F@ F> IF  SqEr F@  MaxSqEr F! THEN
	THEN
;



: IsYeqX ( -- )
	Y F@ X F@ F<> IF
		N @ 0 <=  IF
			Z F@ Zero F@ F=  Q F@ Zero F@ F<=  and IF
				." WARNING:  computing" cr
			ELSE 
				Defect s" computing" BadCond
			THEN
			."   " Z F@ FS. ( %.17e) ." ^" Q F@ FS. ( %.17e) cr
			."     yielded " Y F@ FS. ( %.17e) cr
			."     which compared unequal to correct " X F@ FS. ( %.17e) cr
			."          they differ by " Y F@ X F@ F- FS. ( %.17e) cr
		THEN
		1 N +!  \ ... count discrepancies. 
	THEN
;


: POW ( F: x y -- z ) \ return x ^ y (exponentiation)
	F** ;


: SR3980 ( -- )
	BEGIN
		Ivar @ S>F Q F! 
		Z F@ Q F@ POW  Y F!
		IsYeqX
		1 Ivar +!  Ivar @ M @ > IF  EXIT THEN
		Z F@ X F@ F* X F!
	X F@ W F@ F<  WHILE
	REPEAT 
;

 

: PrintIfNPositive ( -- )
	N @ 0> IF
		." Similar discrepancies have occurred " N ? ."  times." cr
	THEN
;


 

: TstPtUf ( -- )
	0 N !
	Z F@  Zero F@ F<>  IF
		." Since comparison denies Z = 0, evaluating "
		." (Z + Z) / Z should be safe." cr
		\ sigsave = Sigfpe;
		\ if (setjmp(ovfl_buf)) goto very_serious;
		Z F@ Z F@ F+ Z F@ F/  Q9 F!
		." What the machine gets for (Z + Z) / Z is  " ( %.17e)
			Q9 F@ F. cr
		Q9 F@ Two F@ F- FABS  Radix F@ U2 F@ F*  F< IF
			." This is O.K., provided Over/Underflow"
			."  has NOT just been signaled." cr
		ELSE
			Q9 F@ One F@ F<  Q9 F@ Two F@ F> or IF
\ very_serious:
				1 N !
				1 ErrCnt{ Serious } +!
				." This is a VERY SERIOUS DEFECT!" cr
			ELSE
				1 N !
				1 ErrCnt{ Defect } +!
				." This is a DEFECT!" cr
			THEN
		THEN
		0 sigsave !
		Z F@ One F@ F* V9 F!
		V9 F@ Random1 F!
		One F@ Z F@ F* V9 F!
		V9 F@ Random2 F!
		Z F@ One F@ F/ V9 F!

		Z F@ Random1 F@ F=  
		Z F@ Random2 F@ F= and 
		Z F@ V9 F@ F= and IF
			N @ 0> IF Pause THEN
			
		ELSE
			1 N !
			Defect s" What prints as Z = " BadCond
			( %.17e ) Z F@ FS. ." compares different from  "
			Z F@ Random1 F@ F<> IF ." Z * 1 = " ( %.17e) Random1 F@ FS. THEN
			Z F@ Random2 F@ F=  
			Random2 F@ Random1 F@ F= or invert  IF
				." 1 * Z == " Random2 F@ F. cr
			THEN
			Z F@ V9 F@ F= invert IF ." Z / 1 = " ( %.17e) V9 F@ FS. cr THEN
			Random2 F@ Random1 F@ F<> IF 
				1 ErrCnt{ Defect } +!
				Defect  s" Multiplication does not commute!" BadCond
				." Comparison alleges that 1 * Z = " ( %.17e) 
					Random2 F@ FS. cr 
				." differs from Z * 1 = " ( %.17e) Random1 F@ FS. cr
			THEN
			Pause
		THEN
	THEN
;


: part2 ( -- )
	\ =============================================
	10 Milestone !
	\ =============================================
	Failure 
	Three F@ Three F@ F* Nine F@ F= 
	Nine F@ Three F@ F* TwentySeven F@ F= and 
	Four F@ Four F@ F+ Eight F@ F= and
	Eight F@ Four F@ F* ThirtyTwo F@ F= and 
	ThirtyTwo F@ TwentySeven F@ F- Four F@ F- One F@ F- Zero F@ F= and
	s" 9 != 3*3, 27 != 9*3, 32 != 8*4, or 32-27-4-1 != 0" 
	TstCond

	Failure 
	Four F@ One F@ F+ Five F@ F= 
	Four F@ Five F@ F* Three F@ F* Four F@ F* TwoForty F@ F= and
	TwoForty F@ Three F@ F/ Four F@ Four F@ F* Five F@ F* F- Zero F@ F= and
	TwoForty F@ Four F@ F/  Five F@ Three F@ F* Four F@ F* F- Zero F@ F= and
	TwoForty F@ Five F@ F/ Four F@ Three F@ F* Four F@ F* F- Zero F@ F= and
	s" 5 != 4+1, 240/3 != 80, 240/4 != 60, or 240/5 != 48" 
	TstCond

	ErrCnt{ Failure } @ 0=  IF 
		." -1, 0, 1/2, 1, 2, 3, 4, 5, 9, 27, 32 & 240 are O.K." cr
		cr
	THEN
	." Searching for Radix and Precision." cr
	One F@ W F!
	BEGIN
		W F@ W   F@ F+ W F!
		W F@ One F@ F+ Y F!
		Y F@ W   F@ F- Z F!
		Z F@ One F@ F- Y F!
	MinusOne F@  Y F@ FABS F+  Zero F@ F<  WHILE
	REPEAT
 
	\ .. now W is just big enough that |((W+1)-W)-1| >= 1 ...
	Zero F@ PrecisionF F!
	One F@ Y F!
	BEGIN
		W F@  Y F@ F+  Radix F!
		Y F@  Y F@ F+  Y F!
		Radix F@  W F@ F-  Radix F!
	Radix F@ Zero F@ F=  WHILE
	REPEAT

	Radix F@ Two F@ F< IF  One F@ Radix F!  THEN
	." Radix = " Radix F@ F. ." ." cr
	Radix F@ 1E F<> IF
		One F@ W F!
		BEGIN
			PrecisionF F@ One F@ F+ PrecisionF F!
			W F@ Radix F@ F* W F!
			W F@ One F@ F+  Y F!
		Y F@ W F@ F- One F@ F=  WHILE
		REPEAT
	THEN
	\ ... now W == Radix^Precision is barely too big to satisfy (W+1)-W == 1
	\		                              ...
	One F@ W F@ F/     U1 F!
	Radix F@ U1 F@ F*  U2 F!
	." Closest relative separation found is U1 = " U1 F@ FS. ( %.7e) cr cr
	." Recalculating radix and precision " cr

	\ save old values
	Radix F@ E0 F!
	U1 F@    E1 F!
	U2 F@    E9 F!
	PrecisionF F@ E3 F!

	Four F@ Three F@ F/ X F!
	X F@ One F@ F- Third F!
	Half F@  Third F@ F- F6 F!
	F6 F@ F6 F@ F+ X F!
	X F@ Third F@ F- FABS X F!
	X F@ U2 F@ F<  IF  U2 F@ X F!  THEN

	\ ... now X = (unknown no.) ulps of 1+...
	BEGIN
		X F@ U2 F!
		Half F@ U2 F@ F* ThirtyTwo F@ U2 F@ F* U2 F@ F* F+ Y F!
		One F@ Y F@ F+ Y F!
		Y F@ One F@ F- X F! 
	U2 F@ X F@ F<=  X F@ Zero F@ F<=  or invert WHILE
	REPEAT

	\ ... now U2 == 1 ulp of 1 + ... 
	Two F@ Three F@ F/ X F!
	X F@ Half F@ F- F6 F!
	F6 F@ F6 F@ F+ Third F! 
	Third F@ Half F@ F- X F!
	X F@ F6 F@ F+ FABS X F!
	X F@ U1 F@ F< IF U1 F@ X F! THEN

	\ ... now  X == (unknown no.) ulps of 1 -... 
	BEGIN
		X F@ U1 F!
		Half F@ U1 F@ F* ThirtyTwo F@ U1 F@ F* U1 F@ F* F+ Y F!
		Half F@ Y F@ F- Y F!
		Half F@ Y F@ F+ X F!
		Half F@ X F@ F- Y F!
		Half F@ Y F@ F+ X F!
	U1 F@ X F@ F<=  X F@ Zero F@ F<= or invert WHILE
	REPEAT
	\ ... now U1 == 1 ulp of 1 - ... 
	U1 F@ E1 F@ F= IF 
		." confirms closest relative separation U1 ." cr
	ELSE 
		." gets better closest relative separation U1 = " U1 F@ FS. ( %.7e) cr
	THEN
	One F@ U1 F@ F/ W F!
	Half F@ U1 F@ F- Half F@ F+ F9 F!
	0.01E U2 F@ U1 F@ F/ F+ FLOOR Radix F!
	Radix F@ E0 F@ F=  IF 
		." Radix confirmed." cr
	ELSE 
		." MYSTERY: recalculated Radix = " ( %.7e) Radix F@ FS. cr
	THEN

	Defect 
	Radix F@  Eight F@ Eight F@ F+ F<= 
	s" Radix is too big: roundoff problems"
	TstCond

	Flaw 
	Radix F@ Two F@ F=  
	Radix F@ 10E F= or
	Radix F@ One F@ F= or 
	s" Radix is not as good as 2 or 10"
	TstCond
	\ =============================================
	20 Milestone !
	\ =============================================
	
	Failure
	F9 F@ Half F@ F-  Half F@ F< 
	s" (1-U1)-1/2 < 1/2 is FALSE, prog. fails?"
	TstCond

	F9 F@ X F!
	1 Ivar !
	X F@ Half F@ F- Y F!
	Y F@ Half F@ F- Z F!

	Failure
	X F@ One F@ F<>  Z F@ Zero F@ F= or 
	s" Comparison is fuzzy,X=1 but X-1/2-1/2 != 0"
	TstCond

	One F@ U2 F@ F+ X F!
	0 Ivar !
	\ =============================================
	25 Milestone !
	\ =============================================
	\ ... BMinusU2 = nextafter(Radix, 0) 
	Radix F@ One F@ F- BMinusU2 F!
	BMinusU2 F@ U2 F@ F- One F@ F+ BMinusU2 F! 
	\ Purify Integers 
	Radix F@ One F@ F<> IF
		TwoForty F@ U1 F@ FLN F* Radix F@ FLN F/ FNEGATE X F! 
		Half F@  X F@ F+ FLOOR Y F!
		X F@ Y F@ F- FABS Four F@ F* One F@ F<  IF Y F@ X F! THEN
		X F@ TwoForty F@ F/ PrecisionF F!
		Half F@ PrecisionF F@ F+ FLOOR Y F!
		PrecisionF F@ Y F@ F- FABS TwoForty F@ F* Half F@ F< IF Y F@ PrecisionF F! THEN
	THEN
	PrecisionF F@ FLOOR PrecisionF F@ F<>   Radix F@ One F@ F= or IF
		." Precision cannot be characterized by an Integer number" cr
		." of significant digits but, by itself, this is a minor flaw." cr
	THEN
	Radix F@ One F@ F=  IF
		." logarithmic encoding has precision characterized solely by U1." cr
	ELSE 
		." The number of significant digits of the Radix is " PrecisionF F@ F.
		cr
	THEN

	Serious
	U2 F@ Nine F@ F* Nine F@ F* TwoForty F@ F* One F@ F<
	s" Precision worse than 5 decimal figures  "
	TstCond
	\ =============================================
	30 Milestone !
	\ =============================================
	\ Test for extra-precise subepressions 
	\ X = FABS(((Four / Three - One) - One / Four) * Three - One / Four);
	Four F@ Three F@ F/  One F@ F-  One F@ Four F@ F/ F- 
	Three F@ F*  One F@ Four F@ F/ F- FABS  X F!
	BEGIN
		X F@ Z2 F!
		\ X = (One + (Half * Z2 + ThirtyTwo * Z2 * Z2)) - One;
		Half F@ Z2 F@ F* ThirtyTwo F@ Z2 F@ F* Z2 F@ F* F+
		One F@ F+  One F@ F- X F!
	Z2 F@ X F@ F<=   X F@ Zero F@ F<= or invert	WHILE
	REPEAT

	\ X = Y = Z = FABS((Three / Four - Two / Three) * Three - One / Four); 
	Three F@ Four F@ F/  Two F@ Three F@ F/ F- 
	Three F@ F*  One F@ Four F@ F/ F- FABS 
	FDUP Z F! FDUP Y F! X F!
	BEGIN
		Z F@ Z1 F!
		\ Z = (One / Two - ((One / Two - (Half * Z1 + ThirtyTwo * Z1 * Z1))
		\ 	+ One / Two)) + One / Two;
		Half F@ Z1 F@ F*  ThirtyTwo F@ Z1 F@ F* Z1 F@ F* F+
		One F@ Two F@ F/ FSWAP F- 
		One F@ Two F@ F/ F+
		One F@ Two F@ F/ F-  
		One F@ Two F@ F/ F+  Z F!
	Z1 F@ Z F@ F<=  Z F@ Zero F@ F<= or invert  WHILE
	REPEAT
 
	BEGIN
		BEGIN
			Y F@ Y1 F!
			\ Y = (Half - ((Half - (Half * Y1 + ThirtyTwo * Y1 * Y1)) + Half
			\ 	)) + Half;
			Half F@ Y1 F@ F* ThirtyTwo F@ Y1 F@ F* Y1 F@ F* F+
			Half F@ FSWAP F-  Half F@ F+
			Half F@ FSWAP F-  Half F@ F+  Y F!
		Y1 F@ Y F@ F<=   Y F@ Zero F@ F<= or invert WHILE 
		REPEAT 
		X F@ X1 F!
		\ X = ((Half * X1 + ThirtyTwo * X1 * X1) - F9) + F9;
		Half F@ X1 F@ F* ThirtyTwo F@ X1 F@ F* X1 F@ F* F+  
		F9 F@ F-  F9 F@ F+  X F!
	X1 F@ X F@ F<=   X F@ Zero F@ F<= or invert WHILE
	REPEAT
 
	X1 F@ Y1 F@ F<>   X1 F@ Z1 F@ F<> or IF
		Serious s" Disagreements among the values X1, Y1, Z1" BadCond
		." respectively  " X1 F@ FS. ( %.7e)  Y1 F@ FS. ( %.7e) Z1 F@ FS. ( %.7e) cr
		." are symptoms of inconsistencies introduced" cr
		." by extra-precise evaluation of arithmetic subexpressions." cr
		s" Possibly some part of this" notify
		X1 F@ U1 F@ F=  Y1 F@ U1 F@ F= or  Z1 F@ U1 F@ F= or IF
			." That feature is not tested further by this program." cr
		THEN
	ELSE
		Z1 F@ U1 F@ F<>   Z2 F@ U2 F@ F<> or IF
			Z1 F@ U1 F@ F>=  Z2 F@ U2 F@ F>= or IF
				Failure s" " BadCond
				s" Precision" notify
				."    U1 = " U1 F@  FS. ( %.7e)
				." Z1 - U1 = " Z1 F@ U1 F@ F- FS. ( %.7e) cr
				."    U2 = " U2 F@  FS. ( %.7e)
				." Z2 - U2 = " Z2 F@ U2 F@ F-  FS. ( %.7e) cr
			ELSE
				Z1 F@ Zero F@ F<=   Z2 F@ Zero F@ F<= or IF
					." Because of unusual Radix = " Radix F@ F.
					." , or exact rational arithmetic a result" cr
					." Z1 = "  Z1 F@ FS. ( %.7e) ." or Z2 = " Z2 F@ FS. ( %.7e)
					s" of an extra-precision" notify cr
				THEN
				Z1 F@ Z2 F@ F<>   Z1 F@ Zero F@ F> or IF
					Z1 F@ U1 F@ F/ X F!
					Z2 F@ U2 F@ F/ Y F!
					Y F@  X F@ F>  IF Y F@ X F! THEN
					X F@ FLN FNEGATE Q F!
					." Some subexpressions appear to be calculated extra" cr
					." precisely with about " Q F@ Radix F@ FLN F/ ( %g) F. ." extra B-digits, i.e." cr
						
					." roughly " Q F@ 10E FLN F/ ( %g) F. ." extra significant decimals." cr
						
				THEN
				." That feature is not tested further by this program." cr
			THEN
		THEN
	THEN
	Pause
;
\ end of part2

: part3 ( -- )
	\ =============================================
	35 Milestone !
	\ =============================================
	Radix F@ Two F@ F>= IF
		W F@  Radix F@ Radix F@ F* F/  X F! 
		X F@ One F@ F+  Y F!
		Y F@ X F@ F-    Z F!
		Z F@ U2 F@ F+   T F!
		T F@ Z F@ F-    X F!

		Failure  X F@ U2 F@ F=  s" Subtraction is not normalized X=Y,X+Z != Y+Z!"
		TstCond

		X F@ U2 F@ F=  IF 
			." Subtraction appears to be normalized, as it should be."
		THEN
	THEN

	cr ." Checking for guard digit in F*, F/, and F-." cr
	F9 F@ One F@ F*    Y F!
	One F@ F9 F@ F*    Z F!
	F9 F@ Half F@ F-   X F!
	Y F@ Half F@ F- X F@ F-    Y F! 
	Z F@ Half F@ F- X F@ F-    Z F!
	One F@ U2 F@ F+    X F!
	X F@ Radix F@ F*   T F!
	Radix F@ X F@ F*   R F!
	T F@ Radix F@ F-   X F!
	X F@ Radix F@  U2 F@ F* F- X F! 
	R F@ Radix F@ F-   T F! 
	T F@ Radix F@  U2 F@ F* F- T F!
	X F@ Radix F@ One F@ F- F* X F!
	T F@ Radix F@ One F@ F- F* T F!

	X F@ Zero F@ F=       Y F@ Zero F@ F=  and 
	Z F@ Zero F@ F=  and  T F@ Zero F@ F=  and  IF
		Yes GMult !
	ELSE
		No GMult !
		Serious  False  s" F* lacks a Guard Digit, so 1*X != X"
		TstCond
	THEN
	Radix F@ U2 F@ F* Z F!
	One F@ Z F@ F+    X F!
	X F@ Z F@ F+   X F@ X F@ F* F- FABS U2 F@ F-  Y F!
	One F@ U2 F@ F-   X F!
	X F@ U2 F@ F-  X F@ X F@ F* F- FABS U1 F@ F-  Z F!

	Failure  Y F@ Zero F@ F<=   Z F@ Zero F@ F<=  and
	s" F* gets too many final digits wrong." 
	TstCond

	One F@ U2 F@ F- Y F!
	One F@ U2 F@ F+ X F!
	One F@ Y F@ F/  Z F!
	Z F@ X F@ F-    Y F!
	One F@ Three F@ F/        X F!
	Three F@ Nine F@ F/   Z F!
	X F@ Z F@ F-    X F!
	Nine F@ TwentySeven F@ F/ T F!
	Z F@ T F@ F-    Z F!
	
	Defect  X F@ Zero F@ F=   Y F@ Zero F@ F=  and  Z F@ Zero F@ F= and
	s" Division lacks a Guard Digit, so error can exceed 1 ulp\nor  1/3  and  3/9  and  9/27 may disagree"
	TstCond

	F9 F@  One F@ F/ Y F!
	F9 F@ Half F@ F- X F!
	Y  F@ Half F@ F- X F@ F- Y F!
	One F@  U2 F@ F+ X F!
	X  F@  One F@ F/ T F!
	T  F@    X F@ F- X F!
	X F@ Zero F@ F=   Y F@ Zero F@ F= and   Z F@ Zero F@ F= and IF
		Yes GDiv !
	ELSE
		No GDiv !

		Serious  False  s" Division lacks a Guard Digit, so X/1 != X"
		TstCond
	THEN
	One F@  One  F@ U2 F@ F+  F/  X F!
	X   F@  Half F@ F- Half F@ F- Y F!
	
	Serious  Y F@ Zero F@ F<  s" Computed value of 1/1.000..1 >= 1"
	TstCond

	One F@ U2 F@ F- X F!
	One F@ Radix F@ U2 F@ F*  F+  Y F!
	X F@ Radix F@ F* Z F!
	Y F@ Radix F@ F* T F!
	Z F@ Radix F@ F/ R F!
	T F@ Radix F@ F/ StickyBit F!
	R F@ X F@ F- X F!
	StickyBit F@ Y F@ F- Y F!

	Failure  X F@ Zero F@ F=   Y F@ Zero F@ F=  and
	s" F* and/or F/ gets too many last digits wrong"
	TstCond

	One F@ U1 F@ F- Y F!
	One F@ F9 F@ F- X F!
	One F@  Y F@ F- Y F!
	Radix F@ U2 F@ F- T F!
	Radix F@ BMinusU2 F@ F- Z F!
	Radix F@ T F@ F-  T F!
	X F@ U1 F@ F=   Y F@ U1 F@ F= and  Z F@ U2 F@ F= and  T F@ U2 F@ F= and IF
		Yes GAddSub !
	ELSE
		No GAddSub !
		Serious  False  s" F- lacks Guard Digit, so cancellation is obscured"
		TstCond
	THEN
	F9 F@ One F@ F<>   F9 F@ One F@ F- Zero F@ F>=  and IF
		Serious s" comparison alleges  (1-U1) < 1  although " BadCond
		."  subtraction yields  (1-U1) - 1 = 0 , thereby vitiating" cr
		."  such precautions against division by zero as" cr
		."  ...  if (X == 1.0) {.....} else {.../(X-1.0)...}" cr
	THEN
	Yes GMult @ =   Yes GDiv @ = and  Yes GAddSub @ = and IF
		."     F*, F/, and F- appear to have guard digits, as they should." cr
	THEN
	\ =============================================
	40 Milestone !
	\ =============================================
	Pause
	." Checking rounding on multiply, divide and add/subtract." cr
	Other RMult !
	Other RDiv  !
	Other RAddSub !
	Radix F@ Two F@ F/  RadixD2 F!
	Two F@ A1 F!
	False Done !
	BEGIN
		Radix F@ AInvrse F!
		BEGIN
			AInvrse F@ X F!
			AInvrse F@ A1 F@ F/ AInvrse F! 
		AInvrse F@ FLOOR AInvrse F@ F<> invert WHILE
		REPEAT
		X F@ One F@ F=   A1 F@ Three F@ F>  or Done !
		Done @ invert IF  Nine F@ One F@ F+  A1 F! THEN
	Done @ invert WHILE 
	REPEAT
	X F@ One F@ F=  IF Radix F@ A1 F! THEN
	One F@ A1 F@ F/  AInvrse F!
	A1 F@  X F!
	AInvrse F@ Y F!
	False Done !
	BEGIN
		X F@ Y F@ F* Half F@ F- Z F!
		Failure  Z F@ Half F@ F=  s" X * (1/X) differs from 1" TstCond
		X F@ Radix F@ F= Done !
		Radix F@ X F!
		One F@ X F@ F/ Y F!
	Done @ invert WHILE
	REPEAT
	One F@ U2 F@ F+ Y2 F!
	One F@ U2 F@ F- Y1 F!
	OneAndHalf F@ U2 F@ F- X F!
	OneAndHalf F@ U2 F@ F+ Y F!
	X F@ U2 F@ F- Y2 F@ F* Z F!
	Y F@ Y1 F@ F* T F!
	Z F@  X F@ F- Z F!
	T F@  X F@ F- T F!
	X F@ Y2 F@ F* X F!
	Y F@ U2 F@ F+ Y1 F@ F* Y F!
	X F@ OneAndHalf F@ F- X F!
	Y F@ OneAndHalf F@ F- Y F!
	X F@ Zero F@ F=   Y F@ Zero F@ F= and  Z F@ Zero F@ F= and  T F@ Zero F@ F<= and IF
		OneAndHalf F@  U2 F@ F+  Y2 F@ F*  X F!
		OneAndHalf F@  U2 F@ F-  U2 F@ F-  Y F!
		OneAndHalf F@  U2 F@ F+  U2 F@ F+  Z F!
		OneAndHalf F@  U2 F@ F-  Y1 F@ F*  T F!
		X F@   Z F@ U2 F@ F+ F-  X F!
		Y F@  Y1 F@ F*  StickyBit F!
		Z F@  Y2 F@ F*  S F!
		T F@   Y F@ F-  T F!
		U2 F@ Y F@ F- StickyBit F@ F+ Y F!
		S F@  Z F@ U2 F@ F+ U2 F@ F+ F- Z F!
		Y2 F@  U2 F@ F+  Y1 F@ F*  StickyBit F!
		Y2 F@ Y1 F@ F* Y1 F!
		StickyBit F@  Y2 F@ F-  StickyBit F!
		Y1 F@  Half F@ F- Y1 F!
		X F@ Zero F@ F=      Y F@ Zero F@ F= and  Z F@ Zero F@ F= and  
		T F@ Zero F@ F= and  StickyBit F@ Zero F@ F= and  Y1 F@ Half F@ F= and IF
			Rounded RMult !
			." Multiplication appears to round correctly." cr

		ELSE
			X F@ U2 F@ F+ Zero F@ F=  
			Y F@ Zero F@ F< and 
			Z F@ U2 F@ F+ Zero F@ F= and 
			T F@ Zero F@ F< and 
		        StickyBit F@ U2 F@ F+ Zero F@ F= and
			Y1 F@ Half F@ F< and IF
				Chopped RMult !
				." Multiplication appears to chop." cr
			ELSE
			 	." F* is neither chopped nor correctly rounded." cr 
				RMult @ Rounded =   GMult @ No =  and IF 
					s" Multiplication" notify
				THEN
			THEN
		THEN
	ELSE 
		." F* is neither chopped nor correctly rounded." cr ( ABORT)
	THEN
	\ =============================================
	45 Milestone !
	\ =============================================
	One F@ U2 F@ F+ Y2 F!
	One F@ U2 F@ F- Y1 F!
	OneAndHalf F@ U2 F@ F+ U2 F@ F+  Z F!
	Z F@ Y2 F@ F/ X F!
	OneAndHalf F@ U2 F@ F- U2 F@ F-  T F!
	T F@ U2 F@ F-  Y1 F@ F/  Y F!
	Z F@ U2 F@ F+  Y2 F@ F/  Z F!
	X F@   OneAndHalf F@ F-  X F!
	Y F@  T F@ F-  Y F!
	T F@ Y1 F@ F/  T F!
	Z F@  OneAndHalf F@ U2 F@ F+ F- Z F!
	U2 F@ OneAndHalf F@ F-  T F@ F+ T F!

	X F@ Zero F@ F>     Y F@ Zero F@ F> or  
	Z F@ Zero F@ F> or  T F@ Zero F@ F> or  invert IF
		OneAndHalf F@  Y2 F@ F/  X F!
		OneAndHalf F@  U2 F@ F-  Y F!
		OneAndHalf F@  U2 F@ F+  Z F!
		X F@   Y F@ F-  X F!
		OneAndHalf F@  Y1 F@ F/  T F!
		Y F@  Y1 F@ F/  Y F!
		T F@   Z F@ U2 F@ F+ F-  T F!
		Y F@   Z F@ F- Y F!
		Z F@  Y2 F@ F/ Z F!
		Y2 F@ U2 F@ F+ Y2 F@ F/  Y1 F!
		Z F@   OneAndHalf F@ F-  Z F!
		Y1 F@ Y2 F@ F- Y2 F!
		F9 F@ U1 F@ F- F9 F@ F/  Y1 F!
		X F@ Zero F@ F=      Y  F@ Zero F@ F= and  Z  F@ Zero F@ F= and 
		T F@ Zero F@ F= and  Y2 F@ Zero F@ F= and  Y2 F@ Zero F@ F= and
		Y1 F@ Half F@ F- F9 F@ Half F@ F- F= and IF
			Rounded RDiv !
			." Division appears to round correctly." cr
			GDiv @ No =  IF  s" Division" notify  THEN

		ELSE
			X  F@ Zero F@ F<      Y F@ Zero F@ F< and 
			Z  F@ Zero F@ F< and  T F@ Zero F@ F< and
			Y2 F@ Zero F@ F< and  
			Y1 F@ Half F@ F- F9 F@ Half F@ F- F< and IF
				Chopped RDiv !
				." Division appears to chop." cr
			THEN
		THEN
	THEN
	RDiv @ Other = IF ." F/ is neither chopped nor correctly rounded." cr THEN
	One F@ Radix F@ F/ BInvrse F!

	Failure 
	BInvrse F@ Radix F@ F* Half F@ F- Half F@ F= 
	s" Radix * ( 1 / Radix ) differs from 1"
	TstCond
;
\ end part3

: part4_loopA ( -- )
	1 Ivar !
	BEGIN
		Ivar @ NoTrials @ <= WHILE 
		X F@ One F@ F+ X F!
		Defect SqXMinX
		Jvar F@  Zero F@ F> IF EXIT THEN
		1 Ivar +!
	REPEAT
;


: part4 ( -- )
	\ =============================================
	50 Milestone !
	\ =============================================
	Failure  
	F9 F@ U1 F@ F+ Half F@ F- Half F@ F= 
	BMinusU2 F@ U2 F@ F+ One F@ F- Radix F@ One F@ F- F= and
	s" Incomplete carry-propagation in Addition"
	TstCond

	One F@  U1 F@ U1 F@ F* F- X F!
	One F@ U2 F@ One F@  U2 F@ F- F* F+ Y F!
	F9 F@ Half F@ F- Z F!
	X F@ Half F@ F- Z F@ F- X F!
	Y F@ One F@ F- Y F!
	X F@ Zero F@ F=   Y F@ Zero F@ F=  and IF
		Chopped RAddSub !
		." Add/Subtract appears to be chopped." cr
	THEN
	GAddSub @ Yes = IF
		Half F@  U2 F@  F+  U2 F@  F*  X F!
		Half F@  U2 F@  F-  U2 F@  F*  Y F!
		One F@  X F@ F+  X F!
		One F@  Y F@ F+  Y F!
		One F@  U2 F@ F+  X F@ F- X F!
		One F@  Y F@ F- Y F!
		X F@ Zero F@ F=  Y F@ Zero F@ F=  and IF
			Half F@ U2 F@ F+  U1 F@ F*  X F!
			Half F@ U2 F@ F-  U1 F@ F*  Y F!
			One F@  X F@ F-  X F!
			One F@  Y F@ F-  Y F!
			F9  F@  X F@ F-  X F!
			One F@  Y F@ F-  Y F!
			X F@ Zero F@ F=   Y F@ Zero F@ F=  and IF 
				Rounded RAddSub !
				." Addition/Subtraction appears to round correctly." cr
				GAddSub @ No = IF  s" Add/Subtract" notify  THEN
			ELSE 
				." Addition/Subtraction neither rounds nor chops." cr
			THEN
		ELSE 
			." Addition/Subtraction neither rounds nor chops." cr
		THEN
	ELSE 
		." Addition/Subtraction neither rounds nor chops." cr
	THEN
	One F@ S F!
	One F@  Half F@  One F@ Half F@ F+ F* F+  X F!
	One F@  U2 F@ F+ Half F@ F*  Y F!
	X F@  Y F@ F-  Z F!
	Y F@  X F@ F-  T F!
	Z F@  T F@ F+  StickyBit F!
	StickyBit F@ Zero F@ F<> IF
		Zero F@ S F!
		Flaw s" (X - Y) + (Y - X) is non zero!" BadCond
	THEN
	Zero F@ StickyBit F!

	GMult @ Yes =   GDiv @ Yes = and   GAddSub @ Yes = and
	RMult @ Rounded = and    RDiv @ Rounded = and
	RAddSub @ Rounded = and  RadixD2 F@ FLOOR RadixD2 F@ F= and  IF
		." Checking for sticky bit." cr
		Half F@  U1 F@ F+  U2 F@ F*  X F!
		Half F@  U2 F@ F*  Y F!
		One  F@  Y  F@ F+  Z F! 
		One  F@  X  F@ F+  T F!
		Z F@ One F@ F- Zero F@ F<=   T F@ One F@ F- U2 F@ F>=  and IF
			T F@ Y F@ F+  Z F!
			Z F@ X F@ F-  Y F!
			Z F@ T F@ F- U2 F@ F>=   Y F@ T F@ F- Zero F@ F=  and IF
				Half F@ U1 F@ F+  U1 F@ F*  X F!
				Half F@ U1 F@ F*  Y F!
				One  F@ Y  F@ F-  Z F!
				One  F@ X  F@ F-  T F!
				Z F@ One F@ F- Zero F@ F=  T F@ F9 F@ F- Zero F@ F=  and IF
					Half F@ U1 F@ F- U1 F@ F*  Z F!
					F9 F@ Z F@ F-  T F!
					F9 F@ Y F@ F-  Q F!
					T F@ F9 F@ F- Zero F@ F= 
					F9 F@ U1 F@ F- Q F@ F- Zero F@ F=  and IF
						One F@ U2 F@ F+  OneAndHalf F@ F* Z F!
						OneAndHalf F@ U2 F@ F+ Z F@ F- U2 F@ F+  T F!
						One F@ Half F@ Radix F@ F/ F+  X F!
						One F@ Radix F@ U2 F@ F* F+ Y F!
						X F@ Y F@ F* Z F!
						T F@ Zero F@ F=
						X F@ Radix F@ U2 F@ F* F+ 
						Z F@ F- Zero F@ F= and IF
							Radix F@ Two F@ F<> IF
								Two F@ U2 F@ F+ X F! 
								X F@ Two F@ F/ Y F!
								Y F@ One F@ F- Zero F@ F=  IF
									S F@ StickyBit F!
								THEN
							ELSE
								S F@ StickyBit F!
							THEN
						THEN
					THEN
				THEN
			THEN
		THEN
	THEN
	StickyBit F@ One F@ F=  IF
		." Sticky bit apparently used correctly." cr
	ELSE
		." Sticky bit used incorrectly or not at all." cr
	THEN

	Flaw  
	GMult @ No =  GDiv @ No = or  GAddSub @ No = or 
	RMult @ Other = or  RDiv @ Other = or  RAddSub @ Other = or invert
	s" lack(s) of guard digits or failure(s) to correctly round or chop (noted above) count as one flaw in the final tally below"
	TstCond
	\ =============================================
	60 Milestone !
	\ =============================================
	cr
	." Does Multiplication commute?  "
	." Testing on " NoTrials ? ." random pairs." cr
	3.0E FSQRT Random9 F!
	Third F@   Random1 F!
	1 Ivar !
	BEGIN
		Random X F!
		Random Y F!
		Y F@ X F@ F*  Z9 F!
		X F@ Y F@ F*  Z F!
		Z F@ Z9 F@ F- Z9 F!
		1 Ivar +!
	Ivar @  NoTrials @ >   Z9 F@ Zero F@ F<>  or invert WHILE
	REPEAT 
	Ivar @ NoTrials @ = IF
		One F@  Half F@ Three F@ F/ F+ Random1 F!
		U2 F@ U1 F@ F+ One F@ F+ Random2 F!
		Random1 F@ Random2 F@ F* Z F!
		Random2 F@ Random1 F@ F* Y F!
		\ Z9 = (One + Half / Three) * ((U2 + U1) + One) - (One + Half /
		\	Three) * ((U2 + U1) + One)
		One F@ Half F@ Three F@ F/ F+  U2 F@ U1 F@ F+ One F@ F+ F*
		One F@ Half F@ Three F@ F/ F+  U2 F@ U1 F@ F+ One F@ F+ F* F- Z9 F!
	THEN
	Ivar @ NoTrials @ =   Z9 F@ Zero F@ F= or  invert IF
		Defect s" X * Y == Y * X trial fails." BadCond
	ELSE 
		."     No failures found in " NoTrials ? ." integer pairs." cr
	THEN
	\ =============================================
	70 Milestone !
	\ =============================================
	cr ." Running test of square root(x)." cr

	Failure 
	Zero F@  Zero F@ FSQRT F=
	Zero F@ FNEGATE  Zero F@ FNEGATE FSQRT F= and
	One F@ One F@ FSQRT F= and 
	s" Square root of 0.0, -0.0 or 1.0 wrong"
	TstCond

	Zero F@ MinSqEr F!
	Zero F@ MaxSqEr F!
	Zero F@ Jvar F!
	Radix F@  X F!
	U2 F@ OneUlp F!
	Serious SqXMinX
	BInvrse F@ X F!
	BInvrse F@ U1 F@ F* OneUlp F!
	Serious SqXMinX
	U1 F@ X F!
	U1 F@ U1 F@ F* OneUlp F!
	Serious SqXMinX
	Jvar F@ Zero F@ F<> IF  Pause  THEN
	." Testing if sqrt(X * X) == X for " NoTrials ? ." Integers X." cr 
	Zero F@ Jvar F!
	Two F@ X F!
	Radix F@ Y F!
	Radix F@ One F@ F<> IF
		BEGIN
			Y F@ X F!
			Radix F@ Y F@ F* Y F!
			Y F@ X F@ F-  NoTrials @ S>F F>= invert WHILE 
		REPEAT
	THEN
	X F@ U2 F@ F* OneUlp F!
	part4_loopA 

	." Test for sqrt monotonicity."
	1 NEGATE Ivar !
	BMinusU2 F@ X F!
	Radix F@    Y F!
	Radix F@  Radix F@ U2 F@ F* F+ Z F!
	False NotMonot !
	False Monot !
	BEGIN
		NotMonot @  Monot @ or invert WHILE
		1 Ivar +!
		X F@ FSQRT X F!
		Y F@ FSQRT Q F!
		Z F@ FSQRT Z F!
		X F@ Q F@ F>  Q F@ Z F@ F>  or IF
			True NotMonot !
		ELSE
			Q F@ Half F@ F+ FLOOR  Q F!
			Ivar @ 0>  Radix F@ Q F@ Q F@ F* F=  or invert IF
				True Monot !
			ELSE 
				Ivar @ 0> IF
					Ivar @ 1 > IF
						True Monot !
					ELSE
						Y F@ BInvrse F@ F* Y F!
						Y F@ U1 F@ F- X F!
						Y F@ U1 F@ F+ Z F!
					THEN
				ELSE
					Q F@ Y F!
					Y F@ U2 F@ F- X F!
					Y F@ U2 F@ F+ Z F!
				THEN
			THEN
		THEN
	REPEAT
	Monot @ IF 
		cr ." sqrt has passed a test for Monotonicity." cr
	ELSE
		Defect s" " BadCond
		cr ." sqrt(X) is non-monotonic for X near " Y F@ F. ( %.7e) cr
	THEN

;
\ end part4

: part5 ( -- )
	\ =============================================
	80 Milestone !
	\ =============================================
	MinSqEr F@ Half F@ F+ MinSqEr F!
	MaxSqEr F@ Half F@ F- MaxSqEr F!
	One F@ U2 F@ F+ FSQRT One F@ F- U2 F@ F/  Y F!
	Y F@ One F@ F-  U2 F@ Eight F@ F/ F+ SqEr F!
	SqEr F@ MaxSqEr F@ F> IF  SqEr F@  MaxSqEr F! THEN
	Y F@  U2 F@ Eight F@ F/  F+  SqEr F!
	SqEr F@ MinSqEr F@ F< IF  SqEr F@ MinSqEr F!  THEN
	F9 F@ FSQRT U2 F@ F-  One F@ U2 F@ F- F- U1 F@ F/  Y F!
	Y F@  U1 F@ Eight F@ F/  F+  SqEr F!
	SqEr F@ MaxSqEr F@ F> IF  SqEr F@ MaxSqEr F!  THEN
	Y F@ One F@ F+  U1 F@ Eight F@ F/  F+  SqEr F!
	SqEr F@ MinSqEr F@ F< IF  SqEr F@ MinSqEr F!  THEN
	U2 F@ OneUlp F!  
	OneUlp F@ X F!
	\ for( Indx = 1; Indx <= 3; ++Indx) {
	4 1 DO
		X F@ U1 F@ F+ X F@ F+  F9 F@ F+ FSQRT  Y F!
		Y F@ U2 F@ F-  One F@ U2 F@ F- X F@ F+  F-  OneUlp F@ F/  Y F!
		U1 F@ X F@ F- F9 F@ F+  Half F@ F* X F@ F* X F@ F* OneUlp F@ F/ Z F!
		Y F@ Half F@ F+  Z F@ F+  SqEr F!
		SqEr F@ MinSqEr F@ F< IF  SqEr F@ MinSqEr F!  THEN
		Y F@  Half F@ F- Z F@ F+  SqEr F!
		SqEr F@ MaxSqEr F@ F> IF  SqEr F@ MaxSqEr F!  THEN
		I 1 =  I 3 =  or IF
			\ X = OneUlp * Sign (X) * FLOOR(Eight / (Nine * SQRT(OneUlp)))
	OneUlp F@ X F@ Sign F* Eight F@ Nine F@  OneUlp F@ FSQRT F* F/ FLOOR F* X F!
		ELSE
			U1 F@  OneUlp F!
			OneUlp F@ FNEGATE X F!
		THEN
	LOOP
	\ =============================================
	85 Milestone !
	\ =============================================
	False SqRWrng !
	False Anomaly !
	Other RSqrt !  \ ~dgh 
	Radix F@  One F@ F<> IF
		." Testing whether sqrt is rounded or chopped." cr
		\ D = FLOOR(Half + POW(Radix, One + PrecisionF - FLOOR(PrecisionF)));
	Half F@  Radix F@  One F@ PrecisionF F@ F+ PrecisionF F@ FLOOR F- POW F+ FLOOR D F!
	\ ... == Radix^(1 + fract) if (PrecisionF == Integer + fract. 
		D F@ Radix F@ F/ X F!
		D F@ A1 F@ F/ Y F!
		X F@ X F@ FLOOR F<>   Y F@ Y F@ FLOOR F<>  or IF
			True Anomaly !
		ELSE
			Zero F@   X F!
			X    F@  Z2 F!
			One  F@   Y F! 
			Y    F@  Y2 F!
			Radix F@ One F@ F- Z1 F!
			Four F@ D F@ F* FourD F!
			BEGIN
				Y2 F@ Z2 F@ F> IF
					Radix F@ Q F!
					Y F@ Y1 F!
					BEGIN
						\ X1 = FABS(Q + FLOOR(Half - Q / Y1) * Y1)
						Half F@ Q F@ Y1 F@ F/ F- FLOOR Y1 F@ F*
						Q F@ F+ FABS  X1 F!
						Y1 F@ Q F!
						X1 F@ Y1 F!
					X1 F@ Zero F@ F<= invert WHILE 
					REPEAT
					Q F@ One F@ F<= IF
						Y2 F@ Z2 F!
						 Y F@  Z F!
					THEN
				THEN
				Y  F@ Two   F@ F+  Y F!
				X  F@ Eight F@ F+  X F!
				Y2 F@ X     F@ F+ Y2 F!
				Y2 F@ FourD F@ F>=  IF  Y2 F@ FourD F@ F- Y2 F! THEN
			Y F@ D F@ F>=  invert WHILE
			REPEAT
			FourD F@ Z2 F@ F- X8 F!
			X8 F@  Z F@ Z F@ F* F+  FourD F@ F/  Q F!
			X8 F@ Eight F@ F/  X8 F!
			Q F@ Q F@ FLOOR F<> IF
				True Anomaly !
			ELSE
				False Break !
				BEGIN
					Z1 F@ Z F@ F* X F!
					X F@ X F@ Radix F@ F/ FLOOR Radix F@ F* F- X F!
					X F@ One F@ F= IF
						True Break !
					ELSE
						Z1 F@  One F@ F- Z1 F!
					THEN
				Break @  Z1 F@ Zero F@ F<=  or invert WHILE 
				REPEAT
				Z1 F@ Zero F@ F<=   Break @ invert and IF
					True Anomaly !
				ELSE
					Z1 F@  RadixD2 F@ F>  IF Z1 F@ Radix F@ F- Z1 F! THEN
					BEGIN
						NewD
					U2 F@ D F@ F*  F9 F@ F>= invert WHILE 
					REPEAT
					D F@ Radix F@ F* D F@ F-  W F@ D F@ F- F<> IF
						True Anomaly !
					ELSE
						D F@ Z2 F!
						0 Ivar !
						D F@  One F@ Z F@ F+  Half F@ F* F+ Y F!
						D F@ Z F@ F+ Q F@ F+ X F!
						SR3750
						D F@ One F@ Z F@ F- Half F@ F* F+ D F@ F+ Y F!
						D F@  Z F@ F-  D F@ F+  X F!
						X F@ Q F@ F+ X F@ F+ X F!
						SR3750
						NewD
						D F@ Z2 F@ F-  W F@ Z2 F@ F- F<> IF
							True Anomaly !
						ELSE
							D F@ Z2 F@ F-
							Z2 F@ One F@ Z F@ F- Half F@ F* F+
							F+ Y F!
							D F@ Z2 F@ F- Z2 F@ Z F@ F- Q F@ F+ F+  X F!
							SR3750
							One F@ Z F@ F+ Half F@ F* Y F!
							Q F@ X F!
							SR3750
							Ivar @ 0= IF True Anomaly ! THEN
						THEN
					THEN
				THEN
			THEN
		THEN
		Ivar @ 0=  Anomaly @ or IF
			Failure s" Anomalous arithmetic with Integer < " BadCond
			." Radix^Precision = " W F@ FS. ( %.7e) cr
			." fails test whether sqrt rounds or chops." cr
			True SqRWrng !
		THEN
	THEN 
	Anomaly @ invert IF
		MinSqEr F@ Zero F@ F<  MaxSqEr F@ Zero F@ F>  or  invert IF
			Rounded RSqrt !
			." Square root appears to be correctly rounded." cr
		
		ELSE
			MaxSqEr F@ U2 F@ F+  U2 F@ Half F@ F- F> 
			MinSqEr F@ Half F@ F>  or 
			MinSqEr F@ Radix F@ F+ Half F@ F< or  IF
				True SqRWrng !
			ELSE
				Chopped RSqrt !
				." Square root appears to be chopped." cr
			THEN
		THEN
	THEN
	SqRWrng @ IF
		." Square root is neither chopped nor correctly rounded." cr
		." Observed errors run from " MinSqEr F@ Half F@ F- FS. ( %.7e)  
		." to " Half F@ MaxSqEr F@ F+ FS. ( %.7e) ."  ulps." cr 
		Serious  MaxSqEr F@ MinSqEr F@ F-  Radix F@ Radix F@ F*  F<
		s" sqrt gets too many last digits wrong"
		TstCond
	THEN
	\ =============================================
	90 Milestone !
	\ =============================================
	Pause
	." Testing powers Z^i for small Integers Z and i." cr
	0 N !
	\ ... test powers of zero. 
	0 Ivar !
	Zero F@ FNEGATE Z F!
	3 M !
	False Break !
	BEGIN
		One F@ X F!
		SR3980
		Ivar @ 10 <= IF 
			1023 Ivar !
			SR3980
		THEN
		Z F@ MinusOne F@ F= IF
			True Break !
		ELSE
			MinusOne F@ Z F!
			\ PrintIfNPositive
			\ 0 N !
			\ .. if(-1)^N is invalid, replace MinusOne by One. 
			-4 Ivar !
		THEN
	Break @ invert WHILE
	REPEAT
	PrintIfNPositive
	N @ N1 !
	0 N !
	A1 F@ Z F!
	Two F@  W F@ FLN F*  A1 F@ FLN F/  FLOOR  F>D D>S  M !
	False Break !
	BEGIN
		Z F@ X F!
		1 Ivar !
		SR3980
		Z F@ AInvrse F@ F= IF 
			True Break !
		ELSE 
			AInvrse F@ Z F!
		THEN
	Break @ invert WHILE 
	REPEAT
	\ =============================================
		100 Milestone !
	\ =============================================
	\  Powers of Radix have been tested, 
	\         next try a few primes     
	NoTrials @ M !
	Three F@   Z F!
	BEGIN
		Z F@ X F!
		1 Ivar !
		SR3980
		BEGIN
			Z F@ Two F@ F+ Z F!
		Three F@  Z F@ Three F@ F/ FLOOR F*  Z F@ F=  WHILE 
		REPEAT
	Z F@  Eight F@ Three F@ F* F<  WHILE 
	REPEAT
	N @ 0> IF
		." Errors like this may invalidate financial calculations" cr
		." involving interest rates." cr
	THEN
	PrintIfNPositive
	N1 @ N +!
	N @ 0=  IF ." ... no discrepancies found." cr THEN
	N @ 0> IF Pause ELSE cr THEN
	
;
\ end part5


: part6 ( -- )

	\ =============================================
	110 Milestone !
	\ =============================================
	." Seeking Underflow thresholds UfThold and E0." cr
	U1 F@ D F!
	PrecisionF F@  PrecisionF F@ FLOOR  F<> IF
		BInvrse F@ D F!
		PrecisionF F@ X F!
		BEGIN
			D F@  BInvrse F@ F*  D F!
			X F@  One F@ F-  X F!
		X F@ Zero F@ F> WHILE
		REPEAT
	THEN
	One F@ Y F!
	D F@ Z F!
	\ ... D is power of 1/Radix < 1. 
	BEGIN
		Y F@ C F!
		Z F@ Y F!
		Y F@ Y F@ F* Z F!
	Y F@ Z F@ F>  Z F@ Z F@ F+ Z F@ F>  and WHILE 
	REPEAT
	C F@ Y F!
	Y F@ D F@ F* Z F!
	BEGIN
		Y F@ C F!
		Z F@ Y F!
		Y F@ D F@ F* Z F!
	Y F@  Z F@ F>  Z F@ Z F@ F+  Z F@ F>  and WHILE 
	REPEAT
	Radix F@ Two F@ F< IF  Two  ELSE  Radix THEN  F@ HInvrse F!
	One F@ HInvrse F@ F/ H F!
	\ ... 1/HInvrse == H == Min(1/Radix, 1/2) 
	One F@ C F@ F/ CInvrse F!
	C F@ E0 F!
	E0 F@ H F@ F* Z F!
	\ ...1/Radix^(BIG Integer) << 1 << CInvrse == 1/C 
	BEGIN
		E0 F@ Y F!
		Z F@ E0 F!
		E0 F@ H F@ F* Z F!
	E0 F@ Z F@ F>  Z F@ Z F@ F+ Z F@ F>  and WHILE 
	REPEAT
	E0 F@ UfThold F!
	Zero F@ E1 F!
	Zero F@  Q F!
	U2 F@ E9 F!
	One F@ E9 F@ F+ S F!
	C F@ S F@ F* D F!
	D F@ C F@ F<=  IF
		Radix F@ U2 F@ F* E9 F!
		One F@ E9 F@ F+ S F!
		C F@ S F@ F* D F!
		D F@ C F@ F<=  IF
			Failure s" multiplication gets too many last digits wrong." BadCond
			E0 F@ Underflow F!
			Zero F@ Y1 F!
			Z F@ PseudoZero F!
			Pause
		THEN
	ELSE
		D F@ Underflow F!
		Underflow F@ H F@ F* PseudoZero F!
		Zero F@ UfThold F!
		BEGIN
			Underflow F@ Y1 F!
			PseudoZero F@ Underflow F!
			E1 F@ E1 F@ F+  E1 F@ F<=  IF
				Underflow F@ HInvrse F@ F* Y2 F!
				Y1 F@  Y2 F@ F-  FABS E1 F!
				Y1 F@ Q F!
				UfThold F@ Zero F@ F=  Y1 F@ Y2 F@ F<>  and IF Y1 F@ UfThold F! THEN
			THEN
			PseudoZero F@ H F@ F* PseudoZero F!
		Underflow F@ PseudoZero F@ F>
		PseudoZero F@ PseudoZero F@ F+  PseudoZero F@ F> and WHILE 
		REPEAT
	THEN
	\ Comment line 4530 .. 4560 
	PseudoZero F@ Zero F@ F<> IF
		cr 
		PseudoZero F@ Z F!
	\ ... Test PseudoZero for "phoney- zero" violates 
	( ... PseudoZero < Underflow or PseudoZero < PseudoZero + PseudoZero
		   ... )
		PseudoZero F@ Zero F@ F<=  IF
			Failure s" Positive expressions can underflow to an" BadCond
			cr ." allegedly negative value" cr
			." PseudoZero that prints out as: " PseudoZero F@ F. ( %g) cr 
			PseudoZero F@ FNEGATE X F!
			X F@ Zero F@ F<=  IF
				." But -PseudoZero, which should be" cr
				." positive, isn't; it prints out as " X F@ F. ( %g) cr
			THEN
		ELSE
			Flaw s" Underflow can stick at an allegedly positive" BadCond
			cr ." value PseudoZero that prints out as " PseudoZero F@ F. ( %g) cr 
		THEN
		TstPtUf
	THEN
	\ =============================================
	120 Milestone !
	\ =============================================
	CInvrse F@ Y F@ F*  CInvrse F@ Y1 F@ F* F> IF
		H F@ S F@ F* S F!
		Underflow F@ E0 F!
	THEN
	E1 F@ Zero F@ F=  E1 F@ E0 F@ F=  or  invert IF
		Defect  s" " BadCond
		E1 F@ E0 F@ F< IF
			." Products underflow at a higher"
			." threshold than differences." cr
			PseudoZero F@ Zero F@ F=  IF  E1 F@ E0 F!  THEN
		
		ELSE
			." Difference underflows at a higher"
			." threshold than products." cr
		THEN
	THEN
	." Smallest strictly positive number found is E0 = " E0 F@ FS. ( %g) cr
	E0 F@ Z F!
	TstPtUf
	E0 F@ Underflow F!
	N @ 1 = IF  Y F@  Underflow F! THEN
	4 Ivar !
	E1 F@ Zero F@ F=  IF  3 Ivar ! THEN
	UfThold F@ Zero F@ F=  IF  Ivar @ 2 - Ivar !  THEN
	True UfNGrad !
	Ivar @ CASE
		1 OF
		Underflow F@  UfThold F!
		CInvrse F@ Q F@ F*  CInvrse F@ Y F@ F* S F@ F*  F<> IF
			Y F@ UfThold F! 
			Failure s" Either accuracy deteriorates as numbers" BadCond
			cr ." approach a threshold = " UfThold F@ FS. ( %.17e) 
			cr ." coming down from " C F@ FS. ( %.17e) 
			cr ." or else multiplication gets too many last digits wrong." cr
		THEN
		Pause
		ENDOF

		2 OF
		Failure s" Underflow confuses Comparison, which alleges that" BadCond
		cr ." Q == Y while denying that |Q - Y| == 0; these values"
		cr ." print out as Q = " Q F@ FS. ( %.17e) ." , Y = " Y2 F@ FS. ( %.17e)
		cr ." |Q - Y| = " Q F@ Y2 F@ F- FABS FS. ( %.17e) cr 
		Q F@ UfThold F!
		ENDOF

		3 OF
		\ X = X    \ ??
		X F@ X F!
		ENDOF

		4 OF
		Q F@ UfThold F@ F=  E1 F@ E0 F@ F=  and
		UfThold F@  E1 F@ E9 F@ F/ F- FABS   E1 F@ F<=  and IF
			False UfNGrad !
			." Underflow is gradual; it incurs Absolute Error =" cr
			." (roundoff in UfThold) < E0." cr
			E0 F@ CInvrse F@ F* Y F!
			Y F@  OneAndHalf F@ U2 F@ F+ F*  Y F!
			CInvrse F@  One F@ U2 F@ F+ F*  X F!
			Y F@ X F@ F/  Y F!
			Y F@ E0 F@ F=  IEEE !
		THEN
		ENDOF
	ENDCASE
	UfNGrad @ IF
		cr
		\ Sigfpe sigsave !
		\ setjmp(ovfl_buf) 
		0 IF
			." Underflow / UfThold failed!" cr
			R = H F@ H F@ F+  R F!
		ELSE 
			Underflow F@ UfThold F@ F/ FSQRT  R F!
		THEN
		0 sigsave !
		R F@ H F@ F<=  IF
			R F@ UfThold F@ F*  Z F!
			Z F@  One F@  R F@ H F@ F*  One F@ H F@ F+ F* F+  F*  X F!
		ELSE
			UfThold F@ Z F!
			Z F@  One F@  H F@ H F@ F*  One F@ H F@ F+ F* F+ F*  X F! 
		THEN
		X F@ Z F@ F=   X F@ Z F@ F- Zero F@ F<>  or invert IF
			Flaw  s" "  BadCond
			." X = " X F@ FS. ( %.17e) cr 
			."     is not equal to Z = " Z F@ FS. ( %.17e) cr
			X F@ Z F@ F- Z9 F!
			." yet X - Z yields " Z9 F@ FS. ( %.17e) cr
			."    Should this NOT signal Underflow, this is a SERIOUS DEFECT" 
			."    that causes confusion when innocent statements like" cr
			."    if (X == Z)  ...  else ... (f(X) - f(Z)) / (X - Z) ..." cr
			." encounter Division by Zero although actually" cr
			\ Sigfpe sigsave !
			\ setjmp(ovfl_buf) 
			0 IF
				." X / Z fails!" cr
			ELSE 
				." X / Z = 1 + " 
				X F@ Z F@ F/ Half F@ F- Half F@ F- F. ( %g) cr
			THEN
			0 sigsave !
		THEN
	THEN
	." The Underflow threshold is "  UfThold F@ FS. ( %.17e)
	." below which" cr
	." calculation may suffer larger Relative error than merely roundoff." cr
	U1 F@ U1 F@ F*  Y2 F!
	Y2 F@ Y2 F@ F*  Y F!
	Y  F@ U1 F@ F*  Y2 F!
	Y2 F@  UfThold F@ F<=  IF
		Y F@ E0 F@ F> IF
			Defect s" " BadCond
			5 Ivar !
		ELSE
			Serious  s" " BadCond
			4 Ivar !
		THEN
		." Range is too narrow; U1^" Ivar ? ."  Underflows." cr
	THEN
	\ =============================================
;
\ end part6


: part7_loopA ( -- )
	\ for(I = 1;;) {
	1 Ivar !
	BEGIN
		X F@  BInvrse F@ F-  Z F!
		\ Z = (X + One) / (Z - (One - BInvrse))
		X F@ One F@ F+  Z F@  One F@ BInvrse F@ F- F-  F/  Z F!
		X F@ Z F@ POW  Exp2 F@ F- Q F!

		Q F@ FABS  TwoForty F@ U2 F@ F*  F> IF
			1 N !
	 		X F@ BInvrse F@ F-  One F@ BInvrse F@ F-  F-  V9 F!
			Defect s" Calculated " BadCond
			X F@ Z F@ POW FS. ( %.17e) ." for" cr
			." (1 + (" V9 F@ FS. ( %.17e) ." ) ^ (" Z F@ FS. ( %.17e) ." );" cr
			." differs from correct value by " Q F@ FS. ( %.17e) cr
			." This much error may spoil financial" cr
			." calculations involving tiny interest rates." cr
			EXIT \ break;
		ELSE
			Y F@ X F@ F-  Two F@ F*  Y F@ F+  Z F!
			Y F@ X F!
			Z F@ Y F!
			One F@  X F@ F9 F@ F-  X F@ F9 F@ F- F*  F+  Z F!
			Z F@ One F@ F>   Ivar @ NoTrials @ <  and IF
				1 Ivar +!
			ELSE
				X F@ One F@ F> IF
					N @ 0=  IF
					   ." Accuracy seems adequate." cr
					THEN
					EXIT  \ break;
				ELSE
					One F@ U2 F@ F+  X F!
					U2  F@ U2 F@ F+  Y F!
					Y   F@ X  F@ F+  Y F!
					1 Ivar !
				THEN
			THEN
		THEN
	AGAIN
;


: part7 ( -- )

	\ =============================================
	130 Milestone !
	\ =============================================
	\ Y = - FLOOR(Half - TwoForty * LOG(UfThold) / LOG(HInvrse)) / TwoForty;
	Half F@ TwoForty F@ UfThold F@ FLN F* HInvrse F@ FLN F/ F- FLOOR 
	TwoForty F@ F/ FNEGATE Y F!
	Y F@ Y F@ F+ Y2 F!
	." Since underflow occurs below the threshold" cr
	." UfThold = " HInvrse F@ FS. ( %.17e) ." ^"  Y F@ FS. ( %.17e) cr
	." only underflow should afflict the expression" cr
	."      "  HInvrse F@ FS. ( %.17e) ." ^" Y2 F@ FS. ( %.17e) cr
	HInvrse F@  Y2 F@  POW  V9 F!
	." actually calculating yields: " V9 F@ FS. ( %.17e) cr
	V9 F@ Zero F@ F>= 
	V9 F@  Radix F@ Radix F@ F+ E9 F@ F+ UfThold F@ F* F<=  and  invert IF
		Serious s" this is not between 0 and underflow" BadCond
		."   threshold = " UfThold F@ FS. ( %.17e) cr 
	ELSE
		V9 F@   UfThold F@  One F@ E9 F@ F+ F*  F>  invert IF
			." This computed value is O.K." cr
		ELSE
			Defect s" this is not between 0 and underflow" BadCond
			."   threshold = "  UfThold F@ FS. ( %.17e) cr
		THEN
	THEN
	\ =============================================
	140 Milestone !
	\ =============================================
	cr
	\ ...calculate Exp2 == exp(2) == 7.389056099... 
	Zero F@ X F!
	2 Ivar !
	Two F@ Three F@ F*  Y F!
	Zero F@ Q F!
	0 N !
	BEGIN
		X F@ Z F!
		1 Ivar +!
		Y F@  Ivar @ Ivar @ + s>f F/  Y F!
		Y F@ Q F@ F+ R F!
		Z F@ R F@ F+ X F!
		Z F@ X F@ F- R F@ F+  Q F!
	X F@ Z F@ F>  WHILE
	REPEAT
	\ Z = (OneAndHalf + One / Eight) + X / (OneAndHalf * ThirtyTwo);
	OneAndHalf F@  One F@ Eight F@ F/ F+  
	X F@  OneAndHalf F@ ThirtyTwo F@ F* F/ F+  Z F!
	Z F@ Z F@ F*  X F!
	X F@ X F@ F*  Exp2 F!
	F9 F@ X F!
	X F@ U1 F@ F-  Y F!
	." Testing X^((X + 1) / (X - 1)) vs. exp(2) = " Exp2 F@ FS. ( %.17e)
	." as X -> 1." cr

	part7_loopA

	\ =============================================
	150 Milestone !
	\ =============================================
	." Testing powers Z^Q at four nearly extreme values." cr
	0 N !
	A1 F@  Z F!
	Half F@  C F@ FLN  A1 F@ FLN F/  F- FLOOR  Q F!
	False Break !
	BEGIN
		CInvrse F@ X F!
		Z F@ Q F@  POW  Y F!
		IsYeqX
		Q F@ FNEGATE Q F!
		C F@ X F!
		Z F@ Q F@  POW  Y F!
		IsYeqX
		Z F@  One F@ F<  IF
			True Break !
		ELSE 
			AInvrse F@ Z F!
		THEN
	Break @ invert  WHILE
	REPEAT 
	PrintIfNPositive
	N @ 0=  IF  ." ... no discrepancies found." cr  THEN
	cr

	\ =============================================
	160 Milestone !
	\ =============================================
	Pause
	." Searching for Overflow threshold:" cr
	." This may generate an error." cr
	CInvrse F@ FNEGATE  Y F!
	HInvrse F@  Y F@ F*  V9 F!
	\ sigsave = Sigfpe;
	\ if (setjmp(ovfl_buf)) { 0 Ivar !  Y F@ V9 F!  goto overflow; }
	BEGIN
		Y F@ V F!
		V9 F@ Y F!
		HInvrse F@ Y F@ F* V9 F!
	V9 F@ Y F@ F< WHILE
	REPEAT
	1 Ivar !
\ overflow:
	0 sigsave !
	V9 F@ Z F!
	." Can `Z = -Y' overflow?" cr
	." Trying it on Y = " Y F@ FS. ( %.17e) cr
	Y F@ FNEGATE V9 F!
	V9 F@ V0 F!
	V F@ Y F@ F-  V F@ V0 F@ F+  F=  IF
		." Seems O.K." cr
	ELSE
		." finds a "
		Flaw  s" -(-Y) differs from Y."  BadCond
	THEN
	Z F@ Y F@  F<> IF
		Serious  s" "  BadCond
		." overflow past " Y F@ FS. ( %.17e) cr
		."    shrinks to " Z F@ FS. ( %.17e) cr
	THEN
	Ivar @ IF
		V F@  HInvrse F@ U2 F@ F*  HInvrse F@ F- F*  Y F!
		Y F@  One F@ HInvrse F@ F- U2 F@ F* V F@ F*  F+  Z F!
		Z F@ V0 F@  F<  IF  Z F@ Y F!  THEN
		Y F@ V0 F@  F<  IF  Y F@ V F!  THEN
		V0 F@ V F@ F-  V0 F@  F<  IF  V0 F@ V F! THEN
	ELSE
		Y F@  HInvrse F@ * U2 F@ F* HInvrse F@ F-  F*  V F!
		V F@  One F@  HInvrse F@ F-  U2 F@ F*  Y F@ F*  F+  V F!
	THEN
	." Overflow threshold is V  = " V F@ FS. ( %.17e) cr
	Ivar @  IF
		 ." Overflow saturates at V0 = " V0 F@ FS. ( %.17e) cr
	ELSE 
		." There is no saturation value because the system traps on overflow." cr
	THEN
	V F@ One F@ F*  V9 F!
	." No Overflow should be signaled for V * 1 = " V9 F@ FS. ( %.17e) cr
	V F@ One F@ F/  V9 F!
	."                           nor for V / 1 = " V9 F@ FS. ( %.17e) cr
	." Any overflow signal separating this * from the one" cr
	." above is a DEFECT." cr
	\ =============================================
	170 Milestone !
	\ =============================================
	V F@ FNEGATE V F@ F<   V0 F@ FNEGATE V0 F@ F<  and
	UfThold F@ FNEGATE V F@ F<  and  UfThold F@ V F@ F<  and  invert  IF
		Failure  s" Comparisons involving "  BadCond
		." +-" V F@ F. ( %g) ." , +-" V0 F@ F. ( %g) ."  and +-"
		UfThold F@ F. ( %g) ."  are confused by Overflow."
	THEN
	\ =============================================
	175 Milestone !
	\ =============================================
	cr
	4 1 DO  \ for(Indx = 1; Indx <= 3; ++Indx) {
		I CASE
			1  OF  UfThold    ENDOF
			2  OF  E0         ENDOF
			3  OF  PseudoZero ENDOF
		ENDCASE
		F@  Z F!
		Z F@ Zero F@ F<> IF 
			Z F@ FSQRT V9 F!
			V9 F@ V9 F@ F*  Y F!
			Y F@  One F@ Radix F@ E9 F@ F* F- F/  Z F@ F<
			Y F@  One F@  Radix F@ E9 F@ F* F+ Z F@ F* F>  or IF  \ dgh: + E9 --> * E9 
				V9 F@ U1 F@ F> IF Serious ELSE  Defect THEN s" " BadCond

				." Comparison alleges that what prints as Z = "
				Z F@ FS. ( %.17e) cr
				." is too far from sqrt(Z) ^ 2 = " Y F@ FS. ( %.17e) cr
			THEN
		THEN
	LOOP
	\ =============================================
	180 Milestone !
	\ =============================================
	3 1 DO  \ for(Indx = 1; Indx <= 2; ++Indx) {
		I 1 = IF  V ELSE  V0 THEN  F@ Z F!
		Z F@ FSQRT  V9 F!
		One F@ Radix F@ E9 F@ F* F-  V9 F@ F*  X F!
		V9 F@ X F@ F*  V9 F!
		V9 F@  One F@  Two F@ Radix F@ F* E9 F@ F* F- Z F@ F*  F<
		V9 F@ Z F@ F>  or   IF
			V9 F@  Y F!
			X F@  W F@ F<  IF  Serious  ELSE  Defect  THEN
			s" "  BadCond
			." Comparison alleges that Z = "  Z F@ FS. ( %17e) cr
			." is too far from sqrt(Z) ^ 2 (" Y F@ FS. ( %.17e) cr
		THEN
	LOOP
	\ =============================================
	
;
\ end part7


: part8 ( -- )
	\ =============================================
	190 Milestone !
	\ =============================================
	Pause
	UfThold F@ V F@ F* X F!
	Radix F@ Radix F@  F* Y F!
	X F@ Y F@ F* One F@ F<   X F@ Y F@ F>  or IF
		X F@ Y F@ F*  U1 F@ F<  X F@ > Y F@ U1 F@ F/ F>  or IF
			Defect  s" Badly"
		ELSE 
			Flaw s" "
		THEN
		BadCond
		." unbalanced range; UfThold * V = " X F@ FS. ( %.17e) cr
		." is too far from 1." cr
	THEN
	\ =============================================
	200 Milestone !
	\ =============================================
	6 1 DO  \ for (Indx = 1; Indx <= 5; ++Indx)  {
		F9 F@ X F!
		I CASE
			2 OF  One F@ U2 F@ F+  X F!  ENDOF
			3 OF  V F@  X F!             ENDOF
			4 OF  UfThold F@       X F!  ENDOF
			5 OF  Radix F@         X F!  ENDOF
		ENDCASE
		X F@ Y F!
		\ sigsave = Sigfpe;
		\ if (setjmp(ovfl_buf))
		\	printf("  X / X  traps when X = %g\n", X);
		\ else {
			Y F@ X F@ F/  Half F@ F- Half F@ F- V9 F!
			V9 F@ Zero F@ F<> IF 
				V9 F@ U1 F@ FNEGATE F=   I 5 <  and 
				IF  Flaw  ELSE Serious  THEN  s" " BadCond
				."  X / X differs from 1 when X = " X F@ FS. ( %.17e) cr
				."  instead, X / X - 1/2 - 1/2 = " V9 F@ FS. ( %.17e) cr
			THEN
		\ }
		0 sigsave !
	LOOP
	\ =============================================
	210 Milestone !
	\ =============================================
	Zero F@ MyZero F!
	cr
	." What message and/or values does Division by Zero produce?" cr

	NOPAUSE 0= IF
		." This can interupt your program.  You can "
		." skip this part if you wish." cr
		." Do you wish to compute 1 / 0? "
		KEY
		dup [char] Y = swap [char] y =  or
	ELSE True THEN

	IF
		\ sigsave = Sigfpe;
		cr ."    Trying to compute 1 / 0 produces ..."
		\ if (!setjmp(ovfl_buf)) printf("  %.7e .\n", One / MyZero);
		One F@ MyZero F@ F/ FS. cr
		0 sigsave !
	ELSE
		." O.K." cr
	THEN
	NOPAUSE 0= IF
		cr ." Do you wish to compute 0 / 0? "
		KEY
		dup [char] Y = swap [char] y =  or
	ELSE True THEN

	IF
		\ sigsave = Sigfpe;
		cr ."    Trying to compute 0 / 0 produces ..."
		\ if (!setjmp(ovfl_buf)) printf("  %.7e .\n", Zero / MyZero);
		Zero F@ MyZero F@ F/ FS. cr
		0 sigsave !
	ELSE 
		." O.K."  cr
	THEN

	\ =============================================
	220 Milestone !
	\ =============================================
	Pause
	cr
	." FAILUREs  encountered = " ErrCnt{ Failure } ? cr
	." SERIOUS DEFECTs  discovered = " ErrCnt{ Serious } ? cr
	." DEFECTs  discovered = " ErrCnt{ Defect } ? cr 
	." FLAWs  discovered = " ErrCnt{ Flaw } ? cr

	cr
	ErrCnt{ Failure } @   ErrCnt{ Serious } @ + 
	ErrCnt{ Defect }  @ + ErrCnt{ Flaw } @  + 0> IF
		ErrCnt{ Failure } @  ErrCnt{ Serious } @  + 
		ErrCnt{ Defect } @ + 0=  ErrCnt{ Flaw } @ 0> and IF
			." The arithmetic diagnosed seems "
			." Satisfactory though flawed." cr
		THEN
		ErrCnt{ Failure } @ ErrCnt{ Serious } @ + 0= 
		ErrCnt{ Defect } @ 0>  and IF
			." The arithmetic diagnosed may be Acceptable" cr
			." despite inconvenient Defects." cr
		THEN
		ErrCnt{ Failure } @ ErrCnt{ Serious } @ +  0> IF
			." The arithmetic diagnosed has "
			." unacceptable Serious Defects." cr
		THEN
		ErrCnt{ Failure } @  0> IF
			." Potentially fatal FAILURE may have spoiled this"
			." program's subsequent diagnoses." cr
		THEN
	ELSE
		." No failures, defects nor flaws have been discovered." cr
		RMult @ Rounded =   RDiv @ Rounded =  and
		RAddSub @ Rounded =  and  RSqrt @ Rounded =  and  invert IF
			." The arithmetic diagnosed seems Satisfactory." cr
		ELSE
			StickyBit F@ One F@ F>=
			Radix F@ Two F@ F-  Radix F@ Nine F@ F- One F@ F- F* Zero F@ F=  
			and  IF
			." Rounding appears to conform to the proposed IEEE standard P"
			Radix F@ Two F@ F= 
			PrecisionF F@  Four F@ Three F@ F* Two F@ F* F-
			PrecisionF F@ TwentySeven F@ F- TwentySeven F@ F- One F@ F+ 
			F*  Zero F@ F=  and  IF
					." 754"
				ELSE 
					." 854"
				THEN
				IEEE @ IF 
					cr
				ELSE
					cr ." except for possibly Double Rounding"
					." during Gradual Underflow." cr
				THEN
			THEN
			." The arithmetic diagnosed appears to be Excellent!" cr
		THEN
	THEN
	fpecount @ IF
		cr ." A total of " fpecount ? 
		."  floating point exceptions were registered." cr
	THEN
	." END OF TEST." cr
;
\ end part8



: main ( -- )
\ 0 [IF]
\ #ifdef mc
\	char *out;
\	ieee_flags("set", "precision", "double", &out);
\ #endif
\ [THEN]
	\ First two assignments use integer right-hand sides.
	0 S>D D>F             Zero  F!
	1 S>D D>F             One   F!
	One F@ One F@ F+      Two   F!
	Two F@ One F@ F+      Three F!
	Three F@ One F@ F+    Four  F!
	Four F@ One F@ F+     Five  F!
	Four F@ Four F@ F+    Eight F!
	Three F@ Three F@ F*  Nine  F!
	Nine F@ Three F@ F*   TwentySeven F!
	Four F@ Eight F@ F*   ThirtyTwo F!
	Four F@ Five F@ F* Three F@ F* Four F@ F* TwoForty F!
	One F@ FNEGATE        MinusOne F!
	One F@ Two F@ F/      Half F!
	One F@ Half F@ F+     OneAndHalf F!

	0 ErrCnt{ Failure } !
	0 ErrCnt{ Serious } !
	0 ErrCnt{ Defect } !
	0 ErrCnt{ Flaw } !

	1 PageNo !
	\ =============================================
	0 Milestone !
	\ =============================================

	Instructions
	Pause
	Heading
	Pause
	Characteristics
	Pause
	History
	Pause
	
	\ =============================================
	7 Milestone !
	\ =============================================
	." Program is now RUNNING tests on small integers:" cr

	Failure 
	Zero F@  Zero F@ F+ Zero F@  F=  
	One  F@  One  F@ F- Zero F@  F=  and
	One  F@  Zero F@             F>  and
	One  F@  One  F@ F+ Two F@   F=  and 
	s" 0+0 != 0, 1-1 != 0, 1 <= 0, or 1+1 != 2"
	TstCond 

	Zero F@ FNEGATE  Z F!
	Z F@ 0E F<> IF
		1 ErrCnt{ Failure } +!
		." Comparison alleges that -0.0 is Non-zero!" cr
		0.001E U2 F!
		1E Radix F!
		TstPtUf
	THEN
 
	Failure
	Two F@ One F@ F+ Three F@  F=
	Three F@ One F@ F+ Four F@ F=  and
	Four F@ Two F@ Two F@ FNEGATE F* F+ Zero F@ F=  and
	Four F@ Three F@ F- One F@ F- Zero F@ F= and
	s" 3 != 2+1, 4 != 3+1, 4+2*(-2) != 0, or 4-3-1 != 0"
	TstCond 

	Failure
	0E One F@ F- MinusOne F@ F=
	MinusOne F@ One F@ F+ Zero F@ F=  and
	One F@ MinusOne F@ F+ Zero F@ F=  and
	MinusOne F@ One F@ FABS F+ Zero F@ F= and
	MinusOne F@ MinusOne F@ MinusOne F@ F* F+ Zero F@ F= and
	s" -1+1 != 0, (-1)+abs(1) != 0, or -1+(-1)*(-1) != 0"
	TstCond 

	Failure
	Half F@ MinusOne F@ F+ Half F@ F+ Zero F@ F=
	s" 1/2 + (-1) + 1/2 != 0"
	TstCond

	\ =============================================

	part2
	part3
	part4
	part5
	part6
	part7
	part8
;
\ end of main

\ cr cr .( Type "main" to start the tests. )  cr

main

CR .( End of paranoia.fth) CR