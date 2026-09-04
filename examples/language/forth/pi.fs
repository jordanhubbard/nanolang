\ pi.fs
\
\ Rabinowitz/Wagon spigot algorithm for decimal digits of pi.
\
\ Forth 2012 Core and Memory-Allocation word sets. Integer arithmetic only.
\
\ Usage:
\
\     50 PI
\
\ produces:
\
\     3.14159265358979323846264338327950288419716939937510
\
\ Gforth 0.7.3 golden outputs are pinned in tests/forth/pins.json.

variable places        \ requested digits after decimal point
variable alen          \ number of elements in work array
variable aptr          \ address of work array

variable q
variable x
variable j

variable predigit
variable nines

variable started       \ used to discard algorithm's initial zero
variable printed       \ actual pi digits printed

: a[]  ( i -- addr )
    cells aptr @ + ;

: a@   ( i -- n )
    a[] @ ;

: a!   ( n i -- )
    a[] ! ;

: make-array  ( -- )
    \ The algorithm emits the leading digit as well as the requested places.
    places @ 1+ 10 * 3 / 1+
    dup alen !
    cells allocate throw
    aptr !

    alen @ 0 ?do
        2 i a!
    loop ;

: free-array  ( -- )
    aptr @ free throw
    0 aptr ! ;

: digit>char  ( n -- c )
    [char] 0 + ;

: emit-digit  ( n -- )
    \ Throw away the algorithm's initial zero.
    started @ 0= if
        drop
        1 started !
        exit
    then

    digit>char emit
    1 printed +!

    \ The decimal point follows the first real digit.
    printed @ 1 = places @ 0> and if
        [char] . emit
    then ;

: emit-nines  ( -- )
    nines @ 0 ?do
        9 emit-digit
    loop
    0 nines ! ;

: emit-zeros  ( -- )
    nines @ 0 ?do
        0 emit-digit
    loop
    0 nines ! ;

: handle-digit  ( qdigit -- )
    dup 9 = if
        drop
        1 nines +!
        exit
    then

    dup 10 = if
        drop
        predigit @ 1+ emit-digit
        emit-zeros
        0 predigit !
        exit
    then

    predigit @ emit-digit
    predigit !
    emit-nines ;

: pi-pass  ( -- qdigit )
    0 q !
    alen @ j !

    begin
        j @ 0>
    while
        \ x = 10*a[j-1] + q*j
        j @ 1- a@ 10 *
        q @ j @ * +
        x !

        \ /MOD leaves remainder quotient.
        x @ j @ 2* 1- /mod
        q !
        j @ 1- a!
        -1 j +!
    repeat

    \ Split q into a[0] = q mod 10 and the returned quotient.
    q @ 10 /mod
    swap 0 a! ;

: PI  ( u -- )
    places !
    make-array

    0 predigit !
    0 nines !
    0 started !
    0 printed !

    \ One extra pass flushes the delayed digit.
    places @ 1+ 0 ?do
        pi-pass
        handle-digit
    loop

    predigit @ emit-digit
    cr
    free-array ;
