\ unchanged Nano Forth client for nsi:nanolang/log
\ I call nsi:nanolang/log#write. The adapter is not in this source.

: nsi-nanolang-log-write ( n c-addr u -- ior ) 2drop drop 0 ;
: nsi-nanolang-log-write-event ( -- ior ) 0 ;
