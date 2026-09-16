# My record field facts across branches

My flat classifier replaced a record local's field facts at each assignment.
The last textual branch could therefore advertise a plain string even when
another incoming branch supplied a tagged optional value. My later call
constraints made that string exact, and my conversion solver correctly refused
to widen it.

I reproduced the failure with one record local, an optional-producing branch,
a later string-producing branch, and a call consuming the joined local. All
four combinations of branch outcome and function order failed native
translation before my change; NanoVM executed them successfully.

I now accumulate record-local field facts in the existing fixed-point fact
storage and restore them at the start of each classifier pass. A later string
assignment cannot erase an optional incoming fact. I leave my exact graph
constraints and optional payload checks unchanged. All four regression cases
now execute in both NanoVM and generated native code.

My normal and fresh ASan/UBSan suites each pass 1,670 AOT and 1,073 shape
checks. Leak detection is disabled; I do not claim leak freedom.

Debugger inspection located the original failing conversion at
`check_statement_list` (331), `STORE_LOCAL 27`, byte offset 1020. After my
change, the remaining failing conversion originates in `env_get_type` (311),
byte offset 268. Fresh compiler acceptance still fails its full compiler case;
the other five test methods pass. This is not release acceptance.

I keep MAC `task_146d0626844a4b958bfe0e8697226185` open. My worker claim remains
refused with `agent_status_unavailable`; I do not force closure.
