# My hosted fixture identifier check

I qualify #1047 at exact integration head `435dae34cc2ce74d5210806d26ef1ebb9f9e4b7a`. My [hosted fast gate](https://github.com/jordanhubbard/nanolang/actions/runs/42397278342/job/128477253238) passes. It executes all five real-lexer fixture controls, including deliberate mutations of the original `handle` and `effect` fixtures, and inspects 3,120 static fragments with no unacknowledged reserved identifiers. I retain exact-source Git blob identities and the extracted fixture log.

I report 133 lexer-refused fragments separately with their executable tests. I do not call this full grammar/type parity, dynamic fixture execution, retained native admission or complete 5.1 qualification. The implementation, default fast-tier wiring and two requested regressions satisfy this bounded fixture-check issue; the remaining release gates stay open.
