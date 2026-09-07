;; Pinned Nano Scheme subset, drawn from R5RS examples.
;; I do not claim the full report or a third-party suite.
;; Gate: make test-scheme (C harness in tests/scheme/test_scheme.c).
;;
;; In scope here: integers, if, lambda, define, cons/car/cdr, named
;; tail recursion, closures.
;; Out of scope: call/cc, set!, macros, strings, ports, eval.

;; R5RS 4.1.3: procedure calls
;;   (+ 3 4) => 7
;; R5RS 4.1.4: lambda
;;   ((lambda (x) (+ x x)) 4) => 8
;; R5RS 4.1.5: if, only #f is false
;;   (if #t 1 0) => 1
;;   (if 0 1 0)  => 1
;; R5RS 4.1.6: assignments are excluded (set!)
;; R5RS 5.2: define
;;   (define (add a b) (+ a b)) (add 2 3) => 5
;; R5RS 6.3.2: pairs
;;   (car (cons 1 2)) => 1
;; Closures (not a report requirement, ISA pressure):
;;   (define (make-add n) (lambda (x) (+ x n)))
;;   ((make-add 10) 3) => 13
;; Proper tail calls:
;;   (define (sum n acc) (if (= n 0) acc (sum (- n 1) (+ acc n))))
;;   (sum 10000 0) => 50005000 at constant frame depth
