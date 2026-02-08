COQC = coqc
COQFLAGS = -Q . NanoCore

SOURCES = Syntax.v Semantics.v Typing.v Soundness.v Progress.v Determinism.v Equivalence.v EvalFn.v
OBJECTS = $(SOURCES:.v=.vo)

all: $(OBJECTS)

Syntax.vo: Syntax.v
	$(COQC) $(COQFLAGS) $<

Semantics.vo: Semantics.v Syntax.vo
	$(COQC) $(COQFLAGS) $<

Typing.vo: Typing.v Syntax.vo
	$(COQC) $(COQFLAGS) $<

Soundness.vo: Soundness.v Syntax.vo Semantics.vo Typing.vo
	$(COQC) $(COQFLAGS) $<

Progress.vo: Progress.v Syntax.vo Semantics.vo Typing.vo
	$(COQC) $(COQFLAGS) $<

Determinism.vo: Determinism.v Syntax.vo Semantics.vo
	$(COQC) $(COQFLAGS) $<

Equivalence.vo: Equivalence.v Syntax.vo Semantics.vo Typing.vo Progress.vo
	$(COQC) $(COQFLAGS) $<

EvalFn.vo: EvalFn.v Syntax.vo Semantics.vo
	$(COQC) $(COQFLAGS) $<

clean:
	rm -f *.vo *.vok *.vos *.glob .*.aux

.PHONY: all clean
