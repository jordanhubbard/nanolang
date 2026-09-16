COQC = coqc
COQCHK = coqchk
COQFLAGS = -Q . NanoCore

SOURCES = Syntax.v Semantics.v Typing.v Soundness.v Progress.v Determinism.v Equivalence.v EvalFn.v EvalFnTests.v Exhaustiveness.v Contracts.v
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

EvalFnTests.vo: EvalFnTests.v EvalFn.vo
	$(COQC) $(COQFLAGS) $<

Exhaustiveness.vo: Exhaustiveness.v Syntax.vo
	$(COQC) $(COQFLAGS) $<

Contracts.vo: Contracts.v Soundness.vo Progress.vo Determinism.vo Equivalence.vo EvalFn.vo
	$(COQC) $(COQFLAGS) $<

# I rebuild the report on every check, reject assumptions, then independently
# recheck libraries. This does not prove production compiler refinement.
check: all check-assumptions
	$(COQCHK) $(COQFLAGS) $(addprefix NanoCore.,$(SOURCES:.v=)) NanoCore.Assumptions

check-assumptions: Assumptions.v $(OBJECTS) check_assumptions.sh
	@$(COQC) $(COQFLAGS) Assumptions.v > Assumptions.log 2>&1 || { cat Assumptions.log; exit 1; }
	@cat Assumptions.log
	bash check_assumptions.sh Assumptions.v Assumptions.log

# ── OCaml Extraction ──────────────────────────────────────────────────
# Extract the computable evaluator to OCaml source code.
# Generated files go to extracted/ and are used by tools/nanocore_ref/.

Extract.vo: Extract.v EvalFn.vo Syntax.vo Semantics.vo
	$(COQC) $(COQFLAGS) -output-directory extracted $<

extract: Extract.vo

# Build the reference interpreter from extracted OCaml code
nanocore-ref: extract
	$(MAKE) -C ../tools/nanocore_ref all

# Install the reference interpreter to bin/
install-ref: nanocore-ref
	$(MAKE) -C ../tools/nanocore_ref install

clean:
	rm -f *.vo *.vok *.vos *.glob .*.aux Assumptions.log
	rm -rf extracted/*.ml extracted/*.mli extracted/*.cmx extracted/*.cmi extracted/*.o

.PHONY: all check check-assumptions clean extract nanocore-ref install-ref
