# My selected-driver response grammar

I retain the semantic repair at `d303d5b23` under #1158. I query the actual
selected compiler with four bounded, harmless preprocessing forms when a
quoted backslash or escaped newline needs disambiguation. I preserve the
original response path when that query is unknown or expires. Selection
continues to honor my explicit compiler and metadata priorities. My existing
response limits, argument boundaries and tool admission tables remain intact.

My retained GCC 15 observations distinguish its grammar from GCC 13 and the
observed Clang drivers. Single quotes preserve backslashes. Double quotes
consume a backslash before backslash, quote, dollar or backtick and preserve
it otherwise. Outside single quotes, escaped newline is removed. I select
these rules from measured output, without guessing from a major version.

I preserve these independent observations and corrections:

- My initial 60 literal-argv controls cover four actual driver identities,
  nested responses and quoted/unquoted escapes. Hosted controls come from
  [44942021445](https://github.com/jordanhubbard/nanolang/actions/runs/44942021445).
- [44943541684](https://github.com/jordanhubbard/nanolang/actions/runs/44943541684)
  passes all eight original methods and retains my new dollar/backtick failures.
- [44943965916](https://github.com/jordanhubbard/nanolang/actions/runs/44943965916)
  retains all 600 byte/context observations and exactly two remaining GCC 15
  escaped-newline differences.
- [44944360513](https://github.com/jordanhubbard/nanolang/actions/runs/44944360513)
  passes all nine selected methods, with no skips, errors or failures. Its
  corrected 600-observation matrix agrees on native/replay behavior. My private
  assembler admission delta is restored afterward; general qualification is
  explicitly false.

My complete local source-snapshot suite then passes 129 methods in 1285.535
seconds with 24 conditional platform/driver skips, using explicitly selected
system GCC 13.3, GNU assembler 2.42 and Ubuntu Clang 18.1.3. I retain its complete
terminal, the 600-observation local matrix and actual probe image. Tool hashes
and banners in `response-qualified-tool-identities.json` were observed after
the run, rather than asserted to be an immutable pre-run snapshot.

I also retain the earlier complete default-tool run: 129 methods, 34 failures,
three errors and 24 skips under development Clang 23.0.0git. The integrated
assembler report reaches the same unsupported exact-report boundary in both
my unchanged original probe and my response-repair probe. The actual report,
probe images, provider source and tool identities establish this baseline
refusal independently of the successful qualified-tool run. Under #1162 I
classify that refusal without admitting the development compiler. Its exact
native/source/platform qualification would be separate work.

My manifest retains 330 logical inputs, 10,006,504 raw bytes and 2,849,853 unique
compressed bytes with checked raw and compressed SHA-256 identities. I preserve
original test cleanup; successful temporary native objects are not retained
by this receipt. Darwin acceptance, GNU assembler 2.46 native provenance,
complete external-tool qualification and the release remain open.
