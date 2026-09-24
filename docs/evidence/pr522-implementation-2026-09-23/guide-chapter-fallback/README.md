# Published guide chapter fallback

I retain the first hosted failure at `96d1c8589`, job 107540724841.
The renderer already publishes English fallback chapters in each locale;
my source link checker recognized generated references only. I accept an
untranslated chapter target only for a known locale, a declared navigation
entry and an existing English Markdown source within the guide tree.
Missing, unpublished and escaped fallback targets remain errors.

All 19 link-checker and guide tests pass. All six editions render and validate,
and direct source-link checks pass for all five translated starting chapters.
The whole sparse-checkout link check still reports 383 omitted files; every
target exists in the pinned Git tree. I retain that failure and audit instead
of claiming the filesystem gate passed. Full hosted acceptance remains pending.
