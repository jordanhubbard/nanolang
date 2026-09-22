# My native list declaration ownership

My retained4d Puck discovery case reaches actual Stage1 native generation under
strict C99. I emit `typedef struct List_Item List_Item;` before record layouts,
then emit it again in generate_list_for_c_type. Apple Clang rejects the duplicate
with -Wtypedef-redefinition under the unchanged -Werror gate. No rejected product
runs. My C-seed emitter already separates forward typedefs and provider bodies.

I keep the early forward declaration: derived record/union/tuple/callback layouts
may need the list name before its element record and provider body are complete.
I keep standalone generate_list_for_c_type/generate_list_for_type behavior too.
I add an internal generator mode that optionally emits its own forward typedef.
The standalone wrapper requests it; generate_list_specializations, which follows
my full program's early declaration pass, requests only the provider body. Both
paths keep the shared runtime include and exact NL_DEFINE_RECORD_LIST key/type.
The mode changes declaration ownership only, not list layout, bounds, allocation,
argument evaluation, nominal identity or provider linkage.

I audit every caller before changing the shared generator. The only actual
specialization caller uses the full program's early forward pass; the standalone
wrapper is otherwise used by retained generator shadows. Foreign schema-backed
lists keep their existing header selection. I do not change schema header guards
or use a macro that could suppress their complete struct definition.

I retain original shadows and native discovery source. Additive controls require
the standalone result to contain its typedef, the already-forwarded body to omit
it, and the actual complete program output to contain exactly one typedef for the
selected record. Existing strict Apple/GCC/Clang producer gates and original18+9
remain acceptance; textual controls alone do not qualify the generated program.

MAC: task_f201f1249e0a4567ad56a15b024b1408.

My source checkpoint separates generate_list_for_c_type_mode from the retained
standalone wrapper. My additive shadows inspect the actual complete program and
require one forward declaration, as well as separate standalone/body behavior.
This checkpoint has source review pending; I have not executed its gates.
