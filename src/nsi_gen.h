#ifndef NL_NSI_GEN_H
#define NL_NSI_GEN_H

#include "nsi.h"
#include <stdio.h>

/* Bindings and stubs from one NSI document. Ids stay stable; I do not
 * infer a C ABI. Return 0 on success. */

int nl_nsi_gen_nanolang(const NlNsi *nsi, FILE *out);
int nl_nsi_gen_forth(const NlNsi *nsi, FILE *out);
int nl_nsi_gen_python(const NlNsi *nsi, FILE *out);
int nl_nsi_gen_rust(const NlNsi *nsi, FILE *out);
int nl_nsi_gen_cxx(const NlNsi *nsi, FILE *out);
int nl_nsi_gen_dispatch(const NlNsi *nsi, FILE *out);
int nl_nsi_gen_mock(const NlNsi *nsi, FILE *out);
int nl_nsi_gen_docs(const NlNsi *nsi, FILE *out);
int nl_nsi_gen_serialize(const NlNsi *nsi, FILE *out);
int nl_nsi_gen_validate(const NlNsi *nsi, FILE *out);
int nl_nsi_gen_compat_tests(const NlNsi *nsi, FILE *out);
int nl_nsi_gen_language_index(const NlNsi *nsi, FILE *out);
int nl_nsi_gen_nanoisa_imports(const NlNsi *nsi, FILE *out);

#endif
