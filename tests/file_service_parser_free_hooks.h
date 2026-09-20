/* I observe actual owning-TU frees without replacing their implementation. */
#ifndef NANO_FILE_SERVICE_PARSER_FREE_HOOKS_H
#define NANO_FILE_SERVICE_PARSER_FREE_HOOKS_H
#include <stdlib.h>
void file_service_parser_observe_free(void *pointer);
#ifdef NANO_FILE_SERVICE_PARSER_OBSERVE_FREE
#define free file_service_parser_observe_free
#endif
#endif
