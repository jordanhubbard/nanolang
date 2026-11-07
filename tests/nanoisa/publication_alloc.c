/* I count allocations owned by selected publication providers. Untracked
 * frontend/common allocations may cross their free/realloc boundaries. */
#include "publication_alloc.h"
#include <stdint.h>
#include <stdio.h>
static void *slots[65536];
size_t pa_live,pa_calls,pa_fail=SIZE_MAX,pa_denied;
bool pa_active,pa_persistent;
static size_t locate(void *p) {size_t i=0;while(i<pa_live&&slots[i]!=p)i++;return i;}
static bool denied(void) {
    if(!pa_active)return false;
    size_t i=pa_calls++;
    bool deny=i==pa_fail||(pa_persistent&&i>pa_fail)||
        (pa_persistent&&pa_denied&&getenv("PUBLICATION_GROWTH_FAIL"));
    if(deny)pa_denied++;
    return deny;
}
static void remember(void *p) {
    if(!p)return;
    if(pa_live==sizeof slots/sizeof *slots){fputs("I exceeded publication allocation tracking\n",stderr);abort();}
    slots[pa_live++]=p;
}
void *pa_malloc(size_t n){if(denied())return NULL;void *p=malloc(n);if(pa_active)remember(p);return p;}
void *pa_calloc(size_t n,size_t w){if(denied())return NULL;void *p=calloc(n,w);if(pa_active)remember(p);return p;}
void pa_free(void *p){if(!p)return;size_t i=locate(p);if(i<pa_live)slots[i]=slots[--pa_live];free(p);}
void *pa_realloc(void *p,size_t n){
    if(denied())return NULL;
    if(pa_active && n==8192 && getenv("PUBLICATION_GROWTH_FAIL") && !pa_denied){pa_denied++;return NULL;}
    if(!n){pa_free(p);return NULL;}
    size_t i=locate(p);void *q=realloc(p,n);if(!q)return NULL;
    if(i<pa_live)slots[i]=q;else if(pa_active)remember(q);
    return q;
}

/* I follow ownership across cleanup-only providers without injecting there. */
void *pa_cleanup_realloc(void *p,size_t n){
    bool active=pa_active;pa_active=false;
    void *result=pa_realloc(p,n);pa_active=active;return result;
}
