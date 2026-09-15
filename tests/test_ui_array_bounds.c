#include "../modules/ui_widgets/ui_widgets.h"
#include <assert.h>
#include <string.h>
static int mouse_calls, text_calls;
static Uint32 mouse(int *x, int *y) { mouse_calls++; *x = -100; *y = -100; return 0; }
static int color(SDL_Renderer *r, Uint8 a, Uint8 b, Uint8 c, Uint8 d) {
    (void)r; (void)a; (void)b; (void)c; (void)d; return 0;
}
static int fake_rect(SDL_Renderer *r, const SDL_Rect *p) { (void)r; (void)p; return 0; }
static int line(SDL_Renderer *r, int a, int b, int c, int d) {
    (void)r; (void)a; (void)b; (void)c; (void)d; return 0;
}
static SDL_Surface *fake_text(TTF_Font *f, const char *s, SDL_Color c) {
    (void)f; (void)c; assert(!strcmp(s, "visible")); text_calls++; return NULL;
}
#define SDL_GetMouseState mouse
#define SDL_SetRenderDrawColor color
#define SDL_RenderFillRect fake_rect
#define SDL_RenderDrawRect fake_rect
#define SDL_RenderSetClipRect fake_rect
#define SDL_RenderDrawLine line
#define TTF_RenderText_Blended fake_text
#include "../modules/ui_widgets/ui_widgets.c"

static void invoke(DynArray *a, int64_t count, int64_t scroll) {
    SDL_Renderer *r = (SDL_Renderer *)(uintptr_t)1;
    TTF_Font *f = (TTF_Font *)(uintptr_t)1;
    assert(nl_ui_scrollable_list(r, f, a, count, 0, 0, 100, 50, scroll, -1) == -1);
    assert(nl_ui_file_selector(r, f, a, count, 0, 0, 100, 50, scroll, -1) == -1);
}
int main(void) {
    assert(nl_ui_scrollable_list__nano_array_abi == NANO_DYN_ARRAY_ABI_VERSION);
    assert(nl_ui_dropdown__nano_array_abi == NANO_DYN_ARRAY_ABI_VERSION);
    assert(nl_ui_file_selector__nano_array_abi == NANO_DYN_ARRAY_ABI_VERSION);
    char *data[] = {"visible", "outside prefix"};
    DynArray valid = {.length=2, .capacity=2, .elem_size=sizeof(char*),
                      .elem_type=ELEM_STRING, .data=data};
    SDL_Renderer *r = (SDL_Renderer *)(uintptr_t)1;
    TTF_Font *f = (TTF_Font *)(uintptr_t)1;
    for (int i = 0; i < 9; i++) {
        DynArray a = valid; int64_t count = 1;
        switch (i) {
        case 0: a.elem_type = ELEM_INT; break;
        case 1: a.elem_size = 1; break;
        case 2: a.length = -1; break;
        case 3: a.capacity = 1; break;
        case 4: a.data = NULL; break;
        case 5: count = -1; break;
        case 6: count = 3; break;
        case 7: a.capacity = INT64_MAX; break;
        case 8: a.length = a.capacity = count = (int64_t)INT_MAX + 1; break;
        }
        invoke(&a, count, 0);
        assert(nl_ui_dropdown(r, f, &a, count, 0, 0, 100, 25, -1, 1) == -1);
        assert(!mouse_calls && !text_calls);
    }
    invoke(NULL, 0, 0);
    invoke(&valid, 1, -1);
    invoke(&valid, 1, INT64_MAX);
    assert(!mouse_calls);
    invoke(&valid, 1, 0);
    assert(text_calls == 2);
    /* I do not render selected entries outside the requested prefix. */
    assert(nl_ui_dropdown(r, f, &valid, 1, 0, 0, 100, 25, 1, 1) == -1);
    assert(text_calls == 3);
    invoke(&valid, 1, 1);
    assert(text_calls == 3);
    return 0;
}
