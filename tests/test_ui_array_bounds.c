#include "../modules/ui_widgets/ui_widgets.h"
#include <assert.h>
#include <string.h>
#include <float.h>
static int mouse_calls, text_calls;
static int draw_calls;
static Uint32 host_buttons;
static int host_mouse_x = -100, host_mouse_y = -100;
static Uint32 mouse(int *x, int *y) {
    mouse_calls++; *x = host_mouse_x; *y = host_mouse_y; return host_buttons;
}
static int color(SDL_Renderer *r, Uint8 a, Uint8 b, Uint8 c, Uint8 d) {
    (void)r; (void)a; (void)b; (void)c; (void)d; return 0;
}
static int fake_rect(SDL_Renderer *r, const SDL_Rect *p) {
    (void)r;
    if (p) { assert(p->w >= 0 && p->h >= 0); draw_calls++; }
    return 0;
}
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
static void bars(void) {
    SDL_Renderer *r = (SDL_Renderer *)(uintptr_t)1;
    nl_ui_set_scale(1.0);
    host_mouse_x = -100; host_mouse_y = -100; host_buttons = 0;
    const int64_t invalid[][4] = {
        {0,0,0,10}, {0,0,10,0}, {0,0,-1,10},
        {INT64_MIN,0,10,10}, {INT64_MAX,0,10,10},
        {0,INT64_MAX,10,10}, {0,0,INT64_MAX,10},
        {INT_MAX,0,10,10}, {0,INT_MAX,10,10}
    };
    for (size_t i = 0; i < sizeof(invalid)/sizeof(*invalid); i++) {
        const int64_t *g = invalid[i];
        int before = draw_calls, mouse_before = mouse_calls;
        assert(nl_ui_slider(r,g[0],g[1],g[2],g[3],NAN) == 0.0);
        nl_ui_progress_bar(r,g[0],g[1],g[2],g[3],INFINITY);
        assert(nl_ui_seekable_progress_bar(r,g[0],g[1],g[2],g[3],NAN) == -1.0);
        assert(draw_calls == before && mouse_calls == mouse_before);
    }
    const double values[] = {NAN, INFINITY, -INFINITY, DBL_MAX, -DBL_MAX, 0.5};
    const double expected[] = {0.0, 1.0, 0.0, 1.0, 0.0, 0.5};
    for (size_t i = 0; i < sizeof(values)/sizeof(*values); i++) {
        int before = draw_calls;
        assert(nl_ui_slider(r,0,0,100,20,values[i]) == expected[i]);
        nl_ui_progress_bar(r,0,0,100,20,values[i]);
        assert(nl_ui_seekable_progress_bar(r,0,0,100,20,values[i]) == -1.0);
        assert(draw_calls > before);
    }
    assert(nl_ui_slider(r,INT_MIN+4,INT_MIN+4,INT_MAX,20,1.0) == 1.0);
    assert(nl_ui_slider(r,INT_MAX-104,0,100,20,1.0) == 1.0);
    nl_ui_progress_bar(r,INT_MIN,INT_MIN,INT_MAX,INT_MAX,1.0);
    host_mouse_x = 50; host_mouse_y = 10;
    host_buttons = SDL_BUTTON(SDL_BUTTON_LEFT);
    assert(nl_ui_slider(r,0,0,100,20,0.0) == 0.5);
    assert(nl_ui_seekable_progress_bar(r,0,0,100,20,0.0) == -1.0);
    host_buttons = 0;
    assert(nl_ui_seekable_progress_bar(r,0,0,100,20,0.0) == 0.5);
}
int main(void) {
    double invalid_scales[] = {NAN, INFINITY, -INFINITY, 0.0, -1.0, 0.01};
    for (size_t i = 0; i < sizeof(invalid_scales) / sizeof(*invalid_scales); i++) {
        nl_ui_set_scale(invalid_scales[i]);
        assert(g_ui_scale == 1.0);
        assert(scaled_mouse_coordinate(INT_MAX) == INT_MAX);
        assert(scaled_mouse_coordinate(INT_MIN) == INT_MIN);
    }
    nl_ui_set_scale(0.02);
    assert(scaled_mouse_coordinate(INT_MAX) == INT_MAX);
    assert(scaled_mouse_coordinate(INT_MIN) == INT_MIN);
    nl_ui_set_scale(2.0);
    assert(scaled_mouse_coordinate(-5) == -2);
    assert(scaled_mouse_coordinate(5) == 2);
    nl_ui_set_scale(DBL_MAX);
    assert(scaled_mouse_coordinate(INT_MAX) == 0);
    assert(scaled_mouse_coordinate(INT_MIN) == 0);
    assert(point_in_rect(INT_MAX, INT_MAX, INT_MAX - 1, INT_MAX - 1, 10, 10));
    assert(point_in_rect(INT_MIN, INT_MIN, INT_MIN, INT_MIN, 10, 10));
    assert(!point_in_rect(0, 0, INT_MIN, INT_MIN, -1, -1));
    assert(!point_in_rect(0, 0, 1, 1, INT_MAX, INT_MAX));
    host_mouse_x = INT_MAX; host_mouse_y = INT_MIN;
    nl_ui_set_scale(0.02);
    int sx, sy;
    get_mouse_scaled(&sx, &sy);
    assert(sx == INT_MAX && sy == INT_MIN);
    mouse_calls = 0;
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
    nl_ui_set_scale(1.0);
    bars();
    return 0;
}
