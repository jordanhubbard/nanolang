#include "../modules/ui_widgets/ui_widgets.h"
#include <assert.h>
#include <string.h>
#include <float.h>
#include <stdlib.h>
static int fail_token_alloc;
static void *fake_alloc(size_t size) { return fail_token_alloc ? NULL : malloc(size); }
static int mouse_calls, text_calls;
static int draw_calls;
static int render_copies, texture_frees, surface_frees, provide_surface;
static SDL_Surface test_surface;
static const char *expected_text = "visible";
static SDL_Color last_draw_color, last_text_color;
static Uint8 texture_r = 12, texture_g = 34, texture_b = 56, texture_alpha = 200;
static SDL_BlendMode texture_blend = SDL_BLENDMODE_BLEND, renderer_blend = SDL_BLENDMODE_NONE;
static int fail_texture_query, fail_add;
static int measured_w = 10, measured_h = 10, fail_measure, measure_calls;
static int fake_size(TTF_Font *f, const char *s, int *w, int *h) {
    (void)f; (void)s; measure_calls++;
    if (fail_measure) return -1;
    if (w) *w=measured_w;
    if (h) *h=measured_h;
    return 0;
}
static Uint8 copied_alpha, copied_r;
static SDL_BlendMode copied_blend;
static SDL_bool clip_enabled;
static SDL_Rect clip_rect;
static SDL_bool fake_clip_enabled(SDL_Renderer *r) { (void)r; return clip_enabled; }
static void fake_get_clip(SDL_Renderer *r, SDL_Rect *out) { (void)r; *out=clip_rect; }
static int fake_set_clip(SDL_Renderer *r, const SDL_Rect *p) {
    (void)r; clip_enabled=p ? SDL_TRUE : SDL_FALSE;
    if (p) clip_rect=*p;
    return 0;
}
static int get_color(SDL_Texture *t, Uint8 *r, Uint8 *g, Uint8 *b) {
    (void)t; *r=texture_r; *g=texture_g; *b=texture_b; return fail_texture_query;
}
static int set_color(SDL_Texture *t, Uint8 r, Uint8 g, Uint8 b) {
    (void)t; texture_r=r; texture_g=g; texture_b=b; return 0;
}
static int get_alpha(SDL_Texture *t, Uint8 *a) { (void)t; *a=texture_alpha; return 0; }
static int set_alpha(SDL_Texture *t, Uint8 a) { (void)t; texture_alpha=a; return 0; }
static int get_blend(SDL_Texture *t, SDL_BlendMode *b) { (void)t; *b=texture_blend; return 0; }
static int set_blend(SDL_Texture *t, SDL_BlendMode b) {
    (void)t; if (fail_add && b == SDL_BLENDMODE_ADD) return -1;
    texture_blend=b; return 0;
}
static int get_renderer_blend(SDL_Renderer *r, SDL_BlendMode *b) { (void)r; *b=renderer_blend; return 0; }
static int set_renderer_blend(SDL_Renderer *r, SDL_BlendMode b) { (void)r; renderer_blend=b; return 0; }
static Uint32 host_buttons;
static int host_mouse_x = -100, host_mouse_y = -100;
static Uint32 mouse(int *x, int *y) {
    mouse_calls++; *x = host_mouse_x; *y = host_mouse_y; return host_buttons;
}
static int color(SDL_Renderer *r, Uint8 a, Uint8 b, Uint8 c, Uint8 d) {
    (void)r; last_draw_color = (SDL_Color){a,b,c,d}; return 0;
}
static int fake_rect(SDL_Renderer *r, const SDL_Rect *p) {
    (void)r;
    if (p) { assert(p->w >= 0 && p->h >= 0); draw_calls++; }
    return 0;
}
static int line(SDL_Renderer *r, int a, int b, int c, int d) {
    (void)r; (void)a; (void)b; (void)c; (void)d; return 0;
}
static int fake_point(SDL_Renderer *r, int x, int y) {
    (void)r; (void)x; (void)y; draw_calls++; return 0;
}
static SDL_Surface *fake_text(TTF_Font *f, const char *s, SDL_Color c) {
    (void)f; (void)c; assert(!strcmp(s, expected_text)); text_calls++;
    last_text_color = c;
    return provide_surface ? &test_surface : NULL;
}
static SDL_Texture *fake_texture(SDL_Renderer *r, SDL_Surface *s) {
    (void)r; assert(s == &test_surface); return (SDL_Texture *)(uintptr_t)2;
}
static int fake_copy(SDL_Renderer *r, SDL_Texture *t, const SDL_Rect *s, const SDL_Rect *d) {
    (void)r; (void)t; (void)s;
    assert(d->w >= 0 && d->h >= 0); render_copies++;
    copied_alpha=texture_alpha; copied_r=texture_r; copied_blend=texture_blend; return 0;
}
static void fake_destroy(SDL_Texture *t) { (void)t; texture_frees++; }
static void fake_free_surface(SDL_Surface *s) { assert(s == &test_surface); surface_frees++; }
#define SDL_GetMouseState mouse
#define SDL_SetRenderDrawColor color
#define SDL_RenderFillRect fake_rect
#define SDL_RenderDrawRect fake_rect
#define SDL_RenderSetClipRect fake_set_clip
#define SDL_RenderGetClipRect fake_get_clip
#define SDL_RenderIsClipEnabled fake_clip_enabled
#define SDL_RenderDrawLine line
#define SDL_RenderDrawPoint fake_point
#define TTF_RenderText_Blended fake_text
#define TTF_RenderUTF8_Blended fake_text
#define TTF_SizeText fake_size
#define SDL_CreateTextureFromSurface fake_texture
#define SDL_RenderCopy fake_copy
#define SDL_DestroyTexture fake_destroy
#define SDL_FreeSurface fake_free_surface
#define SDL_GetTextureColorMod get_color
#define SDL_SetTextureColorMod set_color
#define SDL_GetTextureAlphaMod get_alpha
#define SDL_SetTextureAlphaMod set_alpha
#define SDL_GetTextureBlendMode get_blend
#define SDL_SetTextureBlendMode set_blend
#define SDL_GetRenderDrawBlendMode get_renderer_blend
#define SDL_SetRenderDrawBlendMode set_renderer_blend
#define malloc fake_alloc
#include "../modules/ui_widgets/ui_widgets.c"
#undef malloc

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
static void spinner(void) {
    SDL_Renderer *r = (SDL_Renderer *)(uintptr_t)1;
    TTF_Font *f = (TTF_Font *)(uintptr_t)1;
    host_mouse_x = -100; host_mouse_y = -100; host_buttons = 0;
    int before = draw_calls;
    assert(nl_ui_number_spinner(r,NULL,5,10,0,0,0,100,20) == 5);
    assert(nl_ui_number_spinner(r,NULL,5,0,10,INT64_MAX,0,100,20) == 5);
    assert(nl_ui_number_spinner(r,NULL,5,0,10,0,0,39,20) == 5);
    assert(nl_ui_number_spinner(r,NULL,5,0,10,0,0,100,9) == 5);
    assert(draw_calls == before);
    assert(nl_ui_number_spinner(r,NULL,INT64_MIN,0,10,0,0,100,20) == 0);
    assert(nl_ui_number_spinner(r,NULL,INT64_MAX,0,10,0,0,100,20) == 10);
    host_mouse_x = 90; host_mouse_y = 10;
    host_buttons = SDL_BUTTON(SDL_BUTTON_LEFT);
    assert(nl_ui_number_spinner(r,NULL,INT64_MAX,INT64_MIN,INT64_MAX,0,0,100,20) == INT64_MAX);
    host_buttons = 0;
    assert(nl_ui_number_spinner(r,NULL,INT64_MAX,INT64_MIN,INT64_MAX,0,0,100,20) == INT64_MAX);
    host_buttons = SDL_BUTTON(SDL_BUTTON_LEFT);
    assert(nl_ui_number_spinner(r,NULL,4,0,10,0,0,100,20) == 4);
    host_buttons = 0;
    assert(nl_ui_number_spinner(r,NULL,4,0,10,0,0,100,20) == 5);
    host_mouse_x = 10; host_buttons = SDL_BUTTON(SDL_BUTTON_LEFT);
    assert(nl_ui_number_spinner(r,NULL,INT64_MIN,INT64_MIN,INT64_MAX,0,0,100,20) == INT64_MIN);
    host_buttons = 0;
    assert(nl_ui_number_spinner(r,NULL,INT64_MIN,INT64_MIN,INT64_MAX,0,0,100,20) == INT64_MIN);
    host_mouse_x = -100;
    expected_text = "5"; provide_surface = 1;
    test_surface.w = 10; test_surface.h = 10;
    assert(nl_ui_number_spinner(r,f,5,0,10,0,0,100,20) == 5);
    assert(render_copies == 1);
    test_surface.w = INT_MAX; test_surface.h = INT_MAX;
    assert(nl_ui_number_spinner(r,f,5,0,10,INT_MIN,INT_MIN,100,20) == 5);
    assert(render_copies == 1 && texture_frees == 2 && surface_frees == 2);
    test_surface.w = -1;
    assert(nl_ui_number_spinner(r,f,5,0,10,0,0,100,20) == 5);
    assert(render_copies == 1 && texture_frees == 3 && surface_frees == 3);
    provide_surface = 0;
}
static void panels_and_labels(void) {
    SDL_Renderer *r = (SDL_Renderer *)(uintptr_t)1;
    TTF_Font *f = (TTF_Font *)(uintptr_t)1;
    int before = draw_calls;
    nl_ui_panel(r,0,0,100,20,INT64_MAX,INT64_MIN,254,INT64_MAX);
    assert(draw_calls == before + 2);
    assert(last_draw_color.r == 255 && last_draw_color.g == 40 &&
           last_draw_color.b == 255 && last_draw_color.a == 255);
    before = draw_calls;
    nl_ui_panel(r,INT64_MAX,0,100,20,0,0,0,0);
    nl_ui_panel(r,INT_MAX,0,100,20,0,0,0,0);
    nl_ui_panel(r,0,0,0,20,0,0,0,0);
    assert(draw_calls == before);
    expected_text = "label"; provide_surface = 1;
    test_surface.w = 10; test_surface.h = 10;
    int copies = render_copies, frees = surface_frees;
    nl_ui_label(r,f,"label",0,0,INT64_MAX,INT64_MIN,128,-1);
    assert(render_copies == copies + 1 && surface_frees == frees + 1);
    assert(last_text_color.r == 255 && last_text_color.g == 0 &&
           last_text_color.b == 128 && last_text_color.a == 0);
    int texts = text_calls;
    nl_ui_label(r,f,"label",INT64_MAX,0,0,0,0,0);
    assert(text_calls == texts);
    nl_ui_label(r,f,"label",INT_MAX,0,0,0,0,0);
    assert(render_copies == copies + 1 && surface_frees == frees + 2);
    test_surface.w = -1;
    nl_ui_label(r,f,"label",0,0,0,0,0,0);
    assert(render_copies == copies + 1 && surface_frees == frees + 3);
    provide_surface = 0;
}
static void buttons(void) {
    SDL_Renderer *r = (SDL_Renderer *)(uintptr_t)1;
    TTF_Font *f = (TTF_Font *)(uintptr_t)1;
    int draws = draw_calls, mice = mouse_calls;
    assert(!nl_ui_button(r,f,"button",INT64_MAX,0,100,20));
    assert(!nl_ui_button(r,f,"button",INT_MAX,0,100,20));
    assert(!nl_ui_button(r,f,"button",0,0,0,20));
    assert(!nl_ui_button(NULL,f,"button",0,0,100,20));
    assert(draw_calls == draws && mouse_calls == mice);
    host_mouse_x = 10; host_mouse_y = 10;
    button_prev_mouse_down = 1; button_current_mouse_down = 0;
    expected_text = "button"; provide_surface = 1;
    test_surface.w = 10; test_surface.h = 10;
    int copies = render_copies, surfaces = surface_frees, textures = texture_frees;
    assert(nl_ui_button(r,f,"button",0,0,100,20) == 1);
    assert(render_copies == copies + 1);
    button_prev_mouse_down = 0;
    assert(!nl_ui_button(r,f,"button",0,0,100,20));
    assert(render_copies == copies + 2);
    test_surface.w = INT_MAX; test_surface.h = INT_MAX;
    assert(!nl_ui_button(r,f,"button",INT_MIN,INT_MIN,100,20));
    test_surface.w = -1;
    assert(!nl_ui_button(r,f,"button",0,0,100,20));
    assert(render_copies == copies + 2);
    assert(surface_frees == surfaces + 4 && texture_frees == textures + 4);
    SDL_Rect dest;
    assert(ui_centered_rect((SDL_Rect){0,0,100,20},10,10,&dest));
    assert(dest.x == 45 && dest.y == 5 && dest.w == 10 && dest.h == 10);
    assert(!ui_centered_rect((SDL_Rect){INT_MAX-10,0,10,10},INT_MAX,10,&dest));
    provide_surface = 0;
}
static void checks_and_radios(void) {
    SDL_Renderer *r = (SDL_Renderer *)(uintptr_t)1;
    TTF_Font *f = (TTF_Font *)(uintptr_t)1;
    int draws = draw_calls, mice = mouse_calls;
    assert(nl_ui_checkbox(r,f,"choice",INT64_MAX,0,1) == 1);
    assert(!nl_ui_radio_button(r,f,"choice",INT64_MIN,0,1));
    assert(!nl_ui_radio_button(r,f,"choice",INT_MAX-20,0,1));
    assert(nl_ui_checkbox(NULL,f,"choice",0,0,1) == 1);
    assert(draw_calls == draws && mouse_calls == mice);
    expected_text = "choice"; provide_surface = 1;
    test_surface.w = 10; test_surface.h = 10;
    host_mouse_x = 10; host_mouse_y = 10;
    checkbox_prev_mouse_down = radio_prev_mouse_down = 1;
    checkbox_current_mouse_down = radio_current_mouse_down = 0;
    int copies = render_copies, frees = surface_frees, textures = texture_frees;
    assert(nl_ui_checkbox(r,f,"choice",0,0,0) == 1);
    assert(nl_ui_checkbox(r,f,"choice",0,0,1) == 0);
    assert(nl_ui_radio_button(r,f,"choice",0,0,1) == 1);
    assert(render_copies == copies + 3);
    checkbox_prev_mouse_down = radio_prev_mouse_down = 0;
    test_surface.w = INT_MAX; test_surface.h = INT_MAX;
    assert(nl_ui_checkbox(r,f,"choice",INT_MIN,INT_MIN,1) == 1);
    assert(!nl_ui_radio_button(r,f,"choice",INT_MIN+1,INT_MIN+1,1));
    assert(nl_ui_checkbox(r,f,"choice",INT_MAX-20,0,1) == 1);
    assert(!nl_ui_radio_button(r,f,"choice",INT_MAX-21,0,1));
    assert(render_copies == copies + 3);
    assert(surface_frees == frees + 7 && texture_frees == textures + 7);
    SDL_Rect dest;
    assert(ui_control_label_rect((SDL_Rect){0,0,20,20},10,10,&dest));
    assert(dest.x == 28 && dest.y == 5);
    assert(!ui_control_label_rect((SDL_Rect){0,0,20,20},-1,10,&dest));
    provide_surface = 0;
}
static void time_displays(void) {
    SDL_Renderer *r = (SDL_Renderer *)(uintptr_t)1;
    TTF_Font *f = (TTF_Font *)(uintptr_t)1;
    const int64_t seconds[] = {0, 61, -61, 4294967296LL, INT64_MAX, INT64_MIN};
    const char *formatted[] = {"00:00", "01:01", "-01:01", "71582788:16",
        "153722867280912930:07", "-153722867280912930:08"};
    provide_surface = 1; test_surface.w = 100; test_surface.h = 10;
    int copies = render_copies, frees = surface_frees;
    for (size_t i = 0; i < sizeof(seconds)/sizeof(*seconds); i++) {
        expected_text = formatted[i];
        nl_ui_time_display(r,f,seconds[i],0,0,INT64_MAX,-1,128,255);
    }
    assert(render_copies == copies + 6 && surface_frees == frees + 6);
    assert(last_text_color.r == 255 && last_text_color.g == 0);
    int texts = text_calls;
    nl_ui_time_display(r,f,INT64_MIN,INT64_MAX,0,0,0,0,0);
    assert(text_calls == texts);
    nl_ui_time_display(r,f,INT64_MIN,INT_MAX,0,0,0,0,0);
    assert(render_copies == copies + 6 && surface_frees == frees + 7);
    provide_surface = 0;
}
static void image_buttons(void) {
    SDL_Renderer *r = (SDL_Renderer *)(uintptr_t)1;
    host_mouse_x=10; host_mouse_y=10;
    button_prev_mouse_down=1; button_current_mouse_down=0;
    int copies=render_copies, mice=mouse_calls;
    assert(!nl_ui_image_button(r,2,INT64_MAX,0,100,20,1.2));
    assert(!nl_ui_image_button(r,2,0,0,1,20,1.2));
    assert(!nl_ui_image_button(r,0,0,0,100,20,1.2));
    assert(render_copies == copies && mouse_calls == mice);
    const double values[] = {NAN, INFINITY, -INFINITY, DBL_MAX, -1, 0.5, 1.5};
    for (size_t i=0; i<sizeof(values)/sizeof(*values); i++) {
        copies=render_copies;
        assert(nl_ui_image_button(r,2,0,0,100,20,values[i]) == 1);
        int additive = values[i] == DBL_MAX || values[i] == 1.5;
        assert(render_copies == copies + 1 + additive);
        if (additive) {
            assert(copied_blend == SDL_BLENDMODE_ADD && copied_r == 255);
            assert(copied_alpha == (values[i] == 1.5 ? 100 : 200));
        } else if (values[i] == 0.5) assert(copied_r == 127);
        else if (values[i] == -1) assert(copied_r == 0);
        assert(texture_r == 12 && texture_g == 34 && texture_b == 56 &&
               texture_alpha == 200 && texture_blend == SDL_BLENDMODE_BLEND);
    }
    fail_add=1; copies=render_copies;
    assert(nl_ui_image_button(r,2,0,0,100,20,1.5) == 1);
    assert(render_copies == copies+1 && texture_blend == SDL_BLENDMODE_BLEND);
    fail_add=0; fail_texture_query=1; copies=render_copies;
    assert(!nl_ui_image_button(r,2,0,0,100,20,1.5));
    assert(render_copies == copies);
    fail_texture_query=0; button_current_mouse_down=1;
    assert(!nl_ui_image_button(r,2,0,0,100,20,1.5));
    assert(renderer_blend == SDL_BLENDMODE_NONE);
    button_current_mouse_down=0;
}
static void tooltips(void) {
    SDL_Renderer *r = (SDL_Renderer *)(uintptr_t)1;
    TTF_Font *f = (TTF_Font *)(uintptr_t)1;
    host_mouse_x=10; host_mouse_y=10;
    expected_text="tip"; provide_surface=1;
    test_surface.w=10; test_surface.h=10;
    int copies=render_copies, frees=surface_frees;
    nl_ui_tooltip(r,f,"tip",0,0,100,20);
    assert(render_copies == copies+1 && surface_frees == frees+1);
    int draws=draw_calls, measures=measure_calls;
    nl_ui_tooltip(r,f,"tip",INT64_MAX,0,100,20);
    assert(draw_calls == draws && measure_calls == measures);
    fail_measure=1;
    nl_ui_tooltip(r,f,"tip",0,0,100,20);
    fail_measure=0; measured_w=INT_MAX;
    nl_ui_tooltip(r,f,"tip",0,0,100,20);
    measured_w=-1;
    nl_ui_tooltip(r,f,"tip",0,0,100,20);
    measured_w=10; measured_h=INT_MAX;
    nl_ui_tooltip(r,f,"tip",0,0,100,20);
    measured_h=10;
    host_mouse_x=INT_MAX; host_mouse_y=INT_MAX;
    nl_ui_tooltip(r,f,"tip",INT_MAX-100,INT_MAX-20,100,20);
    assert(draw_calls == draws);
    host_mouse_x=10; host_mouse_y=10;
    test_surface.w=INT_MAX;
    nl_ui_tooltip(r,f,"tip",0,0,100,20);
    assert(render_copies == copies+1 && surface_frees == frees+2);
    provide_surface=0;
}
static void text_inputs(void) {
    SDL_Renderer *r = (SDL_Renderer *)(uintptr_t)1;
    TTF_Font *f = (TTF_Font *)(uintptr_t)1;
    char unterminated[3] = {'b','a','d'};
    int draws=draw_calls, mice=mouse_calls;
    assert(!nl_ui_text_input(r,f,unterminated,3,0,0,100,20,1));
    assert(!nl_ui_text_input(r,f,"input",0,0,0,100,20,1));
    assert(!nl_ui_text_input(r,f,NULL,6,0,0,100,20,1));
    assert(!nl_ui_text_input(r,f,"input",6,INT64_MAX,0,100,20,1));
    assert(!nl_ui_text_input(r,f,"input",6,0,0,15,20,1));
    assert(draw_calls == draws && mouse_calls == mice);
    expected_text="input"; provide_surface=1;
    test_surface.w=10; test_surface.h=10;
    int copies=render_copies, frees=surface_frees;
    char input[]="input";
    assert(!nl_ui_text_input(r,f,input,sizeof(input),0,0,100,20,1));
    assert(render_copies == copies+1 && !strcmp(input,"input"));
    test_surface.h=INT_MAX;
    assert(!nl_ui_text_input(r,f,input,sizeof(input),INT_MIN,INT_MIN,100,20,0));
    assert(render_copies == copies+1 && surface_frees == frees+2);
    test_surface.w=-1;
    assert(!nl_ui_text_input(r,f,input,sizeof(input),0,0,100,20,0));
    assert(render_copies == copies+1 && surface_frees == frees+3);
    provide_surface=0; measured_w=INT_MAX; measured_h=10;
    for (int i=0; i<120; i++)
        assert(!nl_ui_text_input(r,f,input,sizeof(input),INT_MAX-100,0,100,20,1));
    fail_measure=1;
    for (int i=0; i<60; i++)
        assert(!nl_ui_text_input(r,f,input,sizeof(input),0,0,100,20,1));
    fail_measure=0;
    int measures=measure_calls;
    for (int i=0; i<60; i++)
        assert(!nl_ui_text_input(r,NULL,input,sizeof(input),0,0,100,20,1));
    assert(measure_calls == measures);
    measured_w=10;
}
static void list_geometry(void) {
    SDL_Renderer *r = (SDL_Renderer *)(uintptr_t)1;
    TTF_Font *f = (TTF_Font *)(uintptr_t)1;
    char *items[] = {"row"};
    DynArray a={.length=1,.capacity=1,.elem_type=ELEM_STRING,
                .elem_size=sizeof(char*),.data=items};
    int draws=draw_calls, mice=mouse_calls;
    assert(nl_ui_scrollable_list(r,f,&a,1,INT64_MAX,0,100,50,0,-1) == -1);
    assert(nl_ui_file_selector(r,f,&a,1,INT_MAX,0,100,50,0,-1) == -1);
    assert(nl_ui_scrollable_list(r,f,&a,1,0,0,9,50,0,-1) == -1);
    assert(nl_ui_file_selector(r,f,&a,1,0,0,15,50,0,-1) == -1);
    assert(draw_calls == draws && mouse_calls == mice);
    expected_text="row"; provide_surface=1;
    test_surface.w=10; test_surface.h=10;
    host_mouse_x=10; host_mouse_y=10; host_buttons=SDL_BUTTON(SDL_BUTTON_LEFT);
    clip_enabled=SDL_TRUE; clip_rect=(SDL_Rect){1,2,3,4};
    assert(nl_ui_scrollable_list(r,f,&a,1,0,0,100,50,0,-1) == -1);
    assert(nl_ui_file_selector(r,f,&a,1,0,0,100,50,0,-1) == -1);
    host_buttons=0;
    assert(nl_ui_scrollable_list(r,f,&a,1,0,0,100,50,0,0) == 0);
    assert(nl_ui_file_selector(r,f,&a,1,0,0,100,50,0,0) == 0);
    assert(clip_enabled && clip_rect.x == 1 && clip_rect.y == 2 &&
           clip_rect.w == 3 && clip_rect.h == 4);
    int copies=render_copies, frees=surface_frees, textures=texture_frees;
    test_surface.w=INT_MAX; test_surface.h=INT_MAX;
    assert(nl_ui_scrollable_list(r,f,&a,1,INT_MIN,INT_MIN,100,50,0,0) == -1);
    assert(nl_ui_file_selector(r,f,&a,1,INT_MIN,INT_MIN,100,50,0,0) == -1);
    assert(render_copies == copies && surface_frees == frees+2 && texture_frees == textures+2);
    provide_surface=0; clip_enabled=SDL_FALSE;
    assert(nl_ui_scrollable_list(r,f,&a,1,INT_MAX-100,INT_MAX-50,100,50,0,0) == -1);
    assert(!clip_enabled);
    assert(nl_ui_file_selector(r,f,&a,1,INT_MAX-100,INT_MAX-50,100,50,0,0) == -1);
}
static void dropdown_geometry(void) {
    SDL_Renderer *r = (SDL_Renderer *)(uintptr_t)1;
    TTF_Font *f = (TTF_Font *)(uintptr_t)1;
    char *items[]={"option","option","option","option","option"};
    DynArray a={.length=5,.capacity=5,.elem_type=ELEM_STRING,
                .elem_size=sizeof(char*),.data=items};
    int draws=draw_calls, mice=mouse_calls;
    assert(nl_ui_dropdown(r,f,&a,5,0,0,100,INT_MAX,0,1) == -1);
    assert(nl_ui_dropdown(r,f,&a,5,INT64_MAX,0,100,20,0,0) == -1);
    assert(nl_ui_dropdown(r,f,&a,5,0,INT_MAX-100,100,20,0,1) == -1);
    assert(nl_ui_dropdown(r,f,&a,5,0,0,29,20,0,0) == -1);
    assert(nl_ui_dropdown(r,f,&a,5,0,0,100,7,0,0) == -1);
    assert(draw_calls == draws && mouse_calls == mice);
    expected_text="option"; provide_surface=1;
    test_surface.w=10; test_surface.h=10;
    host_mouse_x=10; host_mouse_y=10; host_buttons=SDL_BUTTON(SDL_BUTTON_LEFT);
    assert(nl_ui_dropdown(r,f,&a,1,0,0,100,20,0,0) == -1);
    host_buttons=0;
    assert(nl_ui_dropdown(r,f,&a,1,0,0,100,20,0,0) == -2);
    host_mouse_y=30; host_buttons=SDL_BUTTON(SDL_BUTTON_LEFT);
    assert(nl_ui_dropdown(r,f,&a,1,0,0,100,20,0,1) == -1);
    host_buttons=0;
    assert(nl_ui_dropdown(r,f,&a,1,0,0,100,20,0,1) == 0);
    host_mouse_x=-10; host_mouse_y=-10; host_buttons=SDL_BUTTON(SDL_BUTTON_LEFT);
    assert(nl_ui_dropdown(r,f,&a,1,0,0,100,20,0,1) == -1);
    host_buttons=0;
    assert(nl_ui_dropdown(r,f,&a,1,0,0,100,20,0,1) == -3);
    int copies=render_copies, frees=surface_frees, textures=texture_frees;
    test_surface.h=INT_MAX;
    assert(nl_ui_dropdown(r,f,&a,1,INT_MIN,INT_MIN,100,20,0,1) == -1);
    assert(render_copies == copies && surface_frees == frees+2 && texture_frees == textures+2);
    provide_surface=0;
    assert(nl_ui_dropdown(r,f,&a,5,0,0,100,INT_MAX/6,-1,1) == -1);
    assert(nl_ui_dropdown(r,f,&a,5,INT_MAX-100,0,100,20,-1,1) == -1);
}
static void code_displays(void) {
    SDL_Renderer *r = (SDL_Renderer *)(uintptr_t)1;
    TTF_Font *f = (TTF_Font *)(uintptr_t)1;
    int draws=draw_calls;
    nl_ui_code_display(r,f,"token",INT64_MAX,0,100,100,0,20);
    nl_ui_code_display(r,f,"token",0,0,100,100,-1,20);
    nl_ui_code_display(r,f,"token",0,0,100,100,0,0);
    nl_ui_code_display(r,f,"token",0,0,100,100,INT64_MAX,20);
    assert(draw_calls == draws);
    char long_token[601];
    memset(long_token,'a',600); long_token[600]=0;
    expected_text=long_token; provide_surface=1;
    test_surface.w=50; test_surface.h=10;
    clip_enabled=SDL_TRUE; clip_rect=(SDL_Rect){1,2,3,4};
    int copies=render_copies, frees=surface_frees;
    nl_ui_code_display(r,f,long_token,0,0,100,100,0,20);
    assert(render_copies == copies+1 && surface_frees == frees+1);
    assert(clip_enabled && clip_rect.x == 1 && clip_rect.w == 3);
    fail_token_alloc=1;
    int texts=text_calls;
    nl_ui_code_display(r,f,long_token,0,0,100,100,0,20);
    assert(text_calls == texts && clip_enabled && clip_rect.x == 1);
    fail_token_alloc=0; expected_text="token"; measured_w=INT_MAX;
    nl_ui_code_display(r,f,"\t token",0,0,100,100,0,20);
    assert(render_copies == copies+1);
    fail_measure=1;
    nl_ui_code_display(r,f," token",0,0,100,100,0,20);
    fail_measure=0; measured_w=10;
    test_surface.w=INT_MAX; test_surface.h=INT_MAX;
    nl_ui_code_display(r,f,"token",INT_MAX-100,INT_MAX-100,100,100,0,20);
    clip_enabled=SDL_FALSE; provide_surface=0;
    nl_ui_code_display(r,f,"token\n token",0,0,100,100,INT_MAX,INT_MAX);
    assert(!clip_enabled);
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
    spinner();
    panels_and_labels();
    buttons();
    checks_and_radios();
    time_displays();
    image_buttons();
    tooltips();
    text_inputs();
    list_geometry();
    dropdown_geometry();
    code_displays();
    return 0;
}
