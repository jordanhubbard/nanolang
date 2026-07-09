/* sdl_term.c — SDL2 + libtmt terminal renderer for NanoLang
 *
 * Each NLTerm owns:
 *   - a libtmt TMT virtual screen
 *   - a TTF_Font for monospace rendering
 *   - a pre-allocated SDL_Texture for each line (lazily rendered)
 *
 * The design is deliberately simple: we re-render only dirty lines,
 * using SDL_RenderFillRect for coloured backgrounds and TTF_RenderUTF8_Blended
 * for foreground text.  Cursor blinking is handled by the caller.
 */

#include "sdl_term.h"
#include "tmt.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <wchar.h>

/* ── Colour table (matches libtmt's tmt_color_t) ────────────────────────── */

typedef struct { uint8_t r, g, b; } RGB;

static const RGB COLORS[9] = {
    {  12,  12,  12 }, /* TMT_COLOR_BLACK   */
    { 197,  15,  31 }, /* TMT_COLOR_RED     */
    {  19, 161,  14 }, /* TMT_COLOR_GREEN   */
    { 193, 156,   0 }, /* TMT_COLOR_YELLOW  */
    {   0,  55, 218 }, /* TMT_COLOR_BLUE    */
    { 136,  23, 152 }, /* TMT_COLOR_MAGENTA */
    {  58, 150, 221 }, /* TMT_COLOR_CYAN    */
    { 204, 204, 204 }, /* TMT_COLOR_WHITE   */
    { 204, 204, 204 }, /* TMT_COLOR_DEFAULT → light grey */
};

static RGB color_fg(tmt_color_t c, const TMTATTRS *a) {
    if (a->bold && c == TMT_COLOR_DEFAULT) return (RGB){255,255,255};
    return COLORS[(int)c];
}
static RGB color_bg(tmt_color_t c) {
    if (c == TMT_COLOR_DEFAULT) return COLORS[0];
    return COLORS[(int)c];
}

/* ── NLTerm struct ─────────────────────────────────────────────────────── */

struct NLTerm {
    TMT        *vt;
    TTF_Font   *font;
    int         cell_w;
    int         cell_h;
    int         rows;
    int         cols;
    int         dirty;   /* set by the libtmt callback */
};

static void tmt_callback(tmt_msg_t m, TMT *vt, const void *a, void *p) {
    (void)vt; (void)a;
    NLTerm *t = (NLTerm *)p;
    if (m == TMT_MSG_UPDATE || m == TMT_MSG_MOVED) {
        t->dirty = 1;
    }
}

/* ── Font helpers ──────────────────────────────────────────────────────── */

static const char *FONT_FALLBACKS[] = {
#ifdef __linux__
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
#elif defined(__APPLE__)
    "/System/Library/Fonts/Menlo.ttc",
    "/Library/Fonts/Courier New.ttf",
#endif
    NULL
};

static TTF_Font *open_mono_font(const char *hint, int pt) {
    if (hint && hint[0]) {
        TTF_Font *f = TTF_OpenFont(hint, pt);
        if (f) return f;
    }
    for (int i = 0; FONT_FALLBACKS[i]; i++) {
        TTF_Font *f = TTF_OpenFont(FONT_FALLBACKS[i], pt);
        if (f) return f;
    }
    return NULL;
}

/* ── Public C API ──────────────────────────────────────────────────────── */

NLTerm *nl_term_create(const char *font_path, int64_t pt_size,
                        int64_t rows, int64_t cols) {
    if (!TTF_WasInit()) TTF_Init();

    NLTerm *t = calloc(1, sizeof(NLTerm));
    if (!t) return NULL;

    t->font = open_mono_font(font_path, (int)pt_size);
    if (!t->font) { free(t); return NULL; }

    /* Measure a single character to get the cell size */
    int w = 0, h = 0;
    TTF_SizeText(t->font, "M", &w, &h);
    t->cell_w = (w > 0) ? w : (int)pt_size;
    t->cell_h = TTF_FontLineSkip(t->font);
    t->rows   = (int)rows;
    t->cols   = (int)cols;
    t->dirty  = 1;

    t->vt = tmt_open((size_t)rows, (size_t)cols, tmt_callback, t, NULL);
    if (!t->vt) {
        TTF_CloseFont(t->font);
        free(t);
        return NULL;
    }
    return t;
}

void nl_term_feed(NLTerm *t, const char *data) {
    if (t && data && data[0]) {
        tmt_write(t->vt, data, 0);
    }
}

void nl_term_render(NLTerm *t, SDL_Renderer *renderer, int64_t ox, int64_t oy) {
    if (!t || !renderer) return;

    const TMTSCREEN *s = tmt_screen(t->vt);
    int cw = t->cell_w, ch = t->cell_h;

    for (size_t r = 0; r < s->nline; r++) {
        if (!s->lines[r]->dirty) continue;

        for (size_t c = 0; c < s->ncol; c++) {
            TMTCHAR *ch_ptr = &s->lines[r]->chars[c];
            const TMTATTRS *a = &ch_ptr->a;

            /* Background */
            RGB bg = a->reverse ? color_fg(a->fg, a) : color_bg(a->bg);
            SDL_SetRenderDrawColor(renderer, bg.r, bg.g, bg.b, 255);
            SDL_Rect cell_rect = {
                (int)(ox + c * cw),
                (int)(oy + r * ch),
                cw, ch
            };
            SDL_RenderFillRect(renderer, &cell_rect);

            /* Foreground character */
            wchar_t wch = ch_ptr->c;
            if (wch == L'\0' || wch == L' ') continue;

            char utf8[8] = {0};
            if (wch < 0x80) {
                utf8[0] = (char)wch;
            } else if (wch < 0x800) {
                utf8[0] = (char)(0xC0 | (wch >> 6));
                utf8[1] = (char)(0x80 | (wch & 0x3F));
            } else {
                utf8[0] = (char)(0xE0 | (wch >> 12));
                utf8[1] = (char)(0x80 | ((wch >> 6) & 0x3F));
                utf8[2] = (char)(0x80 | (wch & 0x3F));
            }

            RGB fg = a->reverse ? color_bg(a->bg) : color_fg(a->fg, a);

            if (a->bold) TTF_SetFontStyle(t->font, TTF_STYLE_BOLD);
            if (a->underline) TTF_SetFontStyle(t->font,
                TTF_GetFontStyle(t->font) | TTF_STYLE_UNDERLINE);

            SDL_Color sdl_fg = {fg.r, fg.g, fg.b, 255};
            SDL_Surface *surf = TTF_RenderUTF8_Blended(t->font, utf8, sdl_fg);
            if (surf) {
                SDL_Texture *tex = SDL_CreateTextureFromSurface(renderer, surf);
                SDL_FreeSurface(surf);
                if (tex) {
                    int tw, th;
                    SDL_QueryTexture(tex, NULL, NULL, &tw, &th);
                    SDL_Rect dst = {(int)(ox + c * cw), (int)(oy + r * ch), tw, th};
                    SDL_RenderCopy(renderer, tex, NULL, &dst);
                    SDL_DestroyTexture(tex);
                }
            }

            TTF_SetFontStyle(t->font, TTF_STYLE_NORMAL);
        }
    }

    /* Draw cursor (simple block) */
    const TMTPOINT *cur = tmt_cursor(t->vt);
    SDL_SetRenderDrawColor(renderer, 220, 220, 220, 180);
    SDL_Rect cursor_rect = {
        (int)(ox + cur->c * cw),
        (int)(oy + cur->r * ch),
        cw, ch
    };
    SDL_RenderFillRect(renderer, &cursor_rect);
}

void nl_term_mark_clean(NLTerm *t) {
    if (t) {
        tmt_clean(t->vt);
        t->dirty = 0;
    }
}

void nl_term_resize(NLTerm *t, int64_t rows, int64_t cols) {
    if (t) {
        t->rows = (int)rows;
        t->cols = (int)cols;
        tmt_resize(t->vt, (size_t)rows, (size_t)cols);
        t->dirty = 1;
    }
}

int64_t nl_term_cursor_col(NLTerm *t) {
    return t ? (int64_t)tmt_cursor(t->vt)->c : 0;
}
int64_t nl_term_cursor_row(NLTerm *t) {
    return t ? (int64_t)tmt_cursor(t->vt)->r : 0;
}
int64_t nl_term_cell_w(NLTerm *t)   { return t ? (int64_t)t->cell_w : 0; }
int64_t nl_term_cell_h(NLTerm *t)   { return t ? (int64_t)t->cell_h : 0; }
int64_t nl_term_pixel_w(NLTerm *t)  { return t ? (int64_t)(t->cell_w * t->cols) : 0; }
int64_t nl_term_pixel_h(NLTerm *t)  { return t ? (int64_t)(t->cell_h * t->rows) : 0; }

void nl_term_destroy(NLTerm *t) {
    if (!t) return;
    if (t->vt)   tmt_close(t->vt);
    if (t->font) TTF_CloseFont(t->font);
    free(t);
}

/* ── Key mapping ─────────────────────────────────────────────────────── */

int64_t nl_term_key_to_seq(int64_t sdl_keycode, int64_t sdl_mod,
                             char *buf, int64_t buf_len) {
    (void)sdl_mod;
    if (buf_len < 8) return 0;

    /* Arrow keys / special */
    switch ((SDL_Keycode)sdl_keycode) {
    case SDLK_RETURN:  case SDLK_KP_ENTER:
        buf[0] = '\r'; buf[1] = 0; return 1;
    case SDLK_BACKSPACE:
        buf[0] = 127; buf[1] = 0; return 1;
    case SDLK_TAB:
        buf[0] = '\t'; buf[1] = 0; return 1;
    case SDLK_ESCAPE:
        buf[0] = 27; buf[1] = 0; return 1;
    case SDLK_UP:
        buf[0]=27; buf[1]='['; buf[2]='A'; buf[3]=0; return 3;
    case SDLK_DOWN:
        buf[0]=27; buf[1]='['; buf[2]='B'; buf[3]=0; return 3;
    case SDLK_RIGHT:
        buf[0]=27; buf[1]='['; buf[2]='C'; buf[3]=0; return 3;
    case SDLK_LEFT:
        buf[0]=27; buf[1]='['; buf[2]='D'; buf[3]=0; return 3;
    case SDLK_HOME:
        buf[0]=27; buf[1]='['; buf[2]='H'; buf[3]=0; return 3;
    case SDLK_END:
        buf[0]=27; buf[1]='['; buf[2]='F'; buf[3]=0; return 3;
    case SDLK_DELETE:
        buf[0]=27; buf[1]='['; buf[2]='3'; buf[3]='~'; buf[4]=0; return 4;
    default: break;
    }

    /* Ctrl + letter */
    if ((sdl_mod & KMOD_CTRL) && sdl_keycode >= SDLK_a && sdl_keycode <= SDLK_z) {
        buf[0] = (char)(sdl_keycode - SDLK_a + 1);
        buf[1] = 0;
        return 1;
    }
    return 0;
}

/* ── NanoLang int64 wrappers ─────────────────────────────────────────── */

int64_t nl_sdl_term_create(const char *fp, int64_t pt, int64_t rows, int64_t cols) {
    return (int64_t)(uintptr_t)nl_term_create(fp, pt, rows, cols);
}
void nl_sdl_term_feed(int64_t h, const char *data) {
    nl_term_feed((NLTerm *)(uintptr_t)h, data);
}
void nl_sdl_term_render(int64_t h, SDL_Renderer *renderer, int64_t x, int64_t y) {
    nl_term_render((NLTerm *)(uintptr_t)h, renderer, x, y);
}
void nl_sdl_term_mark_clean(int64_t h) {
    nl_term_mark_clean((NLTerm *)(uintptr_t)h);
}
void nl_sdl_term_resize(int64_t h, int64_t rows, int64_t cols) {
    nl_term_resize((NLTerm *)(uintptr_t)h, rows, cols);
}
int64_t nl_sdl_term_cursor_col(int64_t h) {
    return nl_term_cursor_col((NLTerm *)(uintptr_t)h);
}
int64_t nl_sdl_term_cursor_row(int64_t h) {
    return nl_term_cursor_row((NLTerm *)(uintptr_t)h);
}
int64_t nl_sdl_term_cell_w(int64_t h) {
    return nl_term_cell_w((NLTerm *)(uintptr_t)h);
}
int64_t nl_sdl_term_cell_h(int64_t h) {
    return nl_term_cell_h((NLTerm *)(uintptr_t)h);
}
int64_t nl_sdl_term_pixel_w(int64_t h) {
    return nl_term_pixel_w((NLTerm *)(uintptr_t)h);
}
int64_t nl_sdl_term_pixel_h(int64_t h) {
    return nl_term_pixel_h((NLTerm *)(uintptr_t)h);
}
void nl_sdl_term_destroy(int64_t h) {
    nl_term_destroy((NLTerm *)(uintptr_t)h);
}

static char nl_key_seq_buf[16];
int64_t nl_sdl_term_key_to_seq(int64_t keycode, int64_t mod,
                                 char *buf, int64_t buf_len) {
    (void)buf; (void)buf_len;
    return nl_term_key_to_seq(keycode, mod, nl_key_seq_buf, sizeof(nl_key_seq_buf));
}

/* Expose the static buffer so NanoLang can read the sequence string */
const char *nl_sdl_term_key_seq_str(void) {
    return nl_key_seq_buf;
}
