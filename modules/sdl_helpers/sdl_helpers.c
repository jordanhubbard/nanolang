/*
 * SDL Helper Functions for nanolang
 * These functions wrap SDL struct operations that nanolang can't do directly
 */

#include <SDL.h>
#ifdef HAVE_SDL_TTF
#include <SDL_ttf.h>
#endif
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#define NL_SDL_EVENT_BUF_CAP 256

static SDL_Event nl_sdl_event_buf[NL_SDL_EVENT_BUF_CAP];
static int nl_sdl_event_buf_len = 0;

static void nl__sdl_drain_events(void) {
    SDL_Event event;
    SDL_PumpEvents();
    while (SDL_PollEvent(&event)) {
        if (nl_sdl_event_buf_len < NL_SDL_EVENT_BUF_CAP) {
            nl_sdl_event_buf[nl_sdl_event_buf_len++] = event;
        }
    }
}

static int nl__sdl_take_first_event(uint32_t type, SDL_Event *out) {
    for (int i = 0; i < nl_sdl_event_buf_len; i++) {
        if (nl_sdl_event_buf[i].type == type) {
            if (out) *out = nl_sdl_event_buf[i];
            memmove(&nl_sdl_event_buf[i],
                    &nl_sdl_event_buf[i + 1],
                    (size_t)(nl_sdl_event_buf_len - i - 1) * sizeof(SDL_Event));
            nl_sdl_event_buf_len--;
            return 1;
        }
    }
    return 0;
}

static void nl__write_u32_be(FILE *f, uint32_t v) {
    unsigned char b[4];
    b[0] = (unsigned char)((v >> 24) & 0xff);
    b[1] = (unsigned char)((v >> 16) & 0xff);
    b[2] = (unsigned char)((v >> 8) & 0xff);
    b[3] = (unsigned char)(v & 0xff);
    fwrite(b, 1, 4, f);
}

static uint32_t nl__crc32(uint32_t crc, const unsigned char *buf, size_t len) {
    static uint32_t table[256];
    static int init = 0;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j) {
                c = (c & 1) ? (0xEDB88320U ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        init = 1;
    }
    crc = ~crc;
    for (size_t i = 0; i < len; ++i) {
        crc = table[(crc ^ buf[i]) & 0xff] ^ (crc >> 8);
    }
    return ~crc;
}

static uint32_t nl__adler32(const unsigned char *data, size_t len) {
    uint32_t s1 = 1;
    uint32_t s2 = 0;
    for (size_t i = 0; i < len; ++i) {
        s1 = (s1 + data[i]) % 65521U;
        s2 = (s2 + s1) % 65521U;
    }
    return (s2 << 16) | s1;
}

static int nl__write_chunk(FILE *f, const char type[4], const unsigned char *data, uint32_t len) {
    nl__write_u32_be(f, len);
    fwrite(type, 1, 4, f);
    if (len) {
        fwrite(data, 1, len, f);
    }
    uint32_t crc = nl__crc32(0, (const unsigned char*)type, 4);
    if (len) {
        crc = nl__crc32(crc, data, (size_t)len);
    }
    nl__write_u32_be(f, crc);
    return ferror(f) ? 0 : 1;
}

static int nl__write_png_rgba(const char *path, int w, int h, const unsigned char *pixels, int stride_bytes) {
    if (!path || w <= 0 || h <= 0 || !pixels || stride_bytes <= 0) return 0;

    const int row_bytes = w * 4;
    const size_t raw_len = (size_t)(row_bytes + 1) * (size_t)h;

    unsigned char *raw = (unsigned char*)malloc(raw_len);
    if (!raw) return 0;

    for (int y = 0; y < h; ++y) {
        unsigned char *dst = raw + (size_t)y * (size_t)(row_bytes + 1);
        const unsigned char *src = pixels + (size_t)y * (size_t)stride_bytes;
        dst[0] = 0; /* filter type 0 */
        memcpy(dst + 1, src, (size_t)row_bytes);
    }

    /* Build zlib stream using uncompressed DEFLATE blocks */
    const size_t max_z = raw_len + 2 + 4 + ((raw_len / 65535) + 1) * 5;
    unsigned char *z = (unsigned char*)malloc(max_z);
    if (!z) {
        free(raw);
        return 0;
    }

    size_t p = 0;
    z[p++] = 0x78; /* CMF */
    z[p++] = 0x01; /* FLG */

    size_t i = 0;
    while (i < raw_len) {
        size_t block_len = raw_len - i;
        if (block_len > 65535) block_len = 65535;
        int bfinal = (i + block_len >= raw_len) ? 1 : 0;
        z[p++] = (unsigned char)bfinal; /* BFINAL + BTYPE=00 */
        z[p++] = (unsigned char)(block_len & 0xff);
        z[p++] = (unsigned char)((block_len >> 8) & 0xff);
        uint16_t nlen = (uint16_t)(~(uint16_t)block_len);
        z[p++] = (unsigned char)(nlen & 0xff);
        z[p++] = (unsigned char)((nlen >> 8) & 0xff);
        memcpy(z + p, raw + i, block_len);
        p += block_len;
        i += block_len;
    }

    uint32_t ad = nl__adler32(raw, raw_len);
    z[p++] = (unsigned char)((ad >> 24) & 0xff);
    z[p++] = (unsigned char)((ad >> 16) & 0xff);
    z[p++] = (unsigned char)((ad >> 8) & 0xff);
    z[p++] = (unsigned char)(ad & 0xff);

    free(raw);

    FILE *f = fopen(path, "wb");
    if (!f) {
        free(z);
        return 0;
    }

    static const unsigned char sig[8] = {137,80,78,71,13,10,26,10};
    fwrite(sig, 1, 8, f);

    unsigned char ihdr[13];
    ihdr[0] = (unsigned char)((w >> 24) & 0xff);
    ihdr[1] = (unsigned char)((w >> 16) & 0xff);
    ihdr[2] = (unsigned char)((w >> 8) & 0xff);
    ihdr[3] = (unsigned char)(w & 0xff);
    ihdr[4] = (unsigned char)((h >> 24) & 0xff);
    ihdr[5] = (unsigned char)((h >> 16) & 0xff);
    ihdr[6] = (unsigned char)((h >> 8) & 0xff);
    ihdr[7] = (unsigned char)(h & 0xff);
    ihdr[8] = 8;  /* bit depth */
    ihdr[9] = 6;  /* color type: RGBA */
    ihdr[10] = 0; /* compression */
    ihdr[11] = 0; /* filter */
    ihdr[12] = 0; /* interlace */

    int ok = 1;
    ok = ok && nl__write_chunk(f, "IHDR", ihdr, 13);
    ok = ok && nl__write_chunk(f, "IDAT", z, (uint32_t)p);
    ok = ok && nl__write_chunk(f, "IEND", NULL, 0);

    free(z);
    fclose(f);
    return ok;
}

/* Wrapper around libc system(3) using the nanolang int64_t ABI. */
int64_t nl_system(const char* cmd) {
    return (int64_t)system(cmd);
}

/* Helper to create SDL_Rect and call SDL_RenderFillRect */
/* Note: renderer is passed as int64_t (pointer value) and cast back to SDL_Renderer* */
int64_t nl_sdl_render_fill_rect(int64_t renderer_ptr, int64_t x, int64_t y, int64_t w, int64_t h) {
    SDL_Renderer *renderer = (SDL_Renderer*)renderer_ptr;
    SDL_Rect rect = {(int)x, (int)y, (int)w, (int)h};
    return SDL_RenderFillRect(renderer, &rect);
}

/* Helper to poll SDL events and return 1 if quit, 0 otherwise */
int64_t nl_sdl_poll_event_quit(void) {
    nl__sdl_drain_events();
    return nl__sdl_take_first_event(SDL_QUIT, NULL) ? 1 : 0;
}

/* Helper to poll for mouse button down events
 * Returns encoded position: row * 1000 + col * 100 + button, or -1 if no click
 * For checkers: we divide x,y by SQUARE_SIZE to get col,row
 * This version returns x * 10000 + y if left button clicked, -1 otherwise
 */
int64_t nl_sdl_poll_mouse_click(void) {
    nl__sdl_drain_events();
    for (int i = 0; i < nl_sdl_event_buf_len; i++) {
        SDL_Event event = nl_sdl_event_buf[i];
        if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT) {
            memmove(&nl_sdl_event_buf[i],
                    &nl_sdl_event_buf[i + 1],
                    (size_t)(nl_sdl_event_buf_len - i - 1) * sizeof(SDL_Event));
            nl_sdl_event_buf_len--;
            return (int64_t)event.button.x * 10000 + (int64_t)event.button.y;
        }
    }
    return -1;
}

/* Helper to poll for mouse state (continuous holding)
 * Returns x * 10000 + y if left button is held, -1 otherwise
 */
int64_t nl_sdl_poll_mouse_state(void) {
    int x, y;
    Uint32 state = SDL_GetMouseState(&x, &y);
    if (state & SDL_BUTTON_LMASK) {
        return (int64_t)x * 10000 + (int64_t)y;
    }
    return -1;
}

/* Helper to poll for mouse button up
 * Returns x * 10000 + y if left button was released, -1 otherwise
 */
int64_t nl_sdl_poll_mouse_up(void) {
    nl__sdl_drain_events();
    for (int i = 0; i < nl_sdl_event_buf_len; i++) {
        SDL_Event event = nl_sdl_event_buf[i];
        if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_LEFT) {
            memmove(&nl_sdl_event_buf[i],
                    &nl_sdl_event_buf[i + 1],
                    (size_t)(nl_sdl_event_buf_len - i - 1) * sizeof(SDL_Event));
            nl_sdl_event_buf_len--;
            return (int64_t)event.button.x * 10000 + (int64_t)event.button.y;
        }
    }
    return -1;
}

/* Helper to poll for mouse motion
 * Returns x * 10000 + y if mouse moved, -1 otherwise
 */
int64_t nl_sdl_poll_mouse_motion(void) {
    SDL_Event event;
    nl__sdl_drain_events();
    if (nl__sdl_take_first_event(SDL_MOUSEMOTION, &event)) {
        return (int64_t)event.motion.x * 10000 + (int64_t)event.motion.y;
    }
    return -1;
}

/* Helper to poll for keyboard events
 * Returns SDL scancode if key pressed, -1 otherwise
 * Common scancodes:
 *   SPACE = 44, ESC = 41, C = 6
 *   0-9 = 30-39, 1 = 30, 2 = 31, etc.
 */
int64_t nl_sdl_poll_keypress(void) {
    SDL_Event event;
    nl__sdl_drain_events();
    if (nl__sdl_take_first_event(SDL_KEYDOWN, &event)) {
        return event.key.keysym.scancode;
    }
    return -1;
}

/* Helper to poll for mouse wheel events
 * Returns wheel.y value: positive = scroll up, negative = scroll down, 0 = no scroll
 * Typical values are -1 or +1
 */
int64_t nl_sdl_poll_mouse_wheel(void) {
    SDL_Event event;
    nl__sdl_drain_events();
    if (nl__sdl_take_first_event(SDL_MOUSEWHEEL, &event)) {
        return (int64_t)event.wheel.y;
    }
    return 0;
}

/* Helper to check if a key is currently held down (keyboard state)
 * Returns 1 if the key with given scancode is held, 0 otherwise
 * Use this for continuous input (thrust, rotation) instead of nl_sdl_poll_keypress
 * Common scancodes:
 *   UP = 82, DOWN = 81, LEFT = 80, RIGHT = 79
 *   SPACE = 44, ESC = 41
 */
int64_t nl_sdl_key_state(int64_t scancode) {
    SDL_PumpEvents();
    const Uint8 *state = SDL_GetKeyboardState(NULL);
    return state[scancode] ? 1 : 0;
}

#ifdef HAVE_SDL_TTF
/* Helper to render text using SDL_ttf
 * Creates SDL_Color struct, renders text, creates texture, and renders to screen
 * Returns 0 on success, -1 on failure
 */
int64_t nl_sdl_render_text_solid(int64_t renderer_ptr, int64_t font_ptr, 
                                  const char* text, int64_t x, int64_t y,
                                  int64_t r, int64_t g, int64_t b, int64_t a) {
    SDL_Renderer *renderer = (SDL_Renderer*)renderer_ptr;
    TTF_Font *font = (TTF_Font*)font_ptr;
    
    if (!font) return -1;
    
    SDL_Color color = {(Uint8)r, (Uint8)g, (Uint8)b, (Uint8)a};
    SDL_Surface *surface = TTF_RenderText_Solid(font, text, color);
    if (!surface) return -1;
    
    SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
    if (!texture) {
        SDL_FreeSurface(surface);
        return -1;
    }
    
    SDL_Rect dest = {(int)x, (int)y, surface->w, surface->h};
    SDL_RenderCopy(renderer, texture, NULL, &dest);
    
    SDL_DestroyTexture(texture);
    SDL_FreeSurface(surface);
    return 0;
}

/* Helper to render text using SDL_ttf with blended mode (anti-aliased)
 * Creates SDL_Color struct, renders text, creates texture, and renders to screen
 * Returns 0 on success, -1 on failure
 */
int64_t nl_sdl_render_text_blended(int64_t renderer_ptr, int64_t font_ptr, 
                                    const char* text, int64_t x, int64_t y,
                                    int64_t r, int64_t g, int64_t b, int64_t a) {
    SDL_Renderer *renderer = (SDL_Renderer*)renderer_ptr;
    TTF_Font *font = (TTF_Font*)font_ptr;
    
    if (!font) return -1;
    
    SDL_Color color = {(Uint8)r, (Uint8)g, (Uint8)b, (Uint8)a};
    SDL_Surface *surface = TTF_RenderText_Blended(font, text, color);
    if (!surface) return -1;
    
    SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
    if (!texture) {
        SDL_FreeSurface(surface);
        return -1;
    }
    
    SDL_Rect dest = {(int)x, (int)y, surface->w, surface->h};
    SDL_RenderCopy(renderer, texture, NULL, &dest);
    
    SDL_DestroyTexture(texture);
    SDL_FreeSurface(surface);
    return 0;
}
#endif /* HAVE_SDL_TTF */

/* Save current renderer output to a BMP file. */
int64_t nl_sdl_save_bmp(int64_t renderer_ptr, int64_t w, int64_t h, const char* path) {
    SDL_Renderer *renderer = (SDL_Renderer*)renderer_ptr;
    if (!renderer || !path || w <= 0 || h <= 0) return -1;

    SDL_Surface *surface = SDL_CreateRGBSurfaceWithFormat(0, (int)w, (int)h, 32, SDL_PIXELFORMAT_ARGB8888);
    if (!surface) return -2;

    if (SDL_RenderReadPixels(renderer, NULL, SDL_PIXELFORMAT_ARGB8888, surface->pixels, surface->pitch) != 0) {
        SDL_FreeSurface(surface);
        return -3;
    }

    int rc = SDL_SaveBMP(surface, path);
    SDL_FreeSurface(surface);
    return (rc == 0) ? 0 : -4;
}

/* Save current renderer output to a PNG file. */
int64_t nl_sdl_save_png(int64_t renderer_ptr, int64_t w, int64_t h, const char* path) {
    SDL_Renderer *renderer = (SDL_Renderer*)renderer_ptr;
    if (!renderer || !path || w <= 0 || h <= 0) return -1;

    SDL_Surface *surface = SDL_CreateRGBSurfaceWithFormat(0, (int)w, (int)h, 32, SDL_PIXELFORMAT_RGBA32);
    if (!surface) return -2;

    if (SDL_RenderReadPixels(renderer, NULL, SDL_PIXELFORMAT_RGBA32, surface->pixels, surface->pitch) != 0) {
        SDL_FreeSurface(surface);
        return -3;
    }

    int ok = nl__write_png_rgba(path, (int)w, (int)h, (const unsigned char*)surface->pixels, surface->pitch);
    SDL_FreeSurface(surface);
    return ok ? 0 : -4;
}

