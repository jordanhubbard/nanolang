#include <SDL.h>

#include <assert.h>
#include <string.h>

static SDL_Event host_events[16];
static int host_event_count, host_event_index;
static Uint32 host_ticks = 100;
static int host_started, host_stopped;

static void fake_pump_events(void) {}
static Uint32 fake_ticks(void) { return host_ticks; }
static int fake_poll_event(SDL_Event *out) {
    if (host_event_index >= host_event_count) return 0;
    *out = host_events[host_event_index++];
    return 1;
}
static void fake_start_text(void) { host_started++; }
static void fake_stop_text(void) { host_stopped++; }

#define SDL_PumpEvents fake_pump_events
#define SDL_GetTicks fake_ticks
#define SDL_PollEvent fake_poll_event
#define SDL_StartTextInput fake_start_text
#define SDL_StopTextInput fake_stop_text
#include "../modules/sdl_helpers/sdl_helpers.c"
#undef SDL_StopTextInput
#undef SDL_StartTextInput
#undef SDL_PollEvent
#undef SDL_GetTicks
#undef SDL_PumpEvents

static void push_text(const char *text) {
    SDL_Event *event = &host_events[host_event_count++];
    memset(event, 0, sizeof(*event));
    event->type = SDL_TEXTINPUT;
    strncpy(event->text.text, text, sizeof(event->text.text) - 1);
}

static void push_key(SDL_Keycode key, SDL_Scancode scancode) {
    SDL_Event *event = &host_events[host_event_count++];
    memset(event, 0, sizeof(*event));
    event->type = SDL_KEYDOWN;
    event->key.keysym.sym = key;
    event->key.keysym.scancode = scancode;
}

int main(void) {
    SDL_Event event;
    nl_sdl_start_text_input();
    assert(host_started == 1);

    push_text("\xc3\xa9");
    push_key(SDLK_BACKSPACE, SDL_SCANCODE_BACKSPACE);
    push_text("!");
    push_key(SDLK_RETURN, SDL_SCANCODE_RETURN);

    /* Generic consumers may run first; my mirrored editing stream remains. */
    assert(nl_sdl_poll_keypress() == SDL_SCANCODE_BACKSPACE);
    assert(strcmp(nl_sdl_poll_text_input(), "\xc3\xa9") == 0);

    assert(nl_sdl_take_text_input_event(&event));
    assert(event.type == SDL_TEXTINPUT && !strcmp(event.text.text, "\xc3\xa9"));
    assert(nl_sdl_take_text_input_event(&event));
    assert(event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_BACKSPACE);
    assert(nl_sdl_take_text_input_event(&event));
    assert(event.type == SDL_TEXTINPUT && !strcmp(event.text.text, "!"));
    assert(nl_sdl_take_text_input_event(&event));
    assert(event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_RETURN);
    assert(!nl_sdl_take_text_input_event(&event));

    host_ticks += 17;
    push_key(SDLK_KP_ENTER, SDL_SCANCODE_KP_ENTER);
    assert(nl_sdl_poll_keypress() == SDL_SCANCODE_KP_ENTER);
    nl_sdl_stop_text_input();
    assert(host_stopped == 1);
    assert(!nl_sdl_take_text_input_event(&event));
    return 0;
}
