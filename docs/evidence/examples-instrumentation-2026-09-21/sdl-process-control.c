#include <SDL.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
static int draw(void) {
    fprintf(stderr, "SDL_Init\n");
    if (SDL_Init(SDL_INIT_VIDEO) != 0) { fprintf(stderr, "%s\n", SDL_GetError()); return 1; }
    fprintf(stderr, "SDL_CreateWindow\n");
    SDL_Window *window = SDL_CreateWindow("SDL process control",100,100,800,600,SDL_WINDOW_SHOWN);
    if (!window) { fprintf(stderr, "%s\n", SDL_GetError()); SDL_Quit(); return 2; }
    fprintf(stderr, "SDL_CreateRenderer\n");
    SDL_Renderer *renderer = SDL_CreateRenderer(window,-1,SDL_RENDERER_ACCELERATED);
    if (!renderer) { fprintf(stderr, "%s\n", SDL_GetError()); SDL_DestroyWindow(window); SDL_Quit(); return 3; }
    SDL_DestroyRenderer(renderer); SDL_DestroyWindow(window); SDL_Quit();
    fprintf(stderr, "complete\n"); return 0;
}
int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1],"fork")==0) {
        pid_t child = fork();
        if (child < 0) return 4;
        if (child == 0) _exit(draw());
        int status; if (waitpid(child,&status,0)<0) return 5;
        if (WIFSIGNALED(status)) { fprintf(stderr,"child signal %d\n",WTERMSIG(status)); return 128+WTERMSIG(status); }
        return WIFEXITED(status) ? WEXITSTATUS(status) : 6;
    }
    return draw();
}
