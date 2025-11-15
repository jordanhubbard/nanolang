/*
 * Checkers Game - C/SDL2 Implementation
 * A simple checkers game that runs as an X11 client
 */

#include <SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define BOARD_SIZE 8
#define SQUARE_SIZE 80
#define WINDOW_WIDTH (BOARD_SIZE * SQUARE_SIZE)
#define WINDOW_HEIGHT (BOARD_SIZE * SQUARE_SIZE + 60)  // Extra space for status

// Piece types
typedef enum {
    EMPTY = 0,
    RED_PIECE = 1,
    RED_KING = 2,
    BLACK_PIECE = 3,
    BLACK_KING = 4
} PieceType;

// Game state
typedef enum {
    PLAYER_TURN,
    AI_TURN,
    GAME_OVER
} GameState;

typedef struct {
    PieceType board[BOARD_SIZE][BOARD_SIZE];
    GameState state;
    int selected_row;
    int selected_col;
    bool has_selection;
    int player_pieces;
    int ai_pieces;
} CheckersGame;

// Initialize the board with starting positions
void init_board(CheckersGame *game) {
    // Clear board
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            game->board[i][j] = EMPTY;
        }
    }
    
    // Place red pieces (top 3 rows)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if ((i + j) % 2 == 1) {
                game->board[i][j] = RED_PIECE;
            }
        }
    }
    
    // Place black pieces (bottom 3 rows)
    for (int i = 5; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if ((i + j) % 2 == 1) {
                game->board[i][j] = BLACK_PIECE;
            }
        }
    }
    
    game->state = PLAYER_TURN;
    game->has_selection = false;
    game->selected_row = -1;
    game->selected_col = -1;
    game->player_pieces = 12;
    game->ai_pieces = 12;
}

// Check if coordinates are valid
bool is_valid_pos(int row, int col) {
    return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

// Check if a square is dark (playable)
bool is_dark_square(int row, int col) {
    return (row + col) % 2 == 1;
}

// Check if a piece belongs to player
bool is_player_piece(PieceType piece) {
    return piece == RED_PIECE || piece == RED_KING;
}

// Check if a piece belongs to AI
bool is_ai_piece(PieceType piece) {
    return piece == BLACK_PIECE || piece == BLACK_KING;
}

// Check if a piece is a king
bool is_king(PieceType piece) {
    return piece == RED_KING || piece == BLACK_KING;
}

// Check if a move is valid (non-jump)
bool is_valid_move(CheckersGame *game, int from_row, int from_col, int to_row, int to_col) {
    if (!is_valid_pos(from_row, from_col) || !is_valid_pos(to_row, to_col)) {
        return false;
    }
    
    if (!is_dark_square(to_row, to_col)) {
        return false;
    }
    
    if (game->board[to_row][to_col] != EMPTY) {
        return false;
    }
    
    PieceType piece = game->board[from_row][from_col];
    
    if (piece == EMPTY) {
        return false;
    }
    
    int row_diff = to_row - from_row;
    int col_diff = abs(to_col - from_col);
    
    if (col_diff != 1) {
        return false;
    }
    
    if (is_player_piece(piece) && game->state == PLAYER_TURN) {
        if (is_king(piece)) {
            return abs(row_diff) == 1;
        } else {
            return row_diff == 1;  // Red moves down
        }
    } else if (is_ai_piece(piece) && game->state == AI_TURN) {
        if (is_king(piece)) {
            return abs(row_diff) == 1;
        } else {
            return row_diff == -1;  // Black moves up
        }
    }
    
    return false;
}

// Check if a jump is valid
bool is_valid_jump(CheckersGame *game, int from_row, int from_col, int to_row, int to_col) {
    if (!is_valid_pos(from_row, from_col) || !is_valid_pos(to_row, to_col)) {
        return false;
    }
    
    if (!is_dark_square(to_row, to_col)) {
        return false;
    }
    
    if (game->board[to_row][to_col] != EMPTY) {
        return false;
    }
    
    PieceType piece = game->board[from_row][from_col];
    
    if (piece == EMPTY) {
        return false;
    }
    
    int row_diff = to_row - from_row;
    int col_diff = to_col - from_col;
    
    if (abs(row_diff) != 2 || abs(col_diff) != 2) {
        return false;
    }
    
    // Check middle square has opponent piece
    int mid_row = from_row + row_diff / 2;
    int mid_col = from_col + col_diff / 2;
    
    if (!is_valid_pos(mid_row, mid_col)) {
        return false;
    }
    
    PieceType mid_piece = game->board[mid_row][mid_col];
    
    if (is_player_piece(piece) && game->state == PLAYER_TURN) {
        if (!is_ai_piece(mid_piece)) {
            return false;
        }
        if (is_king(piece)) {
            return true;
        } else {
            return row_diff == 2;  // Red jumps down
        }
    } else if (is_ai_piece(piece) && game->state == AI_TURN) {
        if (!is_player_piece(mid_piece)) {
            return false;
        }
        if (is_king(piece)) {
            return true;
        } else {
            return row_diff == -2;  // Black jumps up
        }
    }
    
    return false;
}

// Check if there are any jumps available for a piece
bool has_jump(CheckersGame *game, int row, int col) {
    PieceType piece = game->board[row][col];
    if (piece == EMPTY) {
        return false;
    }
    
    int directions[4][2] = {{-2, -2}, {-2, 2}, {2, -2}, {2, 2}};
    int king_directions[4][2] = {{-2, -2}, {-2, 2}, {2, -2}, {2, 2}};
    
    int (*dirs)[2] = is_king(piece) ? king_directions : directions;
    int dir_count = 4;
    
    if (!is_king(piece)) {
        if (is_player_piece(piece)) {
            dirs = (int[][2]){{2, -2}, {2, 2}};
            dir_count = 2;
        } else {
            dirs = (int[][2]){{-2, -2}, {-2, 2}};
            dir_count = 2;
        }
    }
    
    for (int i = 0; i < dir_count; i++) {
        int to_row = row + dirs[i][0];
        int to_col = col + dirs[i][1];
        if (is_valid_jump(game, row, col, to_row, to_col)) {
            return true;
        }
    }
    
    return false;
}

// Check if there are any jumps available for current player
bool has_any_jump(CheckersGame *game) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            PieceType piece = game->board[i][j];
            if ((game->state == PLAYER_TURN && is_player_piece(piece)) ||
                (game->state == AI_TURN && is_ai_piece(piece))) {
                if (has_jump(game, i, j)) {
                    return true;
                }
            }
        }
    }
    return false;
}

// Make a move
bool make_move(CheckersGame *game, int from_row, int from_col, int to_row, int to_col) {
    bool is_jump = abs(to_row - from_row) == 2;
    
    if (is_jump) {
        if (!is_valid_jump(game, from_row, from_col, to_row, to_col)) {
            return false;
        }
        
        // Remove jumped piece
        int mid_row = from_row + (to_row - from_row) / 2;
        int mid_col = from_col + (to_col - from_col) / 2;
        
        if (is_ai_piece(game->board[mid_row][mid_col])) {
            game->ai_pieces--;
        } else {
            game->player_pieces--;
        }
        
        game->board[mid_row][mid_col] = EMPTY;
    } else {
        if (!is_valid_move(game, from_row, from_col, to_row, to_col)) {
            return false;
        }
        
        // Check if jumps are mandatory
        if (has_any_jump(game)) {
            return false;
        }
    }
    
    // Move piece
    PieceType piece = game->board[from_row][from_col];
    game->board[from_row][from_col] = EMPTY;
    game->board[to_row][to_col] = piece;
    
    // Promote to king
    if (piece == RED_PIECE && to_row == BOARD_SIZE - 1) {
        game->board[to_row][to_col] = RED_KING;
    } else if (piece == BLACK_PIECE && to_row == 0) {
        game->board[to_row][to_col] = BLACK_KING;
    }
    
    // Check for game over
    if (game->player_pieces == 0 || game->ai_pieces == 0) {
        game->state = GAME_OVER;
        return true;
    }
    
    // Check for additional jumps
    if (is_jump && has_jump(game, to_row, to_col)) {
        // Player can continue jumping
        return true;
    }
    
    // Switch turns
    game->state = (game->state == PLAYER_TURN) ? AI_TURN : PLAYER_TURN;
    return true;
}

// Simple AI: find first valid move
void ai_move(CheckersGame *game) {
    // First, try to find a jump
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (is_ai_piece(game->board[i][j])) {
                int directions[4][2] = {{-2, -2}, {-2, 2}, {2, -2}, {2, 2}};
                int dir_count = is_king(game->board[i][j]) ? 4 : 2;
                
                if (!is_king(game->board[i][j])) {
                    directions[0][0] = -2; directions[0][1] = -2;
                    directions[1][0] = -2; directions[1][1] = 2;
                }
                
                for (int d = 0; d < dir_count; d++) {
                    int to_row = i + directions[d][0];
                    int to_col = j + directions[d][1];
                    if (is_valid_jump(game, i, j, to_row, to_col)) {
                        make_move(game, i, j, to_row, to_col);
                        return;
                    }
                }
            }
        }
    }
    
    // If no jump, find a regular move
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (is_ai_piece(game->board[i][j])) {
                int directions[4][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
                int dir_count = is_king(game->board[i][j]) ? 4 : 2;
                
                if (!is_king(game->board[i][j])) {
                    directions[0][0] = -1; directions[0][1] = -1;
                    directions[1][0] = -1; directions[1][1] = 1;
                }
                
                for (int d = 0; d < dir_count; d++) {
                    int to_row = i + directions[d][0];
                    int to_col = j + directions[d][1];
                    if (is_valid_move(game, i, j, to_row, to_col)) {
                        make_move(game, i, j, to_row, to_col);
                        return;
                    }
                }
            }
        }
    }
}

// Render the board
void render_board(SDL_Renderer *renderer, CheckersGame *game) {
    // Clear screen
    SDL_SetRenderDrawColor(renderer, 240, 240, 240, 255);
    SDL_RenderClear(renderer);
    
    // Draw board squares
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            SDL_Rect rect = {j * SQUARE_SIZE, i * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE};
            
            if (is_dark_square(i, j)) {
                SDL_SetRenderDrawColor(renderer, 139, 69, 19, 255);  // Brown
            } else {
                SDL_SetRenderDrawColor(renderer, 255, 248, 220, 255);  // Beige
            }
            
            SDL_RenderFillRect(renderer, &rect);
            
            // Highlight selected square
            if (game->has_selection && game->selected_row == i && game->selected_col == j) {
                SDL_SetRenderDrawColor(renderer, 255, 255, 0, 180);  // Yellow highlight
                SDL_RenderFillRect(renderer, &rect);
            }
            
            // Draw piece
            PieceType piece = game->board[i][j];
            if (piece != EMPTY) {
                int center_x = j * SQUARE_SIZE + SQUARE_SIZE / 2;
                int center_y = i * SQUARE_SIZE + SQUARE_SIZE / 2;
                int radius = SQUARE_SIZE / 3;
                
                // Piece color
                if (piece == RED_PIECE || piece == RED_KING) {
                    SDL_SetRenderDrawColor(renderer, 220, 20, 60, 255);  // Red
                } else {
                    SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);  // Black
                }
                
                // Draw circle (approximated with filled circle)
                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dx = -radius; dx <= radius; dx++) {
                        if (dx * dx + dy * dy <= radius * radius) {
                            SDL_RenderDrawPoint(renderer, center_x + dx, center_y + dy);
                        }
                    }
                }
                
                // Draw king indicator (smaller circle on top)
                if (is_king(piece)) {
                    SDL_SetRenderDrawColor(renderer, 255, 215, 0, 255);  // Gold
                    int king_radius = radius / 2;
                    for (int dy = -king_radius; dy <= king_radius; dy++) {
                        for (int dx = -king_radius; dx <= king_radius; dx++) {
                            if (dx * dx + dy * dy <= king_radius * king_radius) {
                                SDL_RenderDrawPoint(renderer, center_x + dx, center_y + dy - radius / 2);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Draw status bar
    SDL_Rect status_rect = {0, BOARD_SIZE * SQUARE_SIZE, WINDOW_WIDTH, 60};
    SDL_SetRenderDrawColor(renderer, 50, 50, 50, 255);
    SDL_RenderFillRect(renderer, &status_rect);
    
    SDL_RenderPresent(renderer);
}

// Handle mouse click
void handle_click(CheckersGame *game, int x, int y) {
    if (game->state != PLAYER_TURN) {
        return;
    }
    
    int col = x / SQUARE_SIZE;
    int row = y / SQUARE_SIZE;
    
    if (!is_valid_pos(row, col)) {
        return;
    }
    
    if (game->has_selection) {
        // Try to make move
        if (make_move(game, game->selected_row, game->selected_col, row, col)) {
            game->has_selection = false;
            game->selected_row = -1;
            game->selected_col = -1;
            
            // AI moves after player
            if (game->state == AI_TURN) {
                ai_move(game);
            }
        } else {
            // Invalid move, try selecting new piece
            if (is_player_piece(game->board[row][col])) {
                game->selected_row = row;
                game->selected_col = col;
            } else {
                game->has_selection = false;
                game->selected_row = -1;
                game->selected_col = -1;
            }
        }
    } else {
        // Select piece
        if (is_player_piece(game->board[row][col])) {
            game->has_selection = true;
            game->selected_row = row;
            game->selected_col = col;
        }
    }
}

int main(int argc, char *argv[]) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL initialization failed: %s\n", SDL_GetError());
        return 1;
    }
    
    // Create window
    SDL_Window *window = SDL_CreateWindow(
        "Checkers",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        SDL_WINDOW_SHOWN
    );
    
    if (!window) {
        fprintf(stderr, "Window creation failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }
    
    // Create renderer
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        fprintf(stderr, "Renderer creation failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    
    // Initialize game
    CheckersGame game;
    init_board(&game);
    
    // Main game loop
    bool running = true;
    SDL_Event event;
    
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            } else if (event.type == SDL_MOUSEBUTTONDOWN) {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    handle_click(&game, event.button.x, event.button.y);
                }
            }
        }
        
        render_board(renderer, &game);
        SDL_Delay(16);  // ~60 FPS
    }
    
    // Cleanup
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}

