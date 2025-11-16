/*
 * Checkers Game - C/SDL2 Implementation
 * A simple checkers game that runs as an X11 client
 */

#include <SDL.h>
#ifdef HAVE_SDL_TTF
#include <SDL_ttf.h>
#endif
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
    // Blinking state for AI moves
    int blink_row;
    int blink_col;
    int pending_from_row;
    int pending_from_col;
    int pending_to_row;
    int pending_to_col;
    Uint32 blink_start_time;
    bool is_blinking;
    bool blink_before_move;
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
    game->is_blinking = false;
    game->blink_row = -1;
    game->blink_col = -1;
    game->blink_start_time = 0;
    game->blink_before_move = false;
    game->pending_from_row = -1;
    game->pending_from_col = -1;
    game->pending_to_row = -1;
    game->pending_to_col = -1;
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

// Evaluate a move's quality (higher is better)
int evaluate_move(CheckersGame *game, int from_row, int from_col, int to_row, int to_col, bool is_jump) {
    int score = 0;
    
    // Jumps are always prioritized
    if (is_jump) {
        score += 1000;
        
        // Multiple jumps are even better
        if (has_jump(game, to_row, to_col)) {
            score += 500;
        }
    }
    
    // King promotion is very valuable
    if (game->board[from_row][from_col] == BLACK_PIECE && to_row == 0) {
        score += 800;
    }
    
    // Advance pieces toward promotion
    if (game->board[from_row][from_col] == BLACK_PIECE) {
        score += (BOARD_SIZE - to_row) * 10;  // Closer to top = better
    }
    
    // Control center of board
    int center_dist = abs(to_col - BOARD_SIZE / 2) + abs(to_row - BOARD_SIZE / 2);
    score += (BOARD_SIZE - center_dist) * 5;
    
    // Avoid edges (pieces on edges are vulnerable)
    if (to_col == 0 || to_col == BOARD_SIZE - 1) {
        score -= 20;
    }
    if (to_row == 0 || to_row == BOARD_SIZE - 1) {
        if (game->board[from_row][from_col] != BLACK_PIECE) {  // Not promoting
            score -= 20;
        }
    }
    
    // Prefer moving kings (they're more valuable)
    if (is_king(game->board[from_row][from_col])) {
        score += 50;
    }
    
    // Avoid positions where opponent can jump us
    // Check if this position is vulnerable
    int vulnerable_score = 0;
    int check_dirs[4][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    for (int d = 0; d < 4; d++) {
        int check_row = to_row + check_dirs[d][0];
        int check_col = to_col + check_dirs[d][1];
        if (is_valid_pos(check_row, check_col) && 
            is_player_piece(game->board[check_row][check_col])) {
            // Check if opponent can jump from here
            int jump_row = to_row - check_dirs[d][0] * 2;
            int jump_col = to_col - check_dirs[d][1] * 2;
            if (is_valid_pos(jump_row, jump_col) && 
                game->board[jump_row][jump_col] == EMPTY) {
                vulnerable_score -= 100;
            }
        }
    }
    score += vulnerable_score;
    
    return score;
}

// Find and return the best move for AI
bool find_best_move(CheckersGame *game, int *best_from_row, int *best_from_col, 
                    int *best_to_row, int *best_to_col) {
    int best_score = -999999;
    bool found_move = false;
    
    // First, collect all possible jumps (mandatory)
    bool has_jumps = false;
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (is_ai_piece(game->board[i][j]) && has_jump(game, i, j)) {
                has_jumps = true;
                break;
            }
        }
        if (has_jumps) break;
    }
    
    // Evaluate all possible moves
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (!is_ai_piece(game->board[i][j])) {
                continue;
            }
            
            PieceType piece = game->board[i][j];
            int directions[4][2] = {{-2, -2}, {-2, 2}, {2, -2}, {2, 2}};
            int move_dirs[4][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
            int dir_count = is_king(piece) ? 4 : 2;
            
            if (!is_king(piece)) {
                directions[0][0] = -2; directions[0][1] = -2;
                directions[1][0] = -2; directions[1][1] = 2;
                move_dirs[0][0] = -1; move_dirs[0][1] = -1;
                move_dirs[1][0] = -1; move_dirs[1][1] = 1;
            }
            
            // Check jumps first if available
            if (!has_jumps || has_jump(game, i, j)) {
                for (int d = 0; d < dir_count; d++) {
                    int to_row = i + directions[d][0];
                    int to_col = j + directions[d][1];
                    if (is_valid_jump(game, i, j, to_row, to_col)) {
                        int score = evaluate_move(game, i, j, to_row, to_col, true);
                        if (score > best_score) {
                            best_score = score;
                            *best_from_row = i;
                            *best_from_col = j;
                            *best_to_row = to_row;
                            *best_to_col = to_col;
                            found_move = true;
                        }
                    }
                }
            }
            
            // Check regular moves if no jumps required
            if (!has_jumps) {
                for (int d = 0; d < dir_count; d++) {
                    int to_row = i + move_dirs[d][0];
                    int to_col = j + move_dirs[d][1];
                    if (is_valid_move(game, i, j, to_row, to_col)) {
                        int score = evaluate_move(game, i, j, to_row, to_col, false);
                        if (score > best_score) {
                            best_score = score;
                            *best_from_row = i;
                            *best_from_col = j;
                            *best_to_row = to_row;
                            *best_to_col = to_col;
                            found_move = true;
                        }
                    }
                }
            }
        }
    }
    
    return found_move;
}

// Improved AI: evaluate moves and choose best one
void ai_move(CheckersGame *game) {
    int from_row, from_col, to_row, to_col;
    
    if (find_best_move(game, &from_row, &from_col, &to_row, &to_col)) {
        // Store move coordinates
        game->pending_from_row = from_row;
        game->pending_from_col = from_col;
        game->pending_to_row = to_row;
        game->pending_to_col = to_col;
        
        // Start blinking before move
        game->is_blinking = true;
        game->blink_before_move = true;
        game->blink_row = from_row;
        game->blink_col = from_col;
        game->blink_start_time = SDL_GetTicks();
    }
}

// Determine the winner
const char* get_winner(CheckersGame *game) {
    if (game->player_pieces == 0) {
        return "Black wins!";
    } else if (game->ai_pieces == 0) {
        return "Red wins!";
    }
    return NULL;
}

// Simple bitmap font rendering (fallback when TTF not available)
void render_text_simple(SDL_Renderer *renderer, const char *text, int x, int y, int scale) {
    // Simple 8x8 bitmap font for basic characters
    // This is a very basic implementation - just enough for "Red wins!" and "Black wins!"
    int char_width = 8 * scale;
    int char_height = 12 * scale;
    int spacing = 2 * scale;
    
    int start_x = x - (strlen(text) * (char_width + spacing)) / 2;
    int current_x = start_x;
    
    for (const char *c = text; *c; c++) {
        int char_x = current_x;
        int char_y = y - char_height / 2;
        
        // Draw simple letter shapes (very basic)
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        
        switch (*c) {
            case 'R':
            case 'r':
                // R shape
                SDL_RenderDrawLine(renderer, char_x, char_y, char_x, char_y + char_height);
                SDL_RenderDrawLine(renderer, char_x, char_y, char_x + char_width, char_y);
                SDL_RenderDrawLine(renderer, char_x + char_width, char_y, char_x + char_width, char_y + char_height/2);
                SDL_RenderDrawLine(renderer, char_x, char_y + char_height/2, char_x + char_width, char_y + char_height/2);
                SDL_RenderDrawLine(renderer, char_x, char_y + char_height/2, char_x + char_width, char_y + char_height);
                break;
            case 'e':
            case 'E':
                // E shape
                SDL_RenderDrawLine(renderer, char_x, char_y, char_x, char_y + char_height);
                SDL_RenderDrawLine(renderer, char_x, char_y, char_x + char_width, char_y);
                SDL_RenderDrawLine(renderer, char_x, char_y + char_height/2, char_x + char_width*3/4, char_y + char_height/2);
                SDL_RenderDrawLine(renderer, char_x, char_y + char_height, char_x + char_width, char_y + char_height);
                break;
            case 'd':
            case 'D':
                // D shape
                SDL_RenderDrawLine(renderer, char_x, char_y, char_x, char_y + char_height);
                SDL_RenderDrawLine(renderer, char_x, char_y, char_x + char_width*3/4, char_y);
                SDL_RenderDrawLine(renderer, char_x + char_width*3/4, char_y, char_x + char_width, char_y + char_height/2);
                SDL_RenderDrawLine(renderer, char_x + char_width, char_y + char_height/2, char_x + char_width*3/4, char_y + char_height);
                SDL_RenderDrawLine(renderer, char_x + char_width*3/4, char_y + char_height, char_x, char_y + char_height);
                break;
            case 'B':
            case 'b':
                // B shape
                SDL_RenderDrawLine(renderer, char_x, char_y, char_x, char_y + char_height);
                SDL_RenderDrawLine(renderer, char_x, char_y, char_x + char_width*3/4, char_y);
                SDL_RenderDrawLine(renderer, char_x + char_width*3/4, char_y, char_x + char_width, char_y + char_height/4);
                SDL_RenderDrawLine(renderer, char_x + char_width, char_y + char_height/4, char_x + char_width*3/4, char_y + char_height/2);
                SDL_RenderDrawLine(renderer, char_x, char_y + char_height/2, char_x + char_width*3/4, char_y + char_height/2);
                SDL_RenderDrawLine(renderer, char_x + char_width*3/4, char_y + char_height/2, char_x + char_width, char_y + char_height*3/4);
                SDL_RenderDrawLine(renderer, char_x + char_width, char_y + char_height*3/4, char_x + char_width*3/4, char_y + char_height);
                SDL_RenderDrawLine(renderer, char_x + char_width*3/4, char_y + char_height, char_x, char_y + char_height);
                break;
            case 'l':
            case 'L':
                // L shape
                SDL_RenderDrawLine(renderer, char_x, char_y, char_x, char_y + char_height);
                SDL_RenderDrawLine(renderer, char_x, char_y + char_height, char_x + char_width, char_y + char_height);
                break;
            case 'a':
            case 'A':
                // A shape
                SDL_RenderDrawLine(renderer, char_x, char_y + char_height, char_x + char_width/2, char_y);
                SDL_RenderDrawLine(renderer, char_x + char_width/2, char_y, char_x + char_width, char_y + char_height);
                SDL_RenderDrawLine(renderer, char_x + char_width/4, char_y + char_height/2, char_x + char_width*3/4, char_y + char_height/2);
                break;
            case 'c':
            case 'C':
                // C shape
                SDL_RenderDrawLine(renderer, char_x + char_width, char_y, char_x, char_y);
                SDL_RenderDrawLine(renderer, char_x, char_y, char_x, char_y + char_height);
                SDL_RenderDrawLine(renderer, char_x, char_y + char_height, char_x + char_width, char_y + char_height);
                break;
            case 'k':
            case 'K':
                // K shape
                SDL_RenderDrawLine(renderer, char_x, char_y, char_x, char_y + char_height);
                SDL_RenderDrawLine(renderer, char_x, char_y + char_height/2, char_x + char_width, char_y);
                SDL_RenderDrawLine(renderer, char_x, char_y + char_height/2, char_x + char_width, char_y + char_height);
                break;
            case 'w':
            case 'W':
                // W shape
                SDL_RenderDrawLine(renderer, char_x, char_y, char_x, char_y + char_height);
                SDL_RenderDrawLine(renderer, char_x, char_y + char_height, char_x + char_width/2, char_y + char_height/2);
                SDL_RenderDrawLine(renderer, char_x + char_width/2, char_y + char_height/2, char_x + char_width, char_y + char_height);
                SDL_RenderDrawLine(renderer, char_x + char_width, char_y + char_height, char_x + char_width, char_y);
                break;
            case 'i':
            case 'I':
                // I shape
                SDL_RenderDrawLine(renderer, char_x + char_width/2, char_y, char_x + char_width/2, char_y + char_height);
                break;
            case 'n':
            case 'N':
                // N shape
                SDL_RenderDrawLine(renderer, char_x, char_y, char_x, char_y + char_height);
                SDL_RenderDrawLine(renderer, char_x, char_y, char_x + char_width, char_y + char_height);
                SDL_RenderDrawLine(renderer, char_x + char_width, char_y + char_height, char_x + char_width, char_y);
                break;
            case 's':
            case 'S':
                // S shape
                SDL_RenderDrawLine(renderer, char_x + char_width, char_y, char_x, char_y);
                SDL_RenderDrawLine(renderer, char_x, char_y, char_x, char_y + char_height/2);
                SDL_RenderDrawLine(renderer, char_x, char_y + char_height/2, char_x + char_width, char_y + char_height/2);
                SDL_RenderDrawLine(renderer, char_x + char_width, char_y + char_height/2, char_x + char_width, char_y + char_height);
                SDL_RenderDrawLine(renderer, char_x + char_width, char_y + char_height, char_x, char_y + char_height);
                break;
            case '!':
                // Exclamation mark
                SDL_RenderDrawLine(renderer, char_x + char_width/2, char_y, char_x + char_width/2, char_y + char_height*2/3);
                SDL_RenderDrawLine(renderer, char_x + char_width/2, char_y + char_height*5/6, char_x + char_width/2, char_y + char_height);
                break;
            case ' ':
                // Space - do nothing
                break;
        }
        
        current_x += char_width + spacing;
    }
}

// Render centered text (uses TTF if available, otherwise simple bitmap)
void render_text(SDL_Renderer *renderer, void *font, const char *text, 
                 int x, int y, SDL_Color color) {
    (void)color;  // Used in TTF path, but not in simple path
#ifdef HAVE_SDL_TTF
    if (font) {
        TTF_Font *ttf_font = (TTF_Font *)font;
        SDL_Surface *surface = TTF_RenderText_Solid(ttf_font, text, color);
        if (surface) {
            SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
            if (texture) {
                int text_width, text_height;
                SDL_QueryTexture(texture, NULL, NULL, &text_width, &text_height);
                
                SDL_Rect dest_rect = {
                    x - text_width / 2,
                    y - text_height / 2,
                    text_width,
                    text_height
                };
                
                SDL_RenderCopy(renderer, texture, NULL, &dest_rect);
                SDL_DestroyTexture(texture);
            }
            SDL_FreeSurface(surface);
            return;
        }
    }
#endif
    // Fallback to simple rendering
    render_text_simple(renderer, text, x, y, 4);
}

// Render the board
void render_board(SDL_Renderer *renderer, CheckersGame *game, void *font) {
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
                // Check if this piece should be blinking
                bool should_blink = game->is_blinking && 
                                    game->blink_row == i && 
                                    game->blink_col == j;
                
                // Calculate blink visibility (blinks every 200ms)
                bool visible = true;
                if (should_blink) {
                    Uint32 elapsed = SDL_GetTicks() - game->blink_start_time;
                    visible = (elapsed / 200) % 2 == 0;
                }
                
                if (visible) {
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
    }
    
    // Draw status bar
    SDL_Rect status_rect = {0, BOARD_SIZE * SQUARE_SIZE, WINDOW_WIDTH, 60};
    SDL_SetRenderDrawColor(renderer, 50, 50, 50, 255);
    SDL_RenderFillRect(renderer, &status_rect);
    
    // Draw game over message if game is over
    if (game->state == GAME_OVER) {
        const char *winner_text = get_winner(game);
        if (winner_text) {
            // Draw semi-transparent overlay
            SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 180);
            SDL_Rect overlay = {0, 0, WINDOW_WIDTH, BOARD_SIZE * SQUARE_SIZE};
            SDL_RenderFillRect(renderer, &overlay);
            SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_NONE);
            
            // Draw winner text centered (works with or without TTF)
            SDL_Color text_color = {255, 255, 255, 255};  // White
            int center_x = WINDOW_WIDTH / 2;
            int center_y = (BOARD_SIZE * SQUARE_SIZE) / 2;
            render_text(renderer, font, winner_text, center_x, center_y, text_color);
        }
    }
    
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
            
            // AI moves after player (only if not already blinking)
            if (game->state == AI_TURN && !game->is_blinking) {
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
    (void)argc;  // Unused
    (void)argv;  // Unused
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
    
    // Initialize SDL_ttf (optional)
    void *font = NULL;
#ifdef HAVE_SDL_TTF
    if (TTF_Init() < 0) {
        fprintf(stderr, "TTF initialization failed: %s\n", TTF_GetError());
        fprintf(stderr, "Continuing without TTF support - using simple text rendering\n");
    } else {
        // Load font (try to use a system font, fallback to default)
        const char *font_paths[] = {
            "/System/Library/Fonts/Helvetica.ttc",  // macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  // Linux
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",  // Linux alternative
            NULL
        };
        
        for (int i = 0; font_paths[i] != NULL; i++) {
            font = TTF_OpenFont(font_paths[i], 72);  // Large font size
            if (font) {
                break;
            }
        }
        
        // If no font found, continue without TTF (will use simple rendering)
        if (!font) {
            fprintf(stderr, "Warning: Could not load font. Using simple text rendering.\n");
        }
    }
#endif
    
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
        
        // Handle blinking animation and AI move execution
        if (game.is_blinking) {
            Uint32 elapsed = SDL_GetTicks() - game.blink_start_time;
            
            if (game.blink_before_move) {
                // Blink before move for 1 second
                if (elapsed >= 1000) {
                    // Execute the move
                    make_move(&game, game.pending_from_row, game.pending_from_col, 
                             game.pending_to_row, game.pending_to_col);
                    
                    // Check if there are more jumps (continue jumping)
                    // Note: make_move may have changed state, so check the landing position
                    int landing_row = game.pending_to_row;
                    int landing_col = game.pending_to_col;
                    if (game.state == AI_TURN && has_jump(&game, landing_row, landing_col)) {
                        // Continue with multiple jumps - find best next jump
                        int from_row = game.pending_to_row;
                        int from_col = game.pending_to_col;
                        int best_to_row, best_to_col;
                        
                        // Find best jump from this position using evaluation
                        int best_score = -999999;
                        bool found_jump = false;
                        
                        PieceType piece = game.board[from_row][from_col];
                        int directions[4][2] = {{-2, -2}, {-2, 2}, {2, -2}, {2, 2}};
                        int dir_count = is_king(piece) ? 4 : 2;
                        
                        if (!is_king(piece)) {
                            directions[0][0] = -2; directions[0][1] = -2;
                            directions[1][0] = -2; directions[1][1] = 2;
                        }
                        
                        for (int d = 0; d < dir_count; d++) {
                            int to_row = from_row + directions[d][0];
                            int to_col = from_col + directions[d][1];
                            if (is_valid_jump(&game, from_row, from_col, to_row, to_col)) {
                                int score = evaluate_move(&game, from_row, from_col, to_row, to_col, true);
                                if (score > best_score) {
                                    best_score = score;
                                    best_to_row = to_row;
                                    best_to_col = to_col;
                                    found_jump = true;
                                }
                            }
                        }
                        
                        if (found_jump) {
                            // Continue blinking at new position
                            game.pending_from_row = from_row;
                            game.pending_from_col = from_col;
                            game.pending_to_row = best_to_row;
                            game.pending_to_col = best_to_col;
                            game.blink_row = from_row;
                            game.blink_col = from_col;
                            game.blink_start_time = SDL_GetTicks();
                            // Continue blinking before next jump
                        } else {
                            // No more jumps, blink at landing position
                            game.blink_before_move = false;
                            game.blink_row = game.pending_to_row;
                            game.blink_col = game.pending_to_col;
                            game.blink_start_time = SDL_GetTicks();
                        }
                    } else {
                        // Move complete, blink at landing position
                        game.blink_before_move = false;
                        game.blink_row = game.pending_to_row;
                        game.blink_col = game.pending_to_col;
                        game.blink_start_time = SDL_GetTicks();
                    }
                }
            } else {
                // Blink after move for 1 second
                if (elapsed >= 1000) {
                    // Stop blinking and switch to player turn if game not over
                    game.is_blinking = false;
                    if (game.state == AI_TURN) {
                        game.state = PLAYER_TURN;
                    }
                }
            }
        }
        
        render_board(renderer, &game, font);
        SDL_Delay(16);  // ~60 FPS
    }
    
    // Cleanup
#ifdef HAVE_SDL_TTF
    if (font) {
        TTF_CloseFont((TTF_Font *)font);
    }
    TTF_Quit();
#endif
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}

