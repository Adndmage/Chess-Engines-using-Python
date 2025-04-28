import pygame
import chess
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chessAlgorithms.minimax import iterative_deepening

# Init pygame
pygame.init()

# Settings
WIDTH, HEIGHT = 640, 640
SQUARE_SIZE = WIDTH // 8

# Colors
LIGHT = (240, 217, 181)
DARK = (181, 136, 99)

# Load piece images
pieces = {}
piece_types = ['P', 'N', 'B', 'R', 'Q', 'K']
colors = ['w', 'b']

images_path = os.path.join(os.path.dirname(__file__), "images")

for color in colors:
    for ptype in piece_types:
        img_path = os.path.join(images_path, f"{color}{ptype}.svg")
        img = pygame.image.load(img_path)
        img = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
        pieces[color + ptype] = img

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Engine GUI")

# Board
board = chess.Board()

# Selected piece
selected_square = None

def draw_board(screen, board):
    for rank in range(8):
        for file in range(8):
            square_color = LIGHT if (rank + file) % 2 == 0 else DARK
            pygame.draw.rect(screen, square_color, pygame.Rect(file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            piece = board.piece_at(chess.square(file, 7 - rank))
            if piece:
                color = 'w' if piece.color == chess.WHITE else 'b'
                piece_str = color + piece.symbol().upper()
                screen.blit(pieces[piece_str], (file * SQUARE_SIZE, rank * SQUARE_SIZE))

def get_square_under_mouse(pos):
    file = pos[0] // SQUARE_SIZE
    rank = 7 - (pos[1] // SQUARE_SIZE)
    return chess.square(file, rank)

def main():
    global selected_square

    running = True
    human_color = chess.WHITE  # You can switch
    clock = pygame.time.Clock()

    while running:
        draw_board(screen, board)
        pygame.display.flip()

        if board.is_game_over():
            print("Game Over:", board.result())
            time.sleep(3)
            running = False
            continue

        if board.turn != human_color:
            # Engine move
            move = iterative_deepening(board, max_depth=10, time_limit=5)
            if move:
                board.push(move)
            continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                square = get_square_under_mouse(pos)

                if selected_square is None:
                    piece = board.piece_at(square)
                    if piece and piece.color == human_color:
                        selected_square = square
                else:
                    move = chess.Move(selected_square, square)
                    if move in board.legal_moves:
                        board.push(move)
                        selected_square = None
                    else:
                        selected_square = None

        clock.tick(60)

if __name__ == "__main__":
    main()
    pygame.quit()