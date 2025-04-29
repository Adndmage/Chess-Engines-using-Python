import pygame as pg
import chess
import chess.svg
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chessAlgorithms.minimax import iterative_deepening

# Constants
WIDTH, HEIGHT = 640, 640
SQUARE_SIZE = WIDTH // 8
WHITE_COLOR = (240, 217, 181)
BLACK_COLOR = (181, 136, 99)

# Pygame standard setup
pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Chess Engine GUI")
clock = pg.time.Clock()
app_running = True

# Load piece images
pieces = {}
piece_types = ['P', 'N', 'B', 'R', 'Q', 'K']
colors = ['w', 'b']

images_path = os.path.join(os.path.dirname(__file__), "images")

for color in colors:
    for ptype in piece_types:
        img_path = os.path.join(images_path, f"{color}{ptype}.svg")
        img = pg.image.load(img_path)
        img = pg.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
        pieces[color + ptype] = img

# Board
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"
board = chess.Board()

# Selected piece
selected_square = None
human_color = chess.BLACK  # You can switch to chess.BLACK

def draw_board(screen, board):
    for rank in range(8):
        for file in range(8):
            square_color = WHITE_COLOR if (rank + file) % 2 == 0 else BLACK_COLOR
            pg.draw.rect(screen, square_color, pg.Rect(file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            piece = board.piece_at(chess.square(7 - file, rank))
            if piece:
                color = 'w' if piece.color == chess.WHITE else 'b'
                piece_str = color + piece.symbol().upper()
                screen.blit(pieces[piece_str], (file * SQUARE_SIZE, rank * SQUARE_SIZE))

def get_square_under_mouse(pos):
    if human_color == "white":
        file = pos[0] // 80
        rank = 7 - (pos[1] // 80)
    else:
        file = 7 - (pos[0] // 80)
        rank = pos[1] // 80

    return chess.square(file, rank)

def main():
    global selected_square

    app_running = True

    while app_running:
        draw_board(screen, board)
        pg.display.flip()

        if board.is_game_over():
            print("Game Over:", board.result())
            time.sleep(3)
            app_running = False
            continue

        if board.turn != human_color:
            # Engine move
            move = iterative_deepening(board, max_depth=10, time_limit=3)
            if move:
                board.push(move)
            continue

        for event in pg.event.get():
            if event.type == pg.QUIT:
                app_running = False

            elif event.type == pg.MOUSEBUTTONDOWN:
                pos = pg.mouse.get_pos()
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
    pg.quit()