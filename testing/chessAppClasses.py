import pygame as pg
import chess
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chessAlgorithms.minimax import iterative_deepening

class ChessGame:
    def __init__(self, players, max_depth, time_limit, engine_type):
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.board = chess.Board(starting_fen)
        self.players = players
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.engine_type = engine_type
        self.running = False
        self.selected_square = None

    # def draw_board(self, screen):
    #     for rank in range(8):
    #         for file in range(8):
    #             square_color = (255, 255, 255) if (rank + file) % 2 == 0 else (0, 0, 0)
    #             pg.draw.rect(screen, square_color, pg.Rect(file * 60, rank * 60, 60, 60))

    #             piece = self.board.piece_at(chess.square(file, 7 - rank))
    #             if piece:
    #                 color = 'w' if piece.color == chess.WHITE else 'b'
    #                 piece_str = color + piece.symbol().upper()
    #                 screen.blit(pieces[piece_str], (file * 60, rank * 60))
    
    def get_selected_square(self, pos):
        file = pos[0] // 60
        rank = 7 - (pos[1] // 60)
        return chess.square(file, rank)
    
    def make_engine_move(self):
        move = iterative_deepening(self.board, self.max_depth, self.time_limit, self.engine_type)
        
        if move:
            self.board.push(move)

class FontSprite(pg.sprite.Sprite):
    def __init__(self, x, y, text, font_name, font_size, placement = "center", color = "#000000"):
        super().__init__()
        
        font = pg.font.SysFont(font_name, font_size)
        self.image = font.render(text, False, color)
        self.rect = self.image.get_rect(midleft = (x, y))

        if placement == "midleft":
            self.rect = self.image.get_rect(midleft = (x, y))
        elif placement == "center":
            self.rect = self.image.get_rect(center = (x, y))
        elif placement == "midbottom":
            self.rect = self.image.get_rect(midbottom = (x, y))
        elif placement == "midtop":
            self.rect = self.image.get_rect(midtop = (x, y))
        elif placement == "midright":
            self.rect = self.image.get_rect(midright = (x, y))