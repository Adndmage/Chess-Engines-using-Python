import pygame as pg
import chess
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chessAlgorithms.minimax import iterative_deepening

class ChessGame:
    def __init__(self, player_1_color, max_depth, time_limit, engine_type_p1, engine_type_p2, fen=None):
        self.board = chess.Board()

        if fen:
            try:
                self.board.set_fen(fen)
            except ValueError:
                print("Invalid FEN string. Using default starting position.")
        
        self.players = ["Player 1", "Player 2"]

        self.player_1_color = player_1_color
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.engine_type_p1 = engine_type_p1
        self.engine_type_p2 = engine_type_p2

        self.move_number = 0
        self.PGN = ""
    
    def start_game(self, player_1_color, max_depth, time_limit, engine_type_p1, engine_type_p2, fen=None):
        self.player_1_color = player_1_color
        self.max_depth = max_depth
        self.time_limit = time_limit
    
    def get_selected_square(self, pos):
        if self.player_1_color == chess.WHITE:
            file = pos[0] // 80
            rank = 7 - (pos[1] // 80)
        else:
            file = 7 - (pos[0] // 80)
            rank = pos[1] // 80

        return chess.square(file, rank)
    
    def make_engine_move(self, engine_type):
        move = iterative_deepening(self.board, self.max_depth, self.time_limit, engine_type)
        
        if move:
            san_move = self.board.san(move)
            add_to_pgn = f"{self.move_number + 1}. {san_move}" if self.board.turn == chess.WHITE else f"{san_move}"
            self.PGN += f"{add_to_pgn} "
            self.move_number += 1 if self.board.turn == chess.WHITE else 0

            print(add_to_pgn)

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