import chess
from Game import *

game = Game(["Human", "AI"])

while not game.board.is_game_over():
    if game.board.turn:
        print(game.board)
        game.human_move()
    else:
        game.computer_move()

print("Game over")
print(game.board)