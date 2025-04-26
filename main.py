import chess
from Game import *

game = Game(["Human", "AI"])

"""
The Arena game loop
"""

move_number = 0
print(game.board)
while not game.board.is_game_over():
    if game.board.turn:
        game.computer_move_NN()
    else:
        game.computer_move_minimax()

    move_number += 1
    print(f"\nMove {move_number}:")
    print(game.board)

print("Game over!")
print("Result:", game.board.result())
print(game.board)