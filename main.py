import chess
from Game import *

game = Game(["Human", "AI"])

"""
The Arena game loop
"""

turnnr = 0
print(game.board)
while not game.board.is_game_over():
    if game.board.turn:
        game.human_move()
    else:
        game.computer_next_best_move(-1)

    turnnr += 1
    print(f"\nMove {turnnr}:")
    print(game.board)

print("Game over!")
print("Result:", game.board.result())
print(game.board)