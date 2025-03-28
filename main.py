import chess
from Game import *

game = Game(["Human", "AI"])

turnnr = 0
print(game.board)
while not game.board.is_game_over():
    if game.board.turn:
        game.computer_move_random()
    else:
        game.computer_move_minimax()

    turnnr += 1
    print(f"\nMove {turnnr}:")
    print(game.board)

print("Game over!")
print("Result:", game.board.result())
print(game.board)