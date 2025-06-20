import pygame as pg
import pygame_widgets
from pygame_widgets.button import Button
from pygame_widgets.dropdown import Dropdown
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
import chess
import time
import os
from chessAppClasses import ChessGame, FontSprite

# Constants
WIDTH, HEIGHT = 640, 640
SQUARE_SIZE = WIDTH // 8
WHITE_COLOR = (240, 217, 181)
BLACK_COLOR = (181, 136, 99)

# Pygame standard setup
pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Chess Training Against Engines")
clock = pg.time.Clock()
game = None
selected_square = None

# Load piece images
pieces = {}
piece_types = ['P', 'N', 'B', 'R', 'Q', 'K']
colors = ['w', 'b']

images_path = os.path.join(os.path.dirname(__file__), "pieces")

for color in colors:
    for ptype in piece_types:
        img_path = os.path.join(images_path, f"{color}{ptype}.svg")
        img = pg.image.load(img_path)
        img = pg.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
        pieces[color + ptype] = img

# Function for drawing the board and pieces
def draw_board(screen, board, perspective=chess.WHITE):
    global selected_square

    for rank in range(8):
        for file in range(8):
            selection_square_perspective = chess.square(file, 7 - rank) if perspective == chess.WHITE else chess.square(7 - file, rank)
            square_color = ("#FF0000") if selection_square_perspective == selected_square else WHITE_COLOR if (rank + file) % 2 == 0 else BLACK_COLOR
            pg.draw.rect(screen, square_color, pg.Rect(file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            if perspective == chess.WHITE:
                piece = board.piece_at(chess.square(file, 7 - rank))
                if piece:
                    color = 'w' if piece.color == chess.WHITE else 'b'
                    piece_str = color + piece.symbol().upper()
                    screen.blit(pieces[piece_str], (file * SQUARE_SIZE, rank * SQUARE_SIZE))
            else:
                piece = board.piece_at(chess.square(7 - file, rank))
                if piece:
                    color = 'w' if piece.color == chess.WHITE else 'b'
                    piece_str = color + piece.symbol().upper()
                    screen.blit(pieces[piece_str], (file * SQUARE_SIZE, rank * SQUARE_SIZE))

def create_game():
    global game
    player_1_color = dropdown_white_player.getSelected() if dropdown_white_player.getSelected() != None else chess.WHITE
    max_depth = slider_max_depth.getValue()
    time_limit = slider_time.getValue()
    engine_type_p1 = dropdown_player_1.getSelected() if dropdown_player_1.getSelected() != None else "Human"
    engine_type_p2 = dropdown_player_2.getSelected() if dropdown_player_2.getSelected() != None else 1
    fen_string = input_starting_position.getText()

    if engine_type_p1 == "Human" and engine_type_p2 == "Human":
        print("At least one player must be an engine.")
        return

    game = ChessGame(player_1_color, max_depth, time_limit, engine_type_p1, engine_type_p2, fen=fen_string)

# Starting screen text
starting_screen_text = pg.sprite.Group()
starting_screen_text.add(FontSprite(WIDTH/2, 60, "Chess Training", "lucidasanstypewriter", 60))
starting_screen_text.add(FontSprite(30, 120, "Starting position:", "lucidasanstypewriter", 16, placement="midleft"))
starting_screen_text.add(FontSprite(30, 200, "Player playing white:", "lucidasanstypewriter", 16, placement="midleft"))
starting_screen_text.add(FontSprite(WIDTH/4, 280, "Player 1:", "lucidasanstypewriter", 16, placement="center"))
starting_screen_text.add(FontSprite(WIDTH*(3/4), 280, "Player 2:", "lucidasanstypewriter", 16, placement="center"))
starting_screen_text.add(FontSprite(WIDTH/4, 360, "Maximum engine depth:", "lucidasanstypewriter", 16, placement="center"))
starting_screen_text.add(FontSprite(WIDTH*(3/4), 360, "Engine thinking time:", "lucidasanstypewriter", 16, placement="center"))

# Starting position text input
input_starting_position = TextBox(screen,
    30, 140, 580, 36, # Coordinates and size
    borderThickness=1,
    borderColour="#000000",
    colour=("#D3D3D3"),
    radius=2,
    font=pg.font.SysFont("lucidasanstypewriterregular", 12),
    placeholderText="Enter FEN (default is standard starting position)"
)

# Starting screen dropdowns

dropdown_white_player = Dropdown(screen,
    30, 220, 580, 36, # Coordinates and size
    name="Choose starting Player",
    choices=["Player 1", "Player 2"],
    values=[chess.WHITE, chess.BLACK],
    borderRadius=2,
    colour=("#D3D3D3"),
    font=pg.font.SysFont("lucidasanstypewriterregular", 14)
)

dropdown_player_1 = Dropdown(screen,
    30, 300, WIDTH/2 - 60, 18, # Coordinates and size
    name="Choose Player or Engine type",
    choices=["Human", "Basic Evaluation", "Material Only", "Neural Network 1", "Neural Network 2", "Neural Network 3", "Neural Network 4", 
             "Neural Network 5", "Neural Network 6", "Neural Network 7", "Neural Network 1 + Material",  "Neural Network 2 + Material", 
             "Neural Network 3 + Material",  "Neural Network 4 + Material", "Neural Network 5 + Material", "Neural Network 6 + Material", "Neural Network 7 + Material"],
    values=["Human", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    borderRadius=2,
    colour=("#D3D3D3"),
    font=pg.font.SysFont("lucidasanstypewriterregular", 12)
)

dropdown_player_2 = Dropdown(screen,
    350, 300, WIDTH/2 - 60, 18, # Coordinates and size
    name="Choose Player or Engine type",
    choices=["Human", "Basic Evaluation", "Material Only", "Neural Network 1", "Neural Network 2", "Neural Network 3", "Neural Network 4", 
             "Neural Network 5", "Neural Network 6", "Neural Network 7", "Neural Network 1 + Material",  "Neural Network 2 + Material", 
             "Neural Network 3 + Material",  "Neural Network 4 + Material", "Neural Network 5 + Material", "Neural Network 6 + Material", "Neural Network 7 + Material"],
    values=["Human", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    borderRadius=2,
    colour=("#D3D3D3"),
    font=pg.font.SysFont("lucidasanstypewriterregular", 12)
)

slider_max_depth = Slider(screen,
    30, 390, WIDTH/2 - 60, 25, # Coordinates and size
    min=1, max=15, step=1,
    initial=10,
)

slider_time = Slider(screen,
    350, 390, WIDTH/2 - 60, 25, # Coordinates and size
    min=1, max=30, step=1,
    initial=5,
)

# Slider text
slider_text = pg.sprite.Group()
text_slider_max_depth = TextBox(screen, 142.5, 424, 35, 30, fontSize=20, borderThickness=1)
text_slider_max_depth.disable()  # Act as label instead of textbox
text_slider_time = TextBox(screen, 462.5, 424, 35, 30, fontSize=20, borderThickness=1)
text_slider_time.disable()  # Act as label instead of textbox

button_begin_game = Button(screen,
    30, 495, WIDTH - 60, 80, # Coordinates and size
    text="Start Game",
    fontSize=20,
    margin=10,
    borderRadius=2,
    borderThickness=2,
    colour="#D3D3D3",
    borderColour="#000000",
    inactiveColour="#FFFFFF",
    pressedColour="#C1E1C1",
    radius=2,
    onClick=create_game
)

# Game loop
def main():
    global selected_square
    global game

    app_running = True

    while app_running:
        events = pg.event.get()
        
        for event in events:
            if event.type == pg.QUIT:
                app_running = False
            
            elif event.type == pg.MOUSEBUTTONDOWN and game is not None:
                pos = pg.mouse.get_pos()
                square = game.get_selected_square(pos)

                human_color = None

                if game.engine_type_p1 == "Human":
                    human_color = chess.WHITE if game.player_1_color == chess.WHITE else chess.BLACK
                elif game.engine_type_p2 == "Human":
                    human_color = chess.BLACK if game.player_1_color == chess.WHITE else chess.WHITE

                if selected_square is None and human_color == game.board.turn:
                    piece = game.board.piece_at(square)
                    if piece and piece.color == human_color:
                        selected_square = square
                elif human_color == game.board.turn:
                    move = chess.Move(selected_square, square)

                    # Check if this is a pawn promotion
                    if game.board.piece_at(selected_square).piece_type == chess.PAWN and (
                        chess.square_rank(square) == 7 if game.board.turn == chess.WHITE else chess.square_rank(square) == 0
                    ):
                        move.promotion = chess.QUEEN  # Default to Queen promotion

                    if move in game.board.legal_moves:
                        san_move = game.board.san(move)
                        add_to_pgn = f"{game.move_number + 1}. {san_move}" if game.board.turn == chess.WHITE else f"{san_move}"
                        game.PGN += f"{add_to_pgn} "
                        game.move_number += 1 if game.board.turn == chess.WHITE else 0

                        print(add_to_pgn)

                        game.board.push(move)
                        selected_square = None
                    else:
                        selected_square = None

        # Starting screen
        if game is not None:
            draw_board(screen, game.board, perspective=game.player_1_color)
            pg.display.flip()

            if game.board.is_game_over():
                print(f"Player 1: Engine {game.engine_type_p1}, Color ({game.player_1_color}) vs Player 2: Engine {game.engine_type_p2}, Color ({chess.WHITE if game.player_1_color == chess.BLACK else chess.BLACK})")
                print("Game Over:", game.board.result())
                print("Game PGN:", game.PGN)
                time.sleep(3)
                game = None
                continue
            
            if game.board.turn == game.player_1_color:
                if game.engine_type_p1 != "Human":
                    game.make_engine_move(game.engine_type_p1)
                    continue
            elif game.board.turn != game.player_1_color:
                if game.engine_type_p2 != "Human":
                    game.make_engine_move(game.engine_type_p2)
                    continue
        else:
            screen.fill("#FFFFFF")
            starting_screen_text.draw(screen)

            # Draw the textinput, dropdowns and sliders
            text_slider_max_depth.setText(str(slider_max_depth.getValue()))
            text_slider_time.setText(str(slider_time.getValue()))
            pygame_widgets.update(events)

            pg.display.flip()

        clock.tick(60)  # Limits FPS to 60

if __name__ == "__main__":
    main()
    pg.quit()