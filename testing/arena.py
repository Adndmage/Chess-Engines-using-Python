import pygame as pg
import pygame_widgets
from pygame_widgets.button import Button
from pygame_widgets.dropdown import Dropdown
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
import chess
import time
import sys
import os
from chessAppClasses import ChessGame, FontSprite
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
pg.display.set_caption("Chess Training Against Engines")
clock = pg.time.Clock()
app_running = True
game = ChessGame(["Human Player", "Engine Player"], max_depth=10, time_limit=5, engine_type=1)

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

# Starting screen text
starting_screen_text = pg.sprite.Group()
starting_screen_text.add(FontSprite(WIDTH/2, 60, "Chess Training", "lucidasanstypewriter", 60))
starting_screen_text.add(FontSprite(30, 140, "Starting position:", "lucidasanstypewriter", 16, placement="midleft"))
starting_screen_text.add(FontSprite(30, 220, "Player playing white:", "lucidasanstypewriter", 16, placement="midleft"))
starting_screen_text.add(FontSprite(WIDTH/4, 300, "Player 1:", "lucidasanstypewriter", 16, placement="center"))
starting_screen_text.add(FontSprite(WIDTH*(3/4), 300, "Player 2:", "lucidasanstypewriter", 16, placement="center"))
starting_screen_text.add(FontSprite(WIDTH/4, 390, "Maximum engine depth:", "lucidasanstypewriter", 16, placement="center"))
starting_screen_text.add(FontSprite(WIDTH*(3/4), 390, "Engine thinking time:", "lucidasanstypewriter", 16, placement="center"))

# Starting position text input
input_starting_position = TextBox(screen,
    30, 160, 580, 36, # Coordinates and size
    borderThickness=1,
    borderColour="#000000",
    colour=("#D3D3D3"),
    radius=2,
    font=pg.font.SysFont("lucidasanstypewriterregular", 16)
)

# Starting screen dropdowns

dropdown_white_player = Dropdown(screen,
    30, 240, 580, 36, # Coordinates and size
    name="Choose starting Player",
    choices=["Player 1", "Player 2"],
    borderRadius=2,
    colour=("#D3D3D3"),
    font=pg.font.SysFont("lucidasanstypewriterregular", 14)
)

dropdown_player_1 = Dropdown(screen,
    30, 320, WIDTH/2 - 60, 25, # Coordinates and size
    name="Choose Player or Engine type",
    choices=["Human", "Basic Evaluation", "Neural Network", "Neural Network + Material"],
    values=["Human", 1, 2, 3],
    borderRadius=2,
    colour=("#D3D3D3"),
    font=pg.font.SysFont("lucidasanstypewriterregular", 12)
)

dropdown_player_2 = Dropdown(screen,
    350, 320, WIDTH/2 - 60, 25, # Coordinates and size
    name="Choose Player or Engine type",
    choices=["Human", "Basic Evaluation", "Neural Network", "Neural Network + Material"],
    values=["Human", 1, 2, 3],
    borderRadius=2,
    colour=("#D3D3D3"),
    font=pg.font.SysFont("lucidasanstypewriterregular", 12)
)

slider_max_depth = Slider(screen,
    30, 420, WIDTH/2 - 60, 25, # Coordinates and size
    min=1, max=15, step=1,
    initial=5,
)

slider_time = Slider(screen,
    350, 420, WIDTH/2 - 60, 25, # Coordinates and size
    min=1, max=30, step=1,
    initial=5,
)

# Slider text
slider_text = pg.sprite.Group()

text_slider_max_depth = TextBox(screen, 142.5, 454, 35, 30, fontSize=20, borderThickness=1)
text_slider_max_depth.disable()  # Act as label instead of textbox
text_slider_time = TextBox(screen, 462.5, 454, 35, 30, fontSize=20, borderThickness=1)
text_slider_time.disable()  # Act as label instead of textbox

# Game loop
while app_running:
    events = pg.event.get()

    for event in events:
        if event.type == pg.QUIT:
            app_running = False
    
    # Starting screen
    if not game.running:
        screen.fill("#FFFFFF")
        starting_screen_text.draw(screen)
    
    # Draw the textinput, dropdowns and sliders
    text_slider_max_depth.setText(str(slider_max_depth.getValue()))
    text_slider_time.setText(str(slider_time.getValue()))
    pygame_widgets.update(events)
    pg.display.flip()
    clock.tick(60)  # Limits FPS to 60
