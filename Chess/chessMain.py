"""
This is main driver file. It's responsibilities are handling user input and displaying current GameState.
Game instructions: https://en.wikipedia.org/wiki/Rules_of_chess
Author: Szymon Kuczyński
Setup:
    1. Get all required files from Git and put them in folder 'Chess'
    2. In the console, navigate to 'Chess' folder
    3. Launch the game by typing 'python chessMain.py'
"""

import pygame as p
import chessEngine
import AiBot

p.init()
WIDTH = HEIGHT = 512  # chessboard resolution
DIMENSION = 8  # dimensions of a chessboard are 8x8
SqSIZE = HEIGHT // DIMENSION
MAxFPS = 15
IMAGES = {}

'''
We are going to load the necessary images once, using our load_pieces function, and then store them in the variable. Loading images every
time from scratch would cause the game to lag. This will initialize a global dictionary of images.
'''


def load_pieces():
    pieces = ['wP', 'bP', 'bB', 'bK', 'bN', 'bQ', 'bR', 'wB', 'wK', 'wN', 'wQ', 'wR']
    '''
    for piece in pieces: using for each loop, we are looping through pieces list and getting each name
    IMAGES[piece]: here we take this piece name and create a path to the image for pygame to load it.
    p.transform size(SqSize, SqSize): loaded image is scaled to the desired resolution 
    '''
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load(piece + ".png"), (SqSIZE, SqSIZE))


"""
This is main driver. It will handle user input and updating the graphics.
screen: we set up the window for our GUI here
clock: we set up the ticks of in game clock, as we need them for generating image frames
screen.fill: we are filling our screen with white color
running: this variable determines if the game is running or not 
"""


def main():
    screen = p.display.set_mode((WIDTH, HEIGHT))
    clock = p.time.Clock()
    screen.fill(p.Color("white"))
    state_of_game = chessEngine.GameState()
    valid_moves = state_of_game.get_valid_moves()
    move_made = False  # flag variable for when a move is made
    anim = False  # flag var for when we should animate a move
    load_pieces()
    running = True
    sq_selected = ()  # no square is selected initially, so the variable is empty, but it will keep track of last
    # click of the user
    player_clicks = []  # keep track of the player clicks (example list structure: [(6,5), (3,3)])
    game_over = False
    player_one = True
    player_two = False

    while running:
        human_turn = (state_of_game.whiteToMove and player_one) or (not state_of_game.whiteToMove and player_two)
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False
            elif e.type == p.MOUSEBUTTONDOWN:
                if not game_over and human_turn:
                    location = p.mouse.get_pos()  # (x,y) location of the mouse
                    col = location[0] // SqSIZE
                    row = location[1] // SqSIZE
                    if sq_selected == (row, col):  # check if user clicked the same square twice
                        sq_selected = ()  # deselect square
                        player_clicks = []  # reset player clicks
                    else:
                        sq_selected = (row, col)
                        player_clicks.append(sq_selected)  # append for 1st and 2nd click
                    if len(player_clicks) == 2:  # situation after 2nd click
                        move = chessEngine.Move(player_clicks[0], player_clicks[1], state_of_game.board)
                        print(move.get_chess_notation())
                        for i in range(len(valid_moves)):
                            if move == valid_moves[i]:
                                state_of_game.make_move(valid_moves[i])
                                move_made = True
                                anim = True
                                sq_selected = ()  # reset user clicks
                                player_clicks = []
                        if not move_made:
                            player_clicks = [
                                sq_selected]  # set current clicks to current square selected in case of changing the move


            elif e.type == p.KEYDOWN:
                if e.key == p.K_z:  # call undo when 'z' is pressed
                    state_of_game.undo_move()
                    move_made = True
                    anim = False
                if e.key == p.K_r:  # reset the game on 'r' press
                    state_of_game = chessEngine.GameState()
                    valid_moves = state_of_game.get_valid_moves()
                    sq_selected = ()
                    player_clicks = []
                    move_made = False
                    anim = False

        # AI move
        if not game_over and not human_turn:
            ai_move = AiBot.find_best_move(state_of_game, valid_moves)
            state_of_game.make_move(ai_move)
            move_made = True
            anim = True

        if move_made:
            if anim:
                animate(state_of_game.moveLog[-1], screen, state_of_game.board, clock)
            valid_moves = state_of_game.get_valid_moves()
            move_made = False
            anim = False

        draw_game_state(screen, state_of_game, valid_moves, sq_selected)
        if state_of_game.check_mate:
            game_over = True
            if state_of_game.whiteToMove:
                draw_text(screen, 'Black wins by checkmate')
            else:
                draw_text(screen, 'White wins by checkmate')
        elif state_of_game.stale_mate:
            game_over = True
            draw_text(screen, 'Stalemate')

        clock.tick(MAxFPS)  # while the game runs we set the number of our in game clock ticks to generate image frames
        p.display.flip()


'''
Highlight selected square and available moves for selected piece
Parameters:
    screen: Pygame screen object
    state_of_game: GameState object representing the current game state
    valid_moves: List of valid chess moves
    sq_selected: Tuple representing the selected square

Returns nothing
'''


def highlight_sq(screen, state_of_game, valid_moves, sq_selected):
    if sq_selected != ():
        r, c = sq_selected
        if state_of_game.board[r][c][0] == (
                'w' if state_of_game.whiteToMove else 'b'):  # sqselected a piece that can be moved
            # highlight selected sq
            s = p.Surface((SqSIZE, SqSIZE))
            s.set_alpha(100)  # transparency value
            s.fill(p.Color('blue'))
            screen.blit(s, (c * SqSIZE, r * SqSIZE))
            # highlight moves from square
            s.fill(p.Color('yellow'))
            for move in valid_moves:
                if move.start_row == r and move.start_col == c:
                    screen.blit(s, (SqSIZE * move.end_col, SqSIZE * move.end_row))


"""
draw_game_state(screen, state_of_game): we draw a board with chess pieces by using functions explained below. 
We need 'screen' for that and 'state_of_game' which takes all the information about the board in this case.
Parameters:
    screen: Pygame screen object
    state_of_game: GameState object representing the current game state
    valid_moves: List of valid chess moves
    sq_selected: Tuple representing the selected square

Returns nothing
"""


def draw_game_state(screen,
                    state_of_game, valid_moves,
                    sq_selected):  # this function is responsible for all the graphics within a current game state
    draw_board(screen)  # this function draws squares on our board
    highlight_sq(screen, state_of_game, valid_moves, sq_selected)
    draw_pieces(screen, state_of_game.board)  # this function draws pieces on our squares


"""
Draw the squares on the board
draw_board(screen): screen is the resolution of our app's window
colors: defines, which colors we will use on our board
for r in range(DIMENSION):
    for c in range (DIMENSION):
        color = colors[((r + c) % 2)]
        p.draw.rect(screen, color, p.Rect(c * SqSIZE, r * SqSIZE, SqSIZE, SqSIZE)): 
By using nested for each loop, we are going through rows (r) and columns (c) in our board, we use 'DIMENSION'
variable to determine where the row and column is. In 'color' we determine by quick math how the rectangles should be
colored (white are odd numbers and black are even numbers). Then, by using 'p.draw.rect' we are drawing rectangles, 
which are adjusted to the resolution by 'screen" variable, have a 'color' and we determine their size and location
by p.Rect. By multiplying 'c' and 'r' by our 'SqSIZE' we know, where exactly the square should be drawn and two 'SqSIZE'
variables determine the size of the drawn square.
Parameters:
    screen: Pygame screen object

Returns nothing
"""


def draw_board(screen):
    global colors
    colors = [p.Color("white"), p.Color("gray")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[((r + c) % 2)]
            p.draw.rect(screen, color, p.Rect(c * SqSIZE, r * SqSIZE, SqSIZE, SqSIZE))


"""
Draw the pieces on the board using the current state_of_game.board.
draw_pieces(screen, board): function takes 'screen' as resolution and 'board' as board representation.
The loop uses the same technique to loop through the board, as the function above.
piece = board[r][c]: here the position of a specific chess piece is determined on the board by taking row and
column as it's location
screen.blit: this function draws a chess piece on the corresponding rectangle (p.Rect function, as explained above) 
by taking it's image (IMAGE[piece]).

Parameters:
    screen: Pygame screen object
    board: List of lists representing the chessboard and the positions of the pieces

Returns nothing
"""


def draw_pieces(screen, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != "--":  # check if the square is not empty
                screen.blit(IMAGES[piece], p.Rect(c * SqSIZE, r * SqSIZE, SqSIZE, SqSIZE))


'''
Animating of moving chess pieces
Parameters:
    move: The chess move to be animated
    screen: Pygame screen object
    board: List of lists representing the chessboard and the positions of the pieces
    clock: Pygame clock object for controlling animation frame rate

Returns nothing
'''


def animate(move, screen, board, clock):
    global colors
    dR = move.end_row - move.start_row
    dC = move.end_col - move.start_col
    frames_per_sq = 10  # frames to move one square
    frame_count = (abs(dR) + abs(dC)) * frames_per_sq
    for frame in range(frame_count + 1):
        r, c = (move.start_row + dR * frame / frame_count, move.start_col + dC * frame / frame_count)
        draw_board(screen)
        draw_pieces(screen, board)
        # erase the piece moved from its ending sq
        color = colors[(move.end_row + move.end_col) % 2]
        end_sq = p.Rect(move.end_col * SqSIZE, move.end_row * SqSIZE, SqSIZE, SqSIZE)
        p.draw.rect(screen, color, end_sq)
        # draw captured piece
        if move.piece_captured != '--':
            screen.blit(IMAGES[move.piece_captured], end_sq)
        # draw moving piece
        screen.blit(IMAGES[move.piece_moved], p.Rect(c * SqSIZE, r * SqSIZE, SqSIZE, SqSIZE))
        p.display.flip()
        clock.tick(60)  # FPS for animation


"""
Draw text on the screen.

Parameters:
    screen: Pygame screen object
    text: The text to be displayed

Returns nothing
"""


def draw_text(screen, text):
    font = p.font.SysFont("Helvitca", 32, True, False)
    text_obj = font.render(text, 0, p.Color('Red'))
    text_loc = p.Rect(0, 0, WIDTH, HEIGHT).move(WIDTH / 2 - text_obj.get_width() / 2,
                                                HEIGHT / 2 - text_obj.get_height() / 2)
    screen.blit(text_obj, text_loc)


if __name__ == "__main__":
    main()
