import random

piece_score = {"K": 0, "Q": 10, "R": 5, "B": 3, "N": 3, "P": 1}
CHECKMATE = 1000
STALEMATE = 0

'''
Find best move based on material alone
Parameters:
state_of_game (GameState): The current game state.
valid_moves (list): List of valid moves in the current position.

Returns:
Move: The best move based on material advantage.
'''


def find_best_move(state_of_game, valid_moves):
    turn_multiplier = 1 if state_of_game.whiteToMove else -1

    opponent_min_max_score = CHECKMATE
    best_player_move = None
    for player_move in valid_moves:
        state_of_game.make_move(player_move)
        opp_moves = state_of_game.get_valid_moves()
        random.shuffle(valid_moves)
        opponent_max_score = -CHECKMATE
        for opp_move in opp_moves:
            state_of_game.make_move(opp_move)
            if state_of_game.check_mate:
                score = -turn_multiplier * CHECKMATE
            elif state_of_game.stale_mate:
                score = STALEMATE
            else:
                score = -turn_multiplier * score_material(state_of_game.board)
            if score > opponent_max_score:
                opponent_max_score = score
            state_of_game.undo_move()
        if opponent_max_score < opponent_min_max_score:
            opponent_min_max_score = opponent_max_score
            best_player_move = player_move
        state_of_game.undo_move()
    return best_player_move


'''
Score the board based on material
Parameters:
board (list): The current state of the chessboard.

Returns:
int: The material advantage score.
'''


def score_material(board):
    score = 0
    for row in board:
        for square in row:
            if square[0] == 'w':
                score += piece_score[square[1]]
            elif square[0] == 'b':
                score -= piece_score[square[1]]
    return score
