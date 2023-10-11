"""
This class responsibilities are storing all the necessary information about the current state of a chess game and
determining valid moves at the current state. Also, move log will be kept.
"""


class GameState():
    def __init__(self):
        """Board is represented as a list of lists (8x8 2-dimensional) where each list will represent a row on the
        chessboard. Blank spaces are represented by '--' string to avoid any conversion problems"""
        self.board = [
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]
        ]
        self.moveFunctions = {'P': self.get_pawn_moves, 'R': self.get_rook_moves, 'N': self.get_knight_moves,
                              'B': self.get_bishop_moves, 'Q': self.get_queen_moves, 'K': self.get_king_moves}

        self.whiteToMove = True
        self.moveLog = []
        self.white_king_location = (7, 4)  # initial position of kings
        self.black_king_location = (0, 4)
        self.check_mate = False
        self.stale_mate = False
        self.enpassant_possible = ()  # coordinates of the square where move is possible
        self.current_castling_rights = CastleRights(True, True, True, True)
        self.castle_rights_log = [CastleRights(self.current_castling_rights.wks, self.current_castling_rights.bks,
                                               self.current_castling_rights.wqs, self.current_castling_rights.bqs)]

    """
    Function takes move as a parameter and executes it (will not work for castling, en-passant and pawn promotion)
    Parameters:
        move (Move): The move to be executed.
    Returns nothing.
    """

    def make_move(self, move):
        self.board[move.start_row][move.start_col] = "--"
        self.board[move.end_row][move.end_col] = move.piece_moved
        self.moveLog.append(move)  # log the move, so it is possible to undo it later / display history
        self.whiteToMove = not self.whiteToMove  # swap players
        # update king loc if moved
        if move.piece_moved == 'wK':
            self.white_king_location = (move.end_row, move.end_col)
        elif move.piece_moved == 'bK':
            self.black_king_location = (move.end_row, move.end_col)

        # pawn promotion
        if move.is_pawn_promotion:
            self.board[move.end_row][move.end_col] = move.piece_moved[0] + 'Q'

        # enpassant move
        if move.is_enpassant_move:
            self.board[move.start_row][move.end_col] = '--'  # capturing a pawn

        # update enpassant_possible var
        if move.piece_moved[1] == 'P' and abs(move.start_row - move.end_row) == 2:  # only 2 square pawn advance
            self.enpassant_possible = ((move.start_row + move.end_row) // 2, move.start_col)
        else:
            self.enpassant_possible = ()

        # castle move
        if move.is_castle_move:
            if move.end_col - move.start_col == 2:  # king side castle
                self.board[move.end_row][move.end_col - 1] = self.board[move.end_row][move.end_col + 1]  # moves rook
                self.board[move.end_row][move.end_col + 1] = '--'  # erase old rook position
            else:  # queen side castle
                self.board[move.end_row][move.end_col + 1] = self.board[move.end_row][move.end_col - 2]  # moves rook
                self.board[move.end_row][move.end_col - 2] = '--'

        # update castling rights
        self.update_castle_rights(move)
        self.castle_rights_log.append(CastleRights(self.current_castling_rights.wks, self.current_castling_rights.bks,
                                                   self.current_castling_rights.wqs, self.current_castling_rights.bqs))

    """
    Update castle rights based on the move made.
    Parameters:
    move (Move): The move that affects castling rights.
    Returns nothing.
    """

    def update_castle_rights(self, move):
        if move.piece_moved == 'wK':
            self.current_castling_rights.wks = False
            self.current_castling_rights.wqs = False
        elif move.piece_moved == 'bK':
            self.current_castling_rights.bks = False
            self.current_castling_rights.bqs = False
        elif move.piece_moved == 'wR':
            if move.start_row == 7:
                if move.start_col == 0:  # left rook
                    self.current_castling_rights.wqs = False
                elif move.start_col == 7:  # right rook
                    self.current_castling_rights.wks = False
        elif move.piece_moved == 'bR':
            if move.start_row == 0:
                if move.start_col == 0:  # left rook
                    self.current_castling_rights.bqs = False
                elif move.start_col == 7:  # right rook
                    self.current_castling_rights.bks = False

    """
    Undo last made move
    Parameters: None
    Returns nothing.
    """

    def undo_move(self):
        if len(self.moveLog) != 0:  # make sure there is move to undo
            move = self.moveLog.pop()
            self.board[move.start_row][move.start_col] = move.piece_moved
            self.board[move.end_row][move.end_col] = move.piece_captured
            self.whiteToMove = not self.whiteToMove  # swap players
            if move.piece_moved == 'wK':
                self.white_king_location = (move.start_row, move.start_col)
            elif move.piece_moved == 'bK':
                self.black_king_location = (move.start_row, move.start_col)
            # undo enpassant
            if move.is_enpassant_move:
                self.board[move.end_row][move.end_col] = '--'
                self.board[move.start_row][move.end_col] = move.piece_captured
                self.enpassant_possible = (move.end_row, move.end_col)
            # undo a 2 square pawn advance
            if move.piece_moved[1] == 'P' and abs(move.start_row - move.end_row) == 2:
                self.enpassant_possible = ()

            # undo castle rights
            self.castle_rights_log.pop()  # get rid of the new rights from the move that is undone
            self.current_castling_rights = self.castle_rights_log[-1]  # set current rights to the last one in list

            # undo castle move
            if move.is_castle_move:
                if move.end_col - move.start_col == 2:  # king side
                    self.board[move.end_row][move.end_col + 1] = self.board[move.end_row][move.end_col - 1]
                    self.board[move.end_row][move.end_col - 1] = '--'
                else:
                    self.board[move.end_row][move.end_col - 2] = self.board[move.end_row][move.end_col + 1]
                    self.board[move.end_row][move.end_col + 1] = '--'

    """
    All moves considering checks
    Parameters: None
    Returns:
    list of Move: A list of all valid moves that don't result in the current player being in check.
    """

    def get_valid_moves(self):
        temp_enpassant_possible = self.enpassant_possible
        temp_castle_rights = CastleRights(self.current_castling_rights.wks, self.current_castling_rights.bks,
                                          self.current_castling_rights.wqs, self.current_castling_rights.bqs)
        # 1. generate all possible moves
        moves = self.get_all_possible_moves()
        if self.whiteToMove:
            self.get_castle_moves(self.white_king_location[0], self.white_king_location[1], moves)
        else:
            self.get_castle_moves(self.black_king_location[0], self.black_king_location[1], moves)
        # 2. for each move, make the move
        for i in range(len(moves) - 1, -1, -1):  # when removing from a list go backwards through that list
            self.make_move(moves[i])
            # 3. generate all opponent moves
            # 4. for each of op moves see if they attack king
            self.whiteToMove = not self.whiteToMove  # make_move switches turns, and we need to switch them again in order for in_check to be valid
            if self.in_check():
                moves.remove(moves[i])
                # 5. if king attacked, move not valid
            self.whiteToMove = not self.whiteToMove
            self.undo_move()
        if len(moves) == 0:
            if self.in_check():
                self.check_mate = True
            else:
                self.stale_mate = True

        self.enpassant_possible = temp_enpassant_possible
        self.current_castling_rights = temp_castle_rights
        return moves

    """Determine if current player is in check
    Parameters: None
    Returns:
    bool: True if the current player is in check, False otherwise.
    """

    def in_check(self):
        if self.whiteToMove:
            return self.square_under_attack(self.white_king_location[0], self.white_king_location[1])
        else:
            return self.square_under_attack(self.black_king_location[0], self.black_king_location[1])

    """Determine if the enemy can attack the square
    Parameters:
    r (int): The row of the square.
    c (int): The column of the square.
    Returns:
    bool: True if the square is under attack, False otherwise.
    """

    def square_under_attack(self, r, c):
        self.whiteToMove = not self.whiteToMove  # swap to opps view
        opponent_moves = self.get_all_possible_moves()
        self.whiteToMove = not self.whiteToMove  # switch turns back
        for move in opponent_moves:
            if move.end_row == r and move.end_col == c:
                return True
        return False

    """
    All moves without considering checks
    Parameters: None
    Returns:
    list: A list of all possible moves.
    """

    def get_all_possible_moves(self):
        moves = []
        for r in range(len(self.board)):  # number of rows
            for c in range(len(self.board[r])):  # number of cols in given row
                turn = self.board[r][c][0]  # access first letter of piece
                if (turn == 'w' and self.whiteToMove) or (turn == 'b' and not self.whiteToMove):
                    piece = self.board[r][c][1]  # get piece type by getting second letter
                    self.moveFunctions[piece](r, c, moves)  # calls appropriate move function based on piece type
        return moves

    """
    Get all pawn moves for the piece located at row, col and add these to the list
    Parameters:
        r (int): The row of the piece.
        c (int): The column of the piece.
        moves (list): The list to store the possible moves.

    Returns nothing
    """

    def get_pawn_moves(self, r, c, moves):
        if self.whiteToMove:  # white pawn moves
            if self.board[r - 1][c] == "--":  # 1 square pawn advance
                moves.append(Move((r, c), (r - 1, c), self.board))
                if r == 6 and self.board[r - 2][c] == "--":  # 2 square pawn advance
                    moves.append(Move((r, c), (r - 2, c), self.board))
            if c - 1 >= 0:  # captures to the left
                if self.board[r - 1][c - 1][0] == 'b':  # enemy piece to capture
                    moves.append(Move((r, c), (r - 1, c - 1), self.board))
                elif (r - 1, c - 1) == self.enpassant_possible:
                    moves.append(Move((r, c), (r - 1, c - 1), self.board, enpassant_possible=True))
            if c + 1 <= 7:  # captures to the right
                if self.board[r - 1][c + 1][0] == 'b':  # enemy piece to capture
                    moves.append(Move((r, c), (r - 1, c + 1), self.board))
                elif (r - 1, c - 1) == self.enpassant_possible:
                    moves.append(Move((r, c), (r - 1, c + 1), self.board, enpassant_possible=True))
        else:  # black pawn moves
            if self.board[r + 1][c] == "--":  # 1 square move
                moves.append(Move((r, c), (r + 1, c), self.board))
                if r == 1 and self.board[r + 2][c] == "--":  # 2 square move
                    moves.append(Move((r, c), (r + 2, c), self.board))
            # captures
            if c - 1 >= 0:  # capture to the left
                if self.board[r + 1][c - 1][0] == 'w':
                    moves.append(Move((r, c), (r + 1, c - 1), self.board))
                elif (r + 1, c - 1) == self.enpassant_possible:
                    moves.append(Move((r, c), (r + 1, c - 1), self.board, enpassant_possible=True))
            if c + 1 <= 7:  # capture to the right
                if self.board[r + 1][c + 1][0] == 'w':
                    moves.append(Move((r, c), (r + 1, c + 1), self.board))
                elif (r + 1, c + 1) == self.enpassant_possible:
                    moves.append(Move((r, c), (r + 1, c + 1), self.board, enpassant_possible=True))

    """
    Get all rook moves for the piece located at row, col and add these to the list
    Parameters:
        r (int): The row of the rook.
        c (int): The column of the rook.
        moves (list): The list to store the possible moves.

    Returns nothing
    """

    def get_rook_moves(self, r, c, moves):
        directions = ((-1, 0), (0, -1), (1, 0), (0, 1))
        enemy_color = "b" if self.whiteToMove else "w"
        for d in directions:
            for i in range(1, 8):
                end_row = r + d[0] * i
                end_col = c + d[1] * i
                if 0 <= end_row < 8 and 0 <= end_col < 8:
                    end_piece = self.board[end_row][end_col]
                    if end_piece == "--":
                        moves.append(Move((r, c), (end_row, end_col), self.board))
                    elif end_piece[0] == enemy_color:
                        moves.append(Move((r, c), (end_row, end_col), self.board))
                        break
                    else:
                        break
                else:
                    break

    """Get all knight moves for the piece located at row, col and add them to the list.

            Parameters:
            r (int): The row of the knight.
            c (int): The column of the knight.
            moves (list): The list to store the possible moves.

            Returns nothing
    """

    def get_knight_moves(self, r, c, moves):
        knight_moves = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
        ally_color = "w" if self.whiteToMove else "b"
        for m in knight_moves:
            end_row = r + m[0]
            end_col = c + m[1]
            if 0 <= end_row < 8 and 0 <= end_col < 8:
                end_piece = self.board[end_row][end_col]
                if end_piece[0] != ally_color:
                    moves.append(Move((r, c), (end_row, end_col), self.board))

    """Get all bishop moves for the piece located at row, col and add them to the list.

            Parameters:
            r (int): The row of the bishop.
            c (int): The column of the bishop.
            moves (list): The list to store the possible moves.

            Returns nothing
    """

    def get_bishop_moves(self, r, c, moves):
        directions = ((-1, -1), (-1, 1), (1, -1), (1, 1))  # 4 diagonals
        enemy_color = "b" if self.whiteToMove else "w"
        for d in directions:
            for i in range(1, 8):
                end_row = r + d[0] * i
                end_col = c + d[1] * i
                if 0 <= end_row < 8 and 0 <= end_col < 8:
                    end_piece = self.board[end_row][end_col]
                    if end_piece == "--":
                        moves.append(Move((r, c), (end_row, end_col), self.board))
                    elif end_piece[0] == enemy_color:
                        moves.append(Move((r, c), (end_row, end_col), self.board))
                        break
                    else:
                        break
                else:
                    break

    """Get all king moves for the piece located at row, col and add them to the list.

            Parameters:
            r (int): The row of the king.
            c (int): The column of the king.
            moves (list): The list to store the possible moves.

            Returns nothing
    """

    def get_king_moves(self, r, c, moves):
        king_moves = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        ally_color = "w" if self.whiteToMove else "b"
        for i in range(8):
            end_row = r + king_moves[i][0]
            end_col = c + king_moves[i][1]
            if 0 <= end_row < 8 and 0 <= end_col < 8:
                end_piece = self.board[end_row][end_col]
                if end_piece[0] != ally_color:
                    moves.append(Move((r, c), (end_row, end_col), self.board))

    '''
    Generate all valid castle moves for the king at r,c and add them to list of moves
    Parameters:
        r (int): The row of the king.
        c (int): The column of the king.
        moves (list): The list to store the possible moves.

    Returns nothing
    '''

    def get_castle_moves(self, r, c, moves):
        if self.square_under_attack(r, c):
            return  # cant castle while in check
        if (self.whiteToMove and self.current_castling_rights.wks) or (
                not self.whiteToMove and self.current_castling_rights.bks):
            self.get_king_side_castle_moves(r, c, moves)
        if (self.whiteToMove and self.current_castling_rights.wqs) or (
                not self.whiteToMove and self.current_castling_rights.bqs):
            self.get_queen_side_castle_moves(r, c, moves)

    """Get king-side castle moves for the king at r, c and add them to the list.

           Parameters:
           r (int): The row of the king.
           c (int): The column of the king.
           moves (list): The list to store the possible moves.

           Returns nothing
    """

    def get_king_side_castle_moves(self, r, c, moves):
        if self.board[r][c + 1] == '--' and self.board[r][c + 2] == '--':
            if not self.square_under_attack(r, c + 1) and not self.square_under_attack(r, c + 2):
                moves.append(Move((r, c), (r, c + 2), self.board, is_castle_move=True))

    """Get queen-side castle moves for the king at r, c and add them to the list.

            Parameters:
            r (int): The row of the king.
            c (int): The column of the king.
            moves (list): The list to store the possible moves.

            Returns nothing
    """

    def get_queen_side_castle_moves(self, r, c, moves):
        if self.board[r][c - 1] == '--' and self.board[r][c - 2] == '--' and self.board[r][c - 3] == '--':
            if not self.square_under_attack(r, c - 1) and not self.square_under_attack(r, c - 2):
                moves.append(Move((r, c), (r, c - 2), self.board, is_castle_move=True))

    """Get all queen moves for the piece located at row, col and add them to the list.

            Parameters:
            r (int): The row of the queen.
            c (int): The column of the queen.
            moves (list): The list to store the possible moves.

            Returns nothing
    """

    def get_queen_moves(self, r, c, moves):
        self.get_rook_moves(r, c, moves)
        self.get_bishop_moves(r, c, moves)


"""Initialize castle rights for both players.

        Parameters:
        wks (bool): White kingside castle rights.
        bks (bool): Black kingside castle rights.
        wqs (bool): White queenside castle rights.
        bqs (bool): Black queenside castle rights.

        Returns nothing
"""


class CastleRights():
    def __init__(self, wks, bks, wqs, bqs):
        self.wks = wks
        self.bks = bks
        self.wqs = wqs
        self.bqs = bqs


"""Initialize a chess move.

        Parameters:
        start_sq (tuple): The (row, col) of the starting square.
        end_sq (tuple): The (row, col) of the ending square.
        board (list): The chessboard.
        enpassant_possible (bool): Flag for en passant move.
        is_castle_move (bool): Flag for castling move.

        Returns:
        None
"""


class Move():
    # maps keys to values, here we write the chess notation to log the moves
    # key : value
    ranks_to_rows = {"1": 7, "2": 6, "3": 5, "4": 4, "5": 3, "6": 2, "7": 1, "8": 0}
    rows_to_ranks = {v: k for k, v in ranks_to_rows.items()}  # reverse the mapping above
    files_to_cols = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
    cols_to_files = {v: k for k, v in files_to_cols.items()}  # reverse the mapping above

    def __init__(self, start_sq, end_sq, board, enpassant_possible=False, is_castle_move=False):
        self.start_row = start_sq[0]
        self.start_col = start_sq[1]
        self.end_row = end_sq[0]
        self.end_col = end_sq[1]
        self.piece_moved = board[self.start_row][self.start_col]
        self.piece_captured = board[self.end_row][self.end_col]

        self.is_pawn_promotion = (self.piece_moved == 'wP' and self.end_row == 0) or (
                self.piece_moved == 'bP' and self.end_row == 7)

        self.is_enpassant_move = enpassant_possible
        if self.is_enpassant_move:
            self.piece_captured = 'wP' if self.piece_moved == 'bP' else 'bP'

        self.is_castle_move = is_castle_move

        self.move_id = self.start_row * 1000 + self.start_col * 100 + self.end_row * 10 + self.end_col
        print(self.move_id)

    """
    Override equals method
    """

    def __eq__(self, other):
        if isinstance(other, Move):
            return self.move_id == other.move_id
        return False

    """Get chess notation for the move.

           Parameters:
           None

           Returns:
           str: Chess notation for the move.
    """
    def get_chess_notation(self):
        return self.get_rank_file(self.start_row, self.start_col) + self.get_rank_file(self.end_row, self.end_col)

    """Get rank and file notation for a square.

            Parameters:
            r (int): The row of the square.
            c (int): The column of the square.

            Returns:
            str: Rank and file notation for the square.
    """
    def get_rank_file(self, r, c):
        return self.cols_to_files[c] + self.rows_to_ranks[r]
