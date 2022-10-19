from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax
from colorama import Fore


class IsolationGame(TwoPlayerGame):

    def __init__(self, players=None):
        '''
        Set up initial data, fill the board with X and O's to display it onto console
        :param players:
        int:number of players
        '''
        self.players = players
        self.board = [[]]
        self.current_player = 1
        self.players_position = [[1, 2], [6, 3]]
        self.current_field = "O"

        for x in range(8):
            self.board.append([])
            for y in range(6):
                if x == 0 or x == 7:
                    self.board[x].append([])
                    self.board[x][y] = "X"
                else:
                    if y != 0 and y != 5:
                        self.board[x].append([])
                        self.board[x][y] = "O"
                    else:
                        self.board[x].append([])
                        self.board[x][y] = "X"

    def possible_moves(self):
        '''
        :return:
        List: list of possible user inputs
        '''
        return ["w", "a", "s", "d"]

    def make_move(self, move):
        '''
        Move through the board and save the current player field
        :param move:
        string: Players move
        '''
        player_index = self.current_player - 1
        prev_pos = self.players_position[player_index]
        self.board[prev_pos[0]][prev_pos[1]] = "X"

        match move:
            case "w":
                self.players_position[player_index][1] = self.check_y_axis_bounds(
                    self.players_position[player_index][1] - 1)
            case "a":
                self.players_position[player_index][0] = self.check_x_axis_bounds(
                    self.players_position[player_index][0] - 1)
            case "s":
                self.players_position[player_index][1] = self.check_y_axis_bounds(
                    self.players_position[player_index][1] + 1)
            case "d":
                self.players_position[player_index][0] = self.check_x_axis_bounds(
                    self.players_position[player_index][0] + 1)

        curr_pos = self.players_position[player_index]
        self.current_field = self.board[curr_pos[0]][curr_pos[1]]

    def check_x_axis_bounds(self, pos):
        '''
        Changes player position to the opposite side of the playing field in case
        of player moves out of bounds in the X axis
        :param pos:
        int: players current position o X axis
        :return:
        int: X axis coordinates for players new position
        '''
        if pos < 0:
            return 7
        elif pos > 7:
            return 0
        return pos

    def check_y_axis_bounds(self, pos):
        '''
        Changes player position to the opposite side of the playing field in case
        of player moves out of bounds in the Y axis
        :param pos:
        int: players current position o Y axis
        :return:
        int: Y axis coordinates for players new position
        '''
        if pos < 0:
            return 5
        elif pos > 5:
            return 0
        return pos

    def win(self):
        '''
        Check if current player is on the allowed field.
        :return:
        boolean: true if player isn't on the field 'O'
        '''
        return self.current_field != "O"

    def is_over(self):
        '''
        Check if the game should continue
        :return:
        boolean: result of win method
        '''
        return self.win()  # Game stops when someone wins.

    def show(self):
        '''
        Draw the board
        '''
        for y in range(6):
            print()
            for x in range(8):
                if x == self.players_position[0][0] and y == self.players_position[0][1]:
                    print(Fore.BLUE + "I", end="")
                elif x == self.players_position[1][0] and y == self.players_position[1][1]:
                    print(Fore.RED + "I", end="")
                else:
                    print(Fore.WHITE + self.board[x][y], end="")

        print(Fore.WHITE)

    def scoring(self):
        '''
        Gives a score to the current game
        :return:
        int: if AI wins return 100, if losses return 0
        '''
        return 100 if self.win() else 0  # For the AI


# Start a match (and store the history of moves when it ends)
ai = Negamax(13)  # The AI will think 13 moves in advance
game = IsolationGame([Human_Player(), AI_Player(ai)])
history = game.play()
