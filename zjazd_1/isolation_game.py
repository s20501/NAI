from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax
from colorama import Fore


class IsolationGame(TwoPlayerGame):

    def __init__(self, players=None):
        self.players = players
        self.board = [[]]
        self.current_player = 1
        self.players_position = [[0, 2], [7, 3]]
        self.current_field = "O"

        # fill board
        for x in range(8):
            self.board.append([])
            for y in range(6):
                self.board[x].append([])
                self.board[x][y] = "O"

    def possible_moves(self): return ["w", "a", "s", "d"]

    # Move through the board and save the current player field
    def make_move(self, move):
        player_index = self.current_player - 1
        match move:
            case "w":
                self.players_position[player_index][1] = self.players_position[player_index][1] - 1
            case "a":
                self.players_position[player_index][0] = self.players_position[player_index][0] - 1
            case "s":
                self.players_position[player_index][1] = self.players_position[player_index][1] + 1
            case "d":
                self.players_position[player_index][0] = self.players_position[player_index][0] + 1

    def win(self): self.current_field != "O"

    def is_over(self): return self.win()  # Game stops when someone wins.

    # Draw the board
    def show(self):
        for y in range(6):
            print()
            for x in range(8):
                if x == self.players_position[0][0] and y == self.players_position[0][1]:
                    print(Fore.BLUE + "I", end="")
                elif x == self.players_position[1][0] and y == self.players_position[1][1]:
                    print(Fore.RED + "I", end="")
                else:
                    print(Fore.WHITE + self.board[x][y], end="")

        print()

    def scoring(self): return 100 if self.win() else 0  # For the AI


# Start a match (and store the history of moves when it ends)
ai = Negamax(13)  # The AI will think 13 moves in advance
game = IsolationGame([Human_Player(), AI_Player(ai)])
history = game.play()
