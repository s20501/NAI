from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax
from colorama import Fore


class IsolationGame(TwoPlayerGame):

    def __init__(self, players=None):
        self.players = players
        self.board = [[]]
        self.current_player = 1
        self.player1_position = [0, 2]
        self.player2_position = [7, 2]

        # fill board
        for x in range(8):
            self.board.append([])
            for y in range(6):
                self.board[x].append([])
                self.board[x][y] = "O"

    def possible_moves(self): return ['w', 'a', 's', 'd']

    def make_move(self, move): self.pile -= int(move)  # remove bones.

    def win(self): return False

    def is_over(self): return self.win()  # Game stops when someone wins.

    def show(self):
        for y in range(6):
            print()
            for x in range(8):
                if x == self.player1_position[0] and y == self.player1_position[1]:
                    print(Fore.BLUE + 'I', end="")
                elif x == self.player2_position[0] and y == self.player2_position[1]:
                    print(Fore.RED + 'I', end="")
                else:
                    print(Fore.WHITE + self.board[x][y], end="")

    def scoring(self): return 100 if self.win() else 0  # For the AI


# Start a match (and store the history of moves when it ends)
ai = Negamax(13)  # The AI will think 13 moves in advance
game = IsolationGame([Human_Player(), AI_Player(ai)])
history = game.play()
