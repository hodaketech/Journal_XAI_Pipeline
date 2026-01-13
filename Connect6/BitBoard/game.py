# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
@modifier: Junguang Jiang
@bitboard-refactor: ChatGPT July 2025
"""

from __future__ import print_function
import numpy as np
import copy

class Board(object):
    """board for the game - Bitboard version"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.n_in_row = int(kwargs.get('n_in_row', 6))
        self.players = [1, 2]  # player1 and player2
        self.chesses = 1  # Số quân còn được đặt ở lượt này
        self.last_moves = []  # Các bước đi của lượt trước
        self.curr_moves = []  # Các bước đi của lượt hiện tại
        # === BITBOARD ===
        self.bitboards = {1: 0, 2: 0}  
        self._all_bits = (1 << (self.width * self.height)) - 1  # mask toàn bộ bàn cờ

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]
        self.bitboards = {1: 0, 2: 0}
        self.last_move = -1
        self.chesses = 1
        self.last_moves = []
        self.curr_moves = []
        self._all_bits = (1 << (self.width * self.height)) - 1

    def move_to_location(self, move):
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """Trả về trạng thái bàn cờ cho current player dưới dạng 4xW x H"""
        square_state = np.zeros((4, self.width, self.height))
        for idx in range(self.width * self.height):
            h = idx // self.width
            w = idx % self.width
            if (self.bitboards[self.current_player] >> idx) & 1:
                square_state[0][h, w] = 1.0
            if (self.bitboards[3 - self.current_player] >> idx) & 1:
                square_state[1][h, w] = 1.0
        for move in self.last_moves:
            square_state[2][move // self.width, move % self.width] = 1.0
        for move in self.curr_moves:
            square_state[3][move // self.width, move % self.width] = 1.0
        return square_state[:, ::-1, :]

    def do_move(self, move):
        """Đặt quân cho current_player"""
        move = int(move)
        bit = 1 << move
        assert (self.bitboards[1] & bit == 0) and (self.bitboards[2] & bit == 0), "Ô đã bị chiếm"
        self.bitboards[self.current_player] |= bit
        self.last_move = move
        self.curr_moves.append(move)
        self.chesses -= 1
        if self.chesses == 0:
            self._change_turn()
            self.chesses = 2
        return self.chesses

    def _change_turn(self):
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        self.last_moves = copy.deepcopy(self.curr_moves)
        self.curr_moves.clear()

    def availables(self):
        """Trả về list các ô còn trống (dạng index 0..N-1)"""
        occupied = self.bitboards[1] | self.bitboards[2]
        empty = self._all_bits & (~occupied)
        return [i for i in range(self.width * self.height) if (empty >> i) & 1]

    def has_a_winner(self):
        width, height, n = self.width, self.height, self.n_in_row

        moved = [i for i in range(width * height) if ((self.bitboards[1] | self.bitboards[2]) >> i) & 1]
        if len(moved) < self.n_in_row + 2:
            return False, -1

        # Kiểm tra từng người chơi
        for player in self.players:
            bb = self.bitboards[player]
            if self._bitboard_has_n_in_row(bb, width, height, n):
                return True, player
        return False, -1

    def _bitboard_has_n_in_row(self, bb, width, height, n):
        """Kiểm tra bitboard bb có n quân liên tiếp không (ngang, dọc, chéo, chéo ngược)"""
        directions = [1, width, width+1, width-1]  # ngang, dọc, chéo xuôi, chéo ngược
        for d in directions:
            m = bb
            for i in range(1, n):
                m = m & (bb >> (d * i))
            if m != 0:
                return True
        return False

    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif len(self.availables()) == 0:
            return True, -1  # hòa
        return False, -1

    def get_current_player(self):
        return self.current_player

    def is_start(self):
        occupied = self.bitboards[1] | self.bitboards[2]
        return occupied == 0

    def __str__(self):
        return f"{self.height}_{self.width}_{self.n_in_row}"


    def get_player_at(self, pos):
        """Trả về player ở ô pos (0 nếu trống)"""
        if (self.bitboards[1] >> pos) & 1:
            return 1
        if (self.bitboards[2] >> pos) & 1:
            return 2
        return 0

class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Vẽ bàn cờ và in trạng thái"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.get_player_at(loc)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3, game_id=None):
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players, is_need_changes = [], [], [], []
        move_id = 0
        while True:
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            is_need_change = self.board.do_move(move)
            is_need_changes.append(is_need_change)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
            move_id += 1
