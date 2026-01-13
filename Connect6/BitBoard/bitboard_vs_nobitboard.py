import random
import numpy as np
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet

width, height, n_in_row = 10, 10, 6
n_playout = 800
use_gpu = False
N_GAMES = 5

random.seed(42)
np.random.seed(42)

policy_bit = PolicyValueNet(width, height, model_file="model/10_10_6_best_policy_myrun_bitboard.model", use_gpu=use_gpu)
policy_nobit = PolicyValueNet(width, height, model_file="model/10_10_6_best_policy_myrun_nobitboard.model", use_gpu=use_gpu)

agent_bit = MCTSPlayer(policy_bit.policy_value_fn, c_puct=5, n_playout=n_playout)
agent_nobit = MCTSPlayer(policy_nobit.policy_value_fn, c_puct=5, n_playout=n_playout)

bit_win, nobit_win, draw = 0, 0, 0

for i in range(N_GAMES):
    board = Board(width=width, height=height, n_in_row=n_in_row)
    game = Game(board)
    # Đảo bên đi trước mỗi trận
    if i % 2 == 0:
        winner = game.start_play(agent_bit, agent_nobit, start_player=0, is_shown=0)
        real_winner = winner
    else:
        winner = game.start_play(agent_nobit, agent_bit, start_player=0, is_shown=0)
        # Đảo lại để bit_win luôn là Bitboard bất kể vị trí chơi
        real_winner = 2 if winner == 1 else 1 if winner == 2 else winner
    if real_winner == 1:
        bit_win += 1
    elif real_winner == 2:
        nobit_win += 1
    else:
        draw += 1
    print(f"Ván {i+1}: Winner = {real_winner}")

print(f"\nKết quả sau {N_GAMES} trận:")
print(f"Bitboard AI thắng: {bit_win}")
print(f"Không Bitboard AI thắng: {nobit_win}")
print(f"Hòa: {draw}")
