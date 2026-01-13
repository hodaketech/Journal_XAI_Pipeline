import time
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet

width = 19     
height = 19
n_in_row = 6
n_playout = 400
model_file = 'model/19_19_6_best_policy_hi_bitboard.model'
use_gpu = False

# Số ván self-play để benchmark
N = 10

# Khởi tạo các đối tượng
board = Board(width=width, height=height, n_in_row=n_in_row)
game = Game(board)
policy = PolicyValueNet(width, height, model_file=model_file, use_gpu=use_gpu)
ai_player = MCTSPlayer(policy.policy_value_fn, c_puct=5, n_playout=n_playout)

all_step_times = []
all_nodes = 0
total_moves = 0
start_all = time.time()
for i in range(N):
    board.init_board()
    game_step_times = []
    game_nodes = 0
    moves = 0
    while not board.game_end()[0]:
        t1 = time.time()
        move = ai_player.get_action(board)
        t2 = time.time()
        board.do_move(move)
        game_step_times.append(t2 - t1)
        moves += 1
        game_nodes += n_playout
    all_step_times += game_step_times
    all_nodes += game_nodes
    total_moves += moves
    print(f"Ván {i+1}: {moves} bước, {sum(game_step_times):.2f}s")
end_all = time.time()
total_time = sum(all_step_times)

print("\n==== TỔNG HỢP KẾT QUẢ ====")
print(f"Tổng số bước: {total_moves}")
print(f"Thời gian trung bình mỗi bước: {total_time/total_moves:.4f} s")
print(f"Số bước/giây: {total_moves/total_time:.2f}")
print(f"Tổng nodes/giây: {all_nodes/total_time:.0f}")
print(f"Tổng thời gian {N} ván: {end_all-start_all:.2f} s")
