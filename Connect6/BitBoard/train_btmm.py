from __future__ import print_function
import random
import numpy as np
import os
import torch
from collections import deque
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from mcts_alphaZero_BTMM import MCTSPlayerBTMM
from policy_value_net_pytorch import PolicyValueNet 

class TrainPipeline():
    def __init__(self, init_model=None, board_width=6, board_height=6,
                 n_in_row=4, n_playout=400, use_gpu=False, is_shown=False,
                 output_file_name="", game_batch_number=1500,
                 use_rws=False, pure_use_rws=False, time_limit=None):
        self.time_limit = time_limit 
        self.board_width = board_width
        self.board_height = board_height
        self.n_in_row = n_in_row
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        self.learn_rate = 2e-3 
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = n_playout
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5
        self.kl_targ = 0.02
        self.check_freq = 1
        self.game_batch_num = game_batch_number
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 1000
        self.use_gpu = use_gpu
        self.is_shown = is_shown
        self.output_file_name = output_file_name
        self.use_rws = use_rws           
        self.pure_use_rws = pure_use_rws 
        self.policy_value_net = PolicyValueNet(self.board_width,
                                               self.board_height,
                                               model_file=init_model,
                                               use_gpu=self.use_gpu
                                               )
        self.mcts_player = MCTSPlayerBTMM(self.policy_value_net.policy_value_fn,
                                        c_puct=self.c_puct,
                                        n_playout=self.n_playout,
                                        use_rws=True, time_limit=self.time_limit)
        self.game_id = None
        self.current_iter = 0 

    def get_equi_data(self, play_data):
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """
        Thu thập dữ liệu tự chơi cho n_games ván, đảm bảo truyền đúng game_id cho từng ván.
        """
        for i in range(n_games):
            game_id = i + 1 
            board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
            game = Game(board)
            mcts_player = MCTSPlayerBTMM(
                self.policy_value_net.policy_value_fn,
                c_puct=self.c_puct,
                n_playout=self.n_playout,
                use_rws=True,
                time_limit=self.time_limit
            )
            mcts_player.game_id = game_id 
            game.game_id = game_id
            winner, play_data = game.start_self_play(
                mcts_player,
                temp=self.temp,
                game_id=game_id
            )

            play_data = list(play_data)
            self.episode_len = len(play_data)
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)



    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:
                break
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        current_mcts_player = MCTSPlayerBTMM(self.policy_value_net.policy_value_fn,
                                            c_puct=self.c_puct,
                                            n_playout=self.n_playout,
                                            use_rws=True,
                                            time_limit=self.time_limit)

        pure_mcts_player = MCTS_Pure(c_puct=5,
                                    n_playout=self.pure_mcts_playout_num,
                                    use_rws=self.pure_use_rws)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                        pure_mcts_player,
                                        start_player=i % 2,
                                        is_shown=self.is_shown)
            win_cnt[winner] += 1
            print(f"[Ván {i+1}] Kết quả: {winner}")
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("Số lượt mô phỏng:{}, Thắng: {}, Thua: {}, Hòa:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("info", exist_ok=True)

        loss_log_path = "info/" + str(self.board) + "_loss_" + self.output_file_name + ".txt"
        win_ratio_log_path = "info/" + str(self.board) + "_win_ration" + self.output_file_name + ".txt"
        if self.current_iter == 0:
            with open(loss_log_path, 'w', encoding="utf-8") as loss_file:
                loss_file.write("Số lần tự chơi,loss,entropy\n")
            with open(win_ratio_log_path, 'w', encoding="utf-8") as win_ratio_file:
                win_ratio_file.write("Số lần tự chơi, Số lượt mô phỏng pure_MCTS, Tỷ lệ thắng\n")

        try:
            for i in range(self.current_iter, self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("Ván tự chơi i:{}, Số bước đã đi:{}".format(i+1, self.episode_len))

                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    with open(loss_log_path, 'a', encoding='utf-8') as loss_file:
                        loss_file.write(f"{i+1},{loss},{entropy}\n")

                if (i+1) % self.check_freq == 0:
                    print("Số ván tự chơi hiện tại: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    with open(win_ratio_log_path, 'a', encoding='utf-8') as win_ratio_file:
                        win_ratio_file.write(f"{i+1},{self.pure_mcts_playout_num},{win_ratio}\n")

                    self.policy_value_net.save_model('./model/' + str(self.board_height)
                                                    + '_' + str(self.board_width)
                                                    + '_' + str(self.n_in_row) +
                                                    '_current_policy_' + self.output_file_name + '.model')
                    if win_ratio >= self.best_win_ratio:
                        print("Đã tạo ra chiến lược tốt hơn!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model('./model/' + str(self.board_height)
                                                        + '_' + str(self.board_width)
                                                        + '_' + str(self.n_in_row) +
                                                        '_best_policy_' + self.output_file_name + '.model')
                        if (self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 50000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                self.current_iter = i + 1
                if (i + 1) % 1 == 0:
                    self.save_checkpoint(f"checkpoints/session_{i+1}.pth")

        except KeyboardInterrupt:
            print('\n\rDừng lại bởi người dùng')

        loss_file.close()
        win_ratio_file.close()

    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.policy_value_net.get_policy_param(),
            'optimizer_state_dict': self.policy_value_net.optimizer.state_dict(),
            'data_buffer': list(self.data_buffer),
            'current_iter': self.current_iter,
            'best_win_ratio': self.best_win_ratio,
        }
        torch.save(checkpoint, filename)
        print(f"[INFO] Đã lưu checkpoint: {filename}")

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')
        self.policy_value_net.policy_value_net.load_state_dict(checkpoint['model_state_dict'])
        self.policy_value_net.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.data_buffer = deque(checkpoint['data_buffer'], maxlen=self.buffer_size)
        self.current_iter = checkpoint.get('current_iter', 0)
        self.best_win_ratio = checkpoint.get('best_win_ratio', 0.0)
        print(f"[INFO] Đã load checkpoint: {filename}, tiếp tục từ ván {self.current_iter + 1}")


def usage():
    print("-s Thiết lập kích thước bàn cờ, mặc định là 6")
    print("-r Thiết lập số quân liên tiếp để thắng, mặc định là 4")
    print("-m Thiết lập số lượt mô phỏng MCTS mỗi nước đi, mặc định là 400")
    print("-o Định danh file lưu model đã huấn luyện (lưu ý: chương trình sẽ tự động tạo tên file dựa trên tham số model)")
    print("-n Thiết lập số ván tự chơi để huấn luyện, mặc định là 1500")
    print("--use_gpu Sử dụng GPU để huấn luyện")
    print("--graphics Hiển thị giao diện khi đánh giá mô hình")
    print("--rws      Bật chế độ RWS cho agent AlphaZero")
    print("--pure_rws Bật chế độ RWS cho agent Pure MCTS")

if __name__ == '__main__':
    import sys, getopt
    import os

    height = 10
    width = 10
    n_in_row = 6
    use_gpu = False
    n_playout = 800
    is_shown = False
    output_file_name = ""
    game_batch_number = 1500
    init_model_name = None
    battle = False
    use_rws = False        
    pure_use_rws = False   
    resume = False
    resume_checkpoint_file = None
    time_limit = None


    opts, args = getopt.getopt(
    sys.argv[1:], "hs:r:m:go:n:i:",
    ["use_gpu", "graphics", "rws", "pure_rws", "btmm", "time_limit="]
)


    for op, value in opts:
        if op == "-h":
            usage()
            sys.exit()
        elif op == "-s":
            height = width = int(value)
        elif op == "-r":
            n_in_row = int(value)
        elif op == "--use_gpu":
            use_gpu = True
        elif op == "-m":
            n_playout = int(value)
        elif op == "-g" or op == "--graphics":
            is_shown = True
        elif op == "-o":
            output_file_name = value
        elif op == "-i":
            init_model_name = value
        elif op == "-n":
            game_batch_number = int(value)
        elif op == "--rws":          # RWS PATCH
            use_rws = True
        elif op == "--pure_rws":     # RWS PATCH
            pure_use_rws = True
        elif op == "--resume":
            resume = True
            resume_checkpoint_file = value
        elif op == "--time_limit":
            time_limit = float(value)


    training_pipeline = TrainPipeline(
        board_height=height, board_width=width,
        n_in_row=n_in_row, use_gpu=use_gpu,
        n_playout=n_playout, is_shown=is_shown,
        output_file_name=output_file_name,
        init_model=init_model_name,
        game_batch_number=game_batch_number,
        use_rws=use_rws,
        pure_use_rws=pure_use_rws,
        time_limit=time_limit 
    )
    if resume and resume_checkpoint_file and os.path.exists(resume_checkpoint_file):
        training_pipeline.load_checkpoint(resume_checkpoint_file)
    training_pipeline.run()
