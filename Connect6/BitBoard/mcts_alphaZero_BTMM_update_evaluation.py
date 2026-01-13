import numpy as np
import copy
from btmm_update_evaluation import BradleyTerryMM 
import torch
import time

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value, flag):
        if self._parent:
            self._parent.update_recursive(-leaf_value, 1 - flag)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS_BTMM(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000, time_limit=None):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._time_limit = time_limit

        self.playout_count_move = 0
        self.playout_count_total = 0

    def _playout(self, state):
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, leaf_value = self._policy(state)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            leaf_value = 0.0 if winner == -1 else (1.0 if winner == state.get_current_player() else -1.0)

        node.update_recursive(-leaf_value if state.chesses == 2 else leaf_value, state.chesses != 2)

    def get_move_probs(self, state, temp=1e-3, game_id=None, move_id=None, log_btmm=False):
        real_game_id = game_id
        real_move_id = move_id
        if real_game_id is None and hasattr(self, "game_id"):
            real_game_id = self.game_id
        if real_move_id is None and hasattr(self, "move_id"):
            real_move_id = self.move_id

        self.playout_count_move = 0

        n = 0
        if self._time_limit is not None:
            start_time = time.time()
            while time.time() - start_time < self._time_limit:
                state_copy = copy.deepcopy(state)
                self._playout(state_copy)
                n += 1
           
                self.playout_count_move += 1
        else:
            for _ in range(self._n_playout):
                state_copy = copy.deepcopy(state)
                self._playout(state_copy)
               
                self.playout_count_move += 1

   
        self.playout_count_total += self.playout_count_move
        print(f"[BTMM] Playout this move = {self.playout_count_move} | Total = {self.playout_count_total}")

        moves_features = {act: self.extract_features(state, act, state.get_current_player())
                        for act in self._root._children}
        win_counts = {act: node._Q for act, node in self._root._children.items()}
        total_counts = {(act, other_act): node._n_visits + other_node._n_visits
                        for act, node in self._root._children.items()
                        for other_act, other_node in self._root._children.items() if act != other_act}

        btmm = BradleyTerryMM(moves_features,
                            player=state.get_current_player(),
                            game_id=real_game_id,
                            move_id=real_move_id)
        btmm.update_gamma(win_counts, total_counts)
        btmm_probs = btmm.get_probs()
        
        btmm_metrics = btmm.get_explanation_metrics()
        
        if log_btmm and real_game_id is not None:
            btmm.append_log_to_file("logs/btmm_gamma_log.csv")
      
            btmm.save_evaluation_log("logs/btmm_evaluation_log.csv")

        acts, probs = zip(*btmm_probs.items())
        probs_cpu = np.array([float(p) for p in probs], dtype=np.float64)
        
   
        if np.sum(probs_cpu) == 0 or np.any(np.isnan(probs_cpu)) or np.any(probs_cpu < 0):
            print(f"[WARNING] Invalid probabilities: sum={np.sum(probs_cpu)}, min={np.min(probs_cpu) if len(probs_cpu) > 0 else 'N/A'}")
            probs_cpu = np.ones_like(probs_cpu) / len(probs_cpu)
        
    
        probs_cpu = np.maximum(probs_cpu, 1e-10)  # Đặt giá trị tối thiểu
        probs_cpu = probs_cpu / np.sum(probs_cpu)  # Chuẩn hóa lại
        

        if temp < 1e-6:
            temp = 1e-3 
            
        try:
            act_probs = softmax(1.0 / temp * np.log(probs_cpu + 1e-10))
        except Exception as e:
            print(f"[ERROR] Softmax failed: {e}, using uniform distribution")
            act_probs = np.ones_like(probs_cpu) / len(probs_cpu)
        
        if np.any(np.isnan(act_probs)):
            print("[WARNING] NaN in act_probs, using uniform distribution")
            act_probs = np.ones_like(act_probs) / len(act_probs)
            
        return acts, act_probs
        

    def extract_features(self, board, move, player):
        directions = [1, board.width, board.width+1, board.width-1]
        features = {}
        bitboard = board.bitboards[player] | (1 << move)

        for d in directions:
            count = 1
            for offset in [1, -1]:
                idx = move
                while True:
                    idx += offset * d
                    if idx < 0 or idx >= board.width * board.height:
                        break
                    if not ((bitboard >> idx) & 1):
                        break
                    count += 1
            features[f"dir_{d}"] = count
            
        features["board_center_distance"] = self._distance_to_center(board, move)
        features["opponent_proximity"] = self._opponent_proximity(board, move, player)
        features["potential_threats"] = self._count_potential_threats(board, move, player)
        
        return features

    def _distance_to_center(self, board, move):
        """Tính khoảng cách đến trung tâm bàn cờ"""
        center_x = board.width / 2
        center_y = board.height / 2
        x = move % board.width
        y = move // board.width
        return abs(x - center_x) + abs(y - center_y)

    def _opponent_proximity(self, board, move, player):
        """Tính khoảng cách đến quân đối thủ gần nhất"""
        opponent = 3 - player  
        opponent_bitboard = board.bitboards[opponent]
        
        min_distance = float('inf')
        x1 = move % board.width
        y1 = move // board.width
        
        # Duyệt qua tất cả các quân đối thủ
        for opponent_move in range(board.width * board.height):
            if (opponent_bitboard >> opponent_move) & 1:
                x2 = opponent_move % board.width
                y2 = opponent_move // board.width
                distance = abs(x1 - x2) + abs(y1 - y2)
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else board.width + board.height

    def _count_potential_threats(self, board, move, player):
        """Đếm số lượng mối đe dọa tiềm năng"""
        threats = 0
        directions = [1, board.width, board.width+1, board.width-1]
        
        for d in directions:
            # Kiểm tra cả 2 hướng
            for offset in [1, -1]:
                consecutive = 0
                idx = move
                for step in range(4):  
                    idx += offset * d
                    if idx < 0 or idx >= board.width * board.height:
                        break
                    if (board.bitboards[player] >> idx) & 1:
                        consecutive += 1
                    else:
                        break
                if consecutive >= 3:
                    threats += 1
        
        return threats

class MCTSPlayerBTMM(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000, use_rws=True, time_limit=None):
        self.mcts = MCTS_BTMM(policy_value_fn, c_puct, n_playout, time_limit=time_limit)
        self.use_rws = use_rws
        self.time_limit = time_limit 

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts._root = TreeNode(None, 1.0)
        self.mcts.playout_count_total = 0

    def get_action(self, board, temp=1e-3, return_prob=0, game_id=None, move_id=None):
        if game_id is None and hasattr(self, "game_id"):
            game_id = self.game_id
        self.mcts.game_id = game_id
        self.mcts.move_id = move_id   
        acts, probs = self.mcts.get_move_probs(board, temp, game_id=game_id, move_id=move_id, log_btmm=True)
        move_probs = np.zeros(board.width * board.height)
        move_probs[list(acts)] = probs

        if self.use_rws:
            move = np.random.choice(acts, p=probs)
        else:
            move = acts[np.argmax(probs)]
        self.mcts._root = TreeNode(None, 1.0)
        return (move, move_probs) if return_prob else move

    def __str__(self):
        return "MCTS_BTMM Player"