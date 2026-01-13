import numpy as np
import copy
import time

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    """A node in the MCTS tree."""

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
            if flag:
                self._parent.update_recursive(-leaf_value, 1 - flag)
            else:
                self._parent.update_recursive(leaf_value, 1 - flag)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000, time_limit=None):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._time_limit = time_limit

        #Đếm playouts
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
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = (1.0 if winner == state.get_current_player() else -1.0)
        if state.chesses == 2:
            node.update_recursive(-leaf_value, 0)
        else:
            node.update_recursive(leaf_value, 1)

    # def get_move_probs(self, state, temp=1e-3):
    #         n = 0
    #         if self._time_limit is not None:
    #             start_time = time.time()
    #             while time.time() - start_time < self._time_limit:
    #                 state_copy = copy.deepcopy(state)
    #                 self._playout(state_copy)
    #                 n += 1
    #         else:
    #             for n in range(self._n_playout):
    #                 state_copy = copy.deepcopy(state)
    #                 self._playout(state_copy)
    #         act_visits = [(act, node._n_visits)
    #                     for act, node in self._root._children.items()]
    #         acts, visits = zip(*act_visits)
    #         act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
    #         return acts, act_probs

    def get_move_probs(self, state, temp=1e-3):
            # === NEW: reset per-move counter ===
            self.playout_count_move = 0

            n = 0
            if self._time_limit is not None:
                start_time = time.time()
                while time.time() - start_time < self._time_limit:
                    state_copy = copy.deepcopy(state)
                    self._playout(state_copy)
                    n += 1
                    # === NEW: count every playout ===
                    self.playout_count_move += 1
            else:
                for n in range(self._n_playout):
                    state_copy = copy.deepcopy(state)
                    self._playout(state_copy)
                    # === NEW: count every playout ===
                    self.playout_count_move += 1

            # === NEW: accumulate total & print ===
            self.playout_count_total += self.playout_count_move
            print(f"[AlphaZero] Playout this move = {self.playout_count_move} | Total = {self.playout_count_total}")

            act_visits = [(act, node._n_visits)
                        for act, node in self._root._children.items()]
            acts, visits = zip(*act_visits)
            act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
            return acts, act_probs
    
    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTSPlayer(object):
    """AI player based on MCTS with RWS or Greedy selection"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0, use_rws=True, time_limit=None):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout, time_limit=time_limit)
        self._is_selfplay = is_selfplay
        self.use_rws = use_rws
        self.time_limit = time_limit

    def set_player_ind(self, p):
        self.player = p


    def reset_player(self):
        self.mcts.update_with_move(-1)
        self.mcts.playout_count_total = 0

    def get_action(self, board, temp=1e-3, return_prob=0, use_rws=None):
        sensible_moves = board.availables()
        move_probs = np.zeros(board.width * board.height)
        _use_rws = self.use_rws if use_rws is None else use_rws
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                self.mcts.update_with_move(move)
            else:
                if _use_rws:
                    move = np.random.choice(acts, p=probs)
                else:
                    move = acts[np.argmax(probs)]
                self.mcts.update_with_move(-1)
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "Alpha Zero MCTS {}".format(self.player)
