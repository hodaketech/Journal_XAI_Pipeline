import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.q_table[state])

    def learn(self, s, a, r, s_, done):
        q_predict = self.q_table[s][a]
        q_target = r if done else r + self.gamma * np.max(self.q_table[s_])
        self.q_table[s][a] += self.alpha * (q_target - q_predict)

class drQAgent:
    def __init__(self, n_states, n_actions):
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.n_actions = n_actions
        self.q_goal = np.zeros((n_states, n_actions))
        self.q_cliff = np.zeros((n_states, n_actions))
        self.q_step = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        total_q = self.q_goal[state] + self.q_cliff[state] + self.q_step[state]
        return np.argmax(total_q)

    def learn(self, s, a, r, s_, done):
        rg, rc, rs = self.decompose_reward(r)
        for q, r_comp in zip([self.q_goal, self.q_cliff, self.q_step], [rg, rc, rs]):
            q_predict = q[s][a]
            q_target = r_comp if done else r_comp + self.gamma * np.max(q[s_])
            q[s][a] += self.alpha * (q_target - q_predict)

    def decompose_reward(self, r):
        if r == 1:
            return 1, 0, 0
        elif r == -100:
            return 0, -100, 0
        else:
            return 0, 0, -1

class HRAAgent(drQAgent):
    def choose_action(self, state):
        weights = [0.5, 0.3, 0.2]  # goal, cliff, step
        total_q = (weights[0] * self.q_goal[state] +
                   weights[1] * self.q_cliff[state] +
                   weights[2] * self.q_step[state])
        return np.argmax(total_q)
