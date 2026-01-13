import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import pandas as pd 
import os  

class RewardPathFinder:
    def __init__(self, grid_size, maze_file):
        self.grid_size = grid_size
        self.start = (0, 0)
        self.end = (grid_size - 1, grid_size - 1)
        self.blocked_points = self.load_maze(maze_file)
        
        self.reward_components = ['turn', 'goal', 'blocked', 'safe']
        self.num_components = len(self.reward_components)
        
        self.q_table = np.zeros((self.num_components, grid_size, grid_size, 4))
        
        self.epsilon = 1.0
        self.alpha = 0.1
        self.gamma = 0.9
        self.best_path = []
        self.consecutive_safe_actions = 0

    def load_maze(self, maze_file):
        blocked_points = []
        with open(maze_file, 'r') as file:
            for i, line in enumerate(file):
                for j, char in enumerate(line.strip().split()):
                    if char == '#':
                        blocked_points.append((i, j))
        return blocked_points

    def reset(self):
        return self.start

    def is_terminal(self, state):
        return state == self.end

    def get_actions(self, state):
        actions = []
        for action in range(4):
            next_state = self.take_action(state, action)
            if 0 <= next_state[0] < self.grid_size and 0 <= next_state[1] < self.grid_size:
                actions.append(action)
        return actions

    def take_action(self, state, action):
        if action == 0:  # up
            return (state[0] - 1, state[1])
        elif action == 1:  # down
            return (state[0] + 1, state[1])
        elif action == 2:  # left
            return (state[0], state[1] - 1)
        elif action == 3:  # right
            return (state[0], state[1] + 1)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.get_actions(state))
        else:
            total_q = np.sum(self.q_table[:, state[0], state[1], :], axis=0)
            return np.argmax(total_q)

    def compute_decomposed_rewards(self, state, action, next_state, last_action):
        rewards = {c: 0 for c in self.reward_components}
        if last_action is not None and (
            (action == 0 and last_action == 1) or (action == 1 and last_action == 0) or
            (action == 2 and last_action == 3) or (action == 3 and last_action == 2)):
            rewards['turn'] = -1
        if self.is_terminal(next_state):
            rewards['goal'] = 30
        if next_state in self.blocked_points:
            rewards['blocked'] = -1
            self.consecutive_safe_actions = 0
        else:
            for a in range(4):
                future_state = self.take_action(next_state, a)
                if (0 <= future_state[0] < self.grid_size and 0 <= future_state[1] < self.grid_size and
                    future_state in self.blocked_points):
                    rewards['blocked'] = -0.5
                    break
            self.consecutive_safe_actions += 1
            if self.consecutive_safe_actions == 2:
                rewards['safe'] = 2
                self.consecutive_safe_actions = 0
        return rewards

    def update_q_table(self, state, action, rewards, next_state):
        total_q_next = np.sum(self.q_table[:, next_state[0], next_state[1], :], axis=0)
        best_next_action = np.argmax(total_q_next)
        for c_idx, component in enumerate(self.reward_components):
            r_c = rewards[component]
            q_c_next = self.q_table[c_idx, next_state[0], next_state[1], best_next_action]
            td_target = r_c + self.gamma * q_c_next
            td_delta = td_target - self.q_table[c_idx, state[0], state[1], action]
            self.q_table[c_idx, state[0], state[1], action] += self.alpha * td_delta

    def reward_difference_explanation(self, state, chosen_action, alternative_action):
        q_chosen = self.q_table[:, state[0], state[1], chosen_action]
        q_alternative = self.q_table[:, state[0], state[1], alternative_action]
        q_diff = q_chosen - q_alternative
        
        explanation = {}
        for c_idx, component in enumerate(self.reward_components):
            explanation[component] = q_diff[c_idx]
        
        total_diff = np.sum(q_diff)
        return explanation, total_diff

    def minimal_sufficient_explanation(self, state, chosen_action, alternative_action):
        explanation, total_diff = self.reward_difference_explanation(state, chosen_action, alternative_action)
        
        d = sum(abs(diff) for comp, diff in explanation.items() if diff < 0)
        positive_components = [(comp, diff) for comp, diff in explanation.items() if diff > 0]
        positive_components.sort(key=lambda x: x[1], reverse=True)
        
        msx_plus = []
        current_sum = 0
        for comp, diff in positive_components:
            current_sum += diff
            msx_plus.append(comp)
            if current_sum > d:
                break
        
        if msx_plus:
            msx_plus_diffs = [explanation[comp] for comp in msx_plus]
            v = sum(msx_plus_diffs) - min(msx_plus_diffs)
        else:
            v = 0
        
        negative_components = [(comp, diff) for comp, diff in explanation.items() if diff < 0]
        negative_components.sort(key=lambda x: abs(x[1]), reverse=True)
        
        msx_minus = []
        current_sum = 0
        for comp, diff in negative_components:
            current_sum += -diff
            msx_minus.append(comp)
            if current_sum > v:
                break
        
        formula_details = {
            'd': d,
            'v': v,
            'msx_plus_sum': sum(explanation[comp] for comp in msx_plus) if msx_plus else 0,
            'msx_minus_sum': sum(-explanation[comp] for comp in msx_minus) if msx_minus else 0
        }
        
        return msx_plus, msx_minus, explanation, formula_details
    

    def train(self, episodes, log_random_episode=False):
        best_total_reward = -float('inf')
        for episode in range(episodes):
            state = self.reset()
            total_reward = 0
            current_path = [state]
            last_action = None
            self.consecutive_safe_actions = 0
            
            while not self.is_terminal(state):
                action = self.choose_action(state)
                next_state = self.take_action(state, action)
                current_path.append(next_state)
                
                rewards = self.compute_decomposed_rewards(state, action, next_state, last_action)
                total_reward += sum(rewards.values())
                self.update_q_table(state, action, rewards, next_state)
                
                state = next_state
                last_action = action
            
            if total_reward > best_total_reward:
                best_total_reward = total_reward
                self.best_path = current_path
                
            # Xuất dữ liệu ra file Excel trong episode cuối
            if episode == episodes - 1 and log_random_episode:
                print(f"\nLogging actions for the last episode (Episode {episode + 1}) to Excel file...")
                table_data = []
                
                # Duyệt qua tất cả các trạng thái từ (0, 0) đến (9, 9)
                for i in range(self.grid_size):  # i từ 0 đến 9
                    for j in range(self.grid_size):  # j từ 0 đến 9
                        state = (i, j)
                        # Sử dụng chính sách tối ưu thay vì choose_action
                        total_q = np.sum(self.q_table[:, state[0], state[1], :], axis=0)
                        chosen_action = np.argmax(total_q)
                        next_state = self.take_action(state, chosen_action)
                        
                        available_actions = self.get_actions(state)
                        if len(available_actions) > 1:
                            alternative_action = random.choice([a for a in available_actions if a != chosen_action])
                        else:
                            alternative_action = chosen_action
                        
                        msx_plus, msx_minus, explanation, formula_details = self.minimal_sufficient_explanation(state, chosen_action, 
                                                                                                                alternative_action)
                        
                        # Thêm dữ liệu vào bảng
                        row = {
                            "State": str(state),
                            "Chosen Action": chosen_action,
                            "Alternative Action": alternative_action,
                            "Δ_turn": round(explanation['turn'], 2),
                            "Δ_goal": round(explanation['goal'], 2),
                            "Δ_blocked": round(explanation['blocked'], 2),
                            "Δ_safe": round(explanation['safe'], 2),
                            "Total Δ": round(sum(explanation.values()), 2),
                            "d": round(formula_details['d'], 2),
                            "MSX+": str(msx_plus),
                            "Sum MSX+": round(formula_details['msx_plus_sum'], 2),
                            "v": round(formula_details['v'], 2),
                            "MSX-": str(msx_minus),
                            "Sum MSX-": round(formula_details['msx_minus_sum'], 2),
                            "Next State": str(next_state)
                        }
                        table_data.append(row)
                
                # Tạo DataFrame từ dữ liệu chính
                df = pd.DataFrame(table_data)
                
                # Lưu vào file Excel với nhiều sheet
                output_file = "episode_1000_log.xlsx"
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    # Lưu dữ liệu chính vào sheet 1
                    df.to_excel(writer, sheet_name='Episode Log', index=False)
                    
                    # Lưu 4 bảng Q-Values vào các sheet 2, 3, 4, 5
                    for c_idx, component in enumerate(self.reward_components):
                        q_table_component = self.q_table[c_idx]  # Q-Table của thành phần hiện tại
                        # Tạo DataFrame cho Q-Table
                        q_data = []
                        for i in range(self.grid_size):
                            for j in range(self.grid_size):
                                row = {
                                    "State": str((i, j)),
                                    "Q_Up": round(q_table_component[i, j, 0], 2),
                                    "Q_Down": round(q_table_component[i, j, 1], 2),
                                    "Q_Left": round(q_table_component[i, j, 2], 2),
                                    "Q_Right": round(q_table_component[i, j, 3], 2)
                                }
                                q_data.append(row)
                        q_df = pd.DataFrame(q_data)
                        # Lưu vào sheet tương ứng (sheet 2, 3, 4, 5)
                        q_df.to_excel(writer, sheet_name=f'Q_{component}', index=False)
                    
                    # Lưu bảng Q tổng hợp vào sheet 6
                    total_q_table = np.sum(self.q_table, axis=0)  # Tổng Q-Table của tất cả các thành phần
                    total_q_data = []
                    for i in range(self.grid_size):
                        for j in range(self.grid_size):
                            row = {
                                "State": str((i, j)),
                                "Q_Up": round(total_q_table[i, j, 0], 2),
                                "Q_Down": round(total_q_table[i, j, 1], 2),
                                "Q_Left": round(total_q_table[i, j, 2], 2),
                                "Q_Right": round(total_q_table[i, j, 3], 2)
                            }
                            total_q_data.append(row)
                    total_q_df = pd.DataFrame(total_q_data)
                    total_q_df.to_excel(writer, sheet_name='Q_Total', index=False)
                
                print(f"Data has been exported to {output_file}")
            
            print(f"Episode {episode + 1}: Total Reward: {total_reward}\n")
            if episode == episodes - 1:
                print("Final Decomposed Q-Table:")
                for c_idx, component in enumerate(self.reward_components):
                    print(f"Component: {component}")
                    for i in range(self.grid_size):
                        for j in range(self.grid_size):
                            print(f"State {i, j}: Q-Values: {self.q_table[c_idx, i, j]}")

    def get_shortest_path(self):
        state = self.start
        shortest_path = [state]
        visited = set()

        while state != self.end:
            visited.add(state)
            total_q = np.sum(self.q_table[:, state[0], state[1], :], axis=0)
            best_action = np.argmax(total_q)
            next_state = self.take_action(state, best_action)

            if next_state in visited or next_state in self.blocked_points or not (
                0 <= next_state[0] < self.grid_size and 0 <= next_state[1] < self.grid_size):
                break

            shortest_path.append(next_state)
            state = next_state

        return shortest_path

    def visualize(self, show_q_values=False):
        total_q = np.sum(self.q_table, axis=0)
        q_values = np.max(total_q, axis=2)
        plt.imshow(q_values, cmap='hot', interpolation='nearest')
        if show_q_values:
            plt.colorbar(label='Total Q-Values')
        else:
            plt.colorbar(label='Heatmap')
        plt.title('Heatmap of Actions' if not show_q_values else 'Total Q-Values Heatmap')
        
        shortest_path = self.get_shortest_path()
        if shortest_path:
            path_x, path_y = zip(*shortest_path)
            plt.plot(path_y, path_x, marker='o', color='cyan', label='Shortest Path')
        
        for point in self.blocked_points:
            plt.scatter(point[1], point[0], marker='s', color='purple', label='Blocked' 
                        if point == self.blocked_points[0] else "", zorder=2)
        
        plt.scatter(self.start[1], self.start[0], marker='o', color='green', label='Start', zorder=3)
        plt.scatter(self.end[1], self.end[0], marker='x', color='blue', label='End', zorder=3)

        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(self.grid_size))
        plt.yticks(np.arange(self.grid_size))
        plt.show()
        plt.close()

        if show_q_values:
            print("Decomposed Q-Values for each action:")
            for c_idx, component in enumerate(self.reward_components):
                print(f"Component: {component}")
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        print(f"State {i, j}: Q-Values: {self.q_table[c_idx, i, j]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rcMain.py <maze_file>")
        sys.exit(1)

    maze_file = sys.argv[1]
    grid_size = 10
    agent = RewardPathFinder(grid_size, maze_file)

    agent.train(1000, log_random_episode=True)

    agent.visualize(show_q_values=True)
    
# python drQ-backup.py maze10.txt
