import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import time

class RewardPathFinder:
    def __init__(self, maze_file):
        self.blocked_points = self.load_maze(maze_file)
        self.grid_size = max(max(x for x, _ in self.blocked_points), max(y for _, y in self.blocked_points)) + 1
        self.start = (0, 0)
        self.end = (self.grid_size - 1, self.grid_size - 1)

        self.reward_components = ['turn', 'goal', 'blocked', 'safe']
        self.num_components = len(self.reward_components)
        self.q_table = np.zeros((self.num_components, self.grid_size, self.grid_size, 4))

        self.epsilon = 1.0
        self.alpha = 0.1
        self.gamma = 0.9
        self.best_path = []
        self.consecutive_safe_actions = 0

    def load_maze(self, maze_file):
        blocked_points = []
        maze = []

        with open(maze_file, 'r') as file:
            for i, line in enumerate(file):
                row = line.strip().split()
                maze.append(row)
                for j, char in enumerate(row):
                    if char == '#':
                        blocked_points.append((i, j))
                    elif char == 'S':
                        self.start = (i, j)
                    elif char == 'G':
                        self.end = (i, j)

        self.grid_size = max(len(maze), len(maze[0]))
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
        if action == 0: return (state[0] - 1, state[1])
        elif action == 1: return (state[0] + 1, state[1])
        elif action == 2: return (state[0], state[1] - 1)
        elif action == 3: return (state[0], state[1] + 1)

    def choose_action(self, state):
        if not (0 <= state[0] < self.grid_size and 0 <= state[1] < self.grid_size):
            return random.choice([0, 1, 2, 3])  # fallback an toàn

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
                if 0 <= future_state[0] < self.grid_size and 0 <= future_state[1] < self.grid_size and future_state in self.blocked_points:
                    rewards['blocked'] = -0.5
                    break
            self.consecutive_safe_actions += 1
            if self.consecutive_safe_actions == 2:
                rewards['safe'] = 2
                self.consecutive_safe_actions = 0
        return rewards

    def update_q_table(self, state, action, rewards, next_state):
        if not (0 <= next_state[0] < self.grid_size and 0 <= next_state[1] < self.grid_size):
            return
        if not (0 <= state[0] < self.grid_size and 0 <= state[1] < self.grid_size):
            return

        total_q_next = np.sum(self.q_table[:, next_state[0], next_state[1], :], axis=0)
        best_next_action = np.argmax(total_q_next)

        for c_idx, component in enumerate(self.reward_components):
            r_c = rewards[component]
            q_c_next = self.q_table[c_idx, next_state[0], next_state[1], best_next_action]
            td_target = r_c + self.gamma * q_c_next
            td_delta = td_target - self.q_table[c_idx, state[0], state[1], action]
            self.q_table[c_idx, state[0], state[1], action] += self.alpha * td_delta
    # Hàm lấy số lần chạy từ file
    def get_run_count(self, grid_size):
        # File lưu số lần chạy
        run_count_file = f"run_count_{grid_size}.txt"
        
        # Nếu file không tồn tại, khởi tạo số lần chạy là 0
        if not os.path.exists(run_count_file):
            with open(run_count_file, 'w') as f:
                f.write("0")
        
        # Đọc số lần chạy từ file
        with open(run_count_file, 'r') as f:
            run_count = int(f.read().strip())
        
        run_count += 1
        
        # Ghi lại số lần chạy mới vào file
        with open(run_count_file, 'w') as f:
            f.write(str(run_count))
        
        return run_count
    def prepare_output_file(self, run_count):
        output_dir = "excel-results"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"grid-{self.grid_size}-no-explain-{run_count}.xlsx")
        return output_file


    def train(self, episodes, log_random_episode=False):
        self.reward_history = []  # lưu tổng reward cho mỗi tập
        max_steps = self.grid_size * self.grid_size * 2

        for episode in range(episodes):
            state = self.reset()
            last_action = None
            self.consecutive_safe_actions = 0
            steps = 0
            total_reward = 0

            while not self.is_terminal(state) and steps < max_steps:
                action = self.choose_action(state)
                next_state = self.take_action(state, action)
                rewards = self.compute_decomposed_rewards(state, action, next_state, last_action)
                self.update_q_table(state, action, rewards, next_state)

                total_reward += sum(rewards.values())
                state = next_state
                last_action = action
                steps += 1

            self.reward_history.append(total_reward)
            self.epsilon = max(self.epsilon * 0.995, 0.05)

            print(f"Episode {episode + 1}: Total Reward: {total_reward}")


        self.plot_metrics()  
        self.export_qtables_to_excel()




    def export_qtables_to_excel(self, output_file=None):
        import pandas as pd
        import os
        os.makedirs("excel-results", exist_ok=True)
        if output_file is None:
            output_file = f"excel-results/non_msx_grid{self.grid_size}.xlsx"

        total_q_table = np.sum(self.q_table, axis=0)

        # --- Sheet: Episode Log ---
        table_data = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = (i, j)
                q_vals = total_q_table[i, j, :]
                total_q = np.sum(q_vals)
                best_action = np.argmax(q_vals)
                next_state = self.take_action(state, best_action)

                row = {
                    "State": str(state),
                    "Chosen Action": best_action,
                    "Q_Up": round(float(q_vals[0]), 2),
                    "Q_Down": round(float(q_vals[1]), 2),
                    "Q_Left": round(float(q_vals[2]), 2),
                    "Q_Right": round(float(q_vals[3]), 2),
                    "Total_Q": round(float(total_q), 2),
                    "Next State": str(next_state)
                }
                table_data.append(row)

        df = pd.DataFrame(table_data)

        # --- Ghi vào Excel ---
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            # Sheet 1: Episode Log
            df.to_excel(writer, sheet_name="Episode Log (no MSX)", index=False)

            # Sheet 2-5: Q_turn, Q_goal, Q_blocked, Q_safe
            for c_idx, component in enumerate(self.reward_components):
                q_data = []
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        row = {
                            "State": str((i, j)),
                            "Q_Up": round(float(self.q_table[c_idx, i, j, 0]), 2),
                            "Q_Down": round(float(self.q_table[c_idx, i, j, 1]), 2),
                            "Q_Left": round(float(self.q_table[c_idx, i, j, 2]), 2),
                            "Q_Right": round(float(self.q_table[c_idx, i, j, 3]), 2),
                        }
                        q_data.append(row)
                pd.DataFrame(q_data).to_excel(writer, sheet_name=f"Q_{component}", index=False)

            # Cuối cùng: Sheet Q_Total
            q_total_data = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    row = {
                        "State": str((i, j)),
                        "Q_Up": round(float(total_q_table[i, j, 0]), 2),
                        "Q_Down": round(float(total_q_table[i, j, 1]), 2),
                        "Q_Left": round(float(total_q_table[i, j, 2]), 2),
                        "Q_Right": round(float(total_q_table[i, j, 3]), 2),
                    }
                    q_total_data.append(row)
            pd.DataFrame(q_total_data).to_excel(writer, sheet_name="Q_Total", index=False)

        print(f"[✓] Excel file exported to: {output_file}")



    
    def get_shortest_path(self):
        from heapq import heappush, heappop

        visited = set()
        came_from = {}
        start = self.start
        end = self.end

        queue = []
        heappush(queue, (-np.max(np.sum(self.q_table[:, start[0], start[1], :], axis=0)), start))  # dùng giá trị Q lớn nhất

        while queue:
            _, current = heappop(queue)
            if current == end:
                break

            visited.add(current)
            total_q = np.sum(self.q_table[:, current[0], current[1], :], axis=0)
            for action in range(4):
                next_state = self.take_action(current, action)
                if (0 <= next_state[0] < self.grid_size and
                    0 <= next_state[1] < self.grid_size and
                    next_state not in self.blocked_points and
                    next_state not in visited):
                    came_from[next_state] = current
                    heappush(queue, (-total_q[action], next_state))

        # reconstruct path
        path = []
        current = end
        while current != start:
            if current not in came_from:
                return []  # Không tìm thấy đường
            path.append(current)
            current = came_from[current]
        path.append(start)
        return path[::-1]  # reverse path

    # Vẽ heatmap cho các hành động
    def visualize(self, show_q_values=False):
        total_q = np.sum(self.q_table, axis=0)
        q_values = np.max(total_q, axis=2)
        plt.imshow(q_values, cmap='hot', interpolation='nearest')
        if show_q_values:
            plt.colorbar(label='Total Q-Values')
        else:
            plt.colorbar(label='Heatmap')
        plt.title('Heatmap of Actions' if not show_q_values else 'Total Q-Values Heatmap')
        
        # Vẽ đường đi ngắn nhất
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
        plt.savefig(f"heatmap_qvalues_{self.grid_size}.png")  # ✅ Lưu heatmap
        plt.close()

        # In ra bảng Q-Values cho từng thành phần
        if show_q_values:
            print("Decomposed Q-Values for each action:")
            for c_idx, component in enumerate(self.reward_components):
                print(f"Component: {component}")
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        print(f"State {i, j}: Q-Values: {self.q_table[c_idx, i, j]}")
    def smooth(self, data, window=50):
        return np.convolve(data, np.ones(window)/window, mode='valid')


    def plot_metrics(self):
        episodes = list(range(1, len(self.reward_history) + 1))
        window = 50  # cửa sổ smoothing

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # --- 1. Reward ---
        axs[0].plot(episodes, self.reward_history, label='Total Reward per Episode', color='lightblue', alpha=0.8)
        if len(self.reward_history) >= window:
            smoothed = np.convolve(self.reward_history, np.ones(window)/window, mode='valid')
            axs[0].plot(episodes[len(episodes)-len(smoothed):], smoothed, label='Smoothed Reward', color='red', linewidth=2)

        axs[0].set_title('Learning Curve: Total Reward per Episode')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Total Reward')
        axs[0].legend()
        axs[0].grid(True)

        # --- 2. Q-Value Heatmap ---
        total_q = np.sum(self.q_table, axis=0)
        q_values = np.max(total_q, axis=2)
        im = axs[1].imshow(q_values, cmap='hot', interpolation='nearest')
        axs[1].set_title('Total Q-Values Heatmap')
        fig.colorbar(im, ax=axs[1], label='Total Q-Values')

        path = self.get_shortest_path()
        if path:
            path_x, path_y = zip(*path)
            axs[1].plot(path_y, path_x, marker='o', color='cyan', label='Shortest Path')

        for point in self.blocked_points:
            axs[1].scatter(point[1], point[0], color='purple', s=25)

        axs[1].scatter(self.start[1], self.start[0], color='green', label='Start')
        axs[1].scatter(self.end[1], self.end[0], color='blue', label='End')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(f"total_reward_{self.grid_size}.png", dpi=300)
        plt.show()




    def plot_policy_arrows(self):
        total_q = np.sum(self.q_table, axis=0)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks(np.arange(0, self.grid_size + 1, 1))
        ax.set_yticks(np.arange(0, self.grid_size + 1, 1))
        ax.set_xticklabels(np.arange(0, self.grid_size + 1, 1))  # x-axis 0 → 10
        ax.set_yticklabels(np.arange(0, self.grid_size + 1, 1))  # y-axis 0 → 10 # ✅ Gán nhãn từ trái sang phải là 0 → 10
        ax.grid(True)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_y = self.grid_size - 1 - i  # chuyển tọa độ để gốc (0,0) nằm trên trái

                if (i, j) in self.blocked_points:
                    ax.add_patch(plt.Rectangle((j, cell_y), 1, 1, color='black'))
                elif (i, j) == self.start:
                    ax.text(j + 0.5, cell_y + 0.5, 'S', ha='center', va='center', fontsize=12, color='green')
                elif (i, j) == self.end:
                    ax.text(j + 0.5, cell_y + 0.5, 'G', ha='center', va='center', fontsize=12, color='blue')
                else:
                    best_action = np.argmax(total_q[i, j, :])
                    dx, dy = [(0, 0.3), (0, -0.3), (-0.3, 0), (0.3, 0)][best_action]
                    ax.arrow(j + 0.5, cell_y + 0.5, dx, dy,
                            head_width=0.2, head_length=0.2, fc='red', ec='red')

        ax.set_title("Optimal Policy Arrows")
        plt.savefig(f"policy_{self.grid_size}.png", dpi=300)
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python drq_no_explain.py <maze_file>")
        sys.exit(1)
    
    maze_file = sys.argv[1]
    grid_size = 10
    agent = RewardPathFinder(maze_file)

    run_count = agent.get_run_count(agent.grid_size)
    output_file = agent.prepare_output_file(run_count)

    # Huấn luyện agent
    start_time = time.time()

    agent.train(3000, log_random_episode=True)

    


    # Hiển thị kết quả và lưu log
    agent.visualize(show_q_values=True)
    agent.plot_policy_arrows()
    
    agent.plot_metrics()
    print(f"Output file: {output_file}")
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[⏱] Training time: {elapsed:.2f} seconds")