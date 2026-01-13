import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import pandas as pd 
import os  
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import time

def generate_maze(grid_size, filename, density=0.25):
    while True:
        maze = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                if (i, j) == (0, 0) or (i, j) == (grid_size - 1, grid_size - 1):
                    row.append('.')
                else:
                    row.append('#' if random.random() < density else '.')
            maze.append(row)

        # Mở đường chéo từ (0,0) tới (n-1,n-1)
        for i in range(grid_size):
            maze[i][i] = '.'

        # Mở thêm một đường ngang ở giữa
        mid = grid_size // 2
        for j in range(grid_size):
            maze[mid][j] = '.'

        # Chuyển thành danh sách blocked_points
        blocked = [(i, j) for i in range(grid_size) for j in range(grid_size) if maze[i][j] == '#']

        # Kiểm tra có đường đi không
        if is_path_available((0, 0), (grid_size - 1, grid_size - 1), blocked, grid_size):
            break  # OK, có đường đi

    # Ghi ra file
    with open(filename, 'w') as f:
        for row in maze:
            f.write(' '.join(row) + '\n')
    print(f"Maze {grid_size}x{grid_size} saved to {filename}")

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
        self.reward_history = []
        self.steps_history = []

    # Hàm tải ma trận từ file
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
    
    # Hàm lấy tất cả các hành động khả thi từ trạng thái hiện tại
    def get_actions(self, state):
        actions = []
        for action in range(4):
            next_state = self.take_action(state, action)
            if 0 <= next_state[0] < self.grid_size and 0 <= next_state[1] < self.grid_size:
                actions.append(action)
        return actions

    # Hàm thực hiện hành động và trả về trạng thái tiếp theo
    def take_action(self, state, action):
        if action == 0:  # up
            return (state[0] - 1, state[1])
        elif action == 1:  # down
            return (state[0] + 1, state[1])
        elif action == 2:  # left
            return (state[0], state[1] - 1)
        elif action == 3:  # right
            return (state[0], state[1] + 1)

    # Hàm chọn hành động dựa trên chính sách epsilon-greedy
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.get_actions(state))
        else:
            total_q = np.sum(self.q_table[:, state[0], state[1], :], axis=0)
            return np.argmax(total_q)

    
    # drQ - Decomposed Reward Q Algorithm    
    def compute_decomposed_rewards(self, state, action, next_state, last_action):
        # Khởi tạo phần thưởng cho các thành phần
        rewards = {c: 0 for c in self.reward_components}
        if last_action is not None and (
            (action == 0 and last_action == 1) or (action == 1 and last_action == 0) or
            (action == 2 and last_action == 3) or (action == 3 and last_action == 2)):
            rewards['turn'] = -1 # Hình phạt cho việc quay đầu
        if self.is_terminal(next_state):
            rewards['goal'] = 30 # Phần thưởng cho việc đến đích
        if next_state in self.blocked_points:
            rewards['blocked'] = -1 # Hình phạt cho việc va chạm với điểm bị chặn
            self.consecutive_safe_actions = 0
        else:
            for a in range(4):
                future_state = self.take_action(next_state, a)
                if (0 <= future_state[0] < self.grid_size and 0 <= future_state[1] < self.grid_size and
                    future_state in self.blocked_points):
                    rewards['blocked'] = -0.5 # Hình phạt cho việc va chạm với điểm bị chặn trong tương lai
                    break
            self.consecutive_safe_actions += 1
            if self.consecutive_safe_actions == 2:
                rewards['safe'] = 2 # Phần thưởng cho việc đi an toàn liên tiếp
                self.consecutive_safe_actions = 0
        return rewards

    # Hàm cập nhật Q-Table
    def update_q_table(self, state, action, rewards, next_state):
        total_q_next = np.sum(self.q_table[:, next_state[0], next_state[1], :], axis=0)
        best_next_action = np.argmax(total_q_next)
        for c_idx, component in enumerate(self.reward_components):
            r_c = rewards[component]
            q_c_next = self.q_table[c_idx, next_state[0], next_state[1], best_next_action]
            td_target = r_c + self.gamma * q_c_next
            td_delta = td_target - self.q_table[c_idx, state[0], state[1], action]
            self.q_table[c_idx, state[0], state[1], action] += self.alpha * td_delta

    # RDX - Reward Difference Explanation Algorithm
    def reward_difference_explanation(self, state, chosen_action, alternative_action):
        # Lấy giá trị Q của hành động được chọn
        q_chosen = self.q_table[:, state[0], state[1], chosen_action]
        
        # Lấy giá trị Q của hành động thay thế
        q_alternative = self.q_table[:, state[0], state[1], alternative_action]
        q_diff = q_chosen - q_alternative
        
        # Lưu sự khác biệt giữa hai hành động cho từng thành phần phần thưởng
        explanation = {}
        for c_idx, component in enumerate(self.reward_components):
            explanation[component] = q_diff[c_idx]
        
        # Tính tổng sự khác biệt
        total_diff = np.sum(q_diff)
        return explanation, total_diff

    # MSX - Minimal Sufficient Explanation Algorithm
    def minimal_sufficient_explanation(self, state, chosen_action, alternative_action):
        # Tính sự khác biệt giữa các hành động
        explanation, total_diff = self.reward_difference_explanation(state, chosen_action, alternative_action)
        
        # d đại diện cho mức độ mà các thành phần tiêu cực (negative components) làm giảm giá trị của hành động được chọn
        d = sum(abs(diff) for comp, diff in explanation.items() if diff < 0)
        
        # Danh sách các thành phần phần thưởng có sự khác biệt dương (positive components), được sắp xếp giảm dần theo giá trị
        positive_components = [(comp, diff) for comp, diff in explanation.items() if diff > 0]
        positive_components.sort(key=lambda x: x[1], reverse=True)
        
        # Các thành phần tích cực tối thiểu cần thiết
        msx_plus = []
        current_sum = 0
        for comp, diff in positive_components:
            current_sum += diff
            msx_plus.append(comp)
            if current_sum > d:
                break
        
        if msx_plus:
            msx_plus_diffs = [explanation[comp] for comp in msx_plus]
            #  v đại diện cho mức độ mà các thành phần tích cực trong msx_plus vượt trội hơn so với giá trị nhỏ nhất
            v = sum(msx_plus_diffs) - min(msx_plus_diffs)
        else:
            v = 0
        
        # Xác định các thành phần tiêu cực
        negative_components = [(comp, diff) for comp, diff in explanation.items() if diff < 0]
        negative_components.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Các thành phần tiêu cực tối thiểu cần thiết
        msx_minus = []
        current_sum = 0
        for comp, diff in negative_components:
            current_sum += -diff
            msx_minus.append(comp)
            if current_sum > v:
                break
        
        # Tạo chi tiết công thức
        formula_details = {
            'd': d,
            'v': v,
            'msx_plus_sum': sum(explanation[comp] for comp in msx_plus) if msx_plus else 0,
            'msx_minus_sum': sum(-explanation[comp] for comp in msx_minus) if msx_minus else 0
        }
        
        return msx_plus, msx_minus, explanation, formula_details
    
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
    
    # Tạo tên file output cho kết quả
    def prepare_output_file(self, grid_size, run_count):
        output_dir = "excel-results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f"grid-{grid_size}-log-{run_count}.xlsx")
        return output_file


    def train(self, episodes, log_random_episode=False):
        self.reward_history = []
        self.steps_history = []
        best_total_reward = -float('inf')
        for episode in range(episodes):
            steps = 0
            state = self.reset()
            total_reward = 0
            current_path = [state]
            last_action = None
            self.consecutive_safe_actions = 0
            
            # Khởi tạo Q-Table cho episode mới
            while not self.is_terminal(state):
                steps = 0  # đếm số bước
                action = self.choose_action(state)
                next_state = self.take_action(state, action)
                current_path.append(next_state)
                
                rewards = self.compute_decomposed_rewards(state, action, next_state, last_action)
                total_reward += sum(rewards.values())
                self.update_q_table(state, action, rewards, next_state)
                
                state = next_state
                last_action = action
                self.reward_history.append(total_reward)
                self.steps_history.append(steps)
                steps += 1

            if total_reward > best_total_reward:
                best_total_reward = total_reward
                self.best_path = current_path
                
            # Xuất dữ liệu ra file Excel trong episode cuối
            if episode == episodes - 1 and log_random_episode:
                print(f"\nLogging actions for the last episode (Episode {episode + 1}) to Excel file...")
                table_data = []

                # Tạo output_file SỚM
                output_file = self.prepare_output_file(self.grid_size, self.get_run_count(self.grid_size))

                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        state = (i, j)
                        total_q = np.sum(self.q_table[:, state[0], state[1], :], axis=0)
                        chosen_action = np.argmax(total_q)
                        next_state = self.take_action(state, chosen_action)

                        available_actions = self.get_actions(state)
                        if len(available_actions) > 1:
                            alternative_action = random.choice([a for a in available_actions if a != chosen_action])
                        else:
                            alternative_action = chosen_action

                        msx_plus, msx_minus, explanation, formula_details = self.minimal_sufficient_explanation(
                            state, chosen_action, alternative_action)

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

                # VẼ BIỂU ĐỒ (Reward + Heatmap)
                self.plot_metrics()
                self.visualize(show_q_values=True)

                # LƯU FILE EXCEL
                df = pd.DataFrame(table_data)
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Episode Log', index=False)

                    for c_idx, component in enumerate(self.reward_components):
                        q_table_component = self.q_table[c_idx]
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
                        q_df.to_excel(writer, sheet_name=f'Q_{component}', index=False)

                    total_q_table = np.sum(self.q_table, axis=0)
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

            
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")
            if episode == episodes - 1:
                print("Final Decomposed Q-Table:")
                for c_idx, component in enumerate(self.reward_components):
                    print(f"Component: {component}")
                    for i in range(self.grid_size):
                        for j in range(self.grid_size):
                            print(f"State {i, j}: Q-Values: {self.q_table[c_idx, i, j]}")

    # Lấy đường đi ngắn nhất từ start đến end
    from collections import deque

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


    def plot_metrics(self, save_path=None):
        episodes = list(range(1, len(self.reward_history) + 1))
        window = 1000  # smoothing window lớn hơn để mượt hơn

        # Cấu hình font
        plt.rcParams.update({'font.size': 12})

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # --- 1. Biểu đồ reward ---
        axs[0].plot(episodes, self.reward_history, label='Total Reward per Episode', color='cornflowerblue', alpha=0.4)
        
        if len(self.reward_history) >= window:
            smoothed = np.convolve(self.reward_history, np.ones(window)/window, mode='valid')
            axs[0].plot(episodes[len(episodes)-len(smoothed):], smoothed, label='Smoothed Reward', color='red', linewidth=2)

        axs[0].set_title('Learning Curve: Total Reward per Episode')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Total Reward')
        axs[0].legend()
        axs[0].grid(True)

        # --- 2. Heatmap Q-values ---
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
            axs[1].scatter(point[1], point[0], marker='s', color='purple', s=15)
        axs[1].scatter(self.start[1], self.start[0], marker='o', color='green', label='Start')
        axs[1].scatter(self.end[1], self.end[0], marker='x', color='blue', label='End')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format='pdf')  # Lưu cho bài báo
            print(f"[✓] Saved plot to {save_path}")
        else:
            plt.show()
    def visualize_optimal_policy(self):
        arrow_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        policy_grid = np.empty((self.grid_size, self.grid_size), dtype=str)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) in self.blocked_points:
                    policy_grid[i, j] = '█'  # blocked
                elif (i, j) == self.end:
                    policy_grid[i, j] = 'G'  # goal
                elif (i, j) == self.start:
                    policy_grid[i, j] = 'S'  # start
                else:
                    total_q = np.sum(self.q_table[:, i, j, :], axis=0)
                    best_action = np.argmax(total_q)
                    policy_grid[i, j] = arrow_map[best_action]

        print("\nOptimal Policy Grid (based on total Q-values):")
        for row in policy_grid:
            print(' '.join(row))
    def plot_policy_arrows(self):
        import matplotlib.pyplot as plt

        total_q = np.sum(self.q_table, axis=0)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks(np.arange(0, self.grid_size + 1, 1))
        ax.set_yticks(np.arange(0, self.grid_size + 1, 1))
        ax.grid(True)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) in self.blocked_points:
                    ax.add_patch(plt.Rectangle((j, self.grid_size - 1 - i), 1, 1, color='black'))
                elif (i, j) == self.start:
                    ax.text(j + 0.5, self.grid_size - 1 - i + 0.5, 'S', ha='center', va='center', fontsize=12, color='green')
                elif (i, j) == self.end:
                    ax.text(j + 0.5, self.grid_size - 1 - i + 0.5, 'G', ha='center', va='center', fontsize=12, color='blue')
                else:
                    best_action = np.argmax(total_q[i, j, :])
                    dx, dy = [(0, 0.3), (0, -0.3), (-0.3, 0), (0.3, 0)][best_action]
                    ax.arrow(j + 0.5, self.grid_size - 1 - i + 0.5, dx, dy, head_width=0.2, head_length=0.2, fc='red', ec='red')

        ax.set_title("Optimal Policy Arrows")
        plt.show()

def is_path_available(start, end, blocked, grid_size):
            visited = set()
            queue = deque([start])
            while queue:
                current = queue.popleft()
                if current == end:
                    return True
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = current[0]+dx, current[1]+dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        if (nx, ny) not in blocked and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
            return False



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python drQ-main.py <grid_size>")
        sys.exit(1)

    # Lấy kích thước mê cung từ đối số dòng lệnh
    grid_size = int(sys.argv[1])
    maze_file = f"auto_gen_maze{grid_size}.txt"

    # Sinh mê cung và lưu vào file
    generate_maze(grid_size, maze_file)

    agent = RewardPathFinder(grid_size, maze_file)

    # Kiểm tra mê cung có đường đi không
    if not is_path_available(agent.start, agent.end, agent.blocked_points, grid_size):
        print("Maze is unsolvable: No valid path from Start to End!")
        sys.exit(1)

    run_count = agent.get_run_count(grid_size)
    output_file = agent.prepare_output_file(grid_size, run_count)

    start_time = time.time()

    agent.train(3000, log_random_episode=True)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[⏱] Training time: {elapsed:.2f} seconds")

    agent.visualize(show_q_values=True)
     # Biểu đồ phần thưởng theo episode
    agent.plot_metrics()
    agent.visualize_optimal_policy()
    agent.plot_policy_arrows()
    print(f"Output file: {output_file}")


# python drQ-main.py maze10.txt
# python drQ-main.py maze20.txt