import random
import sys
from collections import deque

def is_path_available(maze, grid_size):
    visited = set()
    queue = deque([(0, 0)])
    while queue:
        x, y = queue.popleft()
        if (x, y) == (grid_size - 1, grid_size - 1):
            return True
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                if maze[nx][ny] == '.' and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return False

def generate_maze(grid_size, filename, density=0.38):
    attempt = 0
    while True:
        attempt += 1
        maze = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                if (i, j) == (0, 0) or (i, j) == (grid_size - 1, grid_size - 1):
                    row.append('.')  # Start/Goal luôn là đường đi
                else:
                    row.append('#' if random.random() < density else '.')
            maze.append(row)

        # ✅ Mở thêm nhiều ô ngẫu nhiên để tăng khả năng solvable
        for _ in range(grid_size):  # nhiều khe hở hơn
            x, y = random.randint(1, grid_size - 2), random.randint(1, grid_size - 2)
            maze[x][y] = '.'

        # ✅ Optional: mở lại đường chéo nếu maze quá khó
        for i in range(grid_size):
            maze[i][i] = '.'

        # Kiểm tra có đường đi từ Start -> Goal
        if is_path_available(maze, grid_size):
            break

        if attempt > 100:
            raise Exception("Không thể tạo maze phức tạp có thể giải sau 100 lần.")

    with open(filename, 'w') as f:
        for row in maze:
            f.write(' '.join(row) + '\n')

    print(f"[🔥] Maze {grid_size}x{grid_size} đã được lưu vào {filename} (density={density:.2f})")

# --- Chạy bằng dòng lệnh ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Cách dùng: python generate_maze.py <grid_size>")
        sys.exit(1)

    try:
        size = int(sys.argv[1])
        if size not in [10, 20, 30, 40, 50, 60]:
            raise ValueError("Chỉ hỗ trợ size: 10, 20, 30, 40, 50, 60")
    except ValueError as e:
        print("❌", e)
        sys.exit(1)

    filename = f"maze{size}.txt"
    generate_maze(size, filename)

# Chạy bằng dòng lệnh
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Cách dùng: python generate_maze.py <grid_size>")
        sys.exit(1)

    size = int(sys.argv[1])
    filename = f"maze{size}.txt"
    generate_maze(size, filename)


#python generate_maze.py 5
#python generate_maze.py 10
#python generate_maze.py 20