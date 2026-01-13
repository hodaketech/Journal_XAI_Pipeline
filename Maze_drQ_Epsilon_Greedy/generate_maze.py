import random
import sys

def generate_maze(grid_size, filename, density=0.25):
    maze = []

    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            if (i, j) == (0, 0) or (i, j) == (grid_size - 1, grid_size - 1):
                row.append('.')  # Đảm bảo start và end không bị chặn
            else:
                row.append('#' if random.random() < density else '.')
        maze.append(row)

    # Mở đường đi chéo từ (0,0) đến (n-1,n-1)
    for i in range(grid_size):
        maze[i][i] = '.'

    # Mở thêm 1 đường ngang ở giữa để tăng khả năng đi được
    mid = grid_size // 2
    for j in range(grid_size):
        maze[mid][j] = '.'

    # 💾 Ghi ra file
    with open(filename, 'w') as f:
        for row in maze:
            f.write(' '.join(row) + '\n')

    print(f"Maze {grid_size}x{grid_size} đã được lưu vào {filename}")

# --- Chạy bằng dòng lệnh ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Cách dùng: python generate_maze.py <grid_size>")
        sys.exit(1)

# try:
       
#        if size not in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
 #           raise ValueError("Chỉ hỗ trợ size: 5, 10, 20, 30, 40.")
  #  except ValueError as e:
   ##    sys.exit(1)
    size = int(sys.argv[1])
    filename = f"maze{size}.txt"
    generate_maze(size, filename)

#python generate_maze.py 5
#python generate_maze.py 10
#python generate_maze.py 20