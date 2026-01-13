import time

# Đọc trạng thái bitboard từ file
with open('ketqua_ui_bitboard.txt', 'r') as f:
    bb1 = int(f.readline())
    bb2 = int(f.readline())

# Import Board (bitboard)
from game import Board

# Giả sử width, height đúng với bàn bạn vừa chơi
width = 10  # sửa đúng theo bàn cờ của bạn (hoặc đọc từ file 2D)
height = 10
n_in_row = 6

# Phục hồi Board bitboard
board = Board(width=width, height=height, n_in_row=n_in_row)
board.init_board()
board.bitboards[1] = bb1
board.bitboards[2] = bb2

# Benchmark hàm kiểm tra thắng
N = 1000000
start = time.time()
for _ in range(N):
    win, winner = board.has_a_winner()
print(f"[Bitboard] {N} lần kiểm tra thắng mất {time.time() - start:.4f} giây")
