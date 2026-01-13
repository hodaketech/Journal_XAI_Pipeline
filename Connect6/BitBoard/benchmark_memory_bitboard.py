import sys

# Đọc bitboard từ file txt
with open('ketqua_ui_bitboard.txt') as f:
    bb1 = int(f.readline())
    bb2 = int(f.readline())

print("Bitboard player 1:", sys.getsizeof(bb1), "bytes")
print("Bitboard player 2:", sys.getsizeof(bb2), "bytes")
print("Tổng bộ nhớ Bitboard:", sys.getsizeof(bb1) + sys.getsizeof(bb2), "bytes")
