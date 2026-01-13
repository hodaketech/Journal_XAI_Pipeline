import pandas as pd

df = pd.read_csv('auto-validation-dataset-processed.csv')

# Lọc bỏ các dòng có move_id = -1
df = df[df['move_id'] != -1]
df = df.reset_index(drop=True)

# Lọc lấy các record thứ 9, 19, 29, ... (index % 10 == 9)
df_filtered = df[df.index % 10 == 9]
df_filtered = df_filtered.sort_index()

df_filtered.to_csv('logs/auto-validation-dataset_filtered.csv', index=False)

print(f"Đã lọc từ {len(df)} records xuống còn {len(df_filtered)} records")
print(f"Đã loại bỏ các dòng có move_id = -1")
print(f"Giữ lại các record thứ: {', '.join(map(str, df_filtered.index.tolist()))}")