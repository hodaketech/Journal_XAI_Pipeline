import pandas as pd
import numpy as np

# Đọc file CSV
df = pd.read_csv('auto-validation-dataset.csv')

def process_gamma_values_detailed(gamma_str, idx):
    original_gamma_list = [float(x) for x in gamma_str.split('|')]
    
    # KIỂM TRA VÀ XỬ LÝ GAMMA > 1
    gamma_above_1 = [gamma for gamma in original_gamma_list if gamma > 1.0]
    if gamma_above_1:
        if idx >= 30:
            # Lấy gamma của record [i-30]
            previous_gamma_str = df['gamma_values'].iloc[idx - 30]
            previous_gamma_list = [float(x) for x in previous_gamma_str.split('|')]
            
            print(f"Dòng {idx}: PHÁT HIỆN {len(gamma_above_1)} gamma > 1")
            print(f"  Giá trị > 1: {[f'{x:.4f}' for x in gamma_above_1]}")
            print(f"  ĐÃ THAY THẾ BẰNG GAMMA TỪ DÒNG {idx-30}")
            
            return previous_gamma_str
        else:
            # Nếu không có record [i-30], sử dụng record đầu tiên
            first_gamma_str = df['gamma_values'].iloc[0]
            first_gamma_list = [float(x) for x in first_gamma_str.split('|')]
            
            print(f"Dòng {idx}: PHÁT HIỆN {len(gamma_above_1)} gamma > 1")
            print(f"  Giá trị > 1: {[f'{x:.4f}' for x in gamma_above_1]}")
            print(f"  KHÔNG CÓ DÒNG [i-30], SỬ DỤNG DÒNG ĐẦU TIÊN")
            
            return first_gamma_str
    
    # XỬ LÝ GIÁ TRỊ ÂM
    gamma_list = original_gamma_list.copy()
    negative_sum = 0
    negative_count = 0
    original_max = max(gamma_list)
    original_max_index = np.argmax(gamma_list)
    
    for i in range(len(gamma_list)):
        if gamma_list[i] < 0:
            negative_sum += abs(gamma_list[i])
            gamma_list[i] = 0
            negative_count += 1
    
    # Điều chỉnh giá trị lớn nhất
    if negative_sum > 0:
        gamma_list[original_max_index] -= negative_sum
        if gamma_list[original_max_index] < 0:
            gamma_list[original_max_index] = 0
        
        print(f"Dòng {idx}: Đã xử lý {negative_count} giá trị âm, tổng bù: {negative_sum:.4f}")
        print(f"  Giá trị lớn nhất: {original_max:.4f} -> {gamma_list[original_max_index]:.4f}")
    
    return '|'.join([f"{x:.4f}" for x in gamma_list])

# Áp dụng xử lý
print("Bắt đầu xử lý gamma values...")
df['gamma_values'] = [process_gamma_values_detailed(gamma_str, idx) for idx, gamma_str in enumerate(df['gamma_values'])]

# Lưu file
df.to_csv('auto-validation-dataset-processed.csv', index=False)

print("\nXử lý hoàn tất! File đã được lưu thành 'auto-validation-dataset-processed.csv'")

# Thống kê
print(f"\nTổng số dòng: {len(df)}")
print(f"Số lượng gamma values trên mỗi dòng: {len(df['gamma_values'].iloc[0].split('|'))}")

# Đếm số lần thay thế
print("\nTHỐNG KÊ XỬ LÝ:")
processed_gammas = [gamma_str for gamma_str in df['gamma_values']]
original_gammas = pd.read_csv('auto-validation-dataset.csv')['gamma_values']

replacement_count = 0
for idx, (orig, proc) in enumerate(zip(original_gammas, processed_gammas)):
    if orig != proc:
        original_list = [float(x) for x in orig.split('|')]
        if any(gamma > 1.0 for gamma in original_list):
            replacement_count += 1

print(f"Số record có gamma > 1 đã được thay thế: {replacement_count}")