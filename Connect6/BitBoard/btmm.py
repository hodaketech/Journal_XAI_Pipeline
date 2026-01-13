import csv, os

class BradleyTerryMM:
    def __init__(self, moves_features, player=None, game_id=None, move_id=None):
        self.moves_features = moves_features
        self.gamma = {move: 1.0 for move in moves_features}
        self.player = player
        self.game_id = game_id
        self.move_id = move_id
        self.log_data = []

    def update_gamma(self, win_counts, total_counts, iterations=10):
        for iter_num in range(iterations):
            gamma_new = {}
            for move_i in self.gamma:
                numerator = win_counts.get(move_i, 1e-4)
                denominator = 0
                for move_j in self.gamma:
                    if move_j == move_i:
                        continue
                    denom = float(self.gamma[move_i]) + float(self.gamma[move_j])
                    if denom == 0:
                        continue
                    numerator_ij = total_counts.get((move_i, move_j), 0) + total_counts.get((move_j, move_i), 0)
                    denominator += numerator_ij / denom
                denominator += 1e-4
                gamma_new[move_i] = numerator / denominator
            # Chuẩn hóa gamma
            sum_gamma = sum(float(g) for g in gamma_new.values())
            for move in gamma_new:
                gamma_new[move] /= sum_gamma
            self.gamma = gamma_new
            # Log lại thông tin mỗi iteration
            self.log_data.append({
                "game_id": int(self.game_id) if self.game_id is not None else -1,
                "move_id": int(self.move_id) if self.move_id is not None else -1,
                "iteration": iter_num,
                "player": int(self.player) if self.player is not None else -1,
                "num_moves": len(self.gamma),
                "moves": "|".join(map(str, self.gamma.keys())),
                "gamma_values": "|".join(f"{float(g):.4f}" for g in self.gamma.values()),
                "sum_gamma": round(sum(float(g) for g in self.gamma.values()), 4)
            })
        return self.gamma

    def get_probs(self):
        total_gamma = sum(float(g) for g in self.gamma.values())
        return {move: float(self.gamma[move]) / total_gamma for move in self.gamma}

    def append_log_to_file(self, filename="logs/btmm_gamma_log.csv"):
        if not self.log_data:
            return
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file_exists = os.path.isfile(filename)
        with open(filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.log_data[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.log_data)
        self.log_data.clear()
