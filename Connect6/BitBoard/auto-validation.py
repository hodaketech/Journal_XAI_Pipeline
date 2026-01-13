import pandas as pd
import numpy as np
import math

df = pd.read_csv('evaluation-validation-dataset-processed.csv')

df['moves_list'] = df['moves'].apply(lambda x: list(map(int, x.split('|'))))
df['gamma_list'] = df['gamma_values'].apply(lambda x: list(map(float, x.split('|'))))

checks = ['check_sum_gamma', 'check_num_moves', 'check_moves_consistency', 'check_gamma_range', 'check_player_alternation']
for check in checks:
    df[check] = False

df['REVIEW'] = False
df['human_explanation'] = ''
df['error_explanation'] = ''

# Function to explain move selection BASED ON RWS MECHANISM
def explain_move(gamma_values, moves, prev_moves, chosen_move):
    if chosen_move not in prev_moves:
        return "ERROR: The chosen move is not in the list of valid moves"
    
    move_idx = prev_moves.index(chosen_move)
    move_gamma = gamma_values[move_idx]
    
    # Calculate total gamma and cumulative probabilities for RWS
    total_gamma = sum(gamma_values)
    cumulative_probs = []
    current_sum = 0
    for gamma in gamma_values:
        current_sum += gamma
        cumulative_probs.append(current_sum / total_gamma)
    
    # Get comparison information
    other_gammas = [g for j, g in enumerate(gamma_values) if j != move_idx]
    max_other_gamma = max(other_gammas) if other_gammas else 0
    max_other_idx = gamma_values.index(max_other_gamma) if other_gammas else -1
    
    if move_gamma >= 0.3:
        return (f"The player chooses move {chosen_move} with a VERY HIGH probability (gamma={move_gamma:.4f}, accounting for {move_gamma/total_gamma*100:.1f}% of total). "
                f"In the RWS mechanism, this is a large area on the roulette wheel, so the chance of being selected is very high. "
                f"This is the optimal move, evaluated with high value.")
    
    elif move_gamma >= 0.15:
        return (f"The player chooses move {chosen_move} with a HIGH probability (gamma={move_gamma:.4f}, accounting for {move_gamma/total_gamma*100:.1f}% of total). "
                f"In RWS, this is a clearly advantageous choice, with significant area on the roulette wheel. "
                f"The cumulative probability up to this choice is {cumulative_probs[move_idx]:.3f}.")
    
    elif move_gamma >= 0.08:
        if move_gamma > max_other_gamma:
            return (f"The player chooses move {chosen_move} with a MODERATELY HIGH probability (gamma={move_gamma:.4f}, accounting for {move_gamma/total_gamma*100:.1f}% of total). "
                    f"This is a good choice among the options. RWS prioritizes choices with higher gamma, "
                    f"so selecting this move is reasonable.")
        else:
            return (f"The player chooses move {chosen_move} with a MODERATE probability (gamma={move_gamma:.4f}, accounting for {move_gamma/total_gamma*100:.1f}% of total). "
                    f"Although not the highest (highest is move {moves[max_other_idx]} with gamma={max_other_gamma:.4f}), "
                    f"RWS still allows selection due to randomness in the roulette wheel mechanism.")
    
    elif move_gamma >= 0.03:
        return (f"The player chooses move {chosen_move} with a LOW probability (gamma={move_gamma:.4f}, accounting for {move_gamma/total_gamma*100:.1f}% of total). "
                f"In RWS, this is a small area on the roulette wheel. Being selected is mainly due to random factors "
                f"when the pointer lands in this area.")
    
    else:
        return (f"The player chooses move {chosen_move} with a VERY LOW probability (gamma={move_gamma:.4f}, accounting for {move_gamma/total_gamma*100:.1f}% of total). "
                f"This is a rare case in RWS, where the area on the roulette wheel is very small. "
                f"Selecting this move is primarily due to the strong randomness of RWS")

# Function to add DETAILED explanations for validation errors
def add_detailed_error_explanation(row):
    explanations = []
    
    # Check 1: Sum of gamma
    if row['check_sum_gamma']:
        sum_gamma = row['sum_gamma']
        deviation = abs(sum_gamma - 1.0)
        explanations.append(f"Sum of gamma={sum_gamma:.6f} (deviation {deviation:.6f} from 1.0) - possibly due to rounding or calculation error")
    
    # Check 2: Number of moves
    if row['check_num_moves']:
        current_num_moves = row['num_moves']
        explanations.append(f"Current number of moves ({current_num_moves}) does not decrease by exactly 1 compared to previous turn")
    
    # Check 3: Moves consistency
    if row['check_moves_consistency']:
        explanations.append("Moves list is inconsistent with previous turn - some moves may be missing or extra")
    
    # Check 4: Gamma range
    if row['check_gamma_range']:
        gamma_values = row['gamma_list']
        invalid_gammas = [f"gamma[{i}]={g:.6f}" for i, g in enumerate(gamma_values) if g < 0 or g > 1]
        if invalid_gammas:
            explanations.append(f"Invalid gamma values: {', '.join(invalid_gammas)}")
        else:
            explanations.append("Some gamma values are outside the valid range (0-1)")
    
    # Check 5: Player alternation
    if row['check_player_alternation']:
        current_move_id = row['move_id']
        current_player = row['player']
        if current_move_id % 2 == 1:
            explanations.append(f"Move_id {current_move_id} is odd but the player didn't change (still {current_player})")
        else:
            explanations.append(f"Move_id {current_move_id} is even but player change doesn't follow the rule")
    
    return "; ".join(explanations) if explanations else ""

# Check each row - ALL VALIDATION CHECKS COMBINED
for i in range(len(df)):
    row = df.iloc[i]
    
    # Check 1: Sum of gamma ≈ 1
    sum_gamma = row['sum_gamma']
    if not math.isclose(sum_gamma, 1.0):
        df.at[i, 'check_sum_gamma'] = True
    
    # Check 2: Number of moves decreases by 1 compared to previous row
    if i > 0 and df.iloc[i-1]['game_id'] == row['game_id'] and row['move_id'] != 0:
        prev_num_moves = df.iloc[i-1]['num_moves']
        if row['num_moves'] != prev_num_moves - 1:
            df.at[i, 'check_num_moves'] = True
    
    # Check 3: Moves list consistency
    if i > 0 and df.iloc[i-1]['game_id'] == row['game_id'] and row['move_id'] != 0:
        prev_moves = df.iloc[i-1]['moves_list']
        current_moves = row['moves_list']
        missing_moves = set(prev_moves) - set(current_moves)
        if len(missing_moves) != 1:
            df.at[i, 'check_moves_consistency'] = True
        else:
            # Create explanation for the chosen move and assign to previous record
            chosen_move = list(missing_moves)[0]
            prev_gamma = df.iloc[i-1]['gamma_list']
            explanation = explain_move(prev_gamma, prev_moves, prev_moves, chosen_move)
            df.at[i-1, 'human_explanation'] = explanation
    
    # Check 4: Valid gamma values
    gamma_values = row['gamma_list']
    if any(gamma < 0 or gamma > 1 for gamma in gamma_values):
        df.at[i, 'check_gamma_range'] = True
    
    # CHECK 5: Player alternation (COMBINED HERE)
    if i > 0 and df.iloc[i-1]['game_id'] == row['game_id']:
        current_move_id = row['move_id']
        current_player = row['player']
        previous_player = df.iloc[i-1]['player']
        
        # Skip if move_id = 0
        if current_move_id != 0:
            if current_move_id % 2 == 1:  # odd move_id
                if current_player == previous_player:
                    df.at[i, 'check_player_alternation'] = True
            else:  # even move_id
                if current_player != previous_player:
                    df.at[i, 'check_player_alternation'] = True
    
    # ADD VALIDATION ERROR EXPLANATION TO NEW COLUMN
    error_explanation = add_detailed_error_explanation(df.iloc[i])
    if error_explanation:
        df.at[i, 'error_explanation'] = error_explanation
    
    # Add error explanation to human_explanation (if needed)
    if error_explanation:
        current_explanation = df.at[i, 'human_explanation']
        if current_explanation:
            df.at[i, 'human_explanation'] = current_explanation + " | VALIDATION ERROR: " + error_explanation
        else:
            df.at[i, 'human_explanation'] = "VALIDATION ERROR: " + error_explanation
    
    # Mark REVIEW if any error exists
    if any(df.loc[i, check] for check in checks):
        df.at[i, 'REVIEW'] = True

# Create result DataFrame - ADD error_explanation column
result_df = df[['game_id', 'move_id', 'player'] + checks + ['REVIEW', 'human_explanation', 'error_explanation']]

# Create summary
total_moves = len(result_df)
review_needed = result_df['REVIEW'].sum()
no_review_needed = total_moves - review_needed
review_rate = review_needed / total_moves * 100

# Add detailed error statistics
error_stats = {
    'Error Type': ['Sum of gamma', 'Number of moves', 'Moves consistency', 'Gamma range', 'Player alternation'],
    'Count': [
        result_df['check_sum_gamma'].sum(),
        result_df['check_num_moves'].sum(),
        result_df['check_moves_consistency'].sum(),
        result_df['check_gamma_range'].sum(),
        result_df['check_player_alternation'].sum()
    ]
}
error_stats_df = pd.DataFrame(error_stats)

summary_data = {
    'Metric': ['Total moves', 'Need review', 'No review needed', 'Review rate (%)'],
    'Value': [total_moves, review_needed, no_review_needed, f'{review_rate:.2f}%']
}
summary_df = pd.DataFrame(summary_data)

# Write results to Excel file
with pd.ExcelWriter('logs/validation_result.xlsx') as writer:
    result_df.to_excel(writer, sheet_name='Validation Results', index=False)
    summary_df.to_excel(writer, sheet_name='Overview', index=False)
    error_stats_df.to_excel(writer, sheet_name='Error Statistics', index=False)

print("Validation complete! Results saved to 'validation_result.xlsx'")
print(f"Overview: {review_needed} moves need review ({review_rate:.2f}%)")

# Print error statistics
print("\nError statistics:")
for i, row in error_stats_df.iterrows():
    print(f"  - {row['Error Type']}: {row['Count']} errors")