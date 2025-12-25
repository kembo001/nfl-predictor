import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your CSV file
df_combined = pd.read_csv('data/combined_data.csv')


# Filter data
df_filtered = df_combined[df_combined['superbowl_win'] != 'No']

# Define defensive line and offensive line stats based on your actual columns
defensive_line_stats = ['pass_rush_rating', 'sack_rate', 'pressure_rate', 
                        'defensive_pass_epa', 'defensive_rush_epa']

offensive_line_stats = ['pass_block_rating', 'sacks_allowed_rate', 'protection_rate']

# Create a grid of subplots
n_def = len(defensive_line_stats)
n_off = len(offensive_line_stats)

fig, axes = plt.subplots(n_off, n_def, figsize=(24, 12))

# Create each combination
for i, off_stat in enumerate(offensive_line_stats):
    for j, def_stat in enumerate(defensive_line_stats):
        sns.scatterplot(data=df_filtered,
                       x=def_stat,
                       y=off_stat,
                       hue='superbowl_win',
                       hue_order=['SB Winner', 'SB Runner-up', 'Conf Runner-up'],
                       s=80,
                       alpha=0.7,
                       ax=axes[i, j],
                       legend=(i==0 and j==0))  # Only show legend on first plot
        
        axes[i, j].set_title(f'{off_stat} vs {def_stat}', fontsize=10)
        axes[i, j].grid(True, alpha=0.3)
        axes[i, j].set_xlabel(def_stat, fontsize=9)
        axes[i, j].set_ylabel(off_stat, fontsize=9)

plt.suptitle('Offensive Line vs Defensive Line Stats - Trenches Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()