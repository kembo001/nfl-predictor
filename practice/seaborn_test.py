import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create sample data - like football stats
np.random.seed(42)
df = pd.DataFrame({
    'points': [24, 31, 17, 28, 21, 35, 14, 27, 30, 19],
    'yards': [350, 420, 280, 390, 310, 450, 260, 380, 410, 290],
    'turnovers': [1, 0, 3, 1, 2, 0, 2, 1, 0, 2],
    'time_of_possession': [28, 32, 25, 30, 27, 34, 24, 29, 31, 26]
})

# Create the pairplot
sns.pairplot(df)
plt.show()