import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('knn5_time_only_for_0-1.csv')

column_a_counts = df['DIAGNOSIS'].value_counts()

# Count unique values for Column B
column_b_counts = df['TIME_DIAGNOSIS'].value_counts()

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Plot for Column A
sns.barplot(x=column_a_counts.index, y=column_a_counts.values, ax=axes[0], palette="viridis")
axes[0].set_title("NORMAL (0) MCI/DEM (1)")
axes[0].set_xlabel("")
axes[0].set_ylabel("")

# Plot for Column B
sns.barplot(x=column_b_counts.index, y=column_b_counts.values, ax=axes[1], palette="plasma")
axes[1].set_title("DAYS WITH DIAGNOSIS")
axes[1].set_xlabel("")
axes[1].set_ylabel("")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

