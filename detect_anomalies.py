import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os

# 1. Load the Data
# Expand the ~ to the actual home directory path
file_path = os.path.expanduser('~/Desktop/prmon.txt')
df = pd.read_csv(file_path, sep='\s+')

# Setup the exact column names based on prmon's default output
time_col = 'Time' if 'Time' in df.columns else df.columns[0]
features = ['pss', 'nprocs'] # We track Memory (PSS) and Thread count

# 2. Train the Isolation Forest Model
model = IsolationForest(contamination=0.15, random_state=42)
df['anomaly'] = model.fit_predict(df[features])

# 3. Visualization
plt.figure(figsize=(12, 6))

# Plot the baseline memory usage as a blue line
plt.plot(df[time_col], df['pss'], label='Memory (PSS) Usage', color='dodgerblue', linewidth=2)

# Extract and plot the anomalies as red dots
anomalies = df[df['anomaly'] == -1]
plt.scatter(anomalies[time_col], anomalies['pss'], color='red', s=50, label='Detected Anomaly', zorder=5)

# Formatting the graph for the report
plt.title('Automated Anomaly Detection in prmon Time-Series Data', fontsize=14, fontweight='bold')
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Memory PSS (KB)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot for the GitHub Repo
save_path = os.path.expanduser('~/Desktop/anomaly_detection_plot.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print("Success! Plot saved to your Desktop as anomaly_detection_plot.png")

plt.show()
