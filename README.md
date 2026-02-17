# Customer Segmentation with K-Means
# Author: Moses BALUME

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("customers.csv")
print(data.head())

# Select features
X = data[['Income', 'SpendingScore']]

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

print("\nCluster distribution:")
print(data['Cluster'].value_counts())

# Visualization
plt.figure(figsize=(8,5))
plt.scatter(data['Income'], data['SpendingScore'], c=data['Cluster'])
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.show()