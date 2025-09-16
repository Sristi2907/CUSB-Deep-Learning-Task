import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import pandas as pd


# Load the digits dataset (small MNIST: 8x8 images)

digits = load_digits()
X = digits.data
y = digits.target


# Apply PCA to reduce to 2 dimensions

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Get the explained variance ratio for the two components
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by the two components: {explained_variance[0]:.2f}, {explained_variance[1]:.2f}")
print(f"Total explained variance: {sum(explained_variance):.2f} or {sum(explained_variance)*100:.1f}%")

# Put results into a dataframe for easy handling
df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Label": y
})


# Scatter plot of the first two principal components

plt.figure(figsize=(12, 8))

# Create a colormap for the digits
import numpy as np
from matplotlib.colors import ListedColormap

# Use a colormap with distinct colors for each digit
colors = plt.cm.tab10(np.linspace(0, 1, 10))

# Plot each digit with its own color
scatter = plt.scatter(df.PC1, df.PC2, c=df.Label, cmap=ListedColormap(colors), 
                     alpha=0.8, s=50, edgecolors='w', linewidth=0.5)

# Add a colorbar
cbar = plt.colorbar(scatter, ticks=range(10))
cbar.set_label('Digit')

# Add some sample digit images at their PCA coordinates
fig = plt.gcf()
ax = plt.gca()

# Add a few sample images for each digit
for digit in range(10):
    # Get indices for this digit
    mask = y == digit
    indices = np.atleast_1d(mask).nonzero()[0]
    
    # Select a few random samples
    if len(indices) > 0:
        sample_idx = np.random.choice(indices, min(1, len(indices)))
        
        # For each sample, add a small image of the digit
        for idx in sample_idx:
            # Get the image data and reshape it
            img = digits.images[idx]
            
            # Create a small axes for this thumbnail
            x, y = X_pca[idx, 0], X_pca[idx, 1]
            
            # Add text label instead of image for clarity
            ax.text(x, y, str(digit), fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle="circle", fc=colors[digit], alpha=0.8))

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(f"PCA on MNIST Digits Dataset - Two Components\nExplained Variance: {explained_variance[0]:.2f}, {explained_variance[1]:.2f} (Total: {sum(explained_variance):.2f})")

# Add a text annotation about the explained variance
plt.figtext(0.5, 0.01, f"The two principal components explain {sum(explained_variance)*100:.1f}% of the total variance", 
           ha='center', fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
plt.grid(True)

# Save the visualization to a file
output_filename = 'pca_digits_visualization_lab6.1.png'
plt.savefig(output_filename)
plt.close()
print(f"PCA visualization saved to '{output_filename}'")
