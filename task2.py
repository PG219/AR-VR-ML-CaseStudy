import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

image_path = "Fallow-deer-dama-dama.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Display original image
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
plt.show()

# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1,3)) # numpy reshape operation -1 unspecified

# Convert to float type only for supporting cv2.kmean
pixel_vals = np.float32(pixel_vals)


k = 4  # Try 2, 4, 8, etc.
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(pixel_vals)
centers = np.uint8(kmeans.cluster_centers_)

# Replace each pixel with its cluster center color
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(image.shape)

# Show segmented image
plt.imshow(segmented_image)
plt.title(f"Segmented Image (k={k})")
plt.axis("off")
plt.show()

# Convert to LAB color space
image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
pixel_values_lab = image_lab.reshape((-1, 3))
pixel_values_lab = np.float32(pixel_values_lab)

# KMeans on LAB
kmeans_lab = KMeans(n_clusters=4, random_state=42)
labels_lab = kmeans_lab.fit_predict(pixel_values_lab)
centers_lab = np.uint8(kmeans_lab.cluster_centers_)

segmented_lab = centers_lab[labels_lab.flatten()].reshape(image.shape)
segmented_lab_rgb = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2RGB)

# Show LAB result
plt.imshow(segmented_lab_rgb)
plt.title("Segmented Image in LAB")
plt.axis("off")
plt.show()
