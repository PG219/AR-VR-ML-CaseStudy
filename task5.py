import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load and convert image
image_path = "Fallow-deer-dama-dama.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show original image
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
plt.show()

# Reshape image to (num_pixels, 3)
pixel_vals = image.reshape((-1, 3))
pixel_vals = np.float32(pixel_vals)

# KMeans Clustering
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(pixel_vals)
centers = np.uint8(kmeans.cluster_centers_)

# Replace each pixel with its centroid color
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(image.shape)

# Show segmented image
plt.imshow(segmented_image)
plt.title(f"Segmented Image (k={k})")
plt.axis("off")
plt.show()

# Convert segmented image to grayscale
gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on original image
overlay = image.copy()
cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

plt.imshow(overlay)
plt.title("AR Overlay with Contours")
plt.axis("off")
plt.show()

# Draw bounding boxes
overlay_boxes = image.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(overlay_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(overlay_boxes)
plt.title("Bounding Boxes Overlay")
plt.axis("off")
plt.show()
