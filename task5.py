import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Utility to plot and optionally save image
def plot_image(img, title, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def segment_image_kmeans(image, k=4):
    # Reshape and convert to float32
    pixel_vals = image.reshape((-1, 3)).astype(np.float32)

    # Apply KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pixel_vals)
    centers = np.uint8(kmeans.cluster_centers_)

    # Replace each pixel with its cluster center
    segmented = centers[labels.flatten()].reshape(image.shape)
    return segmented

def find_and_draw_contours(image, segmented_image):
    # Convert to grayscale and blur
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours
    contour_overlay = image.copy()
    cv2.drawContours(contour_overlay, contours, -1, (255, 0, 0), 2)

    # Draw bounding boxes
    box_overlay = image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(box_overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return contour_overlay, box_overlay

# Main execution
if __name__ == "__main__":
    # Load and convert image
    image_path = "Fallow-deer-dama-dama.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display original image
    plot_image(image, "Original Image")

    # Segment image
    k = 4
    segmented_image = segment_image_kmeans(image, k)
    plot_image(segmented_image, f"Segmented Image (k={k})")

    # Draw contours and boxes
    contour_overlay, box_overlay = find_and_draw_contours(image, segmented_image)
    plot_image(contour_overlay, "AR Overlay with Contours")
    plot_image(box_overlay, "Bounding Boxes Overlay")
