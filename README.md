# 🧠 AR/VR + Machine Learning Case Study

This repository contains my implementation of 6 tasks that combine **unsupervised machine learning** with **AR/VR concepts**.
I have implemented these tasks using my local computer in PyCharm IDE.

---

## 📌 Overview of Tasks

| Task | Description |
|------|-------------|
| 1    | Clustering 2D synthetic data (KMeans & DBSCAN) |
| 2    | Image segmentation using KMeans |
| 3    | Dimensionality reduction using PCA & t-SNE |
| 4    | Sensor data clustering (UCI HAR Dataset) |
| 5    | Overlaying cluster regions on image (AR-like effect) |
| 6    | Running a real-world unsupervised image segmentation repo |

---

## ✅ Task 1: Clustering 2D Synthetic Data

To understand the dataset, I reffered to skLearn refernces and generated the synthetic dataset.
**What I did:**
- Used `make_blobs` and `make_moons` to generate sample datasets.
I used scatter plots to understand the data and then used Kmeans and DBSCAN to cluster them up.
As the use case, Kmeans clusteres around a data point and DBSCAN clustered in a dense area.

I tried out with different values of K and epsilon and saw how the changes in the values affect the end result.

📊 *KMeans is great for round clusters; DBSCAN handles complex patterns like moons.*

---

## ✅ Task 2: Image Segmentation Using KMeans

**What I did:**
- Loaded an image, reshaped pixels, and clustered them using KMeans.
- Reconstructed the image by replacing pixels with their cluster center color.

**What I learned:**
- Image segmentation can be done **without labels**, just by grouping similar colors.
- Experimenting with different `k` values gave different levels of detail.
- Using LAB color space sometimes gave more human-like clustering results.

🖼️ *This gave me a real sense of how AR filters or object grouping works.*

---

## ✅ Task 3: Dimensionality Reduction with PCA and t-SNE

**What I did:**
- Loaded the Iris dataset from scikit-learn.
- Applied PCA and t-SNE to reduce 64 features to 2D and plotted them.

PCA and t-SNE helped me to reduce high dimenstional data into 2-D. PCA finds new axes (called principal components) that capture most of the variation in the data. We see that t-SNE preserves the local structure and 
helps to visualize complex clusters efficiently.


🔍 *This task built my intuition about how to project sensor/image data into human-readable form.*

---

## ✅ Task 4: Clustering Real-World Sensor Data

**What I did:**
- Used the UCI Human Activity Recognition dataset.
- Applied preprocessing, PCA, and clustering (KMeans + DBSCAN).

**What I learned:**
- Sensor data is high-dimensional and noisy, but clustering can extract patterns.
- Activities like walking vs. sitting formed clear clusters.
- This simulates how AR/VR systems recognize real-time behavior from sensors.

🚶 *This made me appreciate the preprocessing steps before using ML on real-world signals.*

---

## ✅ Task 5: Overlaying Clusters on Image (Simulated AR)

**What I did:**
- After segmenting an image, I detected contours and drew bounding boxes using OpenCV.

**What I learned:**
- How to visually connect clustering output to AR-style overlays.
- Detected regions can simulate object recognition in AR interfaces.
- Drawing real-time overlays is the base of many AR camera effects.

🕶️ *Felt like building the first step of a basic AR object detector!*

---

## ✅ Task 6: Cloning and Running a GitHub Repo

**What I did:**
- Cloned an unsupervised KMeans-based segmentation project from GitHub.
- Ran the code locally, understood the structure, and tested it on my own images.

**What I learned:**
- Real-world projects involve organizing scripts, data, and visualization separately.
- Learned how to navigate a real repo and tweak parameters to experiment.
- Practically understand K-means and Hierarchical clustering algorithms

📦 *This task helped me bridge my learning with open-source practices.*

---

References:
https://www.kaggle.com/code/samuelcortinhas/k-means-from-scratch
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
https://www.geeksforgeeks.org/machine-learning/clustering-in-machine-learning/
https://imageannotation.home.blog/2020/06/18/what-is-the-application-of-image-segmentation/
https://scikit-learn.org/stable/modules/clustering.html
https://medium.com/towardssingularity/k-means-clustering-for-image-segmentation-using-opencv-in-python-17178ce3d6f3


## 💻 How to Run

1. Clone this repo:
```bash
git clone https://github.com/your-username/AR-VR-ML-CaseStudy.git
