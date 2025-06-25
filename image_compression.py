import numpy as np
import cv2
import os
from sklearn.cluster import KMeans


def compress_image(image_path, k=16, output_path="compressed_image.jpg", quality=85, blur_ksize=10):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Reshape image into 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixel_values)
    centers = np.uint8(kmeans.cluster_centers_)  # Convert cluster centers to integers

    # Replace pixels with cluster centers
    compressed_image = centers[labels].reshape(image.shape)

    # Apply Gaussian Blur to smooth out color bands
    blurred_image = cv2.GaussianBlur(compressed_image, (blur_ksize, blur_ksize), 0)

    # Convert back to BGR for OpenCV saving
    blurred_image_bgr = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR)

    # Save the compressed image as JPEG with specified quality
    cv2.imwrite(output_path, blurred_image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    # Get file size
    file_size = os.path.getsize(output_path)  # Size in bytes
    file_size_kb = file_size / 1024  # Convert to KB
    file_size_mb = file_size_kb / 1024  # Convert to MB

    print(f"Compressed file saved as {output_path}")
    print(f"File size: {file_size:.2f} bytes ({file_size_kb:.2f} KB, {file_size_mb:.2f} MB)")


img = cv2.imread(os.path.join(r'C:\Users\hp\Desktop\Machine learning\ml practice\image processing practice\sample images\bird.jpeg'))
compress_image(os.path.join(r'C:\Users\hp\Desktop\Machine learning\ml practice\image processing practice\sample images\Bird-Watching-Guide-Gear-GettyImages-1288108124.webp'), k=30, output_path="compressed_blurred.jpg", quality=75, blur_ksize=7)
