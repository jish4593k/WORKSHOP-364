import cv2
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def stack_images(scale, img_array):
    return cv2.vconcat([cv2.hconcat(img_row) for img_row in img_array])

def get_contours(img):
    img_contour = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:
            cv2.drawContours(img_contour, [cnt], -1, (255, 0, 0), 3)

    return img_contour

# Load the image
img = cv2.imread('resources/shapes.png')

# Convert image to Torch tensor for basic manipulation
img_tensor = torch.from_numpy(img)
img_tensor = img_tensor.permute(2, 0, 1)  # Change the channel order to match Torch conventions

# Perform some Tensor operations (e.g., flip horizontally)
img_tensor = torch.flip(img_tensor, dims=[2])

# Convert the tensor back to a NumPy array for OpenCV
img_processed = img_tensor.numpy().transpose(1, 2, 0).astype(np.uint8)

# Stack the processed image with the original and grayscale versions
img_stack = stack_images(0.6, [
    [img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img_processed],
    [get_contours(img), np.zeros_like(img), np.zeros_like(img)]
])

# Display the stacked images
cv2.imshow('Stacked Images', img_stack)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Example Seaborn plot
data = np.random.randn(100)
sns.histplot(data, kde=True)
plt.title("Seaborn Histogram")
plt.show()
