import cv2
import numpy as np

# Load the image
image = cv2.imread('cropped.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur and edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# edges = cv2.Canny(blurred, 50, 100)
edges = cv2.threshold(blurred,50,200)
cv2.imshow("edges", edges)
cv2.waitKey(1000)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest rectangle
largest_contour = max(contours, key=cv2.contourArea)
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

# Ensure the detected contour has 4 points
if len(approx) == 4:
    points = approx.reshape(4, 2)

    # Order the points (top-left, top-right, bottom-right, bottom-left)
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    ordered_points = order_points(points)

    # Define the width and height of the new image
    width = height = 450  # Standard size for Sudoku grid

    # Define the destination points
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_points, dst)

    # Perform the perspective warp
    warped = cv2.warpPerspective(image, matrix, (width, height))

    # Save and display the output
    cv2.imwrite('sudoku_cropped.png', warped)
    cv2.imshow("Cropped Sudoku", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to find a proper Sudoku grid.")
