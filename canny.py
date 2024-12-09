import cv2
import numpy as np
import os

# Load the image
image = cv2.imread('cropped_tmp.png')
def detectSudokuGridOnImage(image):

    image = cv2.resize(image,(500,500))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocessing: Blur and Edge Detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imwrite(f"{os.getcwd()}/media/canny.png", edges)

    # Optional: Dilate edges to connect gaps
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    sudoku_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:  # Skip very small contours
            continue

        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the approximated contour has 4 sides and is nearly square
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.8 < aspect_ratio < 1.2:  # Ensure it's roughly square
                sudoku_contour = approx
                break

    if sudoku_contour is None:
        print("Sudoku grid not found!")
    else:
        # Draw the contour on the original image (optional)
        result = image.copy()
        cv2.drawContours(result, [sudoku_contour], -1, (0, 255, 0), 3)

        # Perspective transformation (as in the previous script)
        points = sudoku_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]

        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]

        (top_left, top_right, bottom_right, bottom_left) = rect
        widthA = np.linalg.norm(bottom_right - bottom_left)
        widthB = np.linalg.norm(top_right - top_left)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(top_right - bottom_right)
        heightB = np.linalg.norm(top_left - bottom_left)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # Validate Grid Structure
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(warped_gray, 128, 255, cv2.THRESH_BINARY_INV)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        grid_lines = cv2.add(vertical_lines, horizontal_lines)

        if np.count_nonzero(grid_lines) > 100:  # Simple check for grid presence
            print("Grid detected!")
        else:
            print("No grid found in warped image!")

        cv2.imwrite(f"{os.getcwd()}/media/warped.png", warped)
        return warped
