import cv2
import numpy as np
import os
from tensorflow import keras

from YoloSingleImage import applyYoloDetection
from canny import detectSudokuGridOnImage
from SolvingAlgorithm import solve

def writeNumberInCell(img, x, y, number):
    h, w = img.shape[:2]
    cell_height = h // 9
    cell_width = w // 9
    cv2.putText(img, number, (int((x+1/4) * cell_width), int((y+3/4) * cell_height)), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255],2,cv2.LINE_AA)

def getClassForIdx(img, x, y, width, height):
    model = keras.models.load_model(f'{os.getcwd()}/dobryModel2.keras')
    x_size = int(width / 9)
    y_size = int(height / 9)
    x_margin = 8 #x_size // 8
    y_margin = 8 #y_size // 8
    x_start = x_size * x + x_margin
    x_end = x_start + x_size - x_margin
    y_start = y_size * y + y_margin
    y_end = y_start + y_size - y_margin
    if x_end > width or y_end > height:
        print("Fatal error: wrong coordinates")
        return 0, 0
    cropped = img[y_start:y_end, x_start:x_end]
    tmp_img = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_LINEAR)
    tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
    nn_input = np.expand_dims(tmp_img, axis=0)
    nn_input = np.expand_dims(nn_input, axis=-1)



    predictions = model.predict(nn_input, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    classNames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    # print(classNames[predicted_classes[0]])
    # print(np.max(predictions))
    # cv2.imshow("before", tmp_img)
    # cv2.imwrite(f"recognized/{x}_{y}.png",tmp_img)
    # while True:
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    if np.max(predictions) < 0.5:
        return "0"
    if predicted_classes[0] == 10:
        return 0
    return predicted_classes[0]

def process_photo(img_path):
    img = cv2.imread(img_path)
    cv2.imwrite( f"{os.getcwd()}/media/input.png",img)
    ret, img = applyYoloDetection(img)
    if not ret:
        print("Yolo stage failed!")
        return False, "Yolo stage failed"
    cv2.imwrite(f"{os.getcwd()}/media/cropped.png", img)
    post_yolo = detectSudokuGridOnImage(img)
    cropped = post_yolo
    height, width = cropped.shape[:2]
    array = np.zeros((9, 9))
    for y in range(9):
        for x in range(9):
            array[y][x] = getClassForIdx(cropped, x, y, width, height)

    print(array)
    original_array = array.copy()
    solve(array)
    print(array)
    for y in range(9):
        for x in range(9):
            if original_array[y][x] == 0:
                writeNumberInCell(post_yolo, x, y, f"{int(array[y][x])}")
                print(f"Wriging {array[y][x]} at ({x},{y})")

    cv2.imwrite(f"{os.getcwd()}/media/output.png", post_yolo)
    return True, "Operation successfull"