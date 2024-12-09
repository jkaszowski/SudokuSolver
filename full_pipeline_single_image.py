from ultralytics import YOLO
import numpy as np
from tensorflow import keras
import cv2
from SolvingAlgorithm import solve

model = keras.models.load_model('photo_processor/trained_digits_with_background.keras')

def writeNumberInCell(img, x, y, number):
    h, w = img.shape[:2]
    cell_height = h // 9
    cell_width = w // 9
    cv2.putText(img, number, (int((x+1/4) * cell_width), int((y+3/4) * cell_height)), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,100,0],2,cv2.LINE_AA)

def getClassForIdx(img, x, y, width, height):
    x_size = int(width / 9)
    y_size = int(height / 9)
    x_margin = 0 #x_size // 8
    y_margin = 0 #y_size // 8
    x_start = x_size * x + x_margin
    x_end = x_start + x_size - x_margin
    y_start = y_size * y + y_margin
    y_end = y_start + y_size - y_margin
    if x_end > width or y_end > height:
        print("Fatal error: wrong coordinates")
        return 0, 0
    cropped = img[y_start:y_end, x_start:x_end]
    tmp_img = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_LINEAR)
    nn_input = np.expand_dims(tmp_img, axis=0)
    nn_input = np.expand_dims(nn_input, axis=-1)



    predictions = model.predict(nn_input)
    predicted_classes = np.argmax(predictions, axis=1)
    classNames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    print(classNames[predicted_classes[0]])
    print(np.max(predictions))
    cv2.imshow("before", tmp_img)
    while True:
        if cv2.waitKey(1) == ord('q'):
            break
    if np.max(predictions) < 0.7:
        return "0"
    if predicted_classes[0] == 10:
        return 0
    return predicted_classes[0]


def main():
    img = cv2.imread("sudoku_simple.png")
    height, width = img.shape[:2]
    array = np.zeros((9, 9))
    for y in range(9):
        for x in range(9):
            array[y][x] = getClassForIdx(img, x, y, width, height)

    print(array)

    solve(array)
    print(array)
    writeNumberInCell(img,2,2, "8")
    cv2.imwrite("final.png",img )

if __name__ == "__main__":
    main()