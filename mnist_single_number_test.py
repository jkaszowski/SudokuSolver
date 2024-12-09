import numpy as np
from tensorflow import keras
import cv2

model = keras.models.load_model('model.keras')

expectations_colorful = [
    [1,0,0,0,8,0,0,0,9],
    [0,5,0,6,0,1,0,2,0],
    [0,0,0,5,0,3,0,0,0],
    [0,9,6,1,0,4,8,3,0],
    [3,0,0,0,6,0,0,0,5],
    [0,1,5,9,0,8,4,6,0],
    [0,0,0,7,0,5,0,0,0],
    [0,8,0,3,0,9,0,7,0],
    [5,0,0,0,1,0,0,0,3]
]

expectations = [
    [5,3,0,0,7,0,0,0,0],
    [6,0,0,1,9,5,0,0,0],
    [0,9,8,0,0,0,0,6,0],
    [8,0,0,0,6,0,0,0,3],
    [4,0,0,8,0,3,0,0,1],
    [7,0,0,0,2,0,0,0,6],
    [0,6,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,5],
    [0,0,0,0,8,0,0,7,9]
]
# to_check = [["colorful.png"]]
failure = 0

for y in range(9):
    for x in range(9):
        img = cv2.imread(f"recognized/{x}_{y}.png")
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img,verbose=0)
        predicted_classes = np.argmax(predictions,axis=1)
        if np.max(predictions) < 0.5:
            predicted_classes[0] = 0
        if predicted_classes != expectations[y][x]:
            if not (predicted_classes == 10 and expectations[y][x] == 0):
                print(f"Cell ({x},{y}) is {expectations[y][x]} but recognized as {predicted_classes[0]}")
                failure+=1

if(failure >0 ):
    print(f"Errors: {failure}")
    exit(1)