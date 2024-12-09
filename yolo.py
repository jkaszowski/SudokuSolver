from ultralytics import YOLO
import cv2
import math
# start webcam

detection_size = 350
capture_width = 640
capture_height = 480
x_min = capture_width // 2 - detection_size // 2
x_max = capture_width // 2 + detection_size // 2
y_min = capture_height // 2 - detection_size // 2
y_max = capture_height // 2 + detection_size // 2

cap = cv2.VideoCapture(0)
cap.set(3, capture_width)
cap.set(4, capture_height)

# model
model = YOLO("good1.pt")

# object classes
classNames = ["sudoku"]


def applyYoloDetection(model, img):
    results = model(img, stream=True)
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values
            if x_min <= x1 <= x_min+20 and x_max-20 <= x2 <= x_max and y_min <= y1 <= y_min+20 and y_max-20 <= y2 <= y_max:
                cropped = img[y1-10:y2+10, x1-10:x2+10]
                cv2.imwrite("cropped_tmp.png", cropped)
                return True, cropped
            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, f"{classNames[cls]} - {confidence}", org, font, fontScale, color, thickness)

    return False, img

def getSudokuBoardFromYolo():
    while True:
        success, img = cap.read()
        ret, img = applyYoloDetection(model, img)
        if ret == True:
            return img
        cv2.rectangle(img,(x_min,y_min),(x_max,y_max),[255,0,0])
        cv2.rectangle(img, (x_min + 20, y_min+20), (x_max-20, y_max-20), [0, 255, 0])

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            exit(1)

if __name__ == '__main__':
    getSudokuBoardFromYolo()

cap.release()
cv2.destroyAllWindows()
