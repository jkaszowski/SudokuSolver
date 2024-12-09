import cv2
from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('filename')
    args = parser.parse_args()

    model = YOLO("good1.pt")
    img = cv2.imread(args.filename)
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            cropped = img[y1:y2, x1:x2]
            # put box in cam
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # cv2.imwrite("output.png",img)
            cv2.imwrite("cropped.png",cropped)



if __name__ == '__main__':
    main()
