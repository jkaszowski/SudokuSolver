from ultralytics import YOLO
import cv2


# detection_size = 350
# capture_width = 640
# capture_height = 480
# x_min = capture_width // 2 - detection_size // 2
# x_max = capture_width // 2 + detection_size // 2
# y_min = capture_height // 2 - detection_size // 2
# y_max = capture_height // 2 + detection_size // 2



def applyYoloDetection(img):
    model = YOLO("good1.pt")
    classNames = ["sudoku"]
    results = model(img, stream=False, verbose=False)
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            margin_x = int((x2 - x1)/20)
            margin_y = int((y2 - y1)/20)
            cropped = img[y1-margin_y:y2+margin_y,x1-margin_x:x2+margin_x]
            cv2.imwrite("cropped_tmp.png", cropped)
            return True, cropped
    return False, None



# if __name__ == '__main__':
#     img = cv2.imread("input.jpg")
#     ret, img = applyYoloDetection(model, img)
#     if ret:
#         cv2.imshow("result", img)
#         cv2.waitKey(0);
#     else:
#         print("Not found any")
