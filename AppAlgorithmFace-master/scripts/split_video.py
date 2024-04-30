import cv2

path = "/Users/wxy/Desktop/face/mouse.mov"
folder = "/Users/wxy/Desktop/face/alive"
cap = cv2.VideoCapture(path)

ret = True
i = 0
while (cap.isOpened()) and ret:
    i += 1
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (280, 280))
        cv2.imwrite("{}/{}.jpg".format(folder, i), frame)

cap.release()
cv2.destroyAllWindows()
