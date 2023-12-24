import cv2

def inter2high(input_image, method=cv2.INTER_CUBIC):
    dout = cv2.resize(input_image, (100, 100), interpolation=method)
    gray = cv2.cvtColor(dout, cv2.COLOR_BGR2GRAY)
    return gray
