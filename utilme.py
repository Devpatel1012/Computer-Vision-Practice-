import numpy as np
import cv2

def det_limits(color):
    c = np.uint8([[color]])
    hsvc = cv2.cvtColor(c,cv2.COLOR_BGR2HSV)

    lowerlimit = hsvc[0][0][0] - 10,100,100
    upperlimit = hsvc[0][0][0] + 10,255,255

    lowerlimit = np.array(lowerlimit,dtype=np.uint8)
    upperlimit = np.array(upperlimit,dtype=np.uint8)

    return lowerlimit,upperlimit