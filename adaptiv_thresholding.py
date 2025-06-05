import os

import cv2

img = cv2.imread(os.path.join(r"C:\Users\hp\Desktop\Machine learning\ml practice\image processing practice\sample images\handwritten.jpg"))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

adaotive_threshold = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 30)
adaotive_threshold2 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 18)

cv2.imshow('img',img)
cv2.imshow('thresh',adaotive_threshold)
cv2.imshow('thresh2',adaotive_threshold2)

cv2.waitKey(0)