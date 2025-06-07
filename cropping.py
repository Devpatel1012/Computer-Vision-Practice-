import os
import cv2

img = cv2.imread(os.path.join(r'C:\Users\hp\Desktop\Machine learning\ml practice\image processing practice\sample images\dog2.jpg'))
print(img.shape)
cropped_img = img[170:550, 250:600]

cv2.imshow('img',img)
cv2.imshow('cropped_img',cropped_img)
cv2.waitKey(0)