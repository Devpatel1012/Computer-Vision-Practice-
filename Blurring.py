import os

import cv2

img = cv2.imread(os.path.join(r'C:\Users\hp\Desktop\Machine learning\ml practice\image processing practice\sample images\bird.jpeg'))
resized_image =cv2.resize(img,(500,350))
#
k_size = 7
img_blur = cv2.blur(resized_image,(k_size,k_size))
img_guassian_blur = cv2.GaussianBlur(resized_image,(k_size,k_size),3)
img_median_blur  = cv2.medianBlur(resized_image,k_size)

cv2.imwrite(os.path.join(r'C:\Users\hp\Desktop\Machine learning\ml practice\image processing practice\sample images\bird_out.jpeg'),img)

# print(img.shape)
cv2.imshow('resized_img', resized_image)
# cv2.imshow('img_blur', img_blur)
# cv2.imshow('img_guassian_blur', img_guassian_blur)
# cv2.imshow('img_median_blur', img_median_blur)
cv2.waitKey(0)