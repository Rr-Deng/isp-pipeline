import cv2

img = cv2.imread('OIP-C.webp')

cv2.imshow('my image', img)
cv2.waitKey(0)

cv2.imwrite('new_image.png', img)