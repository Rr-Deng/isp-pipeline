import cv2

img = cv2.imread('OIP-C.webp')

img[100,100]=[255,255,255]

roi=img[105:200,105:200]
roi[50:150, 50:150] = [0, 255, 0]

#b,g,r=cv2.split(img)
#r[:]=0
#img=cv2.merge([b,g,r])
img[:,:,0]=0
#img[:, :, [2, 1]] = img[:, :, [1, 2]]

cv2.imshow('my image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('new_image.png', img)