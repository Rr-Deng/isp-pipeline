import cv2

img=cv2.imread('OIP-C.webp')
img1=cv2.imread('doro.jpg')

#b,g,r=cv2.split(img)
#r[:]=0
#img=cv2.merge([b,g,r])
img[:,:,1]=0

#img[:, :, [2, 1]] = img[:, :, [1, 2]]

pixel=img[101,101]
print(pixel)

flipimg=cv2.flip(img,1)

result1=cv2.bitwise_xor(img,flipimg)
result2=cv2.addWeighted(img,0.5,flipimg,0.5,0.5)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret1,thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
thresh2=cv2.adaptiveThreshold(gray,100,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
ret3,thresh3=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
blur=cv2.blur(thresh3,(5,5))
blur1=cv2.GaussianBlur(thresh3,(5,5),0)
blur2=cv2.medianBlur(thresh3,5)
blur3=cv2.bilateralFilter(thresh3,9,75,75)

roi=img[105:200,105:200]
roi[:,:]=[66,66,66]
roi[50:150,50:150] = [0, 255, 0]
img[50:100,50:100]=[0,150,0]

cv2.imshow('my image', img)
#cv2.imshow('gray', gray)
#cv2.imshow('result', result2)
#cv2.imshow('thresh1', thresh1)
#cv2.imshow('thresh2', thresh2)
cv2.imshow('thresh3', thresh3)
cv2.imshow('blur', blur)
cv2.imshow('blur1', blur1)
cv2.imshow('blur2', blur2)
cv2.imshow('blur3', blur3)  

cv2.waitKey(0) 
cv2.destroyAllWindows()

cv2.imwrite('new_image.png', img)
cv2.imwrite('resuilt2.png',result2)
cv2.imwrite('gray.png',gray)
cv2.imwrite('thresh1.png',thresh1)
cv2.imwrite('thresh2.png',thresh2)
cv2.imwrite('thresh3.png',thresh3)