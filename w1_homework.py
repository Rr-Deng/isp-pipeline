import cv2
import numpy as np

'读取图片'
img=cv2.imread('OIP-C.webp')
img1=cv2.imread('city.png')

'通道分离'
#b,g,r=cv2.split(img)
#r[:]=0
#img=cv2.merge([b,g,r])
img[:,:,1]=0

#img[:, :, [2, 1]] = img[:, :, [1, 2]]

'读取像素'
pixel=img[101,101]
print(pixel)

'翻转图片'
flipimg=cv2.flip(img,1)

'逻辑运算'
result1=cv2.bitwise_xor(img,flipimg)
result2=cv2.addWeighted(img,0.5,flipimg,0.5,0.5)

'灰度化'
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

'滤波去噪，模糊图片，基本使用灰度图，美颜磨皮等操作使用彩色图'
blur=cv2.blur(gray1,(5,5))    #均值滤波  马赛克
blur1=cv2.GaussianBlur(gray1,(5,5),0)    #高斯滤波  通用，边缘识别前的预处理
blur2=cv2.medianBlur(gray1,5)    #中值滤波  椒盐噪声（黑白噪点)
blur3=cv2.bilateralFilter(gray1,9,75,75)    #双边滤波  磨皮、需要保留边缘的噪声

'阈值处理，二值化'
ret1,thresh1=cv2.threshold(blur1,127,255,cv2.THRESH_BINARY)    #手动设定阈值  已知阈值前提
thresh2=cv2.adaptiveThreshold(blur1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)    #自适应阈值  用于光照不均匀，比如手机照片直出图
ret3,thresh3=cv2.threshold(blur1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    #OTSU阈值 用于光照均匀，比如扫描件、文档

'图像形态学操作'
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
erode=cv2.erode(thresh2,kernel,iterations=1)
dilate=cv2.dilate(thresh2,kernel,iterations=1)
open_img=cv2.morphologyEx(thresh3,cv2.MORPH_OPEN,kernel)
close_img=cv2.morphologyEx(thresh2,cv2.MORPH_CLOSE,kernel)
gradient_img=cv2.morphologyEx(thresh2,cv2.MORPH_GRADIENT,kernel)

'ROI'
roi=img[105:200,105:200]
roi[:,:]=[66,66,66]
roi[50:150,50:150] = [0, 255, 0]
img[50:100,50:100]=[0,150,0]

'显示图片'
cv2.imshow('my image', img)
#cv2.imshow('gray', gray)
cv2.imshow('gray1', gray1)
#cv2.imshow('result', result2)
#cv2.imshow('thresh1', thresh1)
cv2.imshow('thresh2', thresh2)
#cv2.imshow('thresh3', thresh3)
#cv2.imshow('blur', blur)
cv2.imshow('blur1', blur1)
#cv2.imshow('blur2', blur2)
#cv2.imshow('blur3', blur3)  
#cv2.imshow('city',img1)
cv2.imshow('erode',erode)
cv2.imshow('dilate',dilate)
cv2.imshow('open_img',open_img)
cv2.imshow('close_img',close_img)
cv2.imshow('gradient_img',gradient_img)

'等待按键'
cv2.waitKey(0) 
cv2.destroyAllWindows()

'保存图片'
cv2.imwrite('new_image.png', img)
cv2.imwrite('resuilt2.png',result2)
cv2.imwrite('gray.png',gray)
cv2.imwrite('gray1.jpg',gray1)
cv2.imwrite('thresh1.png',thresh1)
cv2.imwrite('thresh2.png',thresh2)
cv2.imwrite('thresh3.png',thresh3)
cv2.imwrite('city.jpg',img1)
cv2.imwrite('erode.png',erode)
cv2.imwrite('dilate.png',dilate)
cv2.imwrite('open_img.png',open_img)
cv2.imwrite('close_img.png',close_img)
cv2.imwrite('gradient_img.png',gradient_img)
