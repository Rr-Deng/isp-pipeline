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

'图像形态学操作，使用处理后的二值化图像'   #处理二值化图像，填充孔洞，提取轮廓等
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))     #结构元素，矩形核
erode_morph=cv2.erode(thresh2,kernel,iterations=1)     #腐蚀，黑色区域增大，白色减少
dilate_morph=cv2.dilate(thresh2,kernel,iterations=1)   #膨胀，白色区域增大，黑色减少
open_morph=cv2.morphologyEx(thresh3,cv2.MORPH_OPEN,kernel)    #开运算，先腐蚀后膨胀
close_morph=cv2.morphologyEx(thresh2,cv2.MORPH_CLOSE,kernel)    #闭运算，先膨胀后腐蚀
gradient_morph=cv2.morphologyEx(thresh2,cv2.MORPH_GRADIENT,kernel)#梯度运算，提取前景轮廓

'图像边缘检测,canny、锐化使用灰度原图,sobel,laplacian需要高斯模糊处理--找的是突变的像素点'
canny_edge=cv2.Canny(gray1,50,150)  #canny边缘检测，五步：高斯滤波、sobel算子、极大值抑制、双阈值检测、边缘连接

sobel_x=cv2.Sobel(blur1,cv2.CV_64F,1,0,ksize=3)
sobel_y=cv2.Sobel(blur1,cv2.CV_64F,0,1,ksize=3)
sobel_edge=np.sqrt(sobel_x**2+sobel_y**2)    #sobel算子梯度强度（运算结果）
sobel_edge=np.uint8(sobel_edge/sobel_edge.max()*255)   #归一化，将sobel算子运算结果映射到0-255范围

laplacian=cv2.Laplacian(blur1,cv2.CV_64F)   #laplacian算子，提取边缘点
laplacian_sh=cv2.Laplacian(gray1,cv2.CV_64F)    #laplace锐化时选用灰度图原图
sharpened=gray1-laplacian_sh     #加上绝对值，或者减去原始值,为后面锐化做准备
laplacian=np.absolute(laplacian)    #取绝对值，为后面归一化做准备
laplacian_sh=np.absolute(laplacian_sh)    #取绝对值，为后面归一化做准备
laplacian_edge=np.uint8(laplacian/laplacian.max()*255)  #归一化，防止绝对值会超出255上限，转为无符号的八位整数(像素值必须为整数)
laplacian_sh=np.uint8(laplacian_sh/laplacian_sh.max()*255)  #归一化，防止绝对值会超出255上限，转为无符号的八位整数(像素值必须为整数)
sharpened=np.clip(sharpened,0,255).astype(np.uint8)    #锐化运算结果演示，放大相邻像素值的差异，原理是将原始图像减去laplacian提取的边缘细节

'图像轮廓检测，使用二值化图像--找的是连续的像素点，即闭合的一条曲线'


'ROI'
roi=img[105:200,105:200]
roi[:,:]=[66,66,66]
roi[50:150,50:150]=[0, 255, 0]
img[50:100,50:100]=[0,150,0]

'显示图片'
#cv2.imshow('my image',img)
#cv2.imshow('gray',gray)
cv2.imshow('gray1',gray1)
#cv2.imshow('result',result2)
#cv2.imshow('thresh1',thresh1)
#cv2.imshow('thresh2',thresh2)
#cv2.imshow('thresh3',thresh3)
#cv2.imshow('blur',blur)
#cv2.imshow('blur1',blur1)
#cv2.imshow('blur2',blur2)
#cv2.imshow('blur3',blur3)  
#cv2.imshow('city',img1)
#cv2.imshow('erode_morph',erode_morph)
#cv2.imshow('dilate_morph',dilate_morph)
#cv2.imshow('open_morph',open_morph)
#cv2.imshow('close_morph',close_morph)
#cv2.imshow('gradient_morph',gradient_morph)
cv2.imshow('canny_edge',canny_edge)
cv2.imshow('sobel_edge',sobel_edge)
cv2.imshow('laplacian_edge',laplacian_edge)
cv2.imshow('laplacian_sh',laplacian_sh)
cv2.imshow('sharpened',sharpened)

'等待按键'
cv2.waitKey(0) 
cv2.destroyAllWindows()

'保存图片'
cv2.imwrite('new_image.png',img)
cv2.imwrite('resuilt2.png',result2)
cv2.imwrite('gray.png',gray)
cv2.imwrite('gray1.png',gray1)
cv2.imwrite('thresh1.png',thresh1)
cv2.imwrite('thresh2.png',thresh2)
cv2.imwrite('thresh3.png',thresh3)
cv2.imwrite('city.png',img1)
cv2.imwrite('erode_morph.png',erode_morph)
cv2.imwrite('dilate_morph.png',dilate_morph)
cv2.imwrite('open_morph.png',open_morph)
cv2.imwrite('close_morph.png',close_morph)
cv2.imwrite('gradient_morph.png',gradient_morph)
cv2.imwrite('canny_edge.png',canny_edge)
cv2.imwrite('sobel_edge.png',sobel_edge)
cv2.imwrite('laplacian_edge.png',laplacian_edge)
cv2.imwrite('laplacian_sh.png',laplacian_sh)
cv2.imwrite('sharpened.png',sharpened)
