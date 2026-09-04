import cv2
import numpy as np
import matplotlib.pyplot as plt

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
laplacian_sh_edge=np.uint8(laplacian_sh/laplacian_sh.max()*255)  #归一化，防止绝对值会超出255上限，转为无符号的八位整数(像素值必须为整数)
sharpened=np.clip(sharpened,0,255).astype(np.uint8)    #锐化运算结果演示，放大相邻像素值的差异，原理是将原始图像减去laplacian提取的边缘细节

'图像轮廓检测，使用二值化图像--找的是连续的像素点，即闭合的一条曲线'
contours,hierarchy=cv2.findContours(canny_edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    #寻找canny中所有轮廓，contours是轮廓列表，hierarchy是轮廓层级关系
output=np.zeros_like(img1)    #创建全黑的图像，用于绘制轮廓
cv2.drawContours(output,contours,-1,(255,0,0),2)   #绘制所有轮廓
output1=output.copy()       #复制一张，后面对比用
output2=output.copy()       #复制一张，后面对比用
output3=output.copy()       #复制一张，后面对比用
output4=output.copy()       #复制一张，后面对比用
for contour in contours:   #遍历每个轮廓
    #计算轮廓面积和周长
    area=cv2.contourArea(contour)
    length=cv2.arcLength(contour,True)
    print(area,length)
    #绘制轮廓的边界矩形
    x,y,w,h=cv2.boundingRect(contour)
    cv2.rectangle(output1,(x,y),(x+w,y+h),(0,255,0),2)
    #绘制轮廓的最小面积矩形
    rect=cv2.minAreaRect(contour)
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    cv2.drawContours(output2,[box],0,(0,0,255),2)
    #绘制轮廓的最小外接圆
    (x,y),radius=cv2.minEnclosingCircle(contour)
    center=(int(x),int(y))
    radius=int(radius)
    cv2.circle(output3,center,radius,(0,255,255),2)
    #绘制轮廓的近似多边形
    epsilon=0.01*cv2.arcLength(contour,True)
    approx=cv2.approxPolyDP(contour,epsilon,True)
    cv2.drawContours(output4,[approx],0,(255,255,0),2)

'图像直方图'
hist=cv2.calcHist([gray1],[0],None,[256],[0,256])    #计算灰度图直方图，[gray1]是输入图像，[0]是通道索引，None是掩码，[256]是直方图bin数（分为几个区间），[0,256]是直方图范围

plt.plot(hist)
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

equalized_img=cv2.equalizeHist(gray1)    #通过重新分配直方图像素强度来增强图片对比度（让像素强度分布更加均匀）

colors=('b','g','r')
for i,color in enumerate(colors):    #enumerate()函数同时获取索引和元素
    hist1=cv2.calcHist([img1],[i],None,[256],[0,256])    #计算每个通道的直方图
    plt.plot(hist1,color=color)
    
plt.title("Color Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Pixel Count")
plt.show()

b,g,r=cv2.split(img1)        #将通道分离，然后分别直方图均衡化
b_eq=cv2.equalizeHist(b)
g_eq=cv2.equalizeHist(g)
r_eq=cv2.equalizeHist(r)
equalize_img1=cv2.merge((b_eq,g_eq,r_eq))

similarity=cv2.compareHist(hist,hist1,cv2.HISTCMP_CORREL)      #比较两个直方图的相似度，cv2.HISTCMP_CORREL是相关系数，0-1之间，1表示完全相似，0表示完全相反
print("histgram similarity:",similarity)

'ROI'
roi=img[105:200,105:200]
roi[:,:]=[66,66,66]
roi[50:150,50:150]=[0, 255, 0]
img[50:100,50:100]=[0,150,0]

'显示图片'
cv2.imshow('my image',img1)
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
#cv2.imshow('sobel_edge',sobel_edge)
#cv2.imshow('laplacian_edge',laplacian_edge)
#cv2.imshow('laplacian_sh_edge',laplacian_sh_edge)
#cv2.imshow('sharpened',sharpened)
cv2.imshow('output',output)  
cv2.imshow('bounding_rect',output1)
cv2.imshow('min_area_rect',output2)
cv2.imshow('min_enclosing_circle',output3)
cv2.imshow('approx',output4)
cv2.imshow('equalized_img',equalized_img)
cv2.imshow('equalize_img1',equalize_img1)

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
cv2.imwrite('laplacian_sh_edge.png',laplacian_sh_edge)
cv2.imwrite('sharpened.png',sharpened)
cv2.imwrite('output.png',output)
cv2.imwrite('bounding_rect.png',output1)
cv2.imwrite('min_area_rect.png',output2)
cv2.imwrite('min_enclosing_circle.png',output3)
cv2.imwrite('approx.png',output4)
cv2.imwrite('equalized_img.png',equalized_img)
cv2.imwrite('equalize_img1.png',equalize_img1)
