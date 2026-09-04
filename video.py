import cv2
import numpy as np

camera=cv2.VideoCapture(0)
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output.avi',fourcc,20,(640,480))

if not camera.isOpened():
    exit()
while True:
    ret,frame=camera.read()
    if not ret:
        break
    gray_camera=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    blur=cv2.GaussianBlur(gray_camera,(5,5),0)
    sobel_x=cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=3)
    sobel_y=cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=3)
    sobel_edge=np.sqrt(sobel_x**2+sobel_y**2)    #sobel算子梯度强度（运算结果）
    sobel_edge=np.uint8(sobel_edge/sobel_edge.max()*255) 

    out.write(cv2.cvtColor(sobel_edge,cv2.COLOR_GRAY2BGR)) 
    cv2.imshow('sobel_edge',sobel_edge)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
camera.release()
out.release()
cv2.destroyAllWindows()