import cv2
import numpy as np

camera=cv2.VideoCapture(0)
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output.avi',fourcc,20,(640,480))

if not camera.isOpened():
    exit()
while True:
    ret,frame1=camera.read()
    if not ret:
        break
    cv2.imshow('frame',frame1)
    out.write(frame1)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
camera.release()
out.release()

cap=cv2.VideoCapture("output.avi")
ret,frame=cap.read()

x,y,w,h=340,250,100,225
track_window=(x,y,w,h)

roi=frame[y:y+h,x:x+w]

roi_hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
mask=cv2.inRange(roi_hsv,np.array([0,60,32]), np.array([180,255,255]))    #设置掩码，必须是numpy数组
roi_hist=cv2.calcHist([roi_hsv],[0],mask,[180],[0,180])   #计算直方图，也叫颜色指纹，用于后续匹配
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)   #归一化处理

term_crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1)    #限制条件：迭代10次或者窗口移动小于1个像素

while True:
    ret,frame=cap.read()
    if not ret:
        break
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    dst=cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)   #生成概率图

    #ret,track_window=cv2.meanShift(dst,track_window,term_crit)   #根据颜色指纹进行匹配
    #x,y,w,h=track_window
    #img=cv2.rectangle(frame,(x,y),(x+w,y+h),255,2)

    ret,track_window=cv2.CamShift(dst,track_window,term_crit)
    pts=cv2.boxPoints(ret)
    pts=np.intp(pts)
    img=cv2.polylines(frame,[pts],True,255,2)

    cv2.imshow('tracking',img)
    if cv2.waitKey(30)&0xFF==27:
        break

cap.set(cv2.CAP_PROP_POS_FRAMES,0)    #把视频指针重置到开头

#mog=cv2.bgsegm.createBackgroundSubtractorMOG()
mog=cv2.createBackgroundSubtractorMOG2()

while True:
    ret,frame2=cap.read()
    if not ret:
        break
    fg=mog.apply(frame2)
    cv2.imshow('fg',fg)
    if cv2.waitKey(30)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()