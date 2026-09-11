import cv2
import numpy as np
import matplotlib.pyplot as plt
img_address=input('输入图片的文件地址：')
img=cv2.imread(img_address)

if img is None:
    print('图片加载失败')
    exit(0)
else:
    class choosing:
        def __init__(self):
            return None
        def gray(self):
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            cv2.imshow('gray',gray)
            cv2.waitKey(0)
            #cv2.imwrite(f"gray.{self.sorce}",gray)
            cv2.destroyAllWindows()
        def gaussian(self):
            gaussian=cv2.GaussianBlur(img,(5,5),0)
            cv2.imshow('gaussian',gaussian)
            cv2.waitKey(0)
            #cv2.imwrite(f"gaussian.{self.sorce}",gaussian) 
            cv2.destroyAllWindows()
        def canny(self):
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            canny=cv2.Canny(gray,100,200)
            cv2.imshow('canny',canny)
            cv2.waitKey(0)
            #cv2.imwrite(f"canny.{self.sorce}",canny)   
            cv2.destroyAllWindows()
        def sharp(self):
            lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
            lab[:,:,0]=lab[:,:,0].astype(np.float64)
            laplacian=cv2.Laplacian(lab[:,:,0],cv2.CV_64F,ksize=5)
            sharp=laplacian+lab[:,:,0]
            sharp=np.clip(sharp,0,255).astype(np.uint8)
            lab[:,:,0]=sharp
            result=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
            cv2.imshow('sharp',result)
            cv2.waitKey(0)
            #cv2.imwrite(f"sharp.{self.sorce}",result)
            cv2.destroyAllWindows()
        def hist_equalized(self):
            yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
            yuv[:,:,0]=cv2.equalizeHist(yuv[:,:,0])
            bgr=cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
            cv2.imshow('hist_equalized',bgr)
            cv2.waitKey(0)
            #cv2.imwrite(f"hist_equalized.{self.sorce}",bgr)
            cv2.destroyAllWindows()
        def sorce(self):
            sorce=input('你希望保存什么类型的图片：[1]jpg,[2]png,[3]bmp   按q键退出')
            if sorce=='q':
                exit(0)
            mapping={'1':'jpg','2':'png','3':'bmp'}
            if sorce in mapping:
                self.sorce=mapping[sorce]
            else:
                print('输入错误')
        def choice(self):
            while True:
                choice=input('希望对图片进行什么处理:[1]灰度化,[2]高斯模糊,[3]Canny边缘检测,[4]锐化,[5]直方图均衡化   按q键退出')
                if choice=='q':
                    exit(0)
                mapping={'1':self.gray,'2':self.gaussian,'3':self.canny,'4':self.sharp,'5':self.hist_equalized}
                if choice in mapping:
                    mapping[choice]()
                else:
                    print('输入错误')

c=choosing()
c.sorce()
c.choice()


