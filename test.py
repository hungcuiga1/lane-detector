import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import moviepy.editor as mp
from datetime import datetime

image=cv2.imread('ke.jpg')
lane_image = cv2.GaussianBlur(image,(5,5),0)
# hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# low_yellow = np.array([0,86, 140])
# up_yellow = np.array([180, 255, 255])
# mask_yellow = cv2.inRange(hsv,low_yellow,up_yellow)
# result=cv2.bitwise_and(image,image,mask=mask_yellow)
canny2=cv2.Canny(image,100,250)
lines2=cv2.HoughLinesP(canny2,2,np.pi/180,100,np.array([]),minLineLength=10,maxLineGap=1)
# dem=0
for line1 in lines2:
    x11,y11,x22,y22=line1.reshape(4)
    parameters2=np.polyfit((x11,x22),(y11,y22),1)
    cv2.line(image,(x11,y11),(x22,y22),(255,0,0),5)
cv2.imwrite('D:/hung8.jpg',image)
# cv2.imshow('hung',hsv)
cv2.waitKey(0)