import cv2
import numpy as np
lane_image = cv2.imread('hung.jpg')
low_white = np.array([0,0, 171])
up_white = np.array([180, 146, 255])
hsv=cv2.cvtColor(lane_image, cv2.COLOR_BGR2HSV)
mask_white = cv2.inRange(hsv,low_white,up_white)
result3=cv2.bitwise_and(lane_image,lane_image,mask=mask_white)
img2=cv2.cvtColor(result3, cv2.COLOR_BGR2GRAY)
s1=[90,354]
s2=[90,891]
d=354
tim=[]
for i in range(891-354):
    a=img2[90,d]
    tim.append(a)
    d+=1
dem=0
for d in range(len(tim)):
    if tim[d]>170:
        if tim[d-1]<30:
            dem+=1
print(dem)
cv2.waitKey(0)