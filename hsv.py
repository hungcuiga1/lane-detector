import cv2
import numpy as np
image =cv2.imread('test2.png')
lane_image=image[450:,:,:]
hsv=cv2.cvtColor(lane_image, cv2.COLOR_BGR2HSV)
low_white = np.array([0,0, 49])
up_white = np.array([179, 41, 84])
mask_white = cv2.inRange(hsv,low_white,up_white)
result=cv2.bitwise_and(lane_image,lane_image,mask=mask_white)
canny=cv2.Canny(result,250,250)
# pts1=np.float32([[0,500],[620,280],[0,720],[1200,720]])
# pts2=np.float32([[0,0],[400,0],[0,600],[400,600]])
# array=cv2.getPerspectiveTransform(pts1,pts2)
# lane_image=cv2.warpPerspective(image,array,(400,600))
cv2.imshow('hung',result)
cv2.waitKey(0)