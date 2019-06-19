import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import moviepy.editor as mp
from scipy import stats
cap = cv2.VideoCapture("original.mp4")
while True:
#image =cv2.imread('test.png')
    ret,image=cap.read()
#image=imutils.resize(image,width=1280)
    lane_image = cv2.GaussianBlur(image, (5,5), 0)
    hsv=cv2.cvtColor(lane_image, cv2.COLOR_BGR2HSV)
    low_white = np.array([0, 0, 140])
    up_white = np.array([20, 250, 250])
    mask_white = cv2.inRange(hsv,low_white,up_white)
    pts1=np.float32([[550,480],[800,480],[260,680],[1200,680]])
    pts2=np.float32([[0,0],[400,0],[0,600],[400,600]])
    array=cv2.getPerspectiveTransform(pts1,pts2)
    view=cv2.warpPerspective(image,array,(400,600))
    cv2.imshow('hung',view)
    if cv2.waitKey(1) & 0xFFF == ord('q'): #press q to quit
        break
cap.release()
cv2.destroyAllWindows()