import cv2
import numpy as np
# import imutils
# import matplotlib.pyplot as plt
# import moviepy.editor as mp

# cap = cv2.VideoCapture("test2.mp4")
img=cv2.imread('test3.png')
k=0
tb=0
rong=img.shape[0]
dai=img.shape[1]
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for i in range(rong):
    for j in range(dai):
        k=k+img[i,j]
tb=k/(1366*768)
print(k,tb)
# while True:
#     ret,image = cap.read()
#     pts1=np.float32([[0,500],[620,280],[0,720],[1200,720]])
#     pts2=np.float32([[0,0],[800,0],[0,400],[800,400]])
#     array=cv2.getPerspectiveTransform(pts1,pts2)
#     lane_image=cv2.warpPerspective(image,array,(800,400))
#     gray=cv2.cvtColor(lane_image,cv2.COLOR_BGR2GRAY)
#     blur=cv2.GaussianBlur(gray,(1,1),0)
#     canny=cv2.Canny(blur,50,150)
#     #=============================================================================
#     cv2.imshow("hunghung",canny)
    # if cv2.waitKey(1)&0xFFF==ord('q'):
    #     break
# cap.release()
# cv2.destroyAllWindows()
cv2.waitKey(0)