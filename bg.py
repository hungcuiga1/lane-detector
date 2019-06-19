import numpy as np
import cv2

# cap = cv2.VideoCapture('find8.mp4')
# fgbg = cv2.createBackgroundSubtractorMOG2()

# while(1):
#     ret, frame = cap.read()

#     fgmask = fgbg.apply(frame)
 
#     cv2.imshow('fgmask',frame)
#     cv2.imshow('frame',fgmask)

    
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
    

# cap.release()
# cv2.destroyAllWindows()
image=cv2.imread('phanlan.png')
# image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image.shape[1])
image=cv2.resize(image,(200,200))
cv2.imshow('hung',image)
cv2.waitKey(0)