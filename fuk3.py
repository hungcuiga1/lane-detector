import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import moviepy.editor as mp
def canny(image):
    gray=cv2.cvtColor(result2,cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray,(1,1),0)
    canny=cv2.Canny(blur,50,150)
    return canny
#cap = cv2.VideoCapture("test3.png")
# clip = mp.VideoFileClip("original.mp4")
# clip_resized = clip.resize(height=360)
image =cv2.imread('test.png')
#ret,image=cap.read()
low_white = np.array([0,51, 178])
up_white = np.array([180, 255, 255])
hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask_white = cv2.inRange(hsv,low_white,up_white)
result2=cv2.bitwise_and(image,image,mask=mask_white)
#image=imutils.resize(image,width=1280)
#lane_image = cv2.GaussianBlur(image, (1,1), 0)
a=[]
b=[]
canny2=canny(result2)
lines2=cv2.HoughLinesP(canny2,2,np.pi/180,100,np.array([]),minLineLength=30,maxLineGap=1)
print(lines2)
for line in lines2:
    x1,y1,x2,y2=line.reshape(4)
    parameters=np.polyfit((x1,x2),(y1,y2),1)
    cv2.line(result2,(x1,y1),(x2,y2),(0,0,255),5)
    a.append(parameters[0])
    b.append(parameters[1])
print(b,(y1+y2)/(x1-x2))
# plt.imshow(canny_image)
# plt.show()
cv2.imshow('result',result2)
cv2.waitKey(0)
# plt.imshow(canny_image)
# plt.show()
#     if cv2.waitKey(1) & 0xFFF == ord('q'): #press q to quit
#         break
# video.release()
# cv2.destroyAllWindows()