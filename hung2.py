import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import moviepy.editor as mp
# def canny(image):
#     gray=cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)
#     blur=cv2.GaussianBlur(gray,(1,1),0)
#     canny=cv2.Canny(blur,50,150)
#     return canny
cap = cv2.VideoCapture("find10.mp4")
# # clip = mp.VideoFileClip("original.mp4")
# # clip_resized = clip.resize(height=360)
while True:
    ret,image=cap.read()
    lane_image = cv2.GaussianBlur(image,(5,5),0)
    lane_image=image[600:610,:,:]
    hsv=cv2.cvtColor(lane_image, cv2.COLOR_BGR2HSV)
    low_white = np.array([0,0, 221])
    up_white = np.array([180, 146, 255])
    mask_white = cv2.inRange(hsv,low_white,up_white)
    result3=cv2.bitwise_and(lane_image,lane_image,mask=mask_white)
        # triangle =np.array([[(0,0),(0,270),(1280,270),(1280,115)]])
        # mask=np.zeros_like(result2)
        # cv2.fillPoly(mask,triangle,255)
        # regionview=cv2.bitwise_and(lane_image,mask)
    canny2=cv2.Canny(result3,100,250)
    lines2=cv2.HoughLinesP(canny2,2,np.pi/180,100,np.array([]),minLineLength=5,maxLineGap=1)
    try:  
        heso_a=[]
        heso_b=[]
        for line in lines2:
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(lane_image,(x1,y1),(x2,y2),(255,0,0),5)
            parameters=np.polyfit((x1,x2),(y1,y2),1)
            cv2.line(lane_image,(x1,y1),(x2,y2),(255,0,0),5)
            loai=0
            if abs(parameters[0])>0.3:
                for g in range(len(heso_a)):
                    if abs(heso_a[g]-parameters[0])<0.5:
                        loai+=1
                if loai==0:
                    heso_a.append(parameters[0])
                    heso_b.append(parameters[1])
        print(heso_a)
    except TypeError:
        None
    cv2.imshow('hung',lane_image)
    # cv2.imshow('hung2',result2)
    # combo_image = cv2.addWeighted(lane_image,0.8,regionview,1,1)
    if cv2.waitKey(1) & 0xFFF == ord('q'): #press q to quit
        break
cap.release()
cv2.destroyAllWindows()