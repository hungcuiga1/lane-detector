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
    # pts1=np.float32([[446,523],[824,523],[3,651],[1160,651]])
    # pts2=np.float32([[0,0],[400,0],[0,600],[400,600]])
    # array=cv2.getPerspectiveTransform(pts1,pts2)
    # lane_image=cv2.warpPerspective(image,array,(400,600))
    low_white = np.array([0,0, 214])
    up_white = np.array([180, 255, 255])
    hsv=cv2.cvtColor(lane_image, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv,low_white,up_white)
    result2=cv2.bitwise_and(lane_image,lane_image,mask=mask_white)
    canny2=cv2.Canny(result2,100,250)
    lines2=cv2.HoughLinesP(canny2,2,np.pi/180,100,np.array([]),minLineLength=10,maxLineGap=1)
    heso_a_am=[]
    heso_b_am=[]
    heso_a_duong=[]
    heso_b_duong=[]
    heso_a=[]
    heso_b=[]
    frame=0
    try:
        for line in lines2:
            x1,y1,x2,y2=line.reshape(4)
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
        for i in range(len(heso_a)):
            if heso_a[i]<0:
                heso_a_am.append(heso_a[i])
                heso_b_am.append(heso_b[i])
            if parameters[0]>0:
                heso_a_duong.append(heso_a[i])
                heso_b_duong.append(heso_b[i])
        print(heso_a_duong,heso_a_am)
# image =cv2.imread('phanlan.png')
# ret,image=cap.read()
        if heso_a_am and heso_a_duong is not None:
            vi_tri_am=0
            vi_tri_duong=0
            max_am=max(heso_a_am)
            for d in range(len(heso_a_am)):
                if heso_a_am[d]==max_am:
                    vi_tri_am=d
            min_duong=min(heso_a_duong)
            for dd in range(len(heso_a_duong)):
                if heso_a_duong[dd]==min_duong:
                    vi_tri_duong=dd
            #phuong trinh am la:y=heso_a_am[vi_tri_am]*x+heso_b_am[vi_tri_am]
            #phuong trinh duong:y=heso_a_duong[vi_tri_duong]*x+heso_b_duong[vi_tri_duong]
            y_am=heso_b_am[vi_tri_am]  #diem(0,y_am)
            x_am=(720-heso_b_am[vi_tri_am])/heso_a_am  #diem(x_am,0)
            if y_am<720:
                s_am=[0,y_am]
            else:
                s_am=[x_am,720]
            y_duong=heso_a_duong[vi_tri_duong]*1280+heso_b_duong[vi_tri_duong]
            x_duong=(720-heso_b_duong[vi_tri_duong])/heso_a_duong[vi_tri_duong]
            if y_duong<720:
                s_duong=[1280,y_duong]
            else:
                s_duong=[x_duong,720]
            #co s_duong[x,y] va s_am[x,y]
            if s_am[1]<s_duong[1]:
                s_duong[1]=s_am[1]
                s_duong[0]=(s_am[1]-heso_b_duong[vi_tri_duong])/heso_a_duong[vi_tri_duong]
            else:
                if s_am[1]>s_duong[1]:
                    s_am[1]=s_duong[1]
                    s_am[0]=(s_duong[1]-heso_b_am[vi_tri_am])/heso_a_am[vi_tri_am]
                else:
                    None
            #cat y=500
            x_am_500=(500- heso_b_am[vi_tri_am])/heso_a_am[vi_tri_am]
            x_duong_500=(500- heso_b_duong[vi_tri_duong])/heso_a_duong[vi_tri_duong]
            s_am_500=[x_am_500,500]
            s_duong_500=[x_duong_500,500]
            pts1=np.float32([s_am_500,s_duong_500,s_am,s_duong])
            pts2=np.float32([[0,0],[400,0],[0,600],[400,600]])
            array=cv2.getPerspectiveTransform(pts1,pts2)
            lane_image2=cv2.warpPerspective(image,array,(400,600))
    except TypeError:
        None
    cv2.imshow('hung',lane_image2)
#image=imutils.resize(image,width=1280)
# lane_image = cv2.GaussianBlur(image, (1,1), 0)
# lane_image=np.copy(image)
# pts1=np.float32([[446,523],[824,523],[3,651],[1160,651]])
# pts2=np.float32([[0,0],[400,0],[0,600],[400,600]])
# array=cv2.getPerspectiveTransform(pts1,pts2)
# lane_image=cv2.warpPerspective(image,array,(400,600))
# hsv=cv2.cvtColor(lane_image, cv2.COLOR_BGR2HSV)
# low_yellow = np.array([0,0, 215])
# up_yellow = np.array([180, 255, 255])
# mask_yellow = cv2.inRange(hsv,low_yellow,up_yellow)
# result=cv2.bitwise_and(lane_image,lane_image,mask=mask_yellow)
# result=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# histogram=np.sum(result[:,:],axis=0)
# print(histogram[25])
# averaged_lines=average_slope_intercept(lane_image,averaged_lines)
# line_image=display_lines(lane_image,lines)
# combo_image=cv2.addWeighted(lane_image,0.8,line_image,1,1)
# cv2.imshow('result',result)
# plt.figure()
# plt.plot(result)
# plt.show()
# cv2.waitKey(0)
# plt.imshow(canny_image)
# plt.show()
    if cv2.waitKey(1) & 0xFFF == ord('q'): #press q to quit
        break
cap.release()
cv2.destroyAllWindows()