import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import moviepy.editor as mp
# def canny(image):
#     gray=cv2.cvtColor(lane_image,cv2.COLOR_BGR2GRAY)
#     blur=cv2.GaussianBlur(gray,(1,1),0)
#     canny=cv2.Canny(blur,100,100)
#     return canny
# def region_view(image):
#     height=image.shape[0]
#     triangle =np.array([[(220,680),(1200,680),(1200,480),(620,430)]])
#     #triangle =np.array([[(0,500),(0,720),(1200,720),(640,300),(540,300)]])
#     mask=np.zeros_like(image)
#     cv2.fillPoly(mask,triangle,255)
#     regionview=cv2.bitwise_and(image,mask)
#     return regionview
# def find(image,lines):
#     a=[]
#     b=0
#     for line in lines:
#         x1,y1,x2,y2=line.reshape(4)
#         parameters=np.polyfit((x1,x2),(y1,y2),1)
#         slope=parameters[0]
#         intercept=parameters[1]
#         if slope!=0:
#             for i in range(len(a)):
#                 if abs(slope-a[i])<0.3:
#                     b+=1
#             if b==0:
#                 a.append(slope)
#     return a
# def display_lines(image,lines):
#     line_image=np.zeros_like(image)
#     if lines is not None:
#         for line in lines:
#             x1,y1,x2,y2=line.reshape(4)
#             try:
#                 cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
#             except OverflowError:
#                 None
#     return line_image
cap = cv2.VideoCapture("find10.mp4")
# clip = mp.VideoFileClip("original.mp4")
# clip_resized = clip.resize(height=360)
#image =cv2.imread('test.png')
while True:
    ret,image=cap.read()


#image=imutils.resize(image,width=1280)
    lane_image = cv2.GaussianBlur(image, (5,5), 0)
    lane_image = lane_image[450:,:,:]
    cv2.imshow('hung',lane_image)
    #=====================================================================================================
#     hsv=cv2.cvtColor(lane_image, cv2.COLOR_BGR2HSV)
#     low_white = np.array([0, 0, 140])
#     up_white = np.array([20, 255, 255])
#     mask_white = cv2.inRange(hsv,low_white,up_white)
#     canny_image = cv2.Canny(mask_white, 100, 255)
#     cropped_image=region_view(canny_image)
#     lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
#     a=find(lane_image,lines)
#     line_image=display_lines(lane_image,lines)
#     combo_image=cv2.addWeighted(lane_image,0.8,line_image,1,1)
#     #========================================================================================================
#     # lane_image=np.copy(image)
#     # pts1=np.float32([[0,500],[620,280],[0,720],[1200,720]])
#     # pts2=np.float32([[0,0],[400,0],[0,600],[400,600]])
#     # array=cv2.getPerspectiveTransform(pts1,pts2)
#     # lane_image=cv2.warpPerspective(image,array,(400,600))
#     #=======================================================================================
#     #===============================================================================
#     #==================================================================================================
#     # low_blue = np.array([100, 150, 0])
#     # up_blue = np.array([140, 255, 255])
#     try:
#         cv2.putText(combo_image,'so lan:'+str(len(a)), (int(100),int(100)),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,255,255), lineType=cv2.LINE_AA)
#     except TypeError:
#         None   
#         #for g in range(maax):
#             #cv2.putText(combo_image, str(g), (int((abs(x2))),int((abs((y1-y2)/2+y2)))),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,255,255), lineType=cv2.LINE_AA)
#             #cv2.putText(combo_image,'so lan:'+str(g), (int(100),int(100)),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,255,255), lineType=cv2.LINE_AA)
#     cv2.imshow('answer',combo_image)
# #cv2.waitKey(0)
# # plt.imshow(canny_image)
# # plt.show()
    if cv2.waitKey(1) & 0xFFF == ord('q'): #press q to quit
        break
# cap.release()
# cv2.destroyAllWindows()