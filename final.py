import cv2
import numpy as np
# import imutils
# import matplotlib.pyplot as plt
# import moviepy.editor as mp
# # def tachmau(vid_copy):
#     hsv=cv2.cvtColor(vid_copy, cv2.COLOR_BGR2HSV)
#     low_white = np.array([15,50,0])
#     up_white = np.array([40,255,255])
#     mask = cv2.inRange(hsv,low_white,up_white)
#     return mask
# def timvach(mauvang,lines):
#     vach=[]
#     for line in lines:
#         x1,y1,x2,y2=line.reshape(4)
#         parameters=np.polyfit((x1,x2),(y1,y2),1)
#         slope=parameters[0]
#         intercept=parameters[1]
#         vach.append((slope,intercept))
#         vach2=np.average(vach,axis=0)
#         line=vietpt(mauvang,vach2)
#     return np.array([line])
# def vietpt(mauvang,vach2):
#     try:
#         slope, intercept = vach2
#     except TypeError:
#         slope, intercept = 1,image.shape[0]
#     y1=vid_copy.shape[0]
#     y2=int(y1*(4/5))
#     x1=int((y1-intercept)/slope)
#     x2=int((y2-intercept)/slope)
#     return np.array([x1,y1,x2,y2])
# def display_lines(vid_copy,lines):
#     line_image=np.zeros_like(vid_copy)
#     if lines is not None:
#         for line in lines:
#             x1,y1,x2,y2=line.reshape(4)
#             try:
#                 cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
#             except OverflowError:
#                 None  
#     return line_image
video=cv2.VideoCapture("C:/Users/hung phung/Videos/sua_vid/v1_1.mp4")
frame=0
while True:
    ret,vid=video.read()
    # vid_copy=cv2.GaussianBlur(vid, (5,5), 0)
    frame+=1
    #=====================================================
    # mauvang=tachmau(vid_copy)
    # canny_image = cv2.Canny(mauvang, 100, 255)
    # triangle =np.array([[(0,0),(1200,700),(1200,600),(0,600)]])
    # mask2=np.zeros_like(canny_image)
    # cv2.fillPoly(mask2,triangle,255)
    # regionview=cv2.bitwise_and(canny_image,mask2)
    # #======================================================================
    # lines=cv2.HoughLinesP(regionview,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=1)
    # print(lines)
    # averaged_lines=timvach(vid_copy,lines)
    # line_image=display_lines(vid_copy,averaged_lines)
    # combo_image=cv2.addWeighted(vid_copy,0.8,line_image,1,1)
    print(frame)
    cv2.imshow('answer',vid)
    if cv2.waitKey(1) & 0xFFF == ord('q'): #press q to quit
        break
video.release()
cv2.destroyAllWindows()
