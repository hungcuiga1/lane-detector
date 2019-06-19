import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import moviepy.editor as mp
def make_coordinates(image,line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 1,image.shape[0]
    y1=image.shape[0]
    y2=int(y1*(4/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])
def average_slope_intercept(image,lines):
    left_fit1=[]
    left_fit2=[]
    left_fit3=[]
    right_fit1=[]
    right_fit2=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        intercept=parameters[1]
        b=round(slope)
        if b==-2:
            left_fit1.append((slope,intercept))
        if b==-1:
            left_fit2.append((slope,intercept))
        if b==0:
            left_fit3.append((slope,intercept))
        if b==1:
            right_fit1.append((slope,intercept))
        if b==2:
            right_fit2.append((slope,intercept)) 
    # if left_fit1 is not None:  
    left_fit_average1=np.average(left_fit1,axis=0)
    left_line1=make_coordinates(image,left_fit_average1)
    # if left_fit2 is not None:
    left_fit_average2=np.average(left_fit2,axis=0)
    left_line2=make_coordinates(image,left_fit_average2)
    # if left_fit3 is not None:
    left_fit_average3=np.average(left_fit3,axis=0)
    left_line3=make_coordinates(image,left_fit_average3)
    # if right_fit1 is not None:
    right_fit_average1=np.average(right_fit1,axis=0)
    right_line1=make_coordinates(image,right_fit_average1)
    # if right_fit2 is not None:
    right_fit_average2=np.average(right_fit2,axis=0)
    right_line2=make_coordinates(image,right_fit_average2)
    # if left_fit1,left_fit2,left_fit3,right_fit1,right_fit2 is not None:
    return np.array([left_line1,left_line2,left_line3,right_line1,right_line2])
    # left_line=make_coordinates(image,left_fit_average)
    # right_line=make_coordinates(image,right_fit_average)
    # return np.array([left_line,right_line])
def canny(image):
    gray=cv2.cvtColor(lane_image,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(1,1),0)
    canny=cv2.Canny(blur,100,100)
    return canny
def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            try:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
            except OverflowError:
                None
    return line_image
def region_view(image):
    height=image.shape[0]
    triangle =np.array([[(220,680),(1200,680),(1200,480),(620,430)]])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,triangle,255)
    regionview=cv2.bitwise_and(image,mask)
    return regionview
# def get_vertices_for_img(image):
#     height=image.shape[0]
#     vert =np.array([[(0,500),(0,720),(1200,720),(640,300),(540,300)]])
#     return vert
# def find_lane_lines_formula(lines):
#     xs = []
#     ys = []
#     for line in lines:
#         x1,y1,x2,y2=line.reshape(4)
#         xs.append(x1)
#         xs.append(x2)
#         ys.append(y1)
#         ys.append(y2)
    
#     slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
    
#     # Remember, a straight line is expressed as f(x) = Ax + b. Slope is the A, while intercept is the b
#     return (slope, intercept)
# def trace_lane_line(img, lines, top_y, make_copy=True):
#     A, b = find_lane_lines_formula(lines)
#     vert = get_vertices_for_img(lane_image)

#     img_shape = lane_image.shape
#     bottom_y = img_shape[0] - 1
#     # y = Ax + b, therefore x = (y - b) / A
#     x_to_bottom_y = (bottom_y - b) / A
    
#     top_x_to_y = (top_y - b) / A 
    
#     new_lines = [[[int(x_to_bottom_y), int(bottom_y), int(top_x_to_y), int(top_y)]]]
#     return display_lines(lane_image, new_lines, make_copy=make_copy)


    # vert = get_vertices_for_img(img)
    # region_top_left = vert[0][1]
    
    # full_left_lane_img1 = trace_lane_line(img, left_line1, region_top_left[1], make_copy=True)
    # full_left_lane_img2 = trace_lane_line(full_left_lane_img1, left_line2, region_top_left[1], make_copy=True)
    # full_left_lane_img3 = trace_lane_line(full_left_lane_img2, left_line3, region_top_left[1], make_copy=True)
    # full_left_right_lanes_img1 = trace_lane_line(full_left_lane_img3, right_line1, region_top_left[1], make_copy=False)
    # full_left_right_lanes_img2 = trace_lane_line(full_left_right_lanes_img1, right_line2, region_top_left[1], make_copy=False)
    
    # # image1 * α + image2 * β + λ
    # # image1 and image2 must be the same shape.
    # img_with_lane_weight =  cv2.addWeighted(img, 0.7, full_left_right_lanes_img2, 0.3, 0.0)
    
    # return img_with_lane_weight
cap = cv2.VideoCapture("original.mp4")
# clip = mp.VideoFileClip("original.mp4")
# clip_resized = clip.resize(height=360)
while True:
#image =cv2.imread('test.png')
    ret,image=cap.read()
#image=imutils.resize(image,width=1280)
    lane_image = cv2.GaussianBlur(image, (5,5), 0)
    #=====================================================================================================
    hsv=cv2.cvtColor(lane_image, cv2.COLOR_BGR2HSV)
    low_white = np.array([0, 0, 140])
    up_white = np.array([20, 255, 255])
    mask_white = cv2.inRange(hsv,low_white,up_white)
    canny_image = cv2.Canny(mask_white, 100, 255)
    cropped_image=region_view(canny_image)
    lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    averaged_lines=average_slope_intercept(lane_image,lines)
    line_image=display_lines(lane_image,averaged_lines)
    combo_image=cv2.addWeighted(lane_image,0.8,line_image,1,1)
    #========================================================================================================
    # lane_image=np.copy(image)
    # pts1=np.float32([[0,500],[620,280],[0,720],[1200,720]])
    # pts2=np.float32([[0,0],[400,0],[0,600],[400,600]])
    # array=cv2.getPerspectiveTransform(pts1,pts2)
    # lane_image=cv2.warpPerspective(image,array,(400,600))
    #=======================================================================================
    # canny_image=canny(lane_image)
    # cropped_image=region_view(canny_image)
    # lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    # averaged_lines=average_slope_intercept(lane_image,lines)
    # line_image=display_lines(lane_image,averaged_lines)
    # combo_image=cv2.addWeighted(lane_image,0.8,line_image,1,1)
    #===============================================================================
    #==================================================================================================
    hsv=cv2.cvtColor(combo_image, cv2.COLOR_BGR2HSV)
    low_blue = np.array([100, 150, 0])
    up_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv,low_blue,up_blue)
    canny_blue = cv2.Canny(mask_blue, 100, 255)
    cropped_blue=region_view(canny_blue)
    lines_blue=cv2.HoughLinesP(cropped_blue,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    if lines_blue is not None:
        a=np.zeros((9))
        dem=0
        k=0
        for line in lines_blue:
            x1,y1,x2,y2=line.reshape(4)
            parameters=np.polyfit((x1,x2),(y1,y2),1)
            slope2=parameters[0]
            b=round(slope2)
            for i in range(9):
                if a[i]==b:
                    dem+=1
            if dem==0:
                a[k]=b
                k+=1
            dem=0
    maax=0
    for gt in a:
        if gt!=0:
            maax+=1
    print(maax)
    # print(a)
    try:
        for line2 in lines_blue:
            x1,y1,x2,y2=line2.reshape(4)
            cv2.putText(combo_image,'so lan:'+str(maax), (int(100),int(100)),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,255,255), lineType=cv2.LINE_AA)
    except TypeError:
        None   
        #for g in range(maax):
            #cv2.putText(combo_image, str(g), (int((abs(x2))),int((abs((y1-y2)/2+y2)))),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,255,255), lineType=cv2.LINE_AA)
            #cv2.putText(combo_image,'so lan:'+str(g), (int(100),int(100)),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,255,255), lineType=cv2.LINE_AA)
    cv2.imshow('answer',combo_image)
#cv2.waitKey(0)
# plt.imshow(canny_image)
# plt.show()
    if cv2.waitKey(1) & 0xFFF == ord('q'): #press q to quit
        break
cap.release()
cv2.destroyAllWindows()