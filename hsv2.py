import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import moviepy.editor as mp
from datetime import datetime

cap = cv2.VideoCapture("C:/Users/hung phung/Videos/3_làn_đi_ở_2/2_.mp4")
# clip = mp.VideoFileClip("original.mp4")
# clip_resized = clip.resize(height=360)
frame=0
sai_di=0
sai_lan=0
cat=270
while True:
    ret,image=cap.read()
    cao=image.shape[0]
    rong=image.shape[1]
    frame+=1
    if frame==1:
        k=0
        tb=0
        img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for i in range(rong):
            for j in range(cao):
                k=k+img[j,i]
    tb=int(round(k/(cao*rong)))
    #image=imutils.resize(image,width=1280)
    lane_image = cv2.GaussianBlur(image,(5,5),0)
    lane_image=lane_image[cao-cat:,:,:]
    hsv=cv2.cvtColor(lane_image, cv2.COLOR_BGR2HSV)
    low_yellow = np.array([0,78, 157])
    up_yellow = np.array([38, 255, 255])
    mask_yellow = cv2.inRange(hsv,low_yellow,up_yellow)
    result=cv2.bitwise_and(lane_image,lane_image,mask=mask_yellow)
    canny=cv2.Canny(result,100,250)
    lines=cv2.HoughLinesP(canny,2,np.pi/180,100,np.array([]),minLineLength=10,maxLineGap=1)
    b=0
    try:
        for line in lines:
            solan=[]
            solan2=[]
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(lane_image,(x1,y1),(x2,y2),(255,0,0),5)
            b+=1
            parameters=np.polyfit((x1,x2),(y1,y2),1)#y=parameters[0]*x+parameters[1]
            solan.append(parameters[0])
            solan2.append(parameters[1])
            s1=(cat-parameters[1])/parameters[0]#diem tai y=720
            s2=(0-parameters[1])/parameters[0]#tai y=450
            try:
                triangle =np.array([[(int(round(s1-20)),cat),(int(round(s2-20)),0),(rong,0),(rong,cat)]])
            except OverflowError:
                None
            mask=np.zeros_like(lane_image)
            cv2.fillPoly(mask,triangle,255)
            regionview=cv2.bitwise_and(lane_image,mask)
            hsv2=cv2.cvtColor(regionview, cv2.COLOR_BGR2HSV)
            low_white2 = np.array([0,0, tb])
            up_white2 = np.array([180, 50, 255])
            mask_white2 = cv2.inRange(hsv2,low_white2,up_white2)
            result2=cv2.bitwise_and(regionview,regionview,mask=mask_white2)
            result2=cv2.addWeighted(lane_image,0.8,result2,1,1)
            canny2=cv2.Canny(result2,100,250)
            lines2=cv2.HoughLinesP(canny2,2,np.pi/180,100,np.array([]),minLineLength=10,maxLineGap=1)
            dem=0
            try:
                for line1 in lines2:
                    x11,y11,x22,y22=line1.reshape(4)
                    parameters2=np.polyfit((x11,x22),(y11,y22),1)
                    cv2.line(lane_image,(x11,y11),(x22,y22),(255,0,0),5)
                    if abs(parameters2[0])>0.3:
                        for i in range(len(solan)):
                            if abs(parameters2[0]-solan[i])<0.5:
                                dem+=1
                        if dem==0:
                            solan.append(parameters2[0])
                            solan2.append(parameters2[1])
                    danh_gia_lan=len(solan)-1
                    dem_lan=[]
                    try:
                        for d in range(len(solan)):
                            s1=(90-solan2[d])/solan[d]
                            dem_lan.append(s1)
                            cv2.putText(lane_image,'so lan:'+str(len(solan)-1), (int(20),int(20)),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,255,255), lineType=cv2.LINE_AA)
                            minh=min(solan)
                            if solan[d]==minh:
                                s3=(90-solan2[d])/solan[d]
                        dem_lan.sort()
                        for d in range(len(s1)-1):
                            if d!=len(s1):
                                cv2.putText(lane_image,'lan:'+str(d+1), (int(sx[d]),int(90)),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,255,255), lineType=cv2.LINE_AA) 
                        for d in range(len(dem_lan)):
                            if dem_lan[d]==s3:
                                danh_gia_dang_di=d+1
                                cv2.putText(lane_image,'lan dang di:'+str(d+1), (int(20),int(50)),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,255,255), lineType=cv2.LINE_AA)
                    except OverflowError:
                        None
            except TypeError:
                danh_gia_lan=1
                danh_gia_dang_di=1
                cv2.putText(lane_image,'so lan:'+str(1), (int(20),int(20)),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,255,255), lineType=cv2.LINE_AA)
                cv2.putText(lane_image,'lan dang di:'+str(1), (int(20),int(50)),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,255,255), lineType=cv2.LINE_AA)
            if b==1:
                break
    except TypeError:
        image2 = cv2.GaussianBlur(image,(5,5),0)
        lane_image=image2[cao-cat:,:,:]
        low_white = np.array([0,0, tb])
        up_white = np.array([180, 146, 255])
        hsv=cv2.cvtColor(lane_image, cv2.COLOR_BGR2HSV)
        mask_white = cv2.inRange(hsv,low_white,up_white)
        result3=cv2.bitwise_and(lane_image,lane_image,mask=mask_white)
        img2=cv2.cvtColor(result3, cv2.COLOR_BGR2GRAY)
        # triangle =np.array([[(0,0),(0,270),(1280,270),(1280,115)]])
        # mask=np.zeros_like(result2)
        # cv2.fillPoly(mask,triangle,255)
        # regionview=cv2.bitwise_and(lane_image,mask)
        canny2=cv2.Canny(result3,100,250)
        lines2=cv2.HoughLinesP(canny2,2,np.pi/180,100,np.array([]),minLineLength=5,maxLineGap=1)
        heso_a=[]
        heso_b=[]
        sx=[]
        danh_gia_lan=0
        danh_gia_dang_di=0
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
            danh_gia_lan=len(heso_a)-1
            #======================================================================
            # try:
            #     heso_a_am=[]
            #     heso_b_am=[]
            #     heso_a_duong=[]
            #     heso_b_duong=[]
            #     for i in range(len(heso_a)):
            #         if heso_a[i]<0:
            #            heso_a_am.append(heso_a[i])
            #            heso_b_am.append(heso_b[i])
            #         if heso_a[i]>0:
            #            heso_a_duong.append(heso_a[i])
            #            heso_b_duong.append(heso_b[i])
            #     min_duong=min(heso_a_duong)
            #     max_am=max(heso_a_am)
            #     for h in range(len(heso_a_duong)):
            #         if heso_a_duong[h]==min_duong:
            #             ps_trai=heso_b_duong[h]
            #     for h in range(len(heso_a_am)):
            #         if heso_a_am[h]==max_am:
            #             ps_phai=heso_b_am[h]
            #     s_trai=(90-ps_trai)/min_duong
            #     s_phai=(90-ps_phai)/max_am
            #     d=s_trai
            #     tim=[]
            #     for i in range(s_phai-s_trai):
            #         a=img2[90,d]
            #         tim.append(a)
            #         d+=1
            #     dem=0
            #     for d in range(len(tim)):
            #         if tim[d]>170:
            #             if tim[d-1]<30:
            #                dem+=1
            #     print(dem)
            # except ValueError:
            #     None
            #=========================================================
            try:
                for d in range(len(heso_a)):
                    s1=(90-heso_b[d])/heso_a[d]
                    s2=(90-heso_b[d])/heso_a[d]
                    sx.append(s2)
                    cv2.putText(lane_image,'so lan:'+str(len(heso_a)-1), (int(20),int(20)),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,255,255), lineType=cv2.LINE_AA)
                    minh=min(heso_a)
                    if heso_a[d]==minh:
                        s3=(90-heso_b[d])/heso_a[d]
                        # danh_gia_dang_di=d
                        # cv2.putText(lane_image,'lan dang di:'+str(d), (int(20),int(50)),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,255,255), lineType=cv2.LINE_AA)
                sx.sort()
                for d in range(len(sx)-1):
                    if d!=len(sx):
                        cv2.putText(lane_image,'lan:'+str(d+1), (int(sx[d]),int(90)),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,255,255), lineType=cv2.LINE_AA)
                for d in range(len(sx)):
                    if sx[d]==s3:
                        danh_gia_dang_di=d+1
                        cv2.putText(lane_image,'lan dang di:'+str(d+1), (int(20),int(50)),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,255,255), lineType=cv2.LINE_AA)
            except  OverflowError:
                None
        except TypeError:
            danh_gia_lan=len(heso_a)-1
            danh_gia_dang_di=d
            cv2.putText(lane_image,'so lan:'+str(1), (int(20),int(20)),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,255,255), lineType=cv2.LINE_AA)
            cv2.putText(lane_image,'lan dang di:'+str(1), (int(20),int(50)),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,255,255), lineType=cv2.LINE_AA)
        if danh_gia_lan!=3:
            sai_lan+=1
        if danh_gia_dang_di!=2:
            sai_di+=1
    cv2.imshow('hung',lane_image)
    # cv2.imshow('hung2',result2)
    # combo_image = cv2.addWeighted(lane_image,0.8,regionview,1,1)
    if cv2.waitKey(1) & 0xFFF == ord('q'): #press q to quit
        break
# print(sai_lan,sai_di,frame)
cap.release()
cv2.destroyAllWindows()