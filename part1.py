# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 10:35:18 2021

@author: khosro
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import keyboard


def human_skin_detector(img,minr,ming,minb,maxr,maxg,maxb,var ):
    (x,y,z)=np.shape(img)
    red=img[:,:,0]
    green=img[:,:,1]
    blue=img[:,:,2]
    
#    skin_11 =red > (min(sample_skin[:,:,0]) - var)
#    skin_12=red < (max(sample_skin[:,:,0]) +var )
#    skin_1 =numpy.logical_and(skin_11,skin_12)
#    skin_21 = green > (min(sample_skin[:,:,1]) - var)
#    skin_22= green < (max(sample_skin[:,:,1]) +var )
#    skin_2 =numpy.logical_and(skin_21,skin_22)
#    skin_31 =blue > (min(sample_skin[:,:,2]) - var)
#    skin_32=blue < (max(sample_skin[:,:,2]) +var) 
#    skin_3 =numpy.logical_and(skin_31,skin_32)

    skin_1 = np.bitwise_and(red>minr - var , maxr +var > red)
    skin_2 = np.bitwise_and(green> ming - var , maxg +var > green)
    skin_3 = np.bitwise_and(blue > minb - var,maxb +var > blue)
    a=np.bitwise_and(skin_1,skin_2)
    skin=np.bitwise_and(a,skin_3)
    return skin


    
#def depth_image(img):
#    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
#    disparity = stereo.compute(imgL,imgR)


#%%




cap = cv2.VideoCapture(0)

var=7
s=0
x=100
y=100
w=100
h=100
while 1:
    ret, img = cap.read()
    if s==0:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        if keyboard.is_pressed('s'):
            s=1
            sample_skin=img[x:x+w,y:y+h,:]
            minr=np.min(sample_skin[:,:,0])
            ming=np.min(sample_skin[:,:,1])
            minb=np.min(sample_skin[:,:,2])
            maxr=np.max(sample_skin[:,:,0])
            maxg=np.max(sample_skin[:,:,1])
            maxb=np.max(sample_skin[:,:,2])
            
        closing_w=img
        
    else:
        skin_mask_w=human_skin_detector(img,minr,ming,minb,maxr,maxg,maxb,var)
        skin_mask_w = skin_mask_w.astype(np.uint8)
        skin_mask_w*=255
        kernel = np.ones((11,11),np.uint8)
        mask_w = cv2.morphologyEx(skin_mask_w, cv2.MORPH_OPEN,kernel)
        
        kernel_2 = np.ones((3,3),np.uint8)
        closing_w = cv2.morphologyEx(mask_w , cv2.MORPH_CLOSE, kernel_2)
        
        contours, hierarchy = cv2.findContours(closing_w, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            cnt =  max(contours, key = cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
        #cv2.drawContours(closing, [cnt], 0, (0,255,0), 3)

            cv2.rectangle(closing_w,(x,y),(x+w,y+h),(255,0,0),0)
        
    cv2.imshow('img',closing_w )

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        crop=closing_w[y+1:y+h,x+1:x+w]
        break

cap.release()
cv2.destroyAllWindows()



cv2.imshow('sample_skin', sample_skin)
cv2.imshow('crop', crop)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imwrite('sample_skin.jpg',sample_skin)
cv2.imwrite('crop.jpg',crop)