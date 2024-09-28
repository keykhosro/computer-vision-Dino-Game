# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 10:35:18 2021

@author: khosro
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt



def human_skin_detector(img):
    (x,y,z)=np.shape(img)
    red=img[:,:,0]
    green=img[:,:,1]
    blue=img[:,:,2]
    skin_1 = red > 85
    skin_2 = cv2.absdiff(red, blue) > 10
    skin_3 = cv2.absdiff( red,green) > 10
    a=np.bitwise_and(skin_1,skin_2)
    skin=np.bitwise_and(a,skin_3)
    return skin


    
#def depth_image(img):
#    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
#    disparity = stereo.compute(imgL,imgR)



img_1 = cv2.imread('WIN_20210109_10_37_02_Pro.jpg')
img_2 = cv2.imread('WIN_20210109_10_58_28_Pro.jpg')

img_3 = cv2.imread('WIN_20210109_21_21_59_Pro.jpg')

skin_mask=human_skin_detector(img_1)
skin_mask = skin_mask.astype(np.uint8)  #convert to an unsigned byte
skin_mask*=255

#print(skin_mask)
cv2.imshow('skin mask', skin_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('skin_mask_easy.jpg',skin_mask)


kernel_1 = np.ones((15,15),np.uint8)
mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN,kernel_1)

cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel_2 = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_2)

cv2.imshow('mask_2', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('final_mask_easy.jpg',closing)

#%%


#contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#        # cc
#cnt = sorted(contours, key=cv2.contourArea, reverse=True)
#        # ROI will be object with biggest contour
#mask_nani = contours[0]
#        # Know the coordinates of the bounding box of the ROI
#x, y, w, h = cv2.boundingRect(mask)
#
#cv2.rectangle(closing,(x,y),(x+w,y+h),(255,0,0),0)
#
#cv2.imshow('mask_nani', closing)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#%%


contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt =  max(contours, key = cv2.contourArea)
x, y, w, h = cv2.boundingRect(cnt)
#cv2.drawContours(closing, [cnt], 0, (0,255,0), 3)

cv2.rectangle(closing,(x,y),(x+w,y+h),(255,0,0),0)

cv2.imshow('mask_2', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()


#%%




cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    skin_mask_w=human_skin_detector(img)
    skin_mask_w = skin_mask_w.astype(np.uint8)
    skin_mask_w*=255
    kernel = np.ones((15,15),np.uint8)
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
        break

cap.release()
cv2.destroyAllWindows()